// The DSP firmware's AAC decoder (the binary pipe, 3): the application sends 32-byte requests (init, decode a
// buffer of ADTS frames into two planar 16-bit channels, shutdown) and reads a 32-byte answer from the same pipe.
// Games stream music this way (Picross 3D: Round 2's sound/stream/*.aac). Decoded with AudioToolbox.
#include <AudioToolbox/AudioToolbox.h>

#include <algorithm>
#include <cstring>
#include <vector>

#include "dsp_hle.h"

namespace audio::hle {

namespace {

enum : u16 { kCodecAacDecode = 1 };
enum : u16 { kCmdInit = 0, kCmdDecode = 1, kCmdShutdown = 2, kCmdSaveState = 3, kCmdLoadState = 4 };

#pragma pack(push, 1)
struct BinaryMessage {
    u16 codec, cmd;
    u32 result;
    union {
        u8 data[24];
        struct {
            u32 src, size, dst0, dst1, unknown[2];  // physical addresses (FCRAM)
        } request;
        struct {
            u32 rate, channels, size, unknown[2], samples;  // rate as the DSP's enum; samples per channel
        } response;
    };
};
#pragma pack(pop)
static_assert(sizeof(BinaryMessage) == 32);

constexpr u32 kAdtsSyncMask = 0xFFF6, kAdtsSync = 0xFFF0;  // 12 sync bits, layer 0
constexpr u32 kAdtsHeader = 7, kAdtsCrc = 2;
constexpr u32 kFramesPerPacket = 1024;
constexpr int kAdtsRates[16] = {96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000, 7350};
// the firmware's sample rate numbers
constexpr int kDspRates[] = {48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000};

struct Frame {
    const u8 *data;  // raw AAC, after the ADTS header
    u32 size;
};

struct Decoder {
    AudioConverterRef conv = nullptr;
    int rate = 0, channels = 0;
    const std::vector<Frame> *feed = nullptr;
    size_t next = 0;
    AudioStreamPacketDescription desc{};

    void close() {
        if (conv) AudioConverterDispose(conv);
        conv = nullptr;
        rate = channels = 0;
    }

    bool open(int r, int ch) {
        if (conv && r == rate && ch == channels) return true;
        close();
        AudioStreamBasicDescription in{}, out{};
        in.mSampleRate = r;
        in.mFormatID = kAudioFormatMPEG4AAC;
        in.mFramesPerPacket = kFramesPerPacket;
        in.mChannelsPerFrame = ch;
        out.mSampleRate = r;
        out.mFormatID = kAudioFormatLinearPCM;
        out.mFormatFlags = kAudioFormatFlagIsSignedInteger | kAudioFormatFlagIsPacked;
        out.mBitsPerChannel = 16;
        out.mChannelsPerFrame = ch;
        out.mBytesPerFrame = out.mBytesPerPacket = 2 * ch;
        out.mFramesPerPacket = 1;
        if (AudioConverterNew(&in, &out, &conv) != noErr) {
            LOG("dsp: AAC decoder unavailable (%d Hz, %d channels)\n", r, ch);
            conv = nullptr;
            return false;
        }
        rate = r;
        channels = ch;
        return true;
    }

    static OSStatus input(AudioConverterRef, UInt32 *packets, AudioBufferList *io, AudioStreamPacketDescription **descs,
                          void *user) {
        Decoder &d = *(Decoder *)user;
        if (d.next >= d.feed->size()) {
            *packets = 0;
            return 1;  // no more input for this call (not an error: the converter returns what it has)
        }
        const Frame &f = (*d.feed)[d.next++];
        io->mNumberBuffers = 1;
        io->mBuffers[0].mData = (void *)f.data;
        io->mBuffers[0].mDataByteSize = f.size;
        io->mBuffers[0].mNumberChannels = d.channels;
        d.desc = {0, 0, f.size};
        if (descs) *descs = &d.desc;
        *packets = 1;
        return noErr;
    }

    // Decodes the frames into interleaved samples.
    std::vector<s16> decode(const std::vector<Frame> &frames) {
        std::vector<s16> pcm;
        feed = &frames;
        next = 0;
        std::vector<s16> buf(kFramesPerPacket * channels);
        while (next < frames.size()) {
            UInt32 n = kFramesPerPacket;
            AudioBufferList out{};
            out.mNumberBuffers = 1;
            out.mBuffers[0].mNumberChannels = channels;
            out.mBuffers[0].mDataByteSize = (UInt32)(buf.size() * sizeof(s16));
            out.mBuffers[0].mData = buf.data();
            OSStatus st = AudioConverterFillComplexBuffer(conv, input, this, &n, &out, nullptr);
            pcm.insert(pcm.end(), buf.begin(), buf.begin() + n * channels);
            if (st != noErr && st != 1) {
                LOG("dsp: AAC decode error %d\n", (int)st);
                break;
            }
        }
        feed = nullptr;
        return pcm;
    }
};

Decoder g_aac;

u32 dsp_rate(int rate) {
    for (u32 i = 0; i < sizeof(kDspRates) / sizeof(kDspRates[0]); i++)
        if (kDspRates[i] == rate) return i;
    return 0;
}

BinaryMessage decode(const BinaryMessage &req) {
    BinaryMessage res{};
    res.codec = req.codec;
    res.cmd = req.cmd;
    res.response.size = req.request.size;
    const u8 *src = phys_range(req.request.src, req.request.size);
    if (!src) {
        LOG("dsp: AAC decode of a bad buffer %08x+%x\n", req.request.src, req.request.size);
        return res;
    }
    // split the ADTS stream into raw frames
    std::vector<Frame> frames;
    int rate = 0, channels = 0;
    for (u32 o = 0; o + kAdtsHeader <= req.request.size;) {
        const u8 *h = src + o;
        if ((((u32)h[0] << 8 | h[1]) & kAdtsSyncMask) != kAdtsSync) {
            o++;
            continue;
        }
        const bool crc = !(h[1] & 1);
        const int rate_index = (h[2] >> 2) & 0xF;
        const int ch = ((h[2] & 1) << 2) | (h[3] >> 6);
        const u32 len = ((u32)(h[3] & 3) << 11) | ((u32)h[4] << 3) | (h[5] >> 5);
        const u32 head = kAdtsHeader + (crc ? kAdtsCrc : 0);
        if (len <= head || o + len > req.request.size || !kAdtsRates[rate_index] || !ch) break;
        rate = kAdtsRates[rate_index];
        channels = ch;
        frames.push_back({h + head, len - head});
        o += len;
    }
    if (frames.empty() || !g_aac.open(rate, channels)) return res;
    const std::vector<s16> pcm = g_aac.decode(frames);
    const u32 samples = (u32)(pcm.size() / channels);
    for (int c = 0; c < std::min(channels, 2); c++) {
        u8 *dst = (u8 *)phys_range(c ? req.request.dst1 : req.request.dst0, (u64)samples * sizeof(s16));
        if (!dst) {
            continue;
        }
        for (u32 i = 0; i < samples; i++) {
            const s16 v = pcm[i * channels + c];
            memcpy(dst + i * 2, &v, 2);
        }
    }
    res.response.rate = dsp_rate(rate);
    res.response.channels = channels;
    res.response.samples = samples;
    return res;
}

}  // namespace

bool aac_request(const u8 *data, u32 size, std::vector<u8> &answer) {
    BinaryMessage req;
    if (size != sizeof(req)) {
        LOG("dsp: binary pipe write of %u bytes ignored\n", size);
        return false;
    }
    memcpy(&req, data, sizeof(req));
    if (req.codec != kCodecAacDecode) {
        LOG("dsp: binary pipe request for codec %u ignored\n", req.codec);
        return false;
    }
    BinaryMessage res = req;
    switch (req.cmd) {
    case kCmdDecode:
        res = decode(req);
        break;
    case kCmdInit:
    case kCmdShutdown:
        g_aac.close();
        [[fallthrough]];
    case kCmdSaveState:
    case kCmdLoadState:
        res.result = 0;
        break;
    default:
        LOG("dsp: AAC decoder command %u ignored\n", req.cmd);
        return false;
    }
    answer.resize(sizeof(res));
    memcpy(answer.data(), &res, sizeof(res));
    return true;
}

}  // namespace audio::hle
