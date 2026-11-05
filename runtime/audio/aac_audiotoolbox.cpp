// aac_decoder.h with AudioToolbox (Apple).
#include <AudioToolbox/AudioToolbox.h>

#include "aac_decoder.h"

namespace audio::hle {

namespace {

constexpr u32 kFramesPerPacket = 1024;

struct Decoder {
    AudioConverterRef conv = nullptr;
    int rate = 0, channels = 0;
    const std::vector<AacFrame> *feed = nullptr;
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
        const AacFrame &f = (*d.feed)[d.next++];
        io->mNumberBuffers = 1;
        io->mBuffers[0].mData = (void *)(f.adts + f.head);  // raw AAC, after the ADTS header
        io->mBuffers[0].mDataByteSize = f.size - f.head;
        io->mBuffers[0].mNumberChannels = d.channels;
        d.desc = {0, 0, f.size - f.head};
        if (descs) *descs = &d.desc;
        *packets = 1;
        return noErr;
    }

    // Decodes the frames into interleaved samples.
    std::vector<s16> decode(const std::vector<AacFrame> &frames) {
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

}  // namespace

bool aac_open(int rate, int channels) { return g_aac.open(rate, channels); }
std::vector<s16> aac_decode(const std::vector<AacFrame> &frames) { return g_aac.decode(frames); }
void aac_close() { g_aac.close(); }

}  // namespace audio::hle
