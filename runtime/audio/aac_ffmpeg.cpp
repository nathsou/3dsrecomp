// aac_decoder.h with FFmpeg's libavcodec (its native AAC decoder), given the whole ADTS frames.
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/channel_layout.h>
#include <libavutil/frame.h>
}

#include <algorithm>
#include <cmath>

#include "aac_decoder.h"

namespace audio::hle {

namespace {
AVCodecContext *ctx;
AVPacket *pkt;
AVFrame *frame;
int cur_rate, cur_channels;

s16 to_s16(float v) { return (s16)std::clamp(std::lrintf(v * 32768.0f), -32768l, 32767l); }

// appends a decoded frame's samples, interleaved
void take(const AVFrame *f, int channels, std::vector<s16> &pcm) {
    const int n = f->nb_samples, fch = f->ch_layout.nb_channels;
    const size_t at = pcm.size();
    pcm.resize(at + (size_t)n * channels, 0);
    for (int c = 0; c < std::min(channels, fch); c++)
        for (int i = 0; i < n; i++) {
            s16 v = 0;
            switch (f->format) {
            case AV_SAMPLE_FMT_FLTP: v = to_s16(((const float *)f->extended_data[c])[i]); break;
            case AV_SAMPLE_FMT_FLT: v = to_s16(((const float *)f->extended_data[0])[i * fch + c]); break;
            case AV_SAMPLE_FMT_S16P: v = ((const s16 *)f->extended_data[c])[i]; break;
            case AV_SAMPLE_FMT_S16: v = ((const s16 *)f->extended_data[0])[i * fch + c]; break;
            case AV_SAMPLE_FMT_S32P: v = (s16)(((const int32_t *)f->extended_data[c])[i] >> 16); break;
            default: break;
            }
            pcm[at + (size_t)i * channels + c] = v;
        }
}
}  // namespace

void aac_close() {
    avcodec_free_context(&ctx);
    cur_rate = cur_channels = 0;
}

bool aac_open(int rate, int channels) {
    if (ctx && rate == cur_rate && channels == cur_channels) return true;
    aac_close();
    const AVCodec *codec = avcodec_find_decoder(AV_CODEC_ID_AAC);
    if (codec) ctx = avcodec_alloc_context3(codec);
    if (!ctx || avcodec_open2(ctx, codec, nullptr) < 0) {
        LOG("dsp: AAC decoder unavailable (%d Hz, %d channels)\n", rate, channels);
        avcodec_free_context(&ctx);
        return false;
    }
    if (!pkt) pkt = av_packet_alloc();
    if (!frame) frame = av_frame_alloc();
    cur_rate = rate;
    cur_channels = channels;
    return true;
}

std::vector<s16> aac_decode(const std::vector<AacFrame> &frames) {
    std::vector<s16> pcm;
    if (!ctx) return pcm;
    for (const AacFrame &f : frames) {
        pkt->data = (uint8_t *)f.adts;  // (ADTS: the decoder reads the stream's parameters from each header)
        pkt->size = (int)f.size;
        int st = avcodec_send_packet(ctx, pkt);
        if (st < 0) {
            LOG("dsp: AAC decode error %d\n", st);
            break;
        }
        while ((st = avcodec_receive_frame(ctx, frame)) == 0) {
            take(frame, cur_channels, pcm);
            av_frame_unref(frame);
        }
    }
    av_packet_unref(pkt);
    return pcm;
}

}  // namespace audio::hle
