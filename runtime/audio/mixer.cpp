// Final mixer: aux bus send/return through shared memory, delay effect, volumes, quad -> stereo.
#include <algorithm>

#include "dsp_hle.h"

namespace audio::hle {

namespace {
s16 clamp16(s32 v) { return (s16)std::clamp(v, -32768, 32767); }
s32 f2s32(float v) { return (s32)std::clamp(v, -2147483648.0f, 2147483520.0f); }
constexpr size_t kMaxDelaySamples = kFrameSamples * 64;
}  // namespace

void Mixer::Delay::parse(DelayConfig &c) {
    if (!c.dirty) return;
    const bool was = enabled;
    enabled = c.enable != 0;
    c.dirty = 0;
    size_t samples = (size_t)c.frame_count * kFrameSamples;
    if (!enabled || samples == 0) {
        if (was || samples == 0) *this = Delay{};
        return;
    }
    samples = std::min(samples, kMaxDelaySamples);
    g = c.g / 128.0f;
    a = c.a / 128.0f;
    b = c.b / 128.0f;
    if (line[0].size() != samples) {
        for (auto &l : line) l.assign(samples, 0.0f);
        last = {};
        pos = 0;
    }
}

void Mixer::Delay::process(QuadFrame &f) {
    if (!enabled || line[0].empty()) return;
    const size_t n = line[0].size();
    for (int i = 0; i < kFrameSamples; i++) {
        const size_t p = (pos + i) % n;
        for (int c = 0; c < 4; c++) {
            // y[n] = b y[n-1] + a v[n-N],  v[n] = x[n] - g y[n]
            const float y = b * last[c] + a * line[c][p];
            line[c][p] = (float)f[i][c] - g * y;
            last[c] = y;
            f[i][c] = f2s32(y);
        }
    }
    pos = (pos + kFrameSamples) % n;
}

void Mixer::tick(DspConfig &cfg, const IntermediateMix (&read)[2], IntermediateMix (&write)[2],
                 const std::array<QuadFrame, 3> &input) {
    // configuration
    if (const u32 d = cfg.dirty) {
        if (d & CFG_AUX_BUS_ENABLE0_DIRTY) st.aux_enable[0] = cfg.aux_bus_enable[0] != 0;
        if (d & CFG_AUX_BUS_ENABLE1_DIRTY) st.aux_enable[1] = cfg.aux_bus_enable[1] != 0;
        if (d & CFG_MASTER_VOLUME_DIRTY) st.volume[0] = cfg.master_volume;
        if (d & CFG_AUX_RETURN_VOLUME0_DIRTY) st.volume[1] = cfg.aux_return_volume[0];
        if (d & CFG_AUX_RETURN_VOLUME1_DIRTY) st.volume[2] = cfg.aux_return_volume[1];
        if (d & CFG_OUTPUT_FORMAT_DIRTY) st.output_format = cfg.output_format;
        if (d & CFG_DELAY0_DIRTY) st.delay[0].parse(cfg.delay[0]);
        if (d & CFG_DELAY1_DIRTY) st.delay[1].parse(cfg.delay[1]);
        // reverb, surround, clipping/limiter, headphones: not emulated
        cfg.dirty = 0;
    }

    // Aux return: the application's processed samples (from the region it just submitted).
    for (int bus = 0; bus < 2; bus++) {
        if (!st.aux_enable[bus]) continue;
        for (int i = 0; i < kFrameSamples; i++)
            for (int c = 0; c < 4; c++) st.mix[bus + 1][i][c] = read[bus].pcm32[c][i];
    }
    // Aux send: hand this frame's mixes 1/2 to the application, or route them directly.
    st.mix[0] = input[0];
    for (int bus = 0; bus < 2; bus++) {
        if (st.aux_enable[bus]) {
            for (int i = 0; i < kFrameSamples; i++)
                for (int c = 0; c < 4; c++) write[bus].pcm32[c][i] = input[bus + 1][i][c];
        } else {
            st.mix[bus + 1] = input[bus + 1];
        }
    }

    for (int bus = 0; bus < 2; bus++) st.delay[bus].process(st.mix[bus + 1]);

    // Final mix (no limiter/compressor).
    frame.fill({});
    for (int m = 0; m < 3; m++) {
        const float g = st.volume[m];
        const QuadFrame &q = st.mix[m];
        for (int i = 0; i < kFrameSamples; i++) {
            s16 l, r;
            if (st.output_format == 0) {  // mono
                l = r = clamp16(f2s32((g * q[i][0] + g * q[i][1] + g * q[i][2] + g * q[i][3]) / 2));
            } else {  // stereo (surround is downmixed the same way)
                l = clamp16(f2s32(g * q[i][0] + g * q[i][2]));
                r = clamp16(f2s32(g * q[i][1] + g * q[i][3]));
            }
            frame[i][0] = clamp16(frame[i][0] + l);
            frame[i][1] = clamp16(frame[i][1] + r);
        }
    }
}

}  // namespace audio::hle
