// Sample decoders of the DSP firmware: GC-ADPCM, PCM8, PCM16.
#include <algorithm>

#include "dsp_hle.h"

namespace audio::hle {

const u8 *phys_range(u32 paddr, u64 bytes) {
    struct Region { u32 pa, size, va; };
    static constexpr Region regions[] = {
        {mem::FCRAM_PADDR, mem::LINEAR_SIZE, mem::LINEAR_BASE},
        {mem::VRAM_PADDR, mem::VRAM_SIZE, mem::VRAM_BASE},
        {mem::DSP_PADDR, mem::DSP_SIZE, mem::DSP_BASE},
    };
    for (const Region &r : regions) {
        if (paddr >= r.pa && paddr - r.pa < r.size) {
            if ((u64)(paddr - r.pa) + bytes > r.size) return nullptr;
            return g_mem + r.va + (paddr - r.pa);
        }
    }
    return nullptr;
}

void decode_adpcm(const u8 *data, size_t count, const s16 coeffs[16], AdpcmState &st, Sample *out) {
    constexpr size_t kFrameBytes = 8, kFrameSamples = 14;
    int yn1 = st.yn1, yn2 = st.yn2;
    size_t outi = 0;
    for (size_t frame = 0; outi < count; frame++) {
        const u8 *f = data + frame * kFrameBytes;
        const int scale = 1 << (f[0] & 0xF);
        const int pred = (f[0] >> 4) & 7;
        const int c1 = coeffs[pred * 2], c2 = coeffs[pred * 2 + 1];
        for (size_t i = 0; i < kFrameSamples && outi < count; i++) {
            int nib = (i & 1) ? (f[1 + i / 2] & 0xF) : (f[1 + i / 2] >> 4);
            if (nib >= 8) nib -= 16;
            // y[n] = x[n] * scale + c1 y[n-1] + c2 y[n-2], coefficients in Q11, rounded
            s64 acc = ((s64)(nib * scale) << 11) + 0x400 + (s64)c1 * yn1 + (s64)c2 * yn2;
            int v = (int)std::clamp<s64>(acc >> 11, -32768, 32767);
            yn2 = yn1;
            yn1 = v;
            out[outi][0] = out[outi][1] = (s16)v;
            outi++;
        }
    }
    st.yn1 = (s16)yn1;
    st.yn2 = (s16)yn2;
}

void decode_pcm8(unsigned channels, const u8 *data, size_t count, Sample *out) {
    for (size_t i = 0; i < count; i++) {
        if (channels == 2) {
            out[i][0] = (s16)(data[2 * i] << 8);
            out[i][1] = (s16)(data[2 * i + 1] << 8);
        } else {
            out[i][0] = out[i][1] = (s16)(data[i] << 8);
        }
    }
}

void decode_pcm16(unsigned channels, const u8 *data, size_t count, Sample *out) {
    if (channels == 2) {
        memcpy(out, data, count * 4);
    } else {
        for (size_t i = 0; i < count; i++) {
            s16 v;
            memcpy(&v, data + 2 * i, 2);
            out[i][0] = out[i][1] = v;
        }
    }
}

}  // namespace audio::hle
