// Per-source resampling: none (zero-order hold), linear, and "polyphase" (approximated with a
// 4-tap Catmull-Rom cubic, since the firmware's FIR table is unknown).
#include <algorithm>
#include <cmath>

#include "dsp_hle.h"

namespace audio::hle {

namespace {
constexpr int kFracBits = 24;
constexpr u64 kOne = 1ull << kFracBits;
constexpr u64 kFracMask = kOne - 1;

u64 step_of(float rate) { return (u64)(rate * (float)kOne); }

s16 lerp(s16 x0, s16 x1, u64 frac) {
    // Saturated delta, result rounded towards -inf (matches observed firmware within +/-1).
    s64 delta = std::clamp<s64>((s64)x1 - x0, -32768, 32767);
    return (s16)(x0 + (((s64)frac * delta) >> kFracBits));
}

s16 cubic(s16 xm1, s16 x0, s16 x1, s16 x2, u64 frac) {
    const float t = (float)frac * (1.0f / (float)kOne);
    const float a = xm1, b = x0, c = x1, d = x2;
    const float v = b + 0.5f * t * (c - a + t * (2.0f * a - 5.0f * b + 4.0f * c - d +
                                                  t * (3.0f * (b - c) + d - a)));
    return (s16)std::clamp<long>(std::lround(v), -32768, 32767);
}
}  // namespace

size_t interp_input_needed(const InterpState &st, float rate, size_t outputs) {
    if (outputs == 0) return 0;
    const u64 last = st.fpos + (u64)(outputs - 1) * step_of(rate);
    return (size_t)(last >> kFracBits) + 1;
}

size_t interpolate(InterpState &st, Interp mode, float rate, const Sample *in, size_t n, Sample *out,
                   size_t &outi, size_t outn) {
    if (n == 0) return 0;
    // Virtual input: v(-1) = x[n-3], v(0) = x[n-2], v(1) = x[n-1], v(i) = in[i - 2].
    auto v = [&](s64 i) -> const Sample & {
        if (i >= 2) return in[i - 2];
        if (i == 1) return st.xn1;
        if (i == 0) return st.xn2;
        return st.xn3;
    };
    const size_t size = n + 2;
    const u64 step = step_of(rate);
    u64 fpos = st.fpos;
    size_t ip = 0;
    while (outi < outn) {
        ip = (size_t)(fpos >> kFracBits);
        if (ip + 2 >= size) {
            ip = size - 2;
            break;
        }
        const u64 frac = fpos & kFracMask;
        const Sample &x0 = v(ip), &x1 = v(ip + 1);
        Sample &o = out[outi++];
        switch (mode) {
        case Interp::None:
            o = x0;
            break;
        case Interp::Linear:
            o = {lerp(x0[0], x1[0], frac), lerp(x0[1], x1[1], frac)};
            break;
        case Interp::Polyphase:
        default: {
            const Sample &xm1 = v((s64)ip - 1), &x2 = v(ip + 2);
            o = {cubic(xm1[0], x0[0], x1[0], x2[0], frac), cubic(xm1[1], x0[1], x1[1], x2[1], frac)};
            break;
        }
        }
        fpos += step;
    }
    const Sample n3 = v((s64)ip - 1), n2 = v(ip), n1 = v(ip + 1);
    st.xn3 = n3;
    st.xn2 = n2;
    st.xn1 = n1;
    st.fpos = fpos - (u64)ip * kOne;
    return ip;
}

}  // namespace audio::hle
