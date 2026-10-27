// Unit tests for the DSP firmware HLE (runtime/audio/).
//   make && ./audio_test          run the tests (no audio device needed)
//   ./audio_test --play           also play a 2 s test tone through SDL (manual listening check)
//   ./audio_test --play-silent    20 s of silence through the real device, printing queue levels
#include <sys/mman.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <thread>
#include <vector>

#include "audio/dsp_hle.h"
#include "audio/host.h"
#include "audio_iface.h"

// ---- runtime symbols the audio code needs
extern "C" {
u8 *g_mem;
}
int g_verbose = 0;
void rt_log(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
}
void rt_fatal(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    abort();
}

using namespace audio::hle;

static int g_failures = 0, g_checks = 0;
#define CHECK(cond)                                                                  \
    do {                                                                             \
        g_checks++;                                                                  \
        if (!(cond)) {                                                               \
            g_failures++;                                                            \
            fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
        }                                                                            \
    } while (0)
#define CHECK_EQ(a, b)                                                                          \
    do {                                                                                        \
        g_checks++;                                                                             \
        long long _a = (long long)(a), _b = (long long)(b);                                     \
        if (_a != _b) {                                                                         \
            g_failures++;                                                                       \
            fprintf(stderr, "%s:%d: CHECK_EQ failed: %s (%lld) != %s (%lld)\n", __FILE__, __LINE__, \
                    #a, _a, #b, _b);                                                            \
        }                                                                                       \
    } while (0)

// ---------------------------------------------------------------- ADPCM

static const s16 kCoeffs[16] = {0, 0, 2048, 0, 3840, -1920, 3072, -1024,
                                3600, -1700, 1024, 0, 4000, -2000, 2560, -512};

// Straightforward reference decoder (one nibble at a time over the byte stream).
static std::vector<s16> ref_adpcm(const std::vector<u8> &data, size_t count, const s16 *coef, int y1,
                                  int y2) {
    std::vector<s16> out;
    for (size_t n = 0; n < count; n++) {
        size_t frame = n / 14, k = n % 14;
        u8 hdr = data[frame * 8];
        u8 byte = data[frame * 8 + 1 + k / 2];
        int nib = (k % 2 == 0) ? byte >> 4 : byte & 15;
        if (nib & 8) nib -= 16;
        long long x = (long long)nib * (1 << (hdr & 15)) * 2048;
        long long acc = x + 1024 + (long long)coef[2 * ((hdr >> 4) & 7)] * y1 +
                        (long long)coef[2 * ((hdr >> 4) & 7) + 1] * y2;
        long long v = (long long)std::floor((double)acc / 2048.0);
        v = std::max(-32768LL, std::min(32767LL, v));
        y2 = y1;
        y1 = (int)v;
        out.push_back((s16)v);
    }
    return out;
}

// Brute-force GC-ADPCM encoder (for round-trip tests).
static std::vector<u8> encode_adpcm(const std::vector<s16> &pcm, const s16 *coef) {
    std::vector<u8> out;
    int y1 = 0, y2 = 0;
    for (size_t base = 0; base < pcm.size(); base += 14) {
        double best_err = 1e300;
        std::vector<u8> best;
        int best_y1 = 0, best_y2 = 0;
        for (int pred = 0; pred < 8; pred++) {
            for (int sc = 0; sc < 13; sc++) {
                std::vector<u8> fr(8, 0);
                fr[0] = (u8)(pred << 4 | sc);
                int a = y1, b = y2;
                double err = 0;
                for (int k = 0; k < 14; k++) {
                    int target = base + k < pcm.size() ? pcm[base + k] : 0;
                    int best_nib = 0, best_v = 0;
                    long long be = -1;
                    for (int nib = -8; nib < 8; nib++) {
                        long long acc = ((long long)nib * (1 << sc) << 11) + 1024 +
                                        (long long)coef[2 * pred] * a + (long long)coef[2 * pred + 1] * b;
                        int v = (int)std::max(-32768LL, std::min(32767LL, acc >> 11));
                        long long e = (long long)(v - target) * (v - target);
                        if (be < 0 || e < be) be = e, best_nib = nib, best_v = v;
                    }
                    err += be;
                    fr[1 + k / 2] |= (u8)((best_nib & 15) << (k % 2 ? 0 : 4));
                    b = a;
                    a = best_v;
                }
                if (err < best_err) best_err = err, best = fr, best_y1 = a, best_y2 = b;
            }
        }
        out.insert(out.end(), best.begin(), best.end());
        y1 = best_y1;
        y2 = best_y2;
    }
    return out;
}

static void test_adpcm() {
    // Hand-computed: scale 1, predictor 0 (coefficients 0): y = x (with rounding of +0.5).
    {
        u8 frame[8] = {0x00, 0x1F, 0x70, 0x80, 0, 0, 0, 0};
        Sample out[6];
        AdpcmState st;
        decode_adpcm(frame, 6, kCoeffs, st, out);
        CHECK_EQ(out[0][0], 1);
        CHECK_EQ(out[1][0], -1);
        CHECK_EQ(out[2][0], 7);
        CHECK_EQ(out[3][0], 0);
        CHECK_EQ(out[4][0], -8);
        CHECK_EQ(out[0][1], out[0][0]);  // mono duplicated
        CHECK_EQ(st.yn1, 0);
        CHECK_EQ(st.yn2, -8);
    }
    // Predictor 1 (c1 = 1.0): integrator; scale 4 (shift 2).
    {
        u8 frame[8] = {0x12, 0x11, 0x11, 0xF0, 0, 0, 0, 0};
        Sample out[6];
        AdpcmState st{100, 0};
        decode_adpcm(frame, 6, kCoeffs, st, out);
        CHECK_EQ(out[0][0], 104);
        CHECK_EQ(out[3][0], 116);
        CHECK_EQ(out[4][0], 112);  // nibble F = -1 -> -4
        CHECK_EQ(out[5][0], 112);
    }
    // Clamping at the s16 limits.
    {
        u8 frame[8] = {0x1C, 0x77, 0x77, 0x88, 0x88, 0, 0, 0};
        Sample out[8];
        AdpcmState st{32000, 32000};
        decode_adpcm(frame, 8, kCoeffs, st, out);
        CHECK_EQ(out[0][0], 32767);
        CHECK_EQ(out[3][0], 32767);
        CHECK_EQ(out[7][0], -32768);
    }
    // Random streams against the reference decoder, including partial frames and history.
    {
        u32 seed = 12345;
        auto rnd = [&] { return seed = seed * 1664525 + 1013904223, seed >> 8; };
        for (int iter = 0; iter < 50; iter++) {
            size_t count = 1 + rnd() % 500;
            std::vector<u8> data((count + 13) / 14 * 8);
            for (auto &b : data) b = (u8)rnd();
            int y1 = (s16)rnd(), y2 = (s16)rnd();
            std::vector<s16> ref = ref_adpcm(data, count, kCoeffs, y1, y2);
            std::vector<Sample> out(count);
            AdpcmState st{(s16)y1, (s16)y2};
            decode_adpcm(data.data(), count, kCoeffs, st, out.data());
            size_t mismatches = 0;
            for (size_t i = 0; i < count; i++) mismatches += out[i][0] != ref[i] || out[i][1] != ref[i];
            CHECK_EQ(mismatches, 0);
            CHECK_EQ(st.yn1, ref[count - 1]);
            if (count > 1) CHECK_EQ(st.yn2, ref[count - 2]);
        }
    }
    // Round trip of a sine through the encoder: decoding must be faithful (high SNR) and decoding
    // in two pieces (state carried over) must equal decoding at once.
    {
        std::vector<s16> pcm(14 * 200);
        for (size_t i = 0; i < pcm.size(); i++) pcm[i] = (s16)(12000 * std::sin(i * 2 * M_PI * 440 / 32728));
        std::vector<u8> enc = encode_adpcm(pcm, kCoeffs);
        std::vector<Sample> dec(pcm.size());
        AdpcmState st;
        decode_adpcm(enc.data(), pcm.size(), kCoeffs, st, dec.data());
        double sig = 0, noise = 0;
        for (size_t i = 0; i < pcm.size(); i++) {
            sig += (double)pcm[i] * pcm[i];
            noise += (double)(pcm[i] - dec[i][0]) * (pcm[i] - dec[i][0]);
        }
        double snr = 10 * std::log10(sig / std::max(noise, 1.0));
        printf("  adpcm round trip SNR: %.1f dB\n", snr);
        CHECK(snr > 30);
        std::vector<Sample> a(pcm.size());
        AdpcmState s2;
        decode_adpcm(enc.data(), 14 * 77, kCoeffs, s2, a.data());
        decode_adpcm(enc.data() + 77 * 8, pcm.size() - 14 * 77, kCoeffs, s2, a.data() + 14 * 77);
        bool same = true;
        for (size_t i = 0; i < pcm.size(); i++) same &= a[i] == dec[i];
        CHECK(same);
    }
}

static void test_pcm() {
    u8 d8[4] = {0x00, 0x7F, 0x80, 0xFF};
    Sample o[4];
    decode_pcm8(1, d8, 4, o);
    CHECK_EQ(o[0][0], 0);
    CHECK_EQ(o[1][0], 0x7F00);
    CHECK_EQ(o[2][1], -32768);
    CHECK_EQ(o[3][0], -256);
    decode_pcm8(2, d8, 2, o);
    CHECK_EQ(o[0][0], 0);
    CHECK_EQ(o[0][1], 0x7F00);
    CHECK_EQ(o[1][0], -32768);
    s16 d16[4] = {1, -2, 300, -32768};
    decode_pcm16(1, (const u8 *)d16, 4, o);
    CHECK_EQ(o[1][0], -2);
    CHECK_EQ(o[1][1], -2);
    CHECK_EQ(o[3][1], -32768);
    decode_pcm16(2, (const u8 *)d16, 2, o);
    CHECK_EQ(o[0][0], 1);
    CHECK_EQ(o[0][1], -2);
    CHECK_EQ(o[1][0], 300);
    CHECK_EQ(o[1][1], -32768);
}

// ---------------------------------------------------------------- interpolation

static std::vector<Sample> ramp(size_t n, int start = 1) {
    std::vector<Sample> v(n);
    for (size_t i = 0; i < n; i++) v[i] = {(s16)(start + i), (s16)(-(int)(start + i))};
    return v;
}

static void test_interp() {
    // Rate 1: output is the input delayed by two samples, for every mode.
    for (Interp mode : {Interp::None, Interp::Linear, Interp::Polyphase}) {
        InterpState st;
        auto in = ramp(1000);
        std::array<Sample, 160> out{};
        size_t outi = 0;
        size_t used = interpolate(st, mode, 1.0f, in.data(), in.size(), out.data(), outi, 160);
        CHECK_EQ(outi, 160);
        CHECK_EQ(used, 159);
        CHECK_EQ(out[0][0], 0);
        CHECK_EQ(out[1][0], 0);
        bool ok = true;
        for (int i = 2; i < 160; i++) ok &= out[i][0] == i - 1 && out[i][1] == -(i - 1);
        CHECK(ok);
        // continue: next output is in[158]
        outi = 0;
        used = interpolate(st, mode, 1.0f, in.data() + 159, in.size() - 159, out.data(), outi, 160);
        CHECK_EQ(out[0][0], 159);
        CHECK_EQ(used, 160);
    }
    // Linear at rate 0.5: midpoints (rounded down); None holds.
    {
        InterpState st;
        std::vector<Sample> in = {{100, 100}, {201, 201}, {0, 0}, {-101, -101}};
        std::array<Sample, 160> out{};
        size_t outi = 0;
        size_t used = interpolate(st, Interp::Linear, 0.5f, in.data(), in.size(), out.data(), outi, 160);
        // virtual input 0,0,100,201,0,-101 -> positions 0,0.5,...,3.5 (the position must leave
        // one sample of lookahead, so the last two input samples stay in the history)
        const int expect[] = {0, 0, 0, 50, 100, 150, 201, 100};
        CHECK_EQ(outi, 8);
        CHECK_EQ(used, 4);
        for (int i = 0; i < 8; i++) CHECK_EQ(out[i][0], expect[i]);
        // the history continues into the next buffer: 0 -> -101 -> 50
        std::vector<Sample> in2 = {{50, 50}};
        used = interpolate(st, Interp::Linear, 0.5f, in2.data(), 1, out.data(), outi, 160);
        const int expect2[] = {0, -51};
        CHECK_EQ(outi, 10);
        CHECK_EQ(used, 1);
        for (int i = 0; i < 2; i++) CHECK_EQ(out[8 + i][0], expect2[i]);
        st = {};
        outi = 0;
        interpolate(st, Interp::None, 0.5f, in.data(), in.size(), out.data(), outi, 160);
        const int expect_none[] = {0, 0, 0, 0, 100, 100, 201, 201};
        for (int i = 0; i < 8; i++) CHECK_EQ(out[i][0], expect_none[i]);
    }
    // Rate 2: decimation picks every other virtual sample.
    {
        InterpState st;
        auto in = ramp(400);
        std::array<Sample, 160> out{};
        size_t outi = 0;
        size_t used = interpolate(st, Interp::Linear, 2.0f, in.data(), in.size(), out.data(), outi, 160);
        CHECK_EQ(outi, 160);
        CHECK_EQ(out[0][0], 0);
        CHECK_EQ(out[1][0], 1);
        CHECK_EQ(out[2][0], 3);
        CHECK_EQ(out[159][0], 317);
        CHECK_EQ(used, 318);
    }
    // Chunked input (random chunk sizes, as buffers arrive) equals one-shot input.
    for (Interp mode : {Interp::None, Interp::Linear, Interp::Polyphase}) {
        for (float rate : {0.37f, 1.0f, 1.4142f, 3.1f}) {
            std::vector<Sample> in(5000);
            for (size_t i = 0; i < in.size(); i++)
                in[i] = {(s16)(10000 * std::sin(i * 0.05)), (s16)(8000 * std::cos(i * 0.031))};
            std::vector<Sample> a(800), b(800);
            InterpState s1, s2;
            size_t oa = 0, ob = 0;
            interpolate(s1, mode, rate, in.data(), in.size(), a.data(), oa, a.size());
            u32 seed = 7;
            size_t pos = 0;
            while (ob < b.size() && pos < in.size()) {
                seed = seed * 1103515245 + 12345;
                size_t chunk = std::min<size_t>(1 + (seed >> 16) % 37, in.size() - pos);
                size_t limit = std::min(b.size(), ob + 1 + (seed >> 8) % 50);
                pos += interpolate(s2, mode, rate, in.data() + pos, chunk, b.data(), ob, limit);
            }
            CHECK_EQ(oa, a.size());
            bool same = ob == oa;
            for (size_t i = 0; same && i < oa; i++) same &= a[i] == b[i];
            CHECK(same);
        }
    }
    // interp_input_needed gives exactly enough input to fill the output.
    for (float rate : {0.3f, 1.0f, 1.7f, 4.0f}) {
        InterpState st;
        st.fpos = (1ull << 24) * 3 / 4;
        auto in = ramp(2000);
        size_t need = interp_input_needed(st, rate, 160);
        std::array<Sample, 160> out{};
        InterpState s2 = st;
        size_t outi = 0;
        interpolate(s2, Interp::Linear, rate, in.data(), need, out.data(), outi, 160);
        CHECK_EQ(outi, 160);
        s2 = st;
        outi = 0;
        if (need > 0) interpolate(s2, Interp::Linear, rate, in.data(), need - 1, out.data(), outi, 160);
        CHECK(outi < 160);
    }
    // Polyphase (cubic) resampling of a smooth sine is closer to the ideal than linear.
    {
        const float rate = 0.6f;
        const double w = 2 * M_PI * 1000.0 / 32728 / rate;  // 1 kHz output tone at a 0.6 rate
        std::vector<Sample> in(1000);
        for (size_t i = 0; i < in.size(); i++) {
            s16 v = (s16)std::lround(16000 * std::sin(w * i));
            in[i] = {v, v};
        }
        double err[2] = {0, 0};
        int m = 0;
        for (Interp mode : {Interp::Linear, Interp::Polyphase}) {
            InterpState st;
            std::vector<Sample> out(500);
            size_t outi = 0;
            interpolate(st, mode, rate, in.data(), in.size(), out.data(), outi, out.size());
            for (size_t k = 10; k < outi; k++) {
                double x = k * (double)rate - 2;  // two-sample predelay
                double ideal = 16000 * std::sin(w * x);
                err[m] += (out[k][0] - ideal) * (out[k][0] - ideal);
            }
            m++;
        }
        printf("  resample rms error: linear %.1f, polyphase %.1f\n", std::sqrt(err[0] / 490),
               std::sqrt(err[1] / 490));
        CHECK(err[1] < err[0] / 4);
    }
}

// ---------------------------------------------------------------- filters

static void test_filters() {
    StereoFrame f;
    for (int i = 0; i < kFrameSamples; i++) f[i] = {(s16)(i * 100 - 8000), (s16)(-i * 50)};
    StereoFrame orig = f;
    SourceFilters flt;
    flt.enable(true, true);  // pass-through coefficients
    flt.process(f);
    CHECK(f == orig);
    // simple filter b0 = 0.5, a1 = 0: halves
    flt.s_b0 = 0x4000;
    flt.process(f);
    CHECK_EQ(f[10][0], orig[10][0] / 2);
    // simple filter a1 feedback: y = 0.5 x + 0.5 y1 on a step
    SourceFilters lp;
    lp.enable(true, false);
    lp.s_b0 = 0x4000;
    lp.s_a1 = 0x4000;
    StereoFrame step;
    step.fill({10000, 10000});
    lp.process(step);
    CHECK_EQ(step[0][0], 5000);
    CHECK_EQ(step[1][0], 7500);
    CHECK(step[159][0] >= 9998);
    // biquad: y = x - x[n-1] (differentiator) in Q14
    SourceFilters bq;
    bq.enable(false, true);
    bq.b_b0 = 1 << 14;
    bq.b_b1 = -(1 << 14);
    StereoFrame r;
    for (int i = 0; i < kFrameSamples; i++) r[i] = {(s16)(i * 3), (s16)(i * 3)};
    bq.process(r);
    CHECK_EQ(r[0][0], 0);
    CHECK_EQ(r[1][0], 3);
    CHECK_EQ(r[100][1], 3);
    // disabling resets to pass-through
    bq.enable(false, false);
    CHECK_EQ(bq.b_b1, 0);
    CHECK_EQ(bq.b_b0, 1 << 14);
}

// ---------------------------------------------------------------- end to end

static SharedMemory &R(int i) { return *(SharedMemory *)(g_mem + mem::DSP_BASE + (i ? kRegion1 : kRegion0)); }

// Emulates the application side: each frame it fills the region the DSP will read next.
struct App {
    u16 counter = 0;
    int cur = 1;
    // region the application writes for the next frame
    SharedMemory &next() { return R(cur ^ 1); }
    // submit: bump the frame counter of the written region, then run one DSP frame
    bool submit() {
        cur ^= 1;
        R(cur).frame_counter = ++counter;
        return audio::tick();
    }
    // region the DSP wrote its outputs to during the last frame
    SharedMemory &out() { return R(cur ^ 1); }
};

static u8 *phys(u32 pa) { return g_mem + mem::LINEAR_BASE + (pa - mem::FCRAM_PADDR); }

static void dsp_init() {
    u8 cmd[4] = {0, 0, 0, 0};
    audio::pipe_write(2, cmd, 4);
}

static void set_state(u8 s) {
    u8 cmd[4] = {s, 0, 0, 0};
    audio::pipe_write(2, cmd, 4);
}

static void setup_master(SharedMemory &m) {
    m.dsp_config.master_volume = 1.0f;
    m.dsp_config.output_format = 1;
    m.dsp_config.dirty |= CFG_MASTER_VOLUME_DIRTY | CFG_OUTPUT_FORMAT_DIRTY;
}

static void test_pipes_and_state() {
    CHECK_EQ(audio::recv_data(0), 1);
    CHECK(!audio::tick());
    dsp_init();
    CHECK_EQ(audio::recv_data(0), 0);
    CHECK_EQ(audio::pipe_readable(2), 32);
    auto hdr = audio::pipe_read(2, 2);
    CHECK_EQ(hdr.size(), 2);
    CHECK_EQ(hdr[0] | hdr[1] << 8, 15);
    auto rest = audio::pipe_read(2, 100);  // over-read returns what is there
    CHECK_EQ(rest.size(), 30);
    const u16 expect[15] = {0xBFFF, 0x9E92, 0x8680, 0xA792, 0x9430, 0x8400, 0x8540, 0x9492,
                            0x8710, 0x8410, 0xA912, 0xAA12, 0xAAD2, 0xAC52, 0xAC5C};
    for (int i = 0; i < 15; i++) CHECK_EQ(rest[2 * i] | rest[2 * i + 1] << 8, expect[i]);
    CHECK_EQ(audio::pipe_readable(2), 0);
    CHECK(audio::pipe_read(2, 4).empty());
    // Address conversion as done by dsp::DSP: VA = (addr << 1) + 0x1FF40000
    CHECK_EQ((0xBFFFu << 1) + 0x1FF40000u, mem::DSP_BASE + kRegion0 + offsetof(SharedMemory, frame_counter));
    CHECK_EQ((0x9E92u << 1) + 0x1FF40000u, mem::DSP_BASE + kRegion0 + offsetof(SharedMemory, source_config));
    CHECK(audio::tick());
    set_state(1);  // shutdown
    CHECK_EQ(audio::recv_data(0), 1);
    CHECK(!audio::tick());
    CHECK_EQ(audio::pipe_readable(2), 0);
}

static constexpr u32 kSinePA = 0x20100000;

static std::vector<s16> make_sine(size_t n, double freq, double amp) {
    std::vector<s16> v(n);
    for (size_t i = 0; i < n; i++) v[i] = (s16)std::lround(amp * std::sin(2 * M_PI * freq * i / kSampleRate));
    return v;
}

static void test_embedded_sine() {
    dsp_init();
    audio::pipe_read(2, 32);
    App app;
    const size_t N = 8182;
    auto sine = make_sine(N, 440, 20000);
    memcpy(phys(kSinePA), sine.data(), N * 2);

    SharedMemory &m = app.next();
    setup_master(m);
    SourceConfig &c = m.source_config[3];
    c.enable = 1;
    c.rate_multiplier = 1.0f;
    c.interpolation_mode = (u8)Interp::Linear;
    c.gain[0][0] = c.gain[0][1] = 1.0f;
    c.flags1 = 1 | ((u16)Format::PCM16 << 2);
    c.paddr.set(kSinePA);
    c.length.set(N);
    c.buffer_id = 1;
    c.sync_count = 42;
    c.dirty = SRC_ENABLE_DIRTY | SRC_RATE_DIRTY | SRC_INTERP_DIRTY | SRC_GAIN0_DIRTY |
              SRC_FORMAT_DIRTY | SRC_MONO_STEREO_DIRTY | SRC_EMBEDDED_DIRTY | SRC_SYNC_COUNT_DIRTY;
    CHECK(app.submit());
    // dirty flags consumed in the region the DSP read
    CHECK_EQ(R(app.cur).source_config[3].dirty, 0);
    CHECK_EQ(R(app.cur).dsp_config.dirty, 0);
    {
        const SourceStatus &s = app.out().source_status[3];
        CHECK_EQ(s.is_enabled, 1);
        CHECK_EQ(s.current_buffer_id, 1);
        CHECK_EQ(s.current_buffer_id_dirty, 0);  // embedded buffers do not set the dirty flag
        CHECK_EQ(s.sync_count, 42);
        CHECK_EQ(s.buffer_position.get(), 0);
        CHECK_EQ(app.out().source_status[0].is_enabled, 0);
    }
    // First frame after dequeueing is silent; then the sine, delayed by two samples.
    std::vector<s16> left, right;
    auto grab = [&] {
        for (int i = 0; i < kFrameSamples; i++) {
            left.push_back(app.out().final_samples[i][0]);
            right.push_back(app.out().final_samples[i][1]);
        }
    };
    grab();
    int frames = 1, end_frame = -1;
    u32 last_pos = 0;
    bool pos_monotonic = true;
    while (frames < 200) {
        CHECK(app.submit());
        frames++;
        grab();
        const SourceStatus &s = app.out().source_status[3];
        if (s.is_enabled) {
            pos_monotonic &= s.buffer_position.get() >= last_pos;
            last_pos = s.buffer_position.get();
            CHECK_EQ(s.current_buffer_id, 1);
        } else {
            end_frame = frames;
            CHECK_EQ(s.current_buffer_id, 0);
            CHECK_EQ(s.last_buffer_id, 1);
            CHECK_EQ(s.current_buffer_id_dirty, 1);
            CHECK_EQ(s.buffer_position.get(), N);
            break;
        }
    }
    CHECK(pos_monotonic);
    CHECK_EQ(last_pos, N);
    // 8182 samples + 2 predelay = 51.1 frames after the silent first frame -> ends during
    // frame 53, reported disabled in frame 54
    CHECK_EQ(end_frame, 54);
    CHECK(app.submit());
    CHECK_EQ(app.out().source_status[3].current_buffer_id_dirty, 0);
    CHECK_EQ(app.out().source_status[3].is_enabled, 0);

    size_t mism = 0;
    for (size_t i = 0; i < left.size(); i++) {
        long idx = (long)i - kFrameSamples - 2;
        // the last two samples of a buffer remain in the resampler history (two-sample predelay)
        s16 want = (idx >= 0 && idx < (long)N - 2) ? sine[idx] : 0;
        mism += left[i] != want || right[i] != want;
    }
    CHECK_EQ(mism, 0);
    printf("  embedded sine: %zu samples verified, source ended at frame %d\n", left.size(), end_frame);
}

static void test_buffer_queue_and_gains() {
    dsp_init();
    audio::pipe_read(2, 32);
    App app;
    const u32 pa1 = 0x20200000, pa2 = 0x20210000;
    std::vector<s16> b1(300, 1000), b2(200, -2000);
    memcpy(phys(pa1), b1.data(), 600);
    memcpy(phys(pa2), b2.data(), 400);

    SharedMemory &m = app.next();
    setup_master(m);
    SourceConfig &c = m.source_config[0];
    c.enable = 1;
    c.rate_multiplier = 1.0f;
    c.interpolation_mode = (u8)Interp::None;
    c.gain[0][0] = 0.5f;   // front left at half
    c.gain[0][1] = 0.25f;  // front right at a quarter
    c.gain[0][2] = 0.5f;   // back left also goes to the left output
    c.flags1 = 1 | ((u16)Format::PCM16 << 2);
    c.buffers[0].paddr.set(pa1);
    c.buffers[0].length.set(300);
    c.buffers[0].buffer_id = 6;
    c.buffers[1].paddr.set(pa2);
    c.buffers[1].length.set(200);
    c.buffers[1].buffer_id = 7;
    c.buffers_dirty = 3;
    c.dirty = SRC_ENABLE_DIRTY | SRC_RATE_DIRTY | SRC_INTERP_DIRTY | SRC_GAIN0_DIRTY |
              SRC_FORMAT_DIRTY | SRC_MONO_STEREO_DIRTY | SRC_BUFFER_QUEUE_DIRTY;
    CHECK(app.submit());
    CHECK_EQ(R(app.cur).source_config[0].buffers_dirty, 0);
    struct StatusRef {
        App &a;
        const SourceStatus *operator->() const { return &a.out().source_status[0]; }
    } s{app};
    CHECK_EQ(s->current_buffer_id, 6);
    CHECK_EQ(s->current_buffer_id_dirty, 1);  // queued buffers set it when they start
    CHECK(app.submit());  // samples 0..157 of buffer 6
    CHECK_EQ(s->current_buffer_id_dirty, 0);
    CHECK_EQ(s->current_buffer_id, 6);
    CHECK_EQ(app.out().final_samples[100][0], 1000);  // 0.5 + 0.5
    CHECK_EQ(app.out().final_samples[100][1], 250);
    CHECK(app.submit());  // buffer 6 ends, buffer 7 starts in the same frame
    CHECK_EQ(s->current_buffer_id, 7);
    CHECK_EQ(s->current_buffer_id_dirty, 1);
    CHECK_EQ(s->last_buffer_id, 0);
    CHECK_EQ(app.out().final_samples[159][0], -2000);
    CHECK(app.submit());
    CHECK_EQ(s->is_enabled, 1);
    CHECK(app.submit());  // all consumed in the previous/this frame
    while (s->is_enabled) CHECK(app.submit());
    CHECK_EQ(s->last_buffer_id, 7);
    CHECK_EQ(s->current_buffer_id, 0);
    CHECK_EQ(s->current_buffer_id_dirty, 1);

    // Gain ramp: a new gain is reached linearly over one frame.
    dsp_init();
    audio::pipe_read(2, 32);
    App a2;
    std::vector<s16> loud(2000, 10000);
    memcpy(phys(pa1), loud.data(), 4000);
    SharedMemory &m2 = a2.next();
    setup_master(m2);
    SourceConfig &d = m2.source_config[5];
    d.enable = 1;
    d.rate_multiplier = 1.0f;
    d.interpolation_mode = (u8)Interp::None;
    d.gain[0][0] = 1.0f;
    d.flags1 = 1 | ((u16)Format::PCM16 << 2);
    d.paddr.set(pa1);
    d.length.set(2000);
    d.dirty = SRC_ENABLE_DIRTY | SRC_RATE_DIRTY | SRC_INTERP_DIRTY | SRC_GAIN0_DIRTY |
              SRC_EMBEDDED_DIRTY;
    a2.submit();
    a2.submit();
    a2.submit();
    CHECK_EQ(a2.out().final_samples[50][0], 10000);
    SourceConfig &d2 = a2.next().source_config[5];
    d2.gain[0][0] = 0.0f;
    d2.dirty = SRC_GAIN0_DIRTY;
    a2.submit();
    CHECK_EQ(a2.out().final_samples[0][0], 10000);
    CHECK(std::abs(a2.out().final_samples[80][0] - 4969) <= 1);
    CHECK_EQ(a2.out().final_samples[159][0], 0);
    a2.submit();
    CHECK_EQ(a2.out().final_samples[10][0], 0);

    // Master volume / aux return volumes and the aux bus send.
    SharedMemory &m3 = a2.next();
    m3.source_config[5].gain[0][0] = 0.0f;
    m3.source_config[5].gain[1][0] = 1.0f;  // to intermediate mix 1
    m3.source_config[5].dirty = SRC_GAIN0_DIRTY | SRC_GAIN1_DIRTY;
    m3.dsp_config.aux_return_volume[0] = 0.5f;
    m3.dsp_config.dirty = CFG_AUX_RETURN_VOLUME0_DIRTY;
    a2.submit();  // ramp frame
    a2.submit();
    CHECK_EQ(a2.out().final_samples[20][0], 5000);
    SharedMemory &m4 = a2.next();
    m4.dsp_config.aux_bus_enable[0] = 1;
    m4.dsp_config.dirty = CFG_AUX_BUS_ENABLE0_DIRTY;
    memset(m4.intermediate, 0, sizeof(m4.intermediate));
    m4.intermediate[0].pcm32[0][7] = 3000;  // what the application "returns" on aux bus 0
    a2.submit();
    CHECK_EQ(a2.out().intermediate[0].pcm32[0][20], 10000);  // the DSP sent mix 1 to the app
    CHECK_EQ(a2.out().final_samples[20][0], 0);              // and played what the app returned
    CHECK_EQ(a2.out().final_samples[7][0], 1500);
}

static void test_looping_and_play_position() {
    dsp_init();
    audio::pipe_read(2, 32);
    App app;
    const u32 pa = 0x20300000;
    std::vector<s16> buf(400);
    for (int i = 0; i < 400; i++) buf[i] = (s16)(i * 10 + 1);
    memcpy(phys(pa), buf.data(), 800);
    SharedMemory &m = app.next();
    setup_master(m);
    SourceConfig &c = m.source_config[1];
    c.enable = 1;
    c.rate_multiplier = 1.0f;
    c.interpolation_mode = (u8)Interp::Linear;
    c.gain[0][0] = 1.0f;
    c.flags1 = 1 | ((u16)Format::PCM16 << 2);
    c.flags2 = 2;  // looping
    c.paddr.set(pa);
    c.length.set(400);
    c.play_position.set(100);
    c.buffer_id = 9;
    c.dirty = SRC_ENABLE_DIRTY | SRC_RATE_DIRTY | SRC_INTERP_DIRTY | SRC_GAIN0_DIRTY |
              SRC_EMBEDDED_DIRTY | SRC_PLAY_POSITION_DIRTY | SRC_FORMAT_DIRTY;
    app.submit();
    CHECK_EQ(app.out().source_status[1].buffer_position.get(), 100);
    std::vector<s16> got;
    for (int f = 0; f < 20; f++) {
        app.submit();
        for (int i = 0; i < kFrameSamples; i++) got.push_back(app.out().final_samples[i][0]);
        CHECK_EQ(app.out().source_status[1].is_enabled, 1);
        CHECK_EQ(app.out().source_status[1].current_buffer_id, 9);
    }
    // expected stream: 2 samples predelay, buf[100..399], then buf[0..399] forever
    std::vector<s16> want = {0, 0};
    for (int i = 100; i < 400; i++) want.push_back(buf[i]);
    while (want.size() < got.size()) for (int i = 0; i < 400; i++) want.push_back(buf[i]);
    size_t mism = 0;
    for (size_t i = 0; i < got.size(); i++) mism += got[i] != want[i];
    CHECK_EQ(mism, 0);
    // a partial reset (clearing the queue) lets the current pass finish and stops the loop
    SourceConfig &c2 = app.next().source_config[1];
    c2.dirty = SRC_PARTIAL_RESET;
    int frames = 0;
    do {
        app.submit();
    } while (app.out().source_status[1].is_enabled && ++frames < 10);
    CHECK(frames < 10);
    CHECK_EQ(app.out().source_status[1].last_buffer_id, 9);
    // reset flag
    SourceConfig &c3 = app.next().source_config[1];
    c3.dirty = SRC_RESET;
    app.submit();
    CHECK_EQ(app.out().source_status[1].buffer_position.get(), 0);
    CHECK_EQ(app.out().source_status[1].last_buffer_id, 0);
}

static void test_adpcm_source_and_rate() {
    dsp_init();
    audio::pipe_read(2, 32);
    App app;
    // 22050 Hz ADPCM sine resampled to 32728 Hz with polyphase
    const double fs = 22050;
    std::vector<s16> pcm(14 * 300);
    for (size_t i = 0; i < pcm.size(); i++) pcm[i] = (s16)(15000 * std::sin(2 * M_PI * 500 * i / fs));
    auto enc = encode_adpcm(pcm, kCoeffs);
    const u32 pa = 0x20400000;
    memcpy(phys(pa), enc.data(), enc.size());
    SharedMemory &m = app.next();
    setup_master(m);
    memcpy(m.adpcm_coeffs[2], kCoeffs, sizeof(kCoeffs));
    SourceConfig &c = m.source_config[2];
    c.enable = 1;
    c.rate_multiplier = (float)(fs / kSampleRate);
    c.interpolation_mode = (u8)Interp::Polyphase;
    c.gain[0][0] = c.gain[0][1] = 1.0f;
    c.flags1 = 1 | ((u16)Format::ADPCM << 2);
    c.paddr.set(pa);
    c.length.set((u32)pcm.size());
    c.flags2 = 1;  // adpcm dirty (yn = 0)
    c.buffer_id = 3;
    c.dirty = SRC_ENABLE_DIRTY | SRC_RATE_DIRTY | SRC_INTERP_DIRTY | SRC_GAIN0_DIRTY |
              SRC_EMBEDDED_DIRTY | SRC_ADPCM_COEFFS_DIRTY | SRC_FORMAT_DIRTY;
    app.submit();
    std::vector<s16> out;
    int frames = 0;
    while (app.submit() && app.out().source_status[2].is_enabled && frames < 400) {
        for (int i = 0; i < kFrameSamples; i++) out.push_back(app.out().final_samples[i][0]);
        frames++;
    }
    // expected duration: 4200 input samples at 22050 Hz -> ~6234 output samples
    double expect = pcm.size() * kSampleRate / fs;
    printf("  adpcm source: %d frames (%zu samples, expected ~%.0f)\n", frames, out.size(), expect);
    CHECK(std::abs((double)out.size() - expect) < 2 * kFrameSamples);
    // compare against the ideal 500 Hz tone at the output rate (ignoring the edges)
    double err = 0, sig = 0;
    for (size_t k = 200; k + 200 < std::min<size_t>(out.size(), (size_t)expect); k++) {
        double t = (k * (fs / kSampleRate) - 2) / fs;
        double ideal = 15000 * std::sin(2 * M_PI * 500 * t);
        err += (out[k] - ideal) * (out[k] - ideal);
        sig += ideal * ideal;

    }
    double snr = 10 * std::log10(sig / std::max(err, 1.0));
    printf("  adpcm source SNR vs ideal: %.1f dB\n", snr);
    CHECK(snr > 25);
}

// Two queued ADPCM buffers where the second continues the decoder state of the first, plus a
// looping ADPCM embedded buffer with a play position on another source.
static void test_adpcm_chained() {
    dsp_init();
    audio::pipe_read(2, 32);
    App app;
    std::vector<s16> pcm(14 * 100);
    for (size_t i = 0; i < pcm.size(); i++) pcm[i] = (s16)(9000 * std::sin(i * 0.07) + 3000 * std::sin(i * 0.31));
    auto enc = encode_adpcm(pcm, kCoeffs);
    std::vector<Sample> ref(pcm.size());
    AdpcmState rs{};
    decode_adpcm(enc.data(), pcm.size(), kCoeffs, rs, ref.data());
    const u32 pa = 0x20600000;
    memcpy(phys(pa), enc.data(), enc.size());

    SharedMemory &m = app.next();
    setup_master(m);
    memcpy(m.adpcm_coeffs[9], kCoeffs, sizeof(kCoeffs));
    SourceConfig &c = m.source_config[9];
    c.enable = 1;
    c.rate_multiplier = 1.0f;
    c.interpolation_mode = (u8)Interp::None;
    c.gain[0][0] = 1.0f;
    c.flags1 = 1 | ((u16)Format::ADPCM << 2);
    c.buffers[0].paddr.set(pa);
    c.buffers[0].length.set(14 * 40);
    c.buffers[0].adpcm_dirty = 1;
    c.buffers[0].buffer_id = 1;
    c.buffers[1].paddr.set(pa + 40 * 8);
    c.buffers[1].length.set(14 * 60);
    c.buffers[1].buffer_id = 2;
    c.buffers_dirty = 3;
    c.dirty = SRC_ENABLE_DIRTY | SRC_RATE_DIRTY | SRC_INTERP_DIRTY | SRC_GAIN0_DIRTY | SRC_FORMAT_DIRTY |
              SRC_ADPCM_COEFFS_DIRTY | SRC_BUFFER_QUEUE_DIRTY;
    app.submit();  // dequeue frame (silent)
    std::vector<s16> got;
    for (int f = 0; f < 10; f++) {
        app.submit();
        for (int i = 0; i < kFrameSamples; i++) got.push_back(app.out().final_samples[i][0]);
    }
    size_t mism = 0;
    for (size_t i = 0; i < got.size(); i++) {
        long idx = (long)i - 2;
        s16 want = (idx >= 0 && idx < (long)pcm.size() - 2) ? ref[idx][0] : 0;
        mism += got[i] != want;
    }
    CHECK_EQ(mism, 0);
    CHECK_EQ(app.out().source_status[9].is_enabled, 0);
    CHECK_EQ(app.out().source_status[9].last_buffer_id, 2);

    // looping embedded ADPCM with a play position: first pass from 100, then from 0 with the
    // history reloaded (adpcm dirty), so every pass decodes identically
    SharedMemory &m2 = app.next();
    memcpy(m2.adpcm_coeffs[10], kCoeffs, sizeof(kCoeffs));
    SourceConfig &d = m2.source_config[10];
    d.enable = 1;
    d.rate_multiplier = 1.0f;
    d.interpolation_mode = (u8)Interp::Linear;
    d.gain[0][0] = 1.0f;
    d.flags1 = 1 | ((u16)Format::ADPCM << 2);
    d.flags2 = 3;  // adpcm dirty + looping
    d.paddr.set(pa);
    d.length.set(14 * 30);
    d.play_position.set(100);
    d.buffer_id = 4;
    d.dirty = SRC_ENABLE_DIRTY | SRC_RATE_DIRTY | SRC_INTERP_DIRTY | SRC_GAIN0_DIRTY | SRC_FORMAT_DIRTY |
              SRC_ADPCM_COEFFS_DIRTY | SRC_EMBEDDED_DIRTY | SRC_PLAY_POSITION_DIRTY;
    app.submit();
    got.clear();
    for (int f = 0; f < 8; f++) {
        app.submit();
        for (int i = 0; i < kFrameSamples; i++) got.push_back(app.out().final_samples[i][0]);
    }
    std::vector<s16> want = {0, 0};
    for (int i = 100; i < 420; i++) want.push_back(ref[i][0]);
    while (want.size() < got.size()) for (int i = 0; i < 420; i++) want.push_back(ref[i][0]);
    mism = 0;
    for (size_t i = 0; i < got.size(); i++) mism += got[i] != want[i];
    CHECK_EQ(mism, 0);
}

// Stereo PCM16 embedded buffer extended while playing (partial embedded buffer update).
static void test_stereo_partial_update() {
    dsp_init();
    audio::pipe_read(2, 32);
    App app;
    const u32 pa = 0x20700000;
    std::vector<s16> st(2 * 1000);
    for (int i = 0; i < 1000; i++) st[2 * i] = (s16)(i + 1), st[2 * i + 1] = (s16)(-i - 1);
    memcpy(phys(pa), st.data(), st.size() * 2);
    SharedMemory &m = app.next();
    setup_master(m);
    SourceConfig &c = m.source_config[11];
    c.enable = 1;
    c.rate_multiplier = 1.0f;
    c.interpolation_mode = (u8)Interp::None;
    c.gain[0][0] = c.gain[0][1] = 1.0f;
    c.flags1 = 2 | ((u16)Format::PCM16 << 2);
    c.paddr.set(pa);
    c.length.set(200);
    c.buffer_id = 1;
    c.dirty = SRC_ENABLE_DIRTY | SRC_RATE_DIRTY | SRC_INTERP_DIRTY | SRC_GAIN0_DIRTY | SRC_EMBEDDED_DIRTY;
    app.submit();
    app.submit();
    CHECK_EQ(app.out().final_samples[50][0], 49);
    CHECK_EQ(app.out().final_samples[50][1], -49);
    SourceConfig &c2 = app.next().source_config[11];
    c2.length.set(600);
    c2.buffer_id = 1;
    c2.dirty = SRC_PARTIAL_EMBEDDED_DIRTY;
    std::vector<s16> got;
    for (int f = 0; f < 4; f++) {
        app.submit();
        for (int i = 0; i < kFrameSamples; i++) got.push_back(app.out().final_samples[i][1]);
    }
    // continues seamlessly from sample 158 up to the new end (minus the 2-sample tail)
    size_t mism = 0;
    for (size_t i = 0; i < got.size(); i++) {
        int idx = 158 + (int)i;
        s16 want = idx < 598 ? (s16)(-idx - 1) : 0;
        mism += got[i] != want;
    }
    CHECK_EQ(mism, 0);
    app.submit();
    CHECK_EQ(app.out().source_status[11].is_enabled, 0);
    CHECK_EQ(app.out().source_status[11].buffer_position.get(), 600);
}

static void test_sleep_wakeup() {
    dsp_init();
    audio::pipe_read(2, 32);
    App app;
    std::vector<s16> loud(5000, 1234);
    memcpy(phys(0x20500000), loud.data(), 10000);
    SharedMemory &m = app.next();
    setup_master(m);
    SourceConfig &c = m.source_config[7];
    c.enable = 1;
    c.rate_multiplier = 1.0f;
    c.gain[0][0] = 1.0f;
    c.flags1 = 1 | ((u16)Format::PCM16 << 2);
    c.paddr.set(0x20500000);
    c.length.set(5000);
    c.dirty = SRC_ENABLE_DIRTY | SRC_RATE_DIRTY | SRC_GAIN0_DIRTY | SRC_EMBEDDED_DIRTY | SRC_FORMAT_DIRTY;
    app.submit();
    app.submit();
    app.submit();
    u32 pos = app.out().source_status[7].buffer_position.get();
    CHECK(pos > 0);
    set_state(3);  // sleep
    CHECK_EQ(audio::recv_data(0), 1);
    CHECK_EQ(audio::pipe_readable(2), 32);
    audio::pipe_read(2, 32);
    CHECK(!audio::tick());
    memset(g_mem + mem::DSP_BASE, 0xAA, kDspRamSize);  // firmware garbles DSP RAM while asleep
    set_state(2);  // wakeup restores it
    CHECK_EQ(audio::recv_data(0), 0);
    CHECK_EQ(audio::pipe_readable(2), 32);
    CHECK_EQ(app.out().source_status[7].buffer_position.get(), pos);
    app.submit();
    CHECK_EQ(app.out().source_status[7].buffer_position.get(), pos + 160);
    CHECK_EQ(app.out().final_samples[30][0], 1234);
}

static void test_invalid_buffer() {
    dsp_init();
    audio::pipe_read(2, 32);
    App app;
    SharedMemory &m = app.next();
    setup_master(m);
    SourceConfig &c = m.source_config[4];
    c.enable = 1;
    c.rate_multiplier = 1.0f;
    c.flags1 = 1 | ((u16)Format::PCM16 << 2);
    c.paddr.set(0x10000000);  // unmapped physical address
    c.length.set(1000);
    c.buffer_id = 2;
    c.dirty = SRC_ENABLE_DIRTY | SRC_EMBEDDED_DIRTY | SRC_FORMAT_DIRTY;
    app.submit();
    app.submit();
    CHECK_EQ(app.out().source_status[4].is_enabled, 0);  // skipped, then nothing left
    // length running past the end of FCRAM is rejected too
    SourceConfig &c2 = app.next().source_config[4];
    c2.enable = 1;
    c2.paddr.set(0x27FFFF00);
    c2.length.set(1000);
    c2.dirty = SRC_ENABLE_DIRTY | SRC_EMBEDDED_DIRTY;
    app.submit();
    app.submit();
    CHECK_EQ(app.out().source_status[4].is_enabled, 0);
}

static void play_tone(double amp, int seconds) {
    printf("  playing a %d s 440 Hz tone (amplitude %.0f) through SDL...\n", seconds, amp);
    audio::init();
    dsp_init();
    audio::pipe_read(2, 32);
    App app;
    const size_t N = kSampleRate * seconds;
    auto sine = make_sine(N, 440, amp);
    memcpy(phys(kSinePA), sine.data(), N * 2);
    SharedMemory &m = app.next();
    setup_master(m);
    SourceConfig &c = m.source_config[0];
    c.enable = 1;
    c.rate_multiplier = 1.0f;
    c.gain[0][0] = c.gain[0][1] = 1.0f;
    c.flags1 = 1 | ((u16)Format::PCM16 << 2);
    c.paddr.set(kSinePA);
    c.length.set(N);
    c.dirty = SRC_ENABLE_DIRTY | SRC_RATE_DIRTY | SRC_GAIN0_DIRTY | SRC_EMBEDDED_DIRTY | SRC_FORMAT_DIRTY;
    auto period = std::chrono::nanoseconds(1000000000ll * kFrameSamples / kSampleRate);
    auto next = std::chrono::steady_clock::now();
    int min_q = 1 << 30, max_q = 0;
    const int frames = (int)(N / kFrameSamples);
    for (int f = 0; f < frames + 40; f++) {
        app.submit();
        int q = host_queued();
        if (f > 20 && f < frames && q >= 0) min_q = std::min(min_q, q), max_q = std::max(max_q, q);
        if (f % 1000 == 999) printf("    t=%.1fs queue=%d\n", f * kFrameSamples / (double)kSampleRate, q);
        next += period;
        std::this_thread::sleep_until(next);
    }
    printf("  host queue level: min %d max %d samples\n", min_q, max_q);
}

int main(int argc, char **argv) {
    size_t size = (1ull << 32) + 0x10000;
    void *p = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON | MAP_NORESERVE, -1, 0);
    if (p == MAP_FAILED) {
        perror("mmap");
        return 1;
    }
    g_mem = (u8 *)p;

    struct {
        const char *name;
        void (*fn)();
    } tests[] = {
        {"adpcm", test_adpcm},
        {"pcm", test_pcm},
        {"interp", test_interp},
        {"filters", test_filters},
        {"pipes_and_state", test_pipes_and_state},
        {"embedded_sine", test_embedded_sine},
        {"buffer_queue_and_gains", test_buffer_queue_and_gains},
        {"looping_and_play_position", test_looping_and_play_position},
        {"adpcm_source_and_rate", test_adpcm_source_and_rate},
        {"adpcm_chained", test_adpcm_chained},
        {"stereo_partial_update", test_stereo_partial_update},
        {"sleep_wakeup", test_sleep_wakeup},
        {"invalid_buffer", test_invalid_buffer},
    };
    for (auto &t : tests) {
        int before = g_failures;
        printf("[ %s ]\n", t.name);
        t.fn();
        printf("  %s\n", g_failures == before ? "ok" : "FAILED");
    }
    if (argc > 1 && std::string(argv[1]) == "--play") play_tone(8000, 2);
    if (argc > 1 && std::string(argv[1]) == "--play-silent") play_tone(0, 20);  // queue stability
    printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
