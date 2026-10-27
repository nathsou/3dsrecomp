// One DSP voice: configuration parsing, buffer queue, decoding, resampling, filters, gains.
// The status semantics (buffer ids, dirty flag, position, enable) follow Citra/Azahar's DSP HLE,
// which applications have been observed to depend on.
#include <algorithm>

#include "dsp_hle.h"

namespace audio::hle {

// ---- filters

void SourceFilters::reset_simple() {
    s_y1 = {};
    s_a1 = 0;
    s_b0 = 1 << 15;  // pass-through
}

void SourceFilters::reset_biquad() {
    b_x1 = b_x2 = b_y1 = b_y2 = {};
    b_a1 = b_a2 = b_b1 = b_b2 = 0;
    b_b0 = 1 << 14;  // pass-through
}

void SourceFilters::enable(bool simple, bool biquad) {
    simple_on = simple;
    biquad_on = biquad;
    if (!simple) reset_simple();
    if (!biquad) reset_biquad();
}

void SourceFilters::process(StereoFrame &f) {
    if (simple_on) {
        for (Sample &s : f) {
            for (int c = 0; c < 2; c++) {
                s64 acc = (s64)s_b0 * s[c] + (s64)s_a1 * s_y1[c];
                s32 y = (s32)std::clamp<s64>(acc >> 15, -32768, 32767);
                s_y1[c] = y;
                s[c] = (s16)y;
            }
        }
    }
    if (biquad_on) {
        for (Sample &s : f) {
            for (int c = 0; c < 2; c++) {
                s32 x0 = s[c];
                s64 acc = (s64)b_b0 * x0 + (s64)b_b1 * b_x1[c] + (s64)b_b2 * b_x2[c] +
                          (s64)b_a1 * b_y1[c] + (s64)b_a2 * b_y2[c];
                s32 y = (s32)std::clamp<s64>(acc >> 14, -32768, 32767);
                b_x2[c] = b_x1[c];
                b_x1[c] = x0;
                b_y2[c] = b_y1[c];
                b_y1[c] = y;
                s[c] = (s16)y;
            }
        }
    }
}

// ---- source

void Source::enqueue(const Buffer &b) {
    // Lower buffer ids play first; equal ids keep their insertion order.
    auto it = std::upper_bound(st.queue.begin(), st.queue.end(), b.id,
                               [](u16 id, const Buffer &x) { return id < x.id; });
    st.queue.insert(it, b);
}

void Source::parse_config(SourceConfig &cfg, const s16 coeffs[16]) {
    const u32 d = cfg.dirty;
    if (!d) return;

    if (d & SRC_RESET) reset();
    if (d & SRC_PARTIAL_RESET) st.queue.clear();
    if (d & SRC_ENABLE_DIRTY) st.enabled = cfg.enable != 0;
    if (d & SRC_SYNC_COUNT_DIRTY) st.sync_count = cfg.sync_count;
    if (d & SRC_RATE_DIRTY) {
        st.rate = cfg.rate_multiplier;
        if (!(st.rate > 0.0f) || st.rate > 1000.0f) st.rate = 1.0f;  // firmware outputs garbage
    }
    if (d & SRC_ADPCM_COEFFS_DIRTY) std::copy(coeffs, coeffs + 16, st.adpcm_coeffs.begin());
    for (int m = 0; m < 3; m++) {
        if (d & (SRC_GAIN0_DIRTY << m)) {
            st.gain_ramp_start[m] = st.gain[m];
            st.gain_ramp[m] = true;
            for (int c = 0; c < 4; c++) st.gain[m][c] = cfg.gain[m][c];
        }
    }
    if (d & SRC_FILTERS_ENABLED_DIRTY)
        st.filters.enable(cfg.filters_enabled & 1, (cfg.filters_enabled >> 1) & 1);
    if (d & SRC_SIMPLE_FILTER_DIRTY) {
        st.filters.s_b0 = cfg.simple_b0;
        st.filters.s_a1 = cfg.simple_a1;
    }
    if (d & SRC_BIQUAD_FILTER_DIRTY) {
        st.filters.b_a1 = cfg.biquad_a1;
        st.filters.b_a2 = cfg.biquad_a2;
        st.filters.b_b0 = cfg.biquad_b0;
        st.filters.b_b1 = cfg.biquad_b1;
        st.filters.b_b2 = cfg.biquad_b2;
    }
    if (d & SRC_INTERP_DIRTY) st.interp = (Interp)cfg.interpolation_mode;
    if (d & (SRC_FORMAT_DIRTY | SRC_EMBEDDED_DIRTY)) st.format = cfg.format();
    if (d & (SRC_MONO_STEREO_DIRTY | SRC_EMBEDDED_DIRTY)) st.channels = cfg.channels();

    // The play position only applies to the embedded buffer and defaults to 0.
    const u32 play_position = (d & SRC_PLAY_POSITION_DIRTY) ? cfg.play_position.get() : 0;

    if (d & SRC_PARTIAL_EMBEDDED_DIRTY) {
        // The application extended (or shortened) the buffer being played. PCM is read live from
        // guest memory, so only the bookkeeping changes. (ADPCM partial updates: unsupported.)
        if (st.cur_paddr && st.format != Format::ADPCM && (u8)st.format <= (u8)Format::PCM16 &&
            phys_range(st.cur_paddr & ~3u, 1)) {
            st.cur_length = cfg.length;
            st.cur_channels = st.channels;
            st.cur_format = st.format;
            st.cur_looping = (cfg.flags2 & 2) != 0;
            if (st.pos > st.cur_length) st.pos = 0;
            st.have_buffer = st.pos < st.cur_length;
        }
    }

    if (d & SRC_EMBEDDED_DIRTY) {
        if ((s32)cfg.length.get() >= 0) {  // games sometimes pass underflowed lengths
            Buffer b;
            b.paddr = cfg.paddr;
            b.length = cfg.length;
            b.adpcm_yn[0] = cfg.adpcm_yn[0];
            b.adpcm_yn[1] = cfg.adpcm_yn[1];
            b.adpcm_dirty = cfg.flags2 & 1;
            b.looping = (cfg.flags2 & 2) != 0;
            b.id = cfg.buffer_id;
            b.channels = st.channels;
            b.format = st.format;
            b.from_queue = false;
            b.play_position = play_position;
            enqueue(b);
        }
    }

    if (d & SRC_BUFFER_QUEUE_DIRTY) {
        for (int i = 0; i < 4; i++) {
            if (!(cfg.buffers_dirty & (1 << i))) continue;
            const SourceBuffer &q = cfg.buffers[i];
            if ((s32)q.length.get() < 0) continue;
            Buffer b;
            b.paddr = q.paddr;
            b.length = q.length;
            b.adpcm_yn[0] = q.adpcm_yn[0];
            b.adpcm_yn[1] = q.adpcm_yn[1];
            b.adpcm_dirty = q.adpcm_dirty != 0;
            b.looping = q.is_looping != 0;
            b.id = q.buffer_id;
            b.channels = st.channels;
            b.format = st.format;
            b.from_queue = true;
            enqueue(b);
        }
        cfg.buffers_dirty = 0;
    }

    cfg.dirty = 0;
}

bool Source::dequeue() {
    if (st.queue.empty()) return false;
    Buffer b = st.queue.front();
    st.queue.erase(st.queue.begin());

    if (b.adpcm_dirty) {
        st.adpcm.yn1 = (s16)b.adpcm_yn[0];
        st.adpcm.yn2 = (s16)b.adpcm_yn[1];
    }

    // The DSP's DMA ignores the low two address bits.
    const u32 addr = b.paddr & ~3u;
    const u64 len = b.length;
    u64 bytes = 0;
    switch (b.format) {
    case Format::PCM8: bytes = len * b.channels; break;
    case Format::PCM16: bytes = len * 2 * b.channels; break;
    case Format::ADPCM: bytes = (len + 13) / 14 * 8; break;
    default: bytes = ~0ull; break;
    }
    const u8 *p = bytes != ~0ull ? phys_range(addr, std::max<u64>(bytes, 1)) : nullptr;
    if (!p) {
        VLOG("dsp: source buffer id=%u at paddr %08x (len %u fmt %d) is invalid\n", b.id, b.paddr,
             b.length, (int)b.format);
        st.have_buffer = false;
        return true;
    }
    // ADPCM is decoded incrementally by fetch() (from the start, even with a play position), so a
    // long buffer does not stall one frame; PCM is read straight from guest memory.
    st.decoded.clear();
    st.adpcm_decoded = 0;

    // The first playthrough starts at play_position, loops restart at 0.
    st.pos = b.has_played ? 0 : std::min(b.play_position, b.length);
    st.cur_paddr = b.paddr;
    st.cur_length = b.length;
    st.cur_channels = b.channels;
    st.cur_format = b.format;
    st.cur_looping = b.looping;
    st.current_buffer_id = b.id;
    st.last_buffer_id = 0;
    st.buffer_update = b.from_queue && !b.has_played;
    st.have_buffer = st.pos < st.cur_length;

    if (b.looping) {
        b.has_played = true;
        enqueue(b);
    }
    return true;
}

size_t Source::fetch(Sample *out, size_t max) {
    max = std::min<size_t>(max, cur_remaining());
    if (max == 0) return 0;
    if (st.cur_format == Format::ADPCM) {
        const u32 want = st.pos + (u32)max;
        if (st.adpcm_decoded < want) {
            const u32 upto = (u32)std::min<u64>(st.cur_length, (want + 13ull) / 14 * 14);
            const u8 *p = phys_range(st.cur_paddr & ~3u, ((u64)upto + 13) / 14 * 8);
            if (!p) return 0;
            st.decoded.resize(upto);
            decode_adpcm(p + st.adpcm_decoded / 14 * 8, upto - st.adpcm_decoded, st.adpcm_coeffs.data(),
                         st.adpcm, st.decoded.data() + st.adpcm_decoded);
            st.adpcm_decoded = upto;
        }
        std::copy_n(st.decoded.begin() + st.pos, max, out);
        return max;
    }
    const u32 bps = (st.cur_format == Format::PCM16 ? 2 : 1) * st.cur_channels;
    const u8 *p = phys_range((st.cur_paddr & ~3u) + (u64)st.pos * bps, (u64)max * bps);
    if (!p) return 0;
    if (st.cur_format == Format::PCM16) decode_pcm16(st.cur_channels, p, max, out);
    else decode_pcm8(st.cur_channels, p, max, out);
    return max;
}

void Source::generate() {
    frame.fill({});

    if (!st.have_buffer) {
        if (dequeue()) return;
        // Nothing left to play: the source turns itself off.
        st.enabled = false;
        st.buffer_update = true;
        st.last_buffer_id = st.current_buffer_id;
        st.current_buffer_id = 0;
        return;
    }

    thread_local std::vector<Sample> scratch;
    size_t outi = 0;
    int dequeues = 0;
    while (outi < frame.size()) {
        if (!st.have_buffer) {
            if (++dequeues > 64 || !dequeue()) break;
            continue;
        }
        const size_t need = interp_input_needed(st.interp_state, st.rate, frame.size() - outi);
        const size_t take = std::min<size_t>(need, cur_remaining());
        if (scratch.size() < take) scratch.resize(take);
        if (fetch(scratch.data(), take) != take) {  // memory became invalid: drop the buffer
            st.have_buffer = false;
            continue;
        }
        const size_t used = interpolate(st.interp_state, st.interp, st.rate, scratch.data(), take,
                                        frame.data(), outi, frame.size());
        st.pos += (u32)used;
        if (st.pos >= st.cur_length) st.have_buffer = false;
    }

    st.filters.process(frame);
}

SourceStatus Source::tick(SourceConfig &cfg, const s16 coeffs[16]) {
    parse_config(cfg, coeffs);
    if (st.enabled) generate();

    SourceStatus s;
    s.is_enabled = st.enabled;
    s.current_buffer_id_dirty = st.buffer_update ? 1 : 0;
    st.buffer_update = false;
    s.sync_count = st.sync_count;
    s.buffer_position.set(st.pos);
    s.current_buffer_id = st.current_buffer_id;
    s.last_buffer_id = st.last_buffer_id;
    return s;
}

void Source::mix_into(QuadFrame &dst, int mix) {
    const auto &g = st.gain[mix];
    if (!st.enabled) {
        st.gain_ramp_start[mix] = g;
        st.gain_ramp[mix] = false;
        return;
    }
    const bool ramp = st.gain_ramp[mix];
    const auto &g0 = st.gain_ramp_start[mix];
    constexpr float kRampScale = 1.0f / (float)(kFrameSamples - 1);
    for (int i = 0; i < kFrameSamples; i++) {
        float gc[4];
        if (ramp) {
            const float t = (float)i * kRampScale;
            for (int c = 0; c < 4; c++) gc[c] = g0[c] + (g[c] - g0[c]) * t;
        } else {
            for (int c = 0; c < 4; c++) gc[c] = g[c];
        }
        // stereo -> quad: front L/R and back L/R
        dst[i][0] += (s32)(gc[0] * frame[i][0]);
        dst[i][1] += (s32)(gc[1] * frame[i][1]);
        dst[i][2] += (s32)(gc[2] * frame[i][0]);
        dst[i][3] += (s32)(gc[3] * frame[i][1]);
    }
    if (ramp) {
        st.gain_ramp_start[mix] = g;
        st.gain_ramp[mix] = false;
    }
}

}  // namespace audio::hle
