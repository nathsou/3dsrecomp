// Layout of the DSP shared-memory regions as seen by the application (DSP firmware HLE).
//
// The DSP RAM (0x80000 bytes) is mapped at guest VA 0x1FF00000. The application talks to the
// firmware through two copies of a 0x8000-byte "shared memory" block, at byte offsets 0x50000 and
// 0x70000 of DSP RAM (DSP word addresses 0x8000 and 0x18000). The application fills one copy per
// audio frame and bumps the frame counter at its very end; the firmware reads the copy with the
// newer counter and writes its outputs (statuses, final samples...) into the other one.
//
// The DSP is a 16-bit machine: 32-bit integers are stored as two 16-bit words, high word first
// ("middle endian" from the ARM11 side). Floats, 16-bit values and the intermediate mix samples are
// plain little endian.
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>

#include "rt.h"

namespace audio::hle {

constexpr int kSampleRate = 32728;     // native output rate (Hz)
constexpr int kFrameSamples = 160;     // samples per audio frame
constexpr int kNumSources = 24;
constexpr int kNumPipes = 8;
constexpr u32 kRegion0 = 0x50000;      // byte offsets inside DSP RAM
constexpr u32 kRegion1 = 0x70000;
constexpr u32 kDspRamSize = 0x80000;

#pragma pack(push, 1)

// 32-bit value stored as [high u16][low u16].
struct DspU32 {
    u16 hi, lo;
    u32 get() const { return ((u32)hi << 16) | lo; }
    void set(u32 v) { hi = (u16)(v >> 16); lo = (u16)v; }
    operator u32() const { return get(); }
};
static_assert(sizeof(DspU32) == 4);

enum : u32 {  // SourceConfig::dirty bits
    SRC_FORMAT_DIRTY = 1u << 0,
    SRC_MONO_STEREO_DIRTY = 1u << 1,
    SRC_ADPCM_COEFFS_DIRTY = 1u << 2,
    SRC_PARTIAL_EMBEDDED_DIRTY = 1u << 3,
    SRC_PARTIAL_RESET = 1u << 4,
    SRC_ENABLE_DIRTY = 1u << 16,
    SRC_INTERP_DIRTY = 1u << 17,
    SRC_RATE_DIRTY = 1u << 18,
    SRC_BUFFER_QUEUE_DIRTY = 1u << 19,
    SRC_LOOP_RELATED_DIRTY = 1u << 20,
    SRC_PLAY_POSITION_DIRTY = 1u << 21,
    SRC_FILTERS_ENABLED_DIRTY = 1u << 22,
    SRC_SIMPLE_FILTER_DIRTY = 1u << 23,
    SRC_BIQUAD_FILTER_DIRTY = 1u << 24,
    SRC_GAIN0_DIRTY = 1u << 25,
    SRC_GAIN1_DIRTY = 1u << 26,
    SRC_GAIN2_DIRTY = 1u << 27,
    SRC_SYNC_COUNT_DIRTY = 1u << 28,
    SRC_RESET = 1u << 29,
    SRC_EMBEDDED_DIRTY = 1u << 30,
};

enum class Interp : u8 { Polyphase = 0, Linear = 1, None = 2 };
enum class Format : u8 { PCM8 = 0, PCM16 = 1, ADPCM = 2 };

struct SourceBuffer {        // one entry of the buffer queue (20 bytes)
    DspU32 paddr;            // physical address of the sample data
    DspU32 length;           // in samples
    u16 adpcm_ps;            // predictor/scale (unused: frames carry their own header)
    u16 adpcm_yn[2];         // y[n-1], y[n-2]
    u8 adpcm_dirty;          // non-zero: load adpcm_yn when this buffer starts
    u8 is_looping;
    u16 buffer_id;
    u16 pad;
};
static_assert(sizeof(SourceBuffer) == 20);

struct SourceConfig {        // per source, written by the application (192 bytes)
    u32 dirty;
    float gain[3][4];        // [intermediate mix][front L, front R, back L, back R]
    float rate_multiplier;
    u8 interpolation_mode;   // Interp
    u8 pad0;
    u16 filters_enabled;     // bit0 simple, bit1 biquad
    s16 simple_b0, simple_a1;                                    // Q15, H = b0 / (1 - a1 z^-1)
    s16 biquad_a2, biquad_a1, biquad_b2, biquad_b1, biquad_b0;  // Q14, feedback negated
    u16 buffers_dirty;       // bit i -> buffers[i]
    SourceBuffer buffers[4];
    DspU32 loop_related;
    u8 enable;
    u8 pad1;
    u16 sync_count;
    DspU32 play_position;    // start sample of the embedded buffer
    u16 pad2[2];
    // embedded buffer
    DspU32 paddr;
    DspU32 length;
    u16 flags1;              // bits 0-1 mono(1)/stereo(2), bits 2-3 format, bit 5 fade in
    u16 adpcm_ps;
    u16 adpcm_yn[2];
    u16 flags2;              // bit0 adpcm dirty, bit1 looping
    u16 buffer_id;

    unsigned channels() const { return (flags1 & 3) == 2 ? 2 : 1; }
    Format format() const { return (Format)((flags1 >> 2) & 3); }
};
static_assert(sizeof(SourceConfig) == 192);
static_assert(offsetof(SourceConfig, buffers_dirty) == 74);
static_assert(offsetof(SourceConfig, loop_related) == 156);
static_assert(offsetof(SourceConfig, paddr) == 172);

struct SourceStatus {        // per source, written by the DSP (12 bytes)
    u8 is_enabled;
    u8 current_buffer_id_dirty;
    u16 sync_count;
    DspU32 buffer_position;
    u16 current_buffer_id;
    u16 last_buffer_id;
};
static_assert(sizeof(SourceStatus) == 12);

enum : u32 {  // DspConfig::dirty bits
    CFG_AUX_FRONT_BYPASS0_DIRTY = 1u << 6,
    CFG_AUX_FRONT_BYPASS1_DIRTY = 1u << 7,
    CFG_AUX_BUS_ENABLE0_DIRTY = 1u << 8,
    CFG_AUX_BUS_ENABLE1_DIRTY = 1u << 9,
    CFG_DELAY0_DIRTY = 1u << 10,
    CFG_DELAY1_DIRTY = 1u << 11,
    CFG_REVERB0_DIRTY = 1u << 12,
    CFG_REVERB1_DIRTY = 1u << 13,
    CFG_OUTPUT_BUFFER_COUNT_DIRTY = 1u << 15,
    CFG_MASTER_VOLUME_DIRTY = 1u << 16,
    CFG_AUX_RETURN_VOLUME0_DIRTY = 1u << 24,
    CFG_AUX_RETURN_VOLUME1_DIRTY = 1u << 25,
    CFG_OUTPUT_FORMAT_DIRTY = 1u << 26,
    CFG_CLIPPING_MODE_DIRTY = 1u << 27,
    CFG_HEADPHONES_DIRTY = 1u << 28,
    CFG_SURROUND_DEPTH_DIRTY = 1u << 29,
    CFG_SURROUND_SPEAKER_POS_DIRTY = 1u << 30,
    CFG_REAR_RATIO_DIRTY = 1u << 31,
};

struct DelayConfig {         // 20 bytes. H(z) = a z^-N / (1 - b z^-1 + a g z^-N), Q7 coefficients
    u16 dirty;               // bit0 enable, bit1 work buffer, bit2 other
    u16 enable;
    u16 pad;
    u16 outputs;
    DspU32 work_buffer;
    u16 frame_count;         // N = frame_count * 160
    s16 g, a, b;
};
static_assert(sizeof(DelayConfig) == 20);

struct DspConfig {           // written by the application (196 bytes)
    u32 dirty;
    float master_volume;     // volume of intermediate mix 0 in the final mix
    float aux_return_volume[2];  // volumes of intermediate mixes 1 and 2
    u16 output_buffer_count;
    u16 pad0[2];
    u16 output_format;       // 0 mono, 1 stereo, 2 surround
    u16 clipping_mode;
    u16 headphones_connected;
    u16 surround_depth;
    u16 surround_speaker_position;
    u16 pad1;
    u16 rear_ratio;
    u16 aux_front_bypass[2];
    u16 aux_bus_enable[2];
    DelayConfig delay[2];
    u16 reverb[2][26];
    u16 sync_mode;
    u16 pad2;
    u32 dirty2;
};
static_assert(sizeof(DspConfig) == 196);
static_assert(offsetof(DspConfig, delay) == 44);
static_assert(offsetof(DspConfig, sync_mode) == 0xBC);

struct DspStatusBlock {      // 32 bytes
    u16 unknown;
    u16 dropped_frames;
    u16 pad[14];
};
static_assert(sizeof(DspStatusBlock) == 32);

struct IntermediateMix {     // little-endian s32, [channel][sample]
    s32 pcm32[4][kFrameSamples];
};

struct SharedMemory {
    u16 pad0[0x400];
    DspStatusBlock dsp_status;                  // 0x8400
    u16 dsp_debug[0x130];                       // 0x8410
    s16 final_samples[kFrameSamples][2];        // 0x8540
    SourceStatus source_status[kNumSources];    // 0x8680
    u16 compressor[0xD20];                      // 0x8710
    DspConfig dsp_config;                       // 0x9430
    IntermediateMix intermediate[2];            // 0x9492 (aux bus 0 = mix 1, aux bus 1 = mix 2)
    SourceConfig source_config[kNumSources];    // 0x9E92
    s16 adpcm_coeffs[kNumSources][16];          // 0xA792 (Q11)
    u16 unknown10[0x100];                       // 0xA912
    u16 unknown11[0xC0];                        // 0xAA12
    u16 unknown12[0x180];                       // 0xAAD2
    u16 unknown13[0xA];                         // 0xAC52
    u16 unknown14[0x13A3];                      // 0xAC5C
    u16 frame_counter;                          // 0xBFFF
};
static_assert(sizeof(SharedMemory) == 0x8000);
static_assert(offsetof(SharedMemory, dsp_status) == 0x800);
static_assert(offsetof(SharedMemory, final_samples) == 0xA80);
static_assert(offsetof(SharedMemory, source_status) == 0xD00);
static_assert(offsetof(SharedMemory, dsp_config) == 0x2860);
static_assert(offsetof(SharedMemory, intermediate) == 0x2924);
static_assert(offsetof(SharedMemory, source_config) == 0x3D24);
static_assert(offsetof(SharedMemory, adpcm_coeffs) == 0x4F24);
static_assert(offsetof(SharedMemory, frame_counter) == 0x7FFE);

#pragma pack(pop)

// DSP word address of a SharedMemory member in region 0.
constexpr u16 dsp_word_addr(size_t byte_offset) { return (u16)(0x8000 + byte_offset / 2); }

using Sample = std::array<s16, 2>;  // stereo PCM16 sample

}  // namespace audio::hle
