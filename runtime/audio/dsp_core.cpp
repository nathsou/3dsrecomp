// DSP firmware HLE: pipes, power state changes and the per-frame processing (audio:: interface).
#include <mutex>

#include "../audio_iface.h"
#include "dsp_hle.h"
#include "host.h"

namespace audio::hle {

namespace {

enum class Power { Off, On, Sleeping };

struct Dsp {
    std::mutex lock;  // guards everything below (pipe calls come from game threads, tick from a timer)
    Power power = Power::Off;
    std::vector<u8> pipes[kNumPipes];
    std::array<Source, kNumSources> sources;
    Mixer mixer;
    // state saved across sleep
    std::vector<u8> backup_ram;
    std::array<Source, kNumSources> backup_sources;
    Mixer backup_mixer;
    u64 frames = 0;
};

Dsp &dsp() {
    static Dsp *d = new Dsp;  // never destroyed: the timer thread may still tick at exit
    return *d;
}

u8 *dsp_ram() { return g_mem + mem::DSP_BASE; }
SharedMemory &region(int i) { return *(SharedMemory *)(dsp_ram() + (i ? kRegion1 : kRegion0)); }

u16 frame_counter(int i) {
    u16 v;
    memcpy(&v, (const u8 *)&region(i) + offsetof(SharedMemory, frame_counter), 2);
    return v;
}

// The application submits the region whose frame counter is newer (with wraparound).
int read_region_index() {
    const u16 c0 = frame_counter(0), c1 = frame_counter(1);
    if (c0 == 0xFFFF && c1 != 0xFFFE) return 1;
    if (c1 == 0xFFFF && c0 != 0xFFFE) return 0;
    return c0 > c1 ? 0 : 1;
}

void push_u16(std::vector<u8> &p, u16 v) {
    p.push_back((u8)v);
    p.push_back((u8)(v >> 8));
}

void write_struct_addresses(Dsp &d) {
    static constexpr u16 addrs[] = {
        dsp_word_addr(offsetof(SharedMemory, frame_counter)),
        dsp_word_addr(offsetof(SharedMemory, source_config)),
        dsp_word_addr(offsetof(SharedMemory, source_status)),
        dsp_word_addr(offsetof(SharedMemory, adpcm_coeffs)),
        dsp_word_addr(offsetof(SharedMemory, dsp_config)),
        dsp_word_addr(offsetof(SharedMemory, dsp_status)),
        dsp_word_addr(offsetof(SharedMemory, final_samples)),
        dsp_word_addr(offsetof(SharedMemory, intermediate)),
        dsp_word_addr(offsetof(SharedMemory, compressor)),
        dsp_word_addr(offsetof(SharedMemory, dsp_debug)),
        dsp_word_addr(offsetof(SharedMemory, unknown10)),
        dsp_word_addr(offsetof(SharedMemory, unknown11)),
        dsp_word_addr(offsetof(SharedMemory, unknown12)),
        dsp_word_addr(offsetof(SharedMemory, unknown13)),
        dsp_word_addr(offsetof(SharedMemory, unknown14)),
    };
    static_assert(sizeof(addrs) / sizeof(addrs[0]) == 15);
    auto &p = d.pipes[2];
    push_u16(p, 15);
    for (u16 a : addrs) push_u16(p, a);
}

void reset_pipes(Dsp &d) {
    for (auto &p : d.pipes) p.clear();
}

void initialize(Dsp &d) {
    memset(dsp_ram(), 0, kDspRamSize);
    d.mixer.reset();
    for (auto &s : d.sources) s.reset();
    reset_pipes(d);
}

void sleep(Dsp &d) {
    d.backup_ram.assign(dsp_ram(), dsp_ram() + kDspRamSize);
    d.backup_sources = d.sources;
    d.backup_mixer = d.mixer;
}

void wakeup(Dsp &d) {
    if (d.backup_ram.size() == kDspRamSize) memcpy(dsp_ram(), d.backup_ram.data(), kDspRamSize);
    d.backup_ram.clear();
    d.sources = d.backup_sources;
    d.mixer = d.backup_mixer;
    for (auto &s : d.backup_sources) s.reset();
    d.backup_mixer.reset();
    reset_pipes(d);
}

StereoFrame generate_frame(Dsp &d) {
    const int ri = read_region_index();
    SharedMemory &rd = region(ri);
    SharedMemory &wr = region(ri ^ 1);

    std::array<QuadFrame, 3> mixes{};
    for (int i = 0; i < kNumSources; i++) {
        s16 coeffs[16];
        memcpy(coeffs, rd.adpcm_coeffs[i], sizeof(coeffs));
        SourceStatus s = d.sources[i].tick(rd.source_config[i], coeffs);
        memcpy(&wr.source_status[i], &s, sizeof(s));
        for (int m = 0; m < 3; m++) d.sources[i].mix_into(mixes[m], m);
    }

    d.mixer.tick(rd.dsp_config, rd.intermediate, wr.intermediate, mixes);
    DspStatusBlock status{};
    memcpy(&wr.dsp_status, &status, sizeof(status));

    StereoFrame out = d.mixer.output();
    memcpy(wr.final_samples, out.data(), sizeof(wr.final_samples));
    d.frames++;
    return out;
}

}  // namespace

}  // namespace audio::hle

namespace audio {
using namespace hle;

void pipe_write(u32 pipe, const u8 *data, u32 size) {
    Dsp &d = dsp();
    std::lock_guard lk(d.lock);
    if (pipe != 2) {
        // pipe 3 is the AAC decoder (binary pipe): unsupported
        LOG("dsp: write of %u bytes to unsupported pipe %u ignored\n", size, pipe);
        return;
    }
    if (size != 4) {
        LOG("dsp: unexpected audio pipe write of %u bytes\n", size);
        return;
    }
    switch (data[0]) {
    case 0:  // initialize
        VLOG("dsp: initialize\n");
        initialize(d);
        write_struct_addresses(d);
        d.power = Power::On;
        break;
    case 1:  // shutdown
        VLOG("dsp: shutdown\n");
        d.power = Power::Off;
        break;
    case 2:  // wakeup
        VLOG("dsp: wakeup\n");
        wakeup(d);
        write_struct_addresses(d);
        d.power = Power::On;
        break;
    case 3:  // sleep
        VLOG("dsp: sleep\n");
        sleep(d);
        write_struct_addresses(d);
        d.power = Power::Sleeping;
        break;
    default:
        LOG("dsp: unknown state change %u\n", data[0]);
        d.power = Power::Off;
        break;
    }
}

std::vector<u8> pipe_read(u32 pipe, u32 size) {
    Dsp &d = dsp();
    std::lock_guard lk(d.lock);
    if (pipe >= kNumPipes) return {};
    auto &p = d.pipes[pipe];
    size = std::min<u32>({size, (u32)p.size(), 0xFFFFu});
    std::vector<u8> out(p.begin(), p.begin() + size);
    p.erase(p.begin(), p.begin() + size);
    return out;
}

u32 pipe_readable(u32 pipe) {
    Dsp &d = dsp();
    std::lock_guard lk(d.lock);
    return pipe < kNumPipes ? (u32)d.pipes[pipe].size() : 0;
}

u16 recv_data(u32 /*reg*/) {
    Dsp &d = dsp();
    std::lock_guard lk(d.lock);
    return d.power == Power::On ? 0 : 1;
}

bool tick() {
    Dsp &d = dsp();
    StereoFrame out;
    {
        std::lock_guard lk(d.lock);
        if (d.power != Power::On) return false;
        out = generate_frame(d);
    }
    host_output(out.data(), out.size());
    return true;
}

}  // namespace audio
