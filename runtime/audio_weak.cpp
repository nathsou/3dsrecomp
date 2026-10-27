// Weak fallbacks for the audio entry points (the real ones live in runtime/audio/): just enough of the
// DSP pipe protocol for a game to initialise its sound system, without producing sound.
#include <mutex>

#include "audio_iface.h"

namespace audio {
static std::mutex m;
static std::vector<u8> pipe2;
static bool on;

__attribute__((weak)) void init() {}

__attribute__((weak)) void pipe_write(u32 pipe, const u8 *data, u32 size) {
    if (pipe != 2 || size < 1) return;
    std::lock_guard<std::mutex> lk(m);
    u8 state = data[0];
    on = state == 0 || state == 2;
    if (state == 1) return;
    static const u16 addrs[15] = {0xBFFF, 0x9E92, 0x8680, 0xA792, 0x9430, 0x8400, 0x8540, 0x9492,
                                  0x8710, 0x8410, 0xA912, 0xAA12, 0xAAD2, 0xAC52, 0xAC5C};
    auto put = [&](u16 v) {
        pipe2.push_back((u8)v);
        pipe2.push_back((u8)(v >> 8));
    };
    put(15);
    for (u16 a : addrs) put(a);
}

__attribute__((weak)) std::vector<u8> pipe_read(u32 pipe, u32 size) {
    std::lock_guard<std::mutex> lk(m);
    if (pipe != 2) return {};
    size = std::min<u32>(size, (u32)pipe2.size());
    std::vector<u8> out(pipe2.begin(), pipe2.begin() + size);
    pipe2.erase(pipe2.begin(), pipe2.begin() + size);
    return out;
}

__attribute__((weak)) u32 pipe_readable(u32 pipe) {
    std::lock_guard<std::mutex> lk(m);
    return pipe == 2 ? (u32)pipe2.size() : 0;
}

__attribute__((weak)) u16 recv_data(u32) { return on ? 0 : 1; }

__attribute__((weak)) bool tick() {
    if (!on) return false;
    // bump the frame counter of both shared memory regions
    static u16 frame;
    frame++;
    for (u32 region : {0x1FF50000u, 0x1FF60000u}) w16(region + 0x7FFE, frame);
    return true;
}
}  // namespace audio
