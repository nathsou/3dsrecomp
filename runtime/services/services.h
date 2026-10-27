// Cross-service hooks.
#pragma once
#include <vector>

#include "../kernel.h"

namespace kernel {
void register_apt();
void register_fs();
void register_cfg();
void register_gsp();
void register_hid();
void register_dsp();
void register_misc();

std::shared_ptr<SharedMem> load_shared_font();
// a file of a system archive (system_archives/<title id>.romfs), empty if missing
std::vector<u8> system_archive_file(u64 tid, const std::string &path);
void apt_notify(u32 signal);

// gsp
struct FramebufferInfo {
    u32 addr_left, addr_right, stride, format;
    bool valid;
};
FramebufferInfo gsp_framebuffer(int screen);  // what the LCD shows (any thread)
void gsp_vblank();                             // (host thread, lock not held) VBlank interrupts
void gsp_process_gx();                         // (GPU/host thread) run queued GX commands
bool gsp_gx_pending();
struct GxStats { u64 cmdlists, gpu_ns, transfers; };
extern GxStats gx_stats;
// hid
void hid_update(u32 buttons, s16 cx, s16 cy, bool touch, u16 tx, u16 ty);  // (host thread, lock not held)
// dsp
void dsp_tick();  // (host thread, lock not held) one audio frame
}
