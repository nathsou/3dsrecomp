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
void register_ro();
void register_y2r();

std::shared_ptr<SharedMem> load_shared_font();
// a file of a system archive (system_archives/<title id>.romfs), empty if missing
std::vector<u8> system_archive_file(u64 tid, const std::string &path);
// a file of the game's RomFS by path ("/LayoutData/x.szs"), empty if missing
std::vector<u8> romfs_file(const std::string &path);
void apt_notify(u32 signal);
// a Mii (the stored form, CFLiCharData)
constexpr size_t MII_DATA_SIZE = 0x5C;
std::vector<u8> standard_mii();          // the Mii the Mii selector returns
u16 crc16_xmodem(const u8 *p, size_t n);  // the checksum of Mii data and of Mii Maker's database

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
// (host thread, lock not held) sx, sy: the second circle pad (the host's right stick), -0x7FF..0x7FF, up positive
void hid_update(u32 buttons, s16 cx, s16 cy, bool touch, u16 tx, u16 ty, s16 sx, s16 sy);
// ir (lock held): the Circle Pad Pro's state, sent to the game at the polling period it asked for
void ir_poll(u32 buttons, s16 sx, s16 sy);
// dsp
void dsp_tick();  // (host thread, lock not held) one audio frame
}
