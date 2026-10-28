// Software PICA200 GPU.
//
// All functions take GPU physical addresses and operate directly on guest memory (g_mem): every
// rendering result (color / depth buffers, transfer outputs) is written back to guest memory in
// the PICA's native formats before the call returns. Callable from one thread at a time.
#pragma once
#include "cpu.h"

#include <functional>

namespace gpu {
void init();
// Execute a PICA command list (register writes, draws, command-buffer jumps).
void process_command_list(u32 paddr, u32 size_bytes);
// GX MemoryFill. control: bit0 start, bit8 24-bit fill, bit9 32-bit fill (else 16-bit).
void memory_fill(u32 start_paddr, u32 end_paddr, u32 value, u32 control);
// GX DisplayTransfer (tiled<->linear, format conversion, downscale, flip), hw register layout.
void display_transfer(u32 in_paddr, u32 out_paddr, u32 in_dims, u32 out_dims, u32 flags);
// GX TextureCopy.
void texture_copy(u32 in_paddr, u32 out_paddr, u32 size, u32 in_width_gap, u32 out_width_gap,
                  u32 flags);
// Make sure all queued rendering has landed in guest memory. With the software renderer every
// entry point already does this before returning. With the Metal backend, render-target contents
// and the output of display transfers from render targets land lazily: at the next gpu:: call that
// needs them (anything reading or writing that memory), or here. Guest code never observes the
// difference, but a reader outside the GPU module (screenshot code reading a framebuffer from guest
// memory) should call flush() first, on the GX thread; get_display_image_for() needs no flush.
void flush();
// Tell the GPU that guest memory was modified by the CPU or by DMA (e.g. on GSP FlushDataCache /
// RequestDma), so cached decoded textures in that range are re-validated.
void invalidate_memory(u32 paddr, u32 size);
// By default every command list re-validates (hashes) the textures it uses, which is safe even
// if CPU writes are never reported. Once invalidate_memory() is called for all CPU/DMA writes
// that the GPU may read, pass true here to skip that hashing.
void set_trust_invalidations(bool trust);
// Save / restore the GPU's internal state (registers, shader code and uniforms, LUTs, ...).
// A dump tool should call save_state() whenever it snapshots guest memory, so that a replay can
// start from the exact same state (command lists rely on registers set by earlier lists).
bool save_state(const char *path);
bool load_state(const char *path);

// High-resolution display (Metal backend, GPU_SCALE > 1). If the framebuffer at fb_paddr (`lines` rows of 240
// pixels, `stride` bytes apart; 0: any) was last written by a display transfer from a GPU render target (and not
// modified since; the transfer's output may start before it), returns an
// upright RGBA8 image (bytes R,G,B,A) of it at the internal resolution: top screen
// 400*scale x 240*scale, bottom 320*scale x 240*scale. The pointer stays valid until the next call
// from the same thread. Safe to call from another thread than the one driving the GPU.
bool get_display_image_for(u32 fb_paddr, u32 stride, u32 lines, const u32 **rgba, int *w, int *h);
// Without the copy: calls fn(rgba, w, h) under the image lock if the image at fb_paddr is not the one
// with version `have`; returns the version of the image there (0: none).
u64 use_display_image(u32 fb_paddr, u32 stride, u32 lines, u64 have,
                      const std::function<void(const u32 *, int, int)> &fn);
// Same, for the most recent display transfer to screen 0 (top, 400 lines) or 1 (bottom, 320).
bool get_display_image(int screen, const u32 **rgba, int *w, int *h);
// Internal resolution scale of the Metal backend (1-4), changeable while running from any thread (it
// takes effect at the next draw); scale() is 0 without the Metal backend. scale_is_auto(): Metal is
// active and GPU_SCALE was not given, so the frontend should pick the scale from the window size.
void set_scale(int s);
int scale();
bool scale_is_auto();
// Texture filter of the Metal backend (0 off, else a texture upscaler: gpu/texfilter.mm), changeable while
// running from any thread (the GPU thread applies it at the next draw, rebuilding its textures).
// GPU_TEX_FILTER=<mode> sets it at start.
void set_texture_filter(int mode);
int texture_filter_count();
const char *texture_filter_name(int mode);
}  // namespace gpu
