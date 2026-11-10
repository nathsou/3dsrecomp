// The GPU entry points the GSP service drives (implemented in runtime/gpu/).
#pragma once
#include "rt.h"

#include <functional>
#include <string>
#include <vector>

namespace gpu {
void init();
void process_command_list(u32 paddr, u32 size_bytes);
void memory_fill(u32 start_paddr, u32 end_paddr, u32 value, u32 control);
void display_transfer(u32 in_paddr, u32 out_paddr, u32 in_dims, u32 out_dims, u32 flags);
void invalidate_memory(u32 paddr, u32 size);
void vblank();  // (any thread) a new frame: see gpu/gpu.h
bool save_state(const char *path);
// A high-resolution upright RGBA image of what the framebuffer at fb_paddr shows, if the GPU backend has one
// (called from the UI thread at VBlank).
bool get_display_image_for(u32 fb_paddr, u32 stride, u32 lines, const u32 **rgba, int *w, int *h);
// use_display_image: fn gets the image when it is not the one the UI has (have: the version it has; returns the
// version shown, 0 for none): its pixels (pitch: pixels from one row to the next) and, with a backend that has present
// images, copy_to, which puts it into one on the GPU instead (true: done; the renderer may draw it from then on)
struct ScreenImage {
    const u32 *px;
    int w, h, pitch;
    std::function<bool(u64 image)> copy_to;
};
u64 use_display_image(u32 fb_paddr, u32 stride, u32 lines, u64 have, const std::function<void(const ScreenImage &)> &fn);
// Presenting through the GPU backend's own device (the Vulkan backend, with SDL's Vulkan renderer made on it): no
// read-back of the screens' images. set_present_extensions: before init, the instance extensions the window system
// needs; present_device: after it, where the renderer goes (false: not possible, the images go through the CPU).
// present_image_create: a w x h VkImage (VK_FORMAT_R8G8B8A8_UNORM) the renderer can wrap (in
// SHADER_READ_ONLY_OPTIMAL layout), destroyed with present_image_destroy once the renderer's texture of it is; with
// the Metal backend, a CVPixelBufferRef (RGBA bytes) for SDL's Metal renderer, which needs no device of ours. Around
// destroying a renderer made on the device, present_hold(true) / (false) keep the backend from submitting work (SDL
// waits for the whole device then).
void set_present_extensions(const std::vector<std::string> &exts);
struct PresentDevice {
    void *instance, *physical_device, *device;
    u32 queue_family;
};
bool present_device(PresentDevice *pd);
u64 present_image_create(int w, int h);
void present_image_destroy(u64 image);
void present_hold(bool on);
void set_trust_invalidations(bool trust);
// Metal internal resolution scale (1-4; 0 without Metal); auto: GPU_SCALE not given (see gpu/gpu.h)
void set_scale(int s);
int scale();
bool scale_is_auto();
float busy();  // fraction of the last ~2 s the GPU spent on our work (0 without Metal)
// Texture filter of the Metal backend: upscales the game's textures once when they are loaded (0 = off,
// the default; GPU_TEX_FILTER=<mode> at launch). Callable from any thread; takes effect at the next draw.
void set_texture_filter(int mode);
int texture_filter_count();                 // modes, including 0 (off)
const char *texture_filter_name(int mode);  // "Off", "Sharp", "Smooth"
// (Metal) clear render targets to transparent after each display transfer from them, before the next draw into
// them (see runtime/gpu/metal.mm)
void set_clear_after_top_transfer(bool on);
// (Metal) the bottom screen's display images with black made transparent (one-screen play)
void set_key_black_bottom(bool on);
void texture_copy(u32 in_paddr, u32 out_paddr, u32 size, u32 in_width_gap, u32 out_width_gap, u32 flags);
}

namespace gpu {
// build step: write the fragment shaders of pipeline lists as .metal files (see runtime/gpu/metal.mm)
int emit_shaders(int argc, char **argv);
}
