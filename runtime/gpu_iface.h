// The GPU entry points the GSP service drives (implemented in runtime/gpu/).
#pragma once
#include "rt.h"

#include <functional>

namespace gpu {
void init();
void process_command_list(u32 paddr, u32 size_bytes);
void memory_fill(u32 start_paddr, u32 end_paddr, u32 value, u32 control);
void display_transfer(u32 in_paddr, u32 out_paddr, u32 in_dims, u32 out_dims, u32 flags);
void invalidate_memory(u32 paddr, u32 size);
bool save_state(const char *path);
// A high-resolution upright RGBA image of what the framebuffer at fb_paddr shows, if the GPU backend has one
// (called from the UI thread at VBlank).
bool get_display_image_for(u32 fb_paddr, const u32 **rgba, int *w, int *h);
u64 use_display_image(u32 fb_paddr, u64 have, const std::function<void(const u32 *, int, int)> &fn);
void set_trust_invalidations(bool trust);
// Metal internal resolution scale (1-4; 0 without Metal); auto: GPU_SCALE not given (see gpu/gpu.h)
void set_scale(int s);
int scale();
bool scale_is_auto();
float busy();  // fraction of the last ~2 s the GPU spent on our work (0 without Metal)
void texture_copy(u32 in_paddr, u32 out_paddr, u32 size, u32 in_width_gap, u32 out_width_gap, u32 flags);
}

namespace gpu {
// build step: write the fragment shaders of pipeline lists as .metal files (see runtime/gpu/metal.mm)
int emit_shaders(int argc, char **argv);
}
