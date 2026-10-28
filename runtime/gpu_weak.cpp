// Weak fallbacks for the GPU entry points (the real ones live in runtime/gpu/).
#include "gpu_iface.h"

namespace gpu {
__attribute__((weak)) void init() {}
__attribute__((weak)) void process_command_list(u32, u32) {}
__attribute__((weak)) void memory_fill(u32, u32, u32, u32) {}
__attribute__((weak)) void display_transfer(u32, u32, u32, u32, u32) {}
__attribute__((weak)) void invalidate_memory(u32, u32) {}
__attribute__((weak)) bool save_state(const char *) { return false; }
__attribute__((weak)) bool get_display_image_for(u32, u32, u32, const u32 **, int *, int *) { return false; }
__attribute__((weak)) void set_trust_invalidations(bool) {}
__attribute__((weak)) void texture_copy(u32, u32, u32, u32, u32, u32) {}
__attribute__((weak)) int emit_shaders(int, char **) { return 1; }
__attribute__((weak)) void set_texture_filter(int) {}
__attribute__((weak)) int texture_filter_count() { return 1; }
__attribute__((weak)) const char *texture_filter_name(int) { return "Off"; }
}
