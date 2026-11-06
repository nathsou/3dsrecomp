// The hardware GPU backends (pica.h's HwBackend): which one runs, and the frontend's calls that go to it. Metal
// (metal.mm, Apple) and Vulkan (vulkan.cpp, RECOMP_VULKAN) register themselves by defining metal_backend() and
// vulkan_backend(); the weak definitions below stand in for the ones not built. Without a backend the software
// renderer draws everything.
#include <cstring>

#include "gpu/pica.h"

namespace gpu {

const HwBackend *g_hw = nullptr;

__attribute__((weak)) const HwBackend *metal_backend() { return nullptr; }
__attribute__((weak)) const HwBackend *vulkan_backend() { return nullptr; }

namespace {
// settings given before a backend started, applied when one does
int pending_filter = -1;
bool clear_after_top = false, key_black_bottom = false;

// GPU_BACKEND=metal|vulkan names one; otherwise the first built (Metal, then Vulkan)
bool wanted(const HwBackend *b) {
    const char *be = getenv("GPU_BACKEND");
    return b && (!be || !*be || strcmp(be, b->name) == 0);
}
}  // namespace

bool hw_init() {
    static bool tried = false;
    if (tried) return g_hw != nullptr;
    tried = true;
    const char *be = getenv("GPU_BACKEND");
    if (be && strcmp(be, "sw") == 0) return false;
    for (const HwBackend *b : {metal_backend(), vulkan_backend()}) {
        if (!wanted(b) || !b->init()) continue;
        g_hw = b;
        if (pending_filter >= 0) b->set_texture_filter(pending_filter);
        if (clear_after_top) b->set_clear_after_top_transfer(true);
        if (key_black_bottom) b->set_key_black_bottom(true);
        return true;
    }
    if (be && *be) LOG("gpu: no %s backend here, using the software renderer\n", be);
    return false;
}

bool get_display_image_for(u32 fb_paddr, u32 stride, u32 lines, const u32 **rgba, int *w, int *h) {
    return g_hw && g_hw->get_display_image_for(fb_paddr, stride, lines, rgba, w, h);
}
u64 use_display_image(u32 fb_paddr, u32 stride, u32 lines, u64 have, const std::function<void(const ScreenImage &)> &fn) {
    return g_hw ? g_hw->use_display_image(fb_paddr, stride, lines, have, fn) : 0;
}
namespace {
std::vector<std::string> present_exts;
}
void set_present_extensions(const std::vector<std::string> &exts) { present_exts = exts; }
const std::vector<std::string> &present_extensions() { return present_exts; }
bool present_device(PresentDevice *pd) { return g_hw && g_hw->present_device && g_hw->present_device(pd); }
u64 present_image_create(int w, int h) { return g_hw && g_hw->present_image_create ? g_hw->present_image_create(w, h) : 0; }
void present_image_destroy(u64 image) {
    if (g_hw && g_hw->present_image_destroy && image) g_hw->present_image_destroy(image);
}
void present_hold(bool on) {
    if (g_hw && g_hw->present_hold) g_hw->present_hold(on);
}
bool get_display_image(int screen, const u32 **rgba, int *w, int *h) {
    return g_hw && g_hw->get_display_image(screen, rgba, w, h);
}
void set_scale(int s) {
    if (g_hw) g_hw->set_scale(s);
}
int scale() { return g_hw ? g_hw->scale() : 0; }
bool scale_is_auto() { return g_hw && g_hw->scale_is_auto(); }
float busy() { return g_hw ? g_hw->busy() : 0; }
void set_texture_filter(int mode) {
    if (g_hw) g_hw->set_texture_filter(mode);
    else pending_filter = mode;
}
int texture_filter_count() { return g_hw ? g_hw->texture_filter_count() : 1; }
const char *texture_filter_name(int mode) { return g_hw ? g_hw->texture_filter_name(mode) : "Off"; }
void set_clear_after_top_transfer(bool on) {
    clear_after_top = on;
    if (g_hw) g_hw->set_clear_after_top_transfer(on);
}
void set_key_black_bottom(bool on) {
    key_black_bottom = on;
    if (g_hw) g_hw->set_key_black_bottom(on);
}

// build step (no backend started): the backend GPU_BACKEND names, else the first built that precompiles shaders
int emit_shaders(int argc, char **argv) {
    for (const HwBackend *b : {metal_backend(), vulkan_backend()})
        if (wanted(b) && b->emit_shaders) return b->emit_shaders(argc, argv);
    LOG("--emit-shaders: no GPU backend here precompiles shaders\n");
    return 1;
}

}  // namespace gpu
