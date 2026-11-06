// Texture upscaling filters of the Vulkan backend (runtime/gpu/vk_texfilter.cpp): GLSL ports of texfilter.mm's
// kernels, dispatched by vulkan.cpp's texture cache as metal.mm's does.
#pragma once
#include "rt.h"

namespace gpu {
enum VkTexFilter { kVkTfOff = 0, kVkTfSharp = 1, kVkTfSmooth = 2, kVkTfCount = 3 };
enum VkTfKernel { kVkTfKernSharp = 0, kVkTfKernSmooth = 1, kVkTfKernDown = 2, kVkTfKernCount = 3 };
// The compute shader of a kernel: input texture2D at binding 0 (texelFetch), output rgba8 image2D at binding 1, the
// rectangles (uvec4 x, y, w, h; one per z of the grid) at binding 2, the parameters TFP (sw, sh, dw, dh, wrap_s,
// wrap_t, factor, pad) as push constants; 16x16 work groups.
const char *vk_texfilter_glsl(VkTfKernel k);
}  // namespace gpu
