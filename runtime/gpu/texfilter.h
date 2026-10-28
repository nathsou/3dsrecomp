// Texture upscaling filters of the Metal backend (Metal compute kernels, runtime/gpu/texfilter.mm).
#pragma once
#import <Metal/Metal.h>

#include "rt.h"

namespace gpu {
enum TexFilter { kTexFilterOff = 0, kTexFilterSharp = 1, kTexFilterSmooth = 2, kTexFilterCount = 3 };

struct TfRect {
    u32 x, y, w, h;
};

// Encode into cb the upscaling of src (one level, w x h, RGBA8) by `factor` (2 or 4) into level 0 of dst
// (w*factor x h*factor), then levels 1..log2(factor)-1 of dst by alpha-aware 2x2 reduction of level 0.
// wrap_s / wrap_t: how the neighbourhood continues past the edges (0 clamp, 1 repeat, 2 mirror), so that
// tiled textures get no seams. rects (source texels): only update these parts (all if none); the output there
// depends on the source up to 3 texels around them. mid: the intermediate image of the sharp filter at 4x
// (texfilter_new_mid; a partial update needs the one of the previous update), else nil. False if the kernels
// are unavailable.
bool texfilter_encode(id<MTLDevice> dev, id<MTLCommandBuffer> cb, int mode, id<MTLTexture> src, id<MTLTexture> dst,
                      id<MTLTexture> mid, u32 factor, u32 wrap_s, u32 wrap_t, const TfRect *rects = nullptr,
                      u32 nrects = 0);
id<MTLTexture> texfilter_new_mid(id<MTLDevice> dev, u32 w, u32 h);
}  // namespace gpu
