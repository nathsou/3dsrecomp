// What the hardware backends (metal.mm, vulkan.cpp) share: everything about a draw, a render target or a display
// image that is not a call into Metal or Vulkan. Each backend keeps only the encoding of its GPU work.
#pragma once
#include <deque>
#include <unordered_map>
#include <memory>
#include <vector>

#include "gpu/pica.h"

namespace gpu {

// ---- a draw's uniforms: the C++ mirror of fsgen.cpp's Uni (the same layout in MSL and in GLSL's std140)
struct alignas(16) LightU {
    float spec0[4], spec1[4], diffuse[4], ambient[4], pos[4], spot[4], att[4];
};
struct alignas(16) Uni {
    float rt[4];
    float depth[4];
    s32 scissor[4];
    u32 konst[8];
    u32 tevc[4];
    u32 misc[4];
    u32 border[4];
    float bias[4];
    float lscale[8];
    float ambient[4];
    LightU lights[8];
};
static_assert(sizeof(Uni) == 16 * 12 + 8 * 112, "uniform layout");

// a draw's uniforms (target of rt.w x rt.h at `scale`)
void fill_uniforms(Uni &u, const DrawState &ds, const RTConfig &rt, u32 scale);

// The scissor rectangle of a draw, in scaled pixels with y down: x, y, width, height. False: the draw covers nothing.
bool scissor_rect(const DrawState &ds, const RTConfig &rt, u32 scale, u32 out[4]);

// Bytes of a flush's uniform buffer: a block per draw, each distinct LUT snapshot once (allocations aligned to
// kUniformAlign)
constexpr size_t kUniformAlign = 256;
size_t uniform_bytes(const MBatch &mb);

// ---- render targets: GPU copies of images in guest memory, kept in step with it. A backend's target derives from
// TargetState, adding its textures; the backend keeps its targets in a TargetList.
struct TargetState {
    u32 addr = 0, w = 0, h = 0, fmt = 0;
    bool depth = false;
    bool ms_valid = false;   // with MSAA: the multisampled copy matches the single-sample one
    bool synced = false;     // GPU copy corresponds to guest memory as hashed below
    u64 hash = 0;
    u64 checked = 0;         // g_frame when guest memory last matched the hash
    u64 serial = 0;          // unique among all targets ever made
    u64 version = 0;         // changes whenever the GPU copy does
    bool gpu_newer = false;  // changes not yet written to guest memory
    u32 skipped = 0;         // draws left out since its last display transfer (pipelines still compiling)
    bool clear_next = false; // set_clear_after_top_transfer: cleared to transparent before it is drawn into again,
    bool own = false;        // ... and its GPU copy kept (guest memory ignored) until its own next display transfer
};
template <class T> using TargetList = std::vector<std::unique_ptr<T>>;

inline bool overlaps(u32 a, u32 an, u32 b, u32 bn) { return a < b + bn && b < a + an; }
// a target's bytes per pixel in guest memory, and its size there
inline u32 rt_bpp(const TargetState &r) {
    return r.depth ? (r.fmt == 0 ? 2 : r.fmt == 2 ? 3 : 4) : pixel_bytes(cbuf_to_pixfmt(r.fmt));
}
inline u32 rt_bytes(const TargetState &r) { return r.w * r.h * rt_bpp(r); }
inline float depth_quantum(u32 fmt) { return fmt == 0 ? 1.0f / 65536.0f : 1.0f / 16777216.0f; }

// What a target's GPU copy needs before it is drawn into, from sync_target: nothing, a clear to one value (guest
// memory holds one value everywhere: typically a memory fill clearing it), or guest memory's bytes, which the backend
// decodes into it (`upload`).
struct LoadAction {
    bool clear = false;
    u32 color = 0;  // packed RGBA
    float depth = 0;
    u32 stencil = 0;
    bool upload = false;
};
// Checks guest memory against the GPU copy (once a frame: writes reported since clear `synced`, and a CPU write the
// game does not flush out of its data cache would not reach a 3DS GPU before the frame's end either)
LoadAction sync_target(TargetState &r);

// Before writing a target back: the guest memory must still hold what it held when the GPU copy took over. A game
// that freed a target and reused its memory has written it since, and on the 3DS those later CPU writes land on top
// of what the GPU drew: the guest's bytes win, the GPU copy is dropped. (backend: the log's name for it)
bool guest_kept(TargetState &r, const char *backend);

// the GPU copy of an image at addr with this width and format (the tallest; null: none); any: also one that no
// longer matches guest memory (synced again from it before it is drawn into)
template <class T> T *find_view(TargetList<T> &rts, u32 addr, u32 w, u32 fmt, bool depth, bool any = false) {
    T *best = nullptr;
    for (auto &r : rts)
        if (r->depth == depth && (r->synced || any) && r->addr == addr && r->w == w && r->fmt == fmt &&
            (!best || r->h > best->h))
            best = r.get();
    return best;
}
template <class T> T *find_color(TargetList<T> &rts, u32 addr, u32 w, u32 fmt) {
    return find_view(rts, addr, w, fmt, false);
}
// (set_clear_after_top_transfer) the colour targets in the same memory as r: cleared before their next draw
template <class T> void clear_overlapping(TargetList<T> &rts, const T &r) {
    for (auto &o : rts)
        if (!o->depth && overlaps(o->addr, rt_bytes(*o), r.addr, rt_bytes(r))) o->clear_next = o->own = true;
}
// guest memory written by a GPU operation: overlapping targets' GPU copies are no longer trusted
template <class T> void rts_overwritten(TargetList<T> &rts, u32 paddr, u32 size) {
    for (auto &r : rts)
        if (overlaps(r->addr, rt_bytes(*r), paddr, size)) r->synced = false;
}
// the colour target a display transfer reads, starting at one of its tile rows (row0)
template <class T> T *transfer_source(TargetList<T> &rts, u32 in_paddr, u32 in_dims, u32 out_dims, u32 &row0) {
    const u32 in_w = in_dims & 0xffff, out_h0 = out_dims >> 16;
    T *src = nullptr;
    for (auto &r : rts) {
        if (r->depth || !r->synced || r->w != in_w || in_paddr < r->addr) continue;
        const u32 line = r->w * pixel_bytes(cbuf_to_pixfmt(r->fmt));
        const u32 off = in_paddr - r->addr;
        if (off % (line * 8) || off / line + out_h0 > r->h) continue;
        src = r.get();
        row0 = off / line;
    }
    if (src && (out_dims & 0xffff) > src->w) return nullptr;  // (narrower: the transfer keeps the first columns)
    return src;
}
// Before the GPU writes [paddr, paddr + size) into dst (reading src): other GPU copies of that memory go back to it
// (write_back), or are dropped where the write covers them, and are no longer trusted. True if guest memory there
// changed meanwhile. (Queued CPU parts writing there are the caller's to finish first.)
template <class T, class WriteBack> bool claim_memory(TargetList<T> &rts, T *src, T *dst, u32 paddr, u32 size,
                                                     WriteBack write_back) {
    bool changed = false;
    for (auto &r : rts) {
        if (r.get() == src || r.get() == dst || !overlaps(r->addr, rt_bytes(*r), paddr, size)) continue;
        if (r->gpu_newer) {
            if (r->addr >= paddr && r->addr + rt_bytes(*r) <= paddr + size) {
                r->gpu_newer = false;
            } else {
                write_back(*r);
                changed = true;
            }
        }
        r->synced = false;
    }
    return changed;
}

// Guest memory holding GPU output (render targets, or transfer / copy outputs made from them, as long as the bytes
// are unchanged): render-to-texture, not upscaled. Only tracked while a texture filter is on.
struct GpuOutputs {
    struct Out {
        u32 addr, size;
        u64 hash;
    };
    std::deque<Out> outs;
    void note(u32 addr, u32 size);
    template <class T> bool made(TargetList<T> &rts, u32 addr, u32 size) const {
        for (auto &r : rts)
            if (overlaps(r->addr, rt_bytes(*r), addr, size) &&
                (r->gpu_newer || hash_mem(pptr(r->addr), rt_bytes(*r)) == r->hash))
                return true;
        return made_elsewhere(addr, size);
    }
    bool made_elsewhere(u32 addr, u32 size) const;
};

// ---- display images: the upright pictures display transfers make from render targets (what the screens show at
// the internal resolution), by the guest memory the transfer wrote. A backend's image extends DisplayImageBase with
// its buffer and whatever says it is still being made.
struct DisplayImageBase {
    u32 out_addr = 0, out_size = 0;
    u32 out_w = 0, out_bpp = 0, hs = 0, vs = 0;  // output rows' pixels and bytes per pixel; downscaling shifts
    u32 scale = 1;                               // internal scale the image was made at
    bool incomplete = false;                     // some draws were left out (the screen keeps the last image)
    u64 out_hash = 0;                            // guest framebuffer bytes the image corresponds to
    int screen = -1;
    u32 w = 0, h = 0;
    u64 version = 0;  // new for every new image
};
// The part of a display image the screen shows: the framebuffer (`lines` rows of 240 pixels, `stride` bytes apart)
// may start inside a display transfer's output, e.g. NSMB2 transfers 256x416 and shows rows 16 to 415. The image's x
// runs along the output rows, its y backwards along each row's pixels.
template <class D> struct DisplayView {
    const D *d = nullptr;
    u32 x0 = 0, y0 = 0, w = 0, h = 0;
    u64 key = 0;  // changes with the image or the part shown
};
template <class D> struct DisplayImages {
    std::unordered_map<u32, D> by_addr;  // by the output's physical address
    u32 last_screen_addr[2] = {0, 0};    // the latest image of each screen (by its height)
    u64 version = 0;

    // A new image (w x h) of a display transfer's output; images it overwrites are dropped (drop: frees what the
    // backend holds for one). The caller fills in the rest.
    template <class Drop> D &add(u32 out_paddr, u32 out_size, u32 out_dims, u32 flags, u32 w, u32 h, u32 scale, Drop drop) {
        for (auto it = by_addr.begin(); it != by_addr.end();) {
            if (overlaps(it->second.out_addr, it->second.out_size, out_paddr, out_size)) {
                drop(it->second);
                it = by_addr.erase(it);
            } else {
                ++it;
            }
        }
        D &d = by_addr[out_paddr];
        const u32 out_h0 = out_dims >> 16;
        d.out_addr = out_paddr;
        d.out_size = out_size;
        d.w = w;
        d.h = h;
        d.hs = ((flags >> 24) & 3) ? 1 : 0;
        d.vs = ((flags >> 24) & 3) == 2 ? 1 : 0;
        d.out_w = (out_dims & 0xffff) >> d.hs;
        d.out_bpp = pixel_bytes((flags >> 12) & 7);
        d.scale = scale;
        d.version = ++version;
        d.screen = out_h0 == 400 ? 0 : out_h0 == 320 ? 1 : -1;
        if (d.screen >= 0) last_screen_addr[d.screen] = out_paddr;
        return d;
    }
    // Guest memory in [paddr, paddr + size) was written: images made of it are dropped (definite: certainly
    // written; else only if its bytes changed, for images whose guest memory is final, `settled`)
    template <class Settled, class Drop>
    void written(u32 paddr, u32 size, bool definite, const char *backend, Settled settled, Drop drop) {
        for (auto it = by_addr.begin(); it != by_addr.end();) {
            D &d = it->second;
            if (overlaps(d.out_addr, d.out_size, paddr, size) &&
                (definite || (settled(d) && hash_mem(pptr(d.out_addr), d.out_size) != d.out_hash))) {
                if (g_stats)
                    LOG("gpu %s: display image %08x dropped (%s write %08x+%x)\n", backend, d.out_addr,
                        definite ? "GPU" : "CPU", paddr, size);
                drop(d);
                it = by_addr.erase(it);
            } else {
                ++it;
            }
        }
    }
    // the view of the framebuffer at fb (false: no image of it, or not one that can show it)
    bool view(u32 fb, u32 stride, u32 lines, DisplayView<D> &v) const {
        for (const auto &[addr, d] : by_addr) {
            if (fb < d.out_addr || fb >= d.out_addr + d.out_size) continue;
            const u32 row_bytes = d.out_w * d.out_bpp;
            if (!row_bytes || (stride && stride != row_bytes)) return false;
            const u32 off = fb - d.out_addr, r0 = off / row_bytes, c0 = off % row_bytes / d.out_bpp;
            const u32 fx = d.scale << d.vs, fy = d.scale << d.hs;  // image pixels per output row / pixel
            if (c0 + 240 > d.out_w || (r0 + lines) * fx > d.w || (c0 + 240) * fy > d.h) return false;
            v.d = &d;
            v.x0 = r0 * fx;
            v.w = lines * fx;
            v.y0 = d.h - (c0 + 240) * fy;
            v.h = 240 * fy;
            v.key = d.version << 20 ^ off;
            return true;
        }
        return false;
    }
};

}  // namespace gpu
