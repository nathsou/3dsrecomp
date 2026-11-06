// GX transfer engine: MemoryFill, DisplayTransfer and TextureCopy.
#include <algorithm>
#include <array>
#include <utility>

#include "gpu/pica.h"

namespace gpu {

// the n-byte pattern repeated over the first len bytes of s (len rounded down to whole patterns): written once, then
// copied over twice the length each time
static void fill_pattern(u8 *s, u32 len, const u8 *pattern, u32 n) {
    len -= len % n;
    if (!len) return;
    memcpy(s, pattern, n);
    for (u32 done = n; done < len;) {
        const u32 k = std::min(done, len - done);
        memcpy(s + done, s, k);
        done += k;
    }
}

void memory_fill(u32 start, u32 end, u32 value, u32 control) {
    if (!(control & 1)) return;
    ensure_init();
    raster_flush();
    hw_poll();
    u8 *s = pptr(start), *e = pptr(end - 1);
    if (!s || !e || end <= start) {
        LOG("gpu: bad memory fill %08x-%08x\n", start, end);
        return;
    }
    const u32 len = end - start;
    hw_guest_prewrite(start, len);
    const u8 pattern[4] = {(u8)value, (u8)(value >> 8), (u8)(value >> 16), (u8)(value >> 24)};  // little-endian
    fill_pattern(s, len, pattern, control & 0x100 ? 3 : control & 0x200 ? 4 : 2);  // 24, 32 or 16-bit
    texcache_invalidate(start, len);
    hw_guest_written(start, len);
}

namespace {
// decode_pixel / encode_pixel (gpu.cpp) for one format known at compile time, inlined into the loops below
inline u32 x5to8(u32 v) { return (v << 3) | (v >> 2); }
inline u32 x6to8(u32 v) { return (v << 2) | (v >> 4); }
inline u32 x4to8(u32 v) { return (v << 4) | v; }
template <u32 F> inline u32 dec(const u8 *p) {
    if constexpr (F == 0) return rgba(p[3], p[2], p[1], p[0]);
    if constexpr (F == 1) return rgba(p[2], p[1], p[0], 255);
    const u32 v = p[0] | (p[1] << 8);
    if constexpr (F == 2) return rgba(x5to8(v >> 11), x6to8((v >> 5) & 63), x5to8(v & 31), 255);
    if constexpr (F == 3) return rgba(x5to8(v >> 11), x5to8((v >> 6) & 31), x5to8((v >> 1) & 31), (v & 1) * 255);
    return rgba(x4to8(v >> 12), x4to8((v >> 8) & 15), x4to8((v >> 4) & 15), x4to8(v & 15));
}
template <u32 F> inline void enc(u32 c, u8 *p) {
    const u32 r = c & 255, gg = (c >> 8) & 255, b = (c >> 16) & 255, a = c >> 24;
    if constexpr (F == 0) {
        p[0] = a, p[1] = b, p[2] = gg, p[3] = r;
    } else if constexpr (F == 1) {
        p[0] = b, p[1] = gg, p[2] = r;
    } else {
        const u32 v = F == 2   ? ((r >> 3) << 11) | ((gg >> 2) << 5) | (b >> 3)
                      : F == 3 ? ((r >> 3) << 11) | ((gg >> 3) << 6) | ((b >> 3) << 1) | (a >> 7)
                               : ((r >> 4) << 12) | ((gg >> 4) << 8) | ((b >> 4) << 4) | (a >> 4);
        p[0] = (u8)v, p[1] = (u8)(v >> 8);
    }
}
inline u32 avg2(u32 c, u32 c1) {
    u32 o = 0;
    for (u32 sh = 0; sh < 32; sh += 8) o |= ((((c >> sh) & 255) + ((c1 >> sh) & 255)) / 2) << sh;
    return o;
}
inline u32 avg4(u32 c, u32 c1, u32 c2, u32 c3) {
    u32 o = 0;
    for (u32 sh = 0; sh < 32; sh += 8)
        o |= ((((c >> sh) & 255) + ((c1 >> sh) & 255) + ((c2 >> sh) & 255) + ((c3 >> sh) & 255)) / 4) << sh;
    return o;
}

// the conversion for formats IF -> OF (0-4) and scaling SC (0-2), all pixel work inlined
template <u32 IF, u32 OF, u32 SC>
void convert(const u8 *src, u8 *dst, u32 in_w, u32 out_w, u32 out_h, bool flip, bool in_linear, bool dont_swizzle) {
    constexpr u32 ibpp = IF == 0 ? 4 : IF == 1 ? 3 : 2, obpp = OF == 0 ? 4 : OF == 1 ? 3 : 2;
    constexpr u32 hs = SC != 0 ? 1 : 0, vs = SC == 2 ? 1 : 0;
    for (u32 y = 0; y < out_h; y++) {
        const u32 iy = y << vs;
        const u32 oy = flip ? out_h - 1 - y : y;
        for (u32 x = 0; x < out_w; x++) {
            const u32 ix = x << hs;
            u32 so, doff;
            if (in_linear) {
                so = (ix + iy * in_w) * ibpp;
                doff = dont_swizzle ? (x + oy * out_w) * obpp : tiled_offset(x, oy, out_w, obpp);
            } else {
                so = tiled_offset(ix, iy, in_w, ibpp);
                doff = dont_swizzle ? tiled_offset(x, oy, out_w, obpp) : (x + oy * out_w) * obpp;
            }
            const u8 *sp = src + so;
            u32 c = dec<IF>(sp);
            if constexpr (SC == 1) c = avg2(c, dec<IF>(sp + ibpp));
            if constexpr (SC == 2) c = avg4(c, dec<IF>(sp + ibpp), dec<IF>(sp + 2 * ibpp), dec<IF>(sp + 3 * ibpp));
            enc<OF>(c, dst + doff);
        }
    }
}

using ConvertFn = void (*)(const u8 *, u8 *, u32, u32, u32, bool, bool, bool);
template <u32... I> constexpr auto convert_table(std::integer_sequence<u32, I...>) {
    return std::array<ConvertFn, sizeof...(I)>{&convert<I / 15, (I / 3) % 5, I % 3>...};
}
constexpr auto kConvert = convert_table(std::make_integer_sequence<u32, 75>{});  // [in * 15 + out * 3 + scaling]
}  // namespace

// the generic loop: also formats 5-7, which decode to 0 and are not written (as decode_pixel / encode_pixel)
void transfer_convert_ref(const u8 *src, u8 *dst, u32 in_dims, u32 out_dims, u32 flags) {
    const u32 in_w = in_dims & 0xffff;
    const u32 out_w0 = out_dims & 0xffff, out_h0 = out_dims >> 16;
    const bool flip = flags & 1, in_linear = (flags >> 1) & 1;
    const bool dont_swizzle = (flags >> 5) & 1;
    const u32 in_fmt = (flags >> 8) & 7, out_fmt = (flags >> 12) & 7;
    const u32 scaling = (flags >> 24) & 3;
    const u32 ibpp = pixel_bytes(in_fmt), obpp = pixel_bytes(out_fmt);
    const u32 hs = scaling != 0 ? 1 : 0, vs = scaling == 2 ? 1 : 0;
    const u32 out_w = out_w0 >> hs, out_h = out_h0 >> vs;
    for (u32 y = 0; y < out_h; y++) {
        const u32 iy = y << vs;
        const u32 oy = flip ? out_h - 1 - y : y;
        for (u32 x = 0; x < out_w; x++) {
            const u32 ix = x << hs;
            u32 so, doff;
            if (in_linear) {
                so = (ix + iy * in_w) * ibpp;
                doff = dont_swizzle ? (x + oy * out_w) * obpp : tiled_offset(x, oy, out_w, obpp);
            } else {
                so = tiled_offset(ix, iy, in_w, ibpp);
                doff = dont_swizzle ? tiled_offset(x, oy, out_w, obpp) : (x + oy * out_w) * obpp;
            }
            const u8 *sp = src + so;
            u32 c = decode_pixel(in_fmt, sp);
            if (scaling == 1) {
                u32 c1 = decode_pixel(in_fmt, sp + ibpp);
                u32 o = 0;
                for (u32 sh = 0; sh < 32; sh += 8)
                    o |= ((((c >> sh) & 255) + ((c1 >> sh) & 255)) / 2) << sh;
                c = o;
            } else if (scaling == 2) {
                u32 c1 = decode_pixel(in_fmt, sp + ibpp), c2 = decode_pixel(in_fmt, sp + 2 * ibpp),
                    c3 = decode_pixel(in_fmt, sp + 3 * ibpp);
                u32 o = 0;
                for (u32 sh = 0; sh < 32; sh += 8)
                    o |= ((((c >> sh) & 255) + ((c1 >> sh) & 255) + ((c2 >> sh) & 255) +
                           ((c3 >> sh) & 255)) / 4) << sh;
                c = o;
            }
            encode_pixel(out_fmt, c, dst + doff);
        }
    }
}

void transfer_convert(const u8 *src, u8 *dst, u32 in_dims, u32 out_dims, u32 flags) {
    const u32 in_fmt = (flags >> 8) & 7, out_fmt = (flags >> 12) & 7, scaling = (flags >> 24) & 3;
    if (in_fmt > 4 || out_fmt > 4 || scaling > 2) return transfer_convert_ref(src, dst, in_dims, out_dims, flags);
    const u32 hs = scaling != 0 ? 1 : 0, vs = scaling == 2 ? 1 : 0;
    kConvert[in_fmt * 15 + out_fmt * 3 + scaling](src, dst, in_dims & 0xffff, (out_dims & 0xffff) >> hs,
                                                  (out_dims >> 16) >> vs, flags & 1, (flags >> 1) & 1, (flags >> 5) & 1);
}

void display_transfer(u32 in_paddr, u32 out_paddr, u32 in_dims, u32 out_dims, u32 flags) {
    ensure_init();
    raster_flush();
    hw_poll();
    if (flags & 8) {
        LOG("gpu: display_transfer called with texture-copy flag, ignored\n");
        return;
    }
    const u32 in_w = in_dims & 0xffff, in_h = in_dims >> 16;
    const u32 out_w0 = out_dims & 0xffff, out_h0 = out_dims >> 16;
    if (!in_w || !in_h || !out_w0 || !out_h0) return;
    const bool flip = flags & 1, crop = (flags >> 2) & 1;
    const u32 in_fmt = (flags >> 8) & 7, out_fmt = (flags >> 12) & 7;
    const u32 scaling = (flags >> 24) & 3;
    if (in_fmt > 4 || out_fmt > 4 || scaling > 2) {
        LOG("gpu: unsupported display transfer flags %08x\n", flags);
        return;
    }
    const u32 ibpp = pixel_bytes(in_fmt), obpp = pixel_bytes(out_fmt);
    if (flip && crop) out_paddr += (in_w - out_w0) * (out_h0 - 1) * obpp;
    const u8 *src = pptr(in_paddr);
    u8 *dst = pptr(out_paddr);
    if (!src || !dst) {
        LOG("gpu: bad display transfer %08x -> %08x\n", in_paddr, out_paddr);
        return;
    }
    if (hw_gpu_transfer(in_paddr, out_paddr, in_dims, out_dims, flags)) return;
    const u32 hs = scaling != 0 ? 1 : 0, vs = scaling == 2 ? 1 : 0;
    const u32 out_w = out_w0 >> hs, out_h = out_h0 >> vs;
    // from a render target the GPU is still drawing: finished asynchronously
    if (hw_defer_transfer(in_paddr, out_paddr, in_dims, out_dims, flags, out_w * out_h * obpp)) return;
    hw_guest_read(in_paddr, in_w * in_h * ibpp);
    const bool from_gpu = hw_gpu_made(in_paddr, in_w * in_h * ibpp);
    hw_guest_prewrite(out_paddr, out_w * out_h * obpp);
    transfer_convert(src, dst, in_dims, out_dims, flags);
    texcache_invalidate(out_paddr, out_w * out_h * obpp);
    hw_guest_written(out_paddr, out_w * out_h * obpp);
    if (from_gpu) hw_note_gpu_output(out_paddr, out_w * out_h * obpp);
    hw_display_transfer(in_paddr, out_paddr, in_dims, out_dims, flags);
}

void texture_copy(u32 in_paddr, u32 out_paddr, u32 size, u32 in_width_gap, u32 out_width_gap,
                  u32 flags) {
    (void)flags;
    ensure_init();
    raster_flush();
    hw_poll();
    const u8 *src = pptr(in_paddr);
    u8 *dst = pptr(out_paddr);
    if (!src || !dst) {
        LOG("gpu: bad texture copy %08x -> %08x\n", in_paddr, out_paddr);
        return;
    }
    u32 remaining = size & ~15u;
    if (!remaining) return;
    const u32 in_gap = (in_width_gap >> 16) * 16, out_gap = (out_width_gap >> 16) * 16;
    const u32 in_width = in_gap == 0 ? remaining : (in_width_gap & 0xffff) * 16;
    const u32 out_width = out_gap == 0 ? remaining : (out_width_gap & 0xffff) * 16;
    if (!in_width || !out_width) return;
    const u32 out_total = size / out_width * (out_width + out_gap) + out_width;
    if (hw_texture_copy(in_paddr, out_paddr, remaining, in_width, in_gap, out_width, out_gap)) return;
    hw_guest_read(in_paddr, size / in_width * (in_width + in_gap) + in_width);
    const bool from_gpu = hw_gpu_made(in_paddr, size / in_width * (in_width + in_gap) + in_width);
    hw_guest_prewrite(out_paddr, out_total);
    u32 ri = in_width, ro = out_width;
    while (remaining) {
        u32 n = std::min({ri, ro, remaining});
        memmove(dst, src, n);
        src += n;
        dst += n;
        ri -= n;
        ro -= n;
        remaining -= n;
        if (!ri) {
            ri = in_width;
            src += in_gap;
        }
        if (!ro) {
            ro = out_width;
            dst += out_gap;
        }
    }
    texcache_invalidate(out_paddr, out_total);
    hw_guest_written(out_paddr, out_total);
    if (from_gpu) hw_note_gpu_output(out_paddr, out_total);
}

}  // namespace gpu
