// GX transfer engine: MemoryFill, DisplayTransfer and TextureCopy.
#include "gpu/pica.h"

namespace gpu {

void memory_fill(u32 start, u32 end, u32 value, u32 control) {
    if (!(control & 1)) return;
    ensure_init();
    raster_flush();
    metal_poll();
    u8 *s = pptr(start), *e = pptr(end - 1);
    if (!s || !e || end <= start) {
        LOG("gpu: bad memory fill %08x-%08x\n", start, end);
        return;
    }
    const u32 len = end - start;
    metal_guest_prewrite(start, len);
    if (control & 0x100) {
        for (u32 i = 0; i + 3 <= len; i += 3) {
            s[i] = value;
            s[i + 1] = value >> 8;
            s[i + 2] = value >> 16;
        }
    } else if (control & 0x200) {
        for (u32 i = 0; i + 4 <= len; i += 4) memcpy(s + i, &value, 4);
    } else {
        const u16 v = (u16)value;
        for (u32 i = 0; i + 2 <= len; i += 2) memcpy(s + i, &v, 2);
    }
    texcache_invalidate(start, len);
    metal_guest_written(start, len);
}

void transfer_convert(const u8 *src, u8 *dst, u32 in_dims, u32 out_dims, u32 flags) {
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

void display_transfer(u32 in_paddr, u32 out_paddr, u32 in_dims, u32 out_dims, u32 flags) {
    ensure_init();
    raster_flush();
    metal_poll();
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
    const u32 hs = scaling != 0 ? 1 : 0, vs = scaling == 2 ? 1 : 0;
    const u32 out_w = out_w0 >> hs, out_h = out_h0 >> vs;
    // from a render target the GPU is still drawing: finished asynchronously
    if (metal_defer_transfer(in_paddr, out_paddr, in_dims, out_dims, flags, out_w * out_h * obpp)) return;
    metal_guest_read(in_paddr, in_w * in_h * ibpp);
    const bool from_gpu = metal_gpu_made(in_paddr, in_w * in_h * ibpp);
    metal_guest_prewrite(out_paddr, out_w * out_h * obpp);
    transfer_convert(src, dst, in_dims, out_dims, flags);
    texcache_invalidate(out_paddr, out_w * out_h * obpp);
    metal_guest_written(out_paddr, out_w * out_h * obpp);
    if (from_gpu) metal_note_gpu_output(out_paddr, out_w * out_h * obpp);
    metal_display_transfer(in_paddr, out_paddr, in_dims, out_dims, flags);
}

void texture_copy(u32 in_paddr, u32 out_paddr, u32 size, u32 in_width_gap, u32 out_width_gap,
                  u32 flags) {
    (void)flags;
    ensure_init();
    raster_flush();
    metal_poll();
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
    metal_guest_read(in_paddr, size / in_width * (in_width + in_gap) + in_width);
    const bool from_gpu = metal_gpu_made(in_paddr, size / in_width * (in_width + in_gap) + in_width);
    metal_guest_prewrite(out_paddr, out_total);
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
    metal_guest_written(out_paddr, out_total);
    if (from_gpu) metal_note_gpu_output(out_paddr, out_total);
}

}  // namespace gpu
