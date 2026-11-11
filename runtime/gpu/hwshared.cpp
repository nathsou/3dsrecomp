// What the hardware backends share (hwshared.h).
#include "gpu/hwshared.h"

#include <algorithm>
#include <unordered_set>

namespace gpu {

void fill_uniforms(Uni &u, const DrawState &ds, const RTConfig &rt, u32 scale) {
    memset(&u, 0, sizeof u);
    u.rt[0] = (float)rt.w;
    u.rt[1] = (float)rt.h;
    u.rt[2] = (float)scale;
    u.rt[3] = (float)((1u << ds.depth_bits) - 1);
    u.depth[0] = ds.depth_scale;
    u.depth[1] = ds.depth_offset;
    u.depth[2] = ds.depth_bits == 16 ? 1.0f / 65536.0f : 1.0f / 16777216.0f;
    u.scissor[0] = ds.sc_x1;
    u.scissor[1] = ds.sc_y1;
    u.scissor[2] = ds.sc_x2;
    u.scissor[3] = ds.sc_y2;
    for (int k = 0; k < 6; k++) u.konst[k] = ds.tev[k].konst;
    u.tevc[0] = ds.tev_init_out;
    u.tevc[1] = ds.tev_init_buf;
    u.tevc[2] = ds.tev_init_next;
    u.misc[0] = ds.blend_color;
    u.misc[1] = ds.fog_color;
    u.misc[2] = ds.alpha_ref;
    u.misc[3] = ds.tex[0].shadow_bias;
    for (int k = 0; k < 3; k++) {
        u.border[k] = ds.tex[k].border;
        u.bias[k] = ds.tex[k].bias;
    }
    if (!ds.lighting) return;
    const LightingState &l = ds.light;
    const LutSel *sels[7] = {&l.d0, &l.d1, &l.sp, &l.fr, &l.rb, &l.rg, &l.rr};
    for (int k = 0; k < 7; k++) u.lscale[k] = sels[k]->scale;
    for (int k = 0; k < 3; k++) u.ambient[k] = l.global_ambient[k];
    for (u32 k = 0; k < l.num_lights; k++) {
        const LightState &s = l.lights[k];
        LightU &d = u.lights[k];
        for (int c = 0; c < 3; c++) {
            d.spec0[c] = s.spec0[c];
            d.spec1[c] = s.spec1[c];
            d.diffuse[c] = s.diffuse[c];
            d.ambient[c] = s.ambient[c];
            d.pos[c] = s.pos[c];
            d.spot[c] = s.spot[c];
        }
        d.att[0] = s.da_scale;
        d.att[1] = s.da_bias;
    }
}

bool scissor_rect(const DrawState &ds, const RTConfig &rt, u32 scale, u32 out[4]) {
    if (ds.scissor_mode != 3) {  // (the other modes clip in the fragment shader, against u.scissor)
        out[0] = out[1] = 0;
        out[2] = rt.w * scale;
        out[3] = rt.h * scale;
        return true;
    }
    const s32 x1 = std::clamp<s32>(ds.sc_x1, 0, rt.w), x2 = std::clamp<s32>(ds.sc_x2, 0, rt.w);
    const s32 y1 = std::clamp<s32>(ds.sc_y1, 0, rt.h), y2 = std::clamp<s32>(ds.sc_y2, 0, rt.h);
    if (x2 <= x1 || y2 <= y1) return false;
    out[0] = (u32)x1 * scale;
    out[1] = (u32)(rt.h - y2) * scale;  // (PICA coordinates have y up)
    out[2] = (u32)(x2 - x1) * scale;
    out[3] = (u32)(y2 - y1) * scale;
    return true;
}

size_t uniform_bytes(const MBatch &mb) {
    std::unordered_set<const void *> luts;
    size_t n = mb.segs.size() * (kUniformAlign + sizeof(Uni));
    for (const MSeg &seg : mb.segs) {
        const DrawState &ds = *seg.ds;
        if (ds.lighting && luts.insert(ds.light.luts.get()).second) n += kUniformAlign + sizeof(LightLuts);
        if (ds.fog && luts.insert(ds.fog_lut.get()).second) n += kUniformAlign + sizeof(FogLut);
    }
    return n;
}

LoadAction sync_target(TargetState &r) {
    LoadAction la;
    const u8 *p = pptr(r.addr);
    if (!p) return la;
    const u64 frame = g_frame.load(std::memory_order_relaxed);
    if (r.synced && r.checked == frame) return la;
    const u64 hv = hash_mem(p, rt_bytes(r));
    r.checked = frame;
    if (r.synced && hv == r.hash) return la;
    if (r.own) {  // (cleared after a transfer to the top screen: the top screen's write-back is not its picture)
        r.hash = hv;
        r.synced = true;
        return la;
    }
    r.synced = true;
    r.hash = hv;
    r.gpu_newer = false;
    // a target filled with one value: bytes repeat every pixel
    const u32 bpp = rt_bpp(r);
    if (rt_bytes(r) > bpp && memcmp(p, p + bpp, rt_bytes(r) - bpp) == 0) {
        la.clear = true;
        if (!r.depth) {
            la.color = decode_pixel(cbuf_to_pixfmt(r.fmt), p);
        } else {
            const u32 z = r.fmt == 0 ? (u32)(p[0] | p[1] << 8) : (u32)(p[0] | p[1] << 8 | p[2] << 16);
            la.depth = (float)z * depth_quantum(r.fmt);
            la.stencil = r.fmt == 3 ? p[3] : 0;
        }
        return la;
    }
    r.ms_valid = false;
    r.version++;
    la.upload = true;
    return la;
}

bool guest_kept(TargetState &r, const char *backend) {
    if (hash_mem(pptr(r.addr), rt_bytes(r)) == r.hash) return true;
    static int logged;
    if (logged < 8 || g_stats) {
        logged++;
        LOG("gpu %s: %s target %08x (%ux%u) was rewritten by the game since it was drawn; not written back\n", backend,
            r.depth ? "depth" : "colour", r.addr, r.w, r.h);
    }
    r.gpu_newer = false;
    r.synced = false;
    return false;
}

void GpuOutputs::note(u32 addr, u32 size) {
    if (!pptr(addr) || !pptr(addr + size - 1)) return;
    for (auto it = outs.begin(); it != outs.end(); ++it)
        if (it->addr == addr) {
            outs.erase(it);
            break;
        }
    outs.push_back({addr, size, hash_mem(pptr(addr), size)});
    constexpr size_t kKept = 64;
    if (outs.size() > kKept) outs.pop_front();
}

bool GpuOutputs::made_elsewhere(u32 addr, u32 size) const {
    for (const Out &g : outs)
        if (overlaps(g.addr, g.size, addr, size) && hash_mem(pptr(g.addr), g.size) == g.hash) return true;
    return false;
}

}  // namespace gpu
