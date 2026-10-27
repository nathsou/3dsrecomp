// Vertex loading, vertex shader invocation (parallel for batched draws), geometry shader pipeline
// and primitive assembly.
#include <algorithm>

#include "gpu/pica.h"

namespace gpu {

// ---------------------------------------------------------------- vertex loader
namespace {
struct LoaderAttr {
    u32 offset = 0, stride = 0;
    u8 fmt = 0, elems = 0;
    bool is_default = false;
};
struct Loader {
    u32 base;
    u32 num;
    LoaderAttr a[16];
};

void setup_loader(Loader &L) {
    const u32 *regs = g.regs;
    L.base = bits(regs[R::ATTR_BASE], 1, 28) * 16;
    const u64 fmts = (u64)regs[R::ATTR_FMT_LO] | ((u64)(regs[R::ATTR_FMT_HI] & 0xffff) << 32);
    const u32 mask = bits(regs[R::ATTR_FMT_HI], 16, 12);
    L.num = bits(regs[R::ATTR_FMT_HI], 28, 4) + 1;
    auto fmt_of = [&](u32 i) { return (u32)((fmts >> (4 * i)) & 3); };
    auto elems_of = [&](u32 i) { return (u32)((fmts >> (4 * i + 2)) & 3) + 1; };
    auto esize = [&](u32 i) { u32 f = fmt_of(i); return f == 3 ? 4u : f == 2 ? 2u : 1u; };
    for (u32 i = 0; i < 16; i++) {
        L.a[i] = LoaderAttr{};
        L.a[i].is_default = i >= 12 || ((mask >> i) & 1);
    }
    for (u32 l = 0; l < 12; l++) {
        const u32 r0 = regs[R::ATTR_LOADER + 3 * l];
        const u32 r1 = regs[R::ATTR_LOADER + 3 * l + 1];
        const u32 r2 = regs[R::ATTR_LOADER + 3 * l + 2];
        const u64 comps = (u64)r1 | ((u64)(r2 & 0xffff) << 32);
        const u32 data_offset = r0 & 0x0fffffff;
        const u32 byte_count = bits(r2, 16, 8);
        const u32 ncomp = bits(r2, 28, 4);
        u32 offset = 0;
        for (u32 c = 0; c < ncomp && c < 12; c++) {
            u32 ai = (comps >> (4 * c)) & 0xf;
            if (ai < 12) {
                u32 es = esize(ai);
                offset = (offset + es - 1) / es * es;
                L.a[ai].offset = data_offset + offset;
                L.a[ai].stride = byte_count;
                L.a[ai].fmt = fmt_of(ai);
                L.a[ai].elems = elems_of(ai);
                offset += es * elems_of(ai);
            } else {
                offset = (offset + 3) & ~3u;
                offset += (ai - 11) * 4;
            }
        }
    }
}

inline void load_vertex(const Loader &L, u32 vertex, V4 *in, const V4 *defaults = g.default_attr) {
    for (u32 i = 0; i < L.num; i++) {
        const LoaderAttr &A = L.a[i];
        if (A.is_default) {
            in[i] = defaults[i];
            continue;
        }
        if (A.elems == 0) {
            in[i] = V4{{0, 0, 0, 1}};  // "vertex retention" (not tracked)
            continue;
        }
        const u8 *p = pptr(L.base + A.offset + A.stride * vertex);
        if (!p) {
            in[i] = V4{{0, 0, 0, 1}};
            continue;
        }
        float *o = in[i].v;
        switch (A.fmt) {
        case 0: for (u32 c = 0; c < A.elems; c++) o[c] = (float)(s8)p[c]; break;
        case 1: for (u32 c = 0; c < A.elems; c++) o[c] = (float)p[c]; break;
        case 2:
            for (u32 c = 0; c < A.elems; c++) {
                s16 v;
                memcpy(&v, p + 2 * c, 2);
                o[c] = (float)v;
            }
            break;
        default: memcpy(o, p, 4 * A.elems); break;
        }
        for (u32 c = A.elems; c < 4; c++) o[c] = c == 3 ? 1.0f : 0.0f;
    }
}

// ---------------------------------------------------------------- output mapping
struct OutMap {
    s8 slot_src[24];  // attrib*4+comp in the output buffer, or -1 (=1.0)
};

void setup_outmap(OutMap &m) {
    for (auto &s : m.slot_src) s = -1;
    const u32 total = g.regs[R::VS_OUT_TOTAL] & 7;
    for (u32 a = 0; a < total; a++) {
        const u32 map = g.regs[R::VS_OUT_MAP + a];
        for (u32 c = 0; c < 4; c++) {
            u32 sem = (map >> (8 * c)) & 0x1f;
            if (sem < 24) m.slot_src[sem] = (s8)(a * 4 + c);
        }
    }
}

inline void map_vertex(const OutMap &m, const V4 *buf, OutVertex &o) {
    float *d = (float *)&o;
    const float *s = buf[0].v;
    for (u32 i = 0; i < 24; i++) d[i] = m.slot_src[i] >= 0 ? s[m.slot_src[i]] : 1.0f;
    for (u32 i = 0; i < 4; i++) {
        float c = fabsf(o.color[i]);
        o.color[i] = c < 1.0f ? c : 1.0f;
    }
}

// collect shader outputs in output-mask order
inline u32 gather_outputs(const ShaderUnit &u, u32 mask, V4 *buf) {
    u32 n = 0;
    for (u32 reg = 0; reg < 16; reg++)
        if (mask & (1u << reg)) buf[n++] = u.r[REG_OUT + reg];
    return n;
}

inline void load_inputs(ShaderUnit &u, u32 base, const V4 *in) {
    const u32 maxi = g.regs[base + R::SH_INPUT_CFG] & 0xf;
    const u64 map = (u64)g.regs[base + R::SH_PERM_LO] | ((u64)g.regs[base + R::SH_PERM_HI] << 32);
    for (u32 a = 0; a <= maxi; a++) u.r[REG_IN + ((map >> (4 * a)) & 0xf)] = in[a];
}

// ---------------------------------------------------------------- primitive assembly
const DrawState *cur_ds;

void submit_triangle(const OutVertex &a, const OutVertex &b, const OutVertex &c) {
    if (cur_ds && !cur_ds->skip) raster_triangle(cur_ds, a, b, c);
}

void pa_submit(const OutVertex &v) {
    PrimAssembler &pa = g.pa;
    switch (pa.topology) {
    case 0:
    case 3:
        if (pa.index < 2) {
            pa.buf[pa.index++] = v;
        } else {
            pa.index = 0;
            if (pa.topology == 3 && pa.winding) {
                submit_triangle(pa.buf[1], pa.buf[0], v);
                pa.winding = false;
            } else {
                submit_triangle(pa.buf[0], pa.buf[1], v);
            }
        }
        break;
    case 1:
    case 2:
        if (pa.strip_ready) submit_triangle(pa.buf[0], pa.buf[1], v);
        pa.buf[pa.index] = v;
        pa.strip_ready |= pa.index == 1;
        pa.index = pa.topology == 1 ? !pa.index : 1;
        break;
    default: break;
    }
}

// ---------------------------------------------------------------- geometry shader pipeline
struct GsPipe {
    bool active = false;
    u32 mode = 0;
    u32 vs_out_num = 0;
    V4 buffer[16];
    u32 cur = 0, end = 0;  // point: into buffer; fixed/variable: into gs uniforms
    u32 begin = 0;
    bool need_index = true;
    u32 main_num = 0, total_num = 0;
    OutMap map;
} gsp;

void gs_emit_prim(Emitter *e) {
    if (e->winding) g.pa.winding = true;
    for (u32 i = 0; i < 3; i++) {
        OutVertex v;
        map_vertex(gsp.map, e->buf[i], v);
        pa_submit(v);
    }
}

void gs_reconfigure() {
    gsp.active = (g.regs[R::GS_MISC] & 3) == 2;
    if (!gsp.active) return;
    const u32 *regs = g.regs;
    gsp.mode = regs[R::GS_CONFIG] & 0xff;
    gsp.vs_out_num = (regs[R::VS_OUTMAP_TOTAL1] & 0xf) + 1;
    g.gs_emitter.output_mask = regs[R::GS + R::SH_OUTMASK] & 0xffff;
    g.gs_emitter.emit_prim = gs_emit_prim;
    g.gs_unit.emitter = &g.gs_emitter;
    g.gs_unit.r[REG_ONE] = V4{{1, 1, 1, 1}};
    setup_outmap(gsp.map);
    switch (gsp.mode) {
    case 0:  // point
        gsp.cur = 0;
        gsp.end = (regs[R::GS + R::SH_INPUT_CFG] & 0xf) + 1;
        break;
    case 1:  // variable primitive
        gsp.need_index = true;
        break;
    default: {  // fixed primitive
        u32 vnum = bits(regs[R::GS_CONFIG], 8, 4) + 1;
        gsp.begin = gsp.cur = bits(regs[R::GS_CONFIG], 16, 8);
        gsp.end = gsp.begin + gsp.vs_out_num * vnum;
        break;
    }
    }
}

void gs_run() {
    const ShaderProgram *p = shader_get_program(g.gs);
    shader_copy_uniforms(g.gs_unit, g.gs);
    shader_run(p, g.gs_unit, g.gs, g.regs[R::GS + R::SH_ENTRY] & 0xffff);
    g.gs.b[15] = true;
}

void gs_submit_index(u32 val) {
    g.gs.f[0] = V4{{(float)val, (float)val, (float)val, (float)val}};
    gsp.cur = 1;
    gsp.main_num = g.regs[R::GS_VAR_MAIN_NUM] + 1;
    gsp.total_num = val;
    gsp.need_index = false;
}

void gs_submit_vertex(const V4 *in) {
    switch (gsp.mode) {
    case 0:
        for (u32 i = 0; i < gsp.vs_out_num && gsp.cur < 16; i++) gsp.buffer[gsp.cur++] = in[i];
        if (gsp.cur >= gsp.end) {
            gsp.cur = 0;
            load_inputs(g.gs_unit, R::GS, gsp.buffer);
            gs_run();
        }
        break;
    case 1:
        if (gsp.main_num) {
            for (u32 i = 0; i < gsp.vs_out_num && gsp.cur < 96; i++) g.gs.f[gsp.cur++] = in[i];
            gsp.main_num--;
        } else if (gsp.cur < 96) {
            g.gs.f[gsp.cur++] = in[0];
        }
        if (--gsp.total_num == 0) {
            gsp.need_index = true;
            gs_run();
        }
        break;
    default:
        for (u32 i = 0; i < gsp.vs_out_num && gsp.cur < 96; i++) g.gs.f[gsp.cur++] = in[i];
        if (gsp.cur >= gsp.end) {
            gsp.cur = gsp.begin;
            gs_run();
        }
        break;
    }
}

// ---------------------------------------------------------------- batched vertex shading
std::vector<ShaderUnit> units;
std::vector<u64> unit_draw;
u64 draw_counter = 0;
std::vector<u32> idx_list, uniq, slot_of;
std::vector<u32> stamp;
u32 stamp_cur = 0;
std::vector<OutVertex> outv;
std::vector<SV> outsv;
std::vector<u32> outcode;

ShaderUnit &unit_for(u32 tid) {
    if (unit_draw[tid] != draw_counter) {
        shader_unit_init(units[tid], g.vs);
        unit_draw[tid] = draw_counter;
    }
    return units[tid];
}
}  // namespace

void pa_reconfigure(u32 topology) {
    pa_reset();
    g.pa.topology = topology;
}
void pa_reset() {
    g.pa.index = 0;
    g.pa.strip_ready = false;
    g.pa.winding = false;
}

void vtx_reset_immediate() {
    g.imm_attr = 0;
    g.imm_reset_gs = true;
    g.imm_n = 0;
}

void vtx_immediate_vertex() {
    vtx_finish_jobs();
    cur_ds = draw_state();
    static ShaderUnit u;
    shader_unit_init(u, g.vs);
    load_inputs(u, R::VS, g.imm_vertex);
    shader_run(shader_get_program(g.vs), u, g.vs, g.regs[R::VS + R::SH_ENTRY] & 0xffff);
    V4 buf[16];
    gather_outputs(u, g.regs[R::VS + R::SH_OUTMASK] & 0xffff, buf);
    if (g.imm_reset_gs) {
        gs_reconfigure();
        g.imm_reset_gs = false;
    }
    if (gsp.active) {
        gs_submit_vertex(buf);
    } else {
        OutMap m;
        setup_outmap(m);
        OutVertex v;
        map_vertex(m, buf, v);
        pa_submit(v);
    }
}

// ---------------------------------------------------------------- deferred draws
// Draws without a geometry shader are recorded as jobs (with a snapshot of everything the vertex
// stage needs) and processed in parallel when the batch is flushed: vertex shading in chunks of
// vertices, triangle setup in chunks of triangles, then binning in submission order.
namespace {
struct DrawJob {
    const DrawState *ds;
    Loader L;
    OutMap m;
    const ShaderProgram *prog;
    u32 entry, outmask, in_maxi;
    u64 in_map;
    V4 f[96];
    bool b[16];
    u8 iu[4][4];
    V4 defaults[16];  // fixed vertex attributes at draw time
    std::vector<u32> uniq;   // unique vertex ids to shade
    std::vector<u32> tris;   // slot triples
    std::vector<OutVertex> outv;
    std::vector<SV> outsv;
    std::vector<u32> outcode;
    s32 pa_buf[2];           // slots that remain buffered in the primitive assembler (-1 none)
};
std::vector<std::unique_ptr<DrawJob>> jobs;
u32 njobs = 0;
u32 pending_indices = 0;

struct Task {
    u32 job, begin, end;
    TriSink out;
};
std::vector<Task> shade_tasks, setup_tasks;

std::vector<ShaderUnit> junits;
std::vector<const DrawJob *> junit_job;

ShaderUnit &unit_for_job(u32 tid, const DrawJob &j) {
    if (junit_job[tid] != &j) {
        // fresh unit per draw (as before deferral): unwritten registers read as zero
        ShaderUnit &u = junits[tid];
        memset(&u, 0, sizeof(u));
        memcpy(&u.r[REG_UNI], j.f, sizeof(j.f));
        u.r[REG_ONE] = V4{{1, 1, 1, 1}};
        junit_job[tid] = &j;
    }
    return junits[tid];
}

std::vector<ShaderUnit4 *> junits4;

void shade_range(DrawJob &j, u32 begin, u32 end, u32 tid) {
    const bool fresh = junit_job[tid] != &j;
    ShaderUnit &u = unit_for_job(tid, j);
    V4 in[16], buf[16];
    u32 k = begin;
    if (shader_program_simd(j.prog)) {
        ShaderUnit4 &u4 = *junits4[tid];
        if (fresh) shader_unit4_reset(&u4);
        u32 inmask = 0;
        for (u32 a = 0; a <= j.in_maxi; a++) inmask |= 1u << ((j.in_map >> (4 * a)) & 0xf);
        V4 in4[4][16], out4[4][16];
        for (; k + 4 <= end; k += 4) {
            for (u32 l = 0; l < 4; l++) {
                load_vertex(j.L, j.uniq[k + l], in, j.defaults);
                for (u32 a = 0; a <= j.in_maxi; a++) in4[l][(j.in_map >> (4 * a)) & 0xf] = in[a];
            }
            if (!shader_run4(j.prog, u4, j.f, j.b, j.iu, j.entry, in4, inmask, out4, j.outmask)) {
                // lanes diverged: scalar interpreter for these four vertices
                if (g_trace) LOG("gpu trace: simd fallback (prog %p)\n", (const void *)j.prog);
                for (u32 l = 0; l < 4; l++) {
                    for (u32 r = 0; r < 16; r++)
                        if (inmask & (1u << r)) u.r[REG_IN + r] = in4[l][r];
                    shader_run(j.prog, u, j.b, j.iu, j.entry);
                    for (u32 r = 0; r < 16; r++) out4[l][r] = u.r[REG_OUT + r];
                }
            }
            for (u32 l = 0; l < 4; l++) {
                u32 o = 0;
                for (u32 r = 0; r < 16; r++)
                    if (j.outmask & (1u << r)) buf[o++] = out4[l][r];
                map_vertex(j.m, buf, j.outv[k + l]);
                if (!metal_active()) raster_prepare_vertex(*j.ds, j.outv[k + l], j.outsv[k + l], j.outcode[k + l]);
            }
        }
    }
    for (; k < end; k++) {
        load_vertex(j.L, j.uniq[k], in, j.defaults);
        for (u32 a = 0; a <= j.in_maxi; a++) u.r[REG_IN + ((j.in_map >> (4 * a)) & 0xf)] = in[a];
        shader_run(j.prog, u, j.b, j.iu, j.entry);
        gather_outputs(u, j.outmask, buf);
        map_vertex(j.m, buf, j.outv[k]);
        if (!metal_active()) raster_prepare_vertex(*j.ds, j.outv[k], j.outsv[k], j.outcode[k]);
    }
}

void setup_range(DrawJob &j, u32 begin, u32 end, TriSink &out) {
    raster_set_sink(&out);
    const u32 *t = j.tris.data();
    for (u32 k = begin; k < end; k++) {
        const u32 a = t[3 * k], b = t[3 * k + 1], c = t[3 * k + 2];
        if (j.outcode[a] | j.outcode[b] | j.outcode[c])
            raster_triangle(j.ds, j.outv[a], j.outv[b], j.outv[c]);
        else
            raster_triangle_sv(j.ds, j.outsv[a], j.outsv[b], j.outsv[c]);
    }
    raster_set_sink(nullptr);
}
}  // namespace

bool vtx_jobs_pending() { return njobs != 0; }

void vtx_finish_jobs() {
    if (!njobs) return;
    if (junits.empty()) {
        junits.resize(pool_threads());
        junit_job.assign(pool_threads(), nullptr);
        for (auto &u : junits) memset(&u, 0, sizeof(u));
        for (u32 t = 0; t < pool_threads(); t++) junits4.push_back(shader_unit4_new());
    }
    for (auto &p : junit_job) p = nullptr;
    if (g_trace) LOG("gpu trace: finish %u jobs, %u indices\n", njobs, pending_indices);
    double t0 = g_stats ? now_ms() : 0;
    // vertex shading
    constexpr u32 VCHUNK = 64, TCHUNK = 96;
    shade_tasks.clear();
    for (u32 ji = 0; ji < njobs; ji++) {
        DrawJob &j = *jobs[ji];
        const u32 nu = (u32)j.uniq.size();
        j.outv.resize(nu);
        j.outsv.resize(nu);
        j.outcode.resize(nu);
        for (u32 b = 0; b < nu; b += VCHUNK) shade_tasks.push_back({ji, b, std::min(nu, b + VCHUNK), {}});
    }
    parallel_for((u32)shade_tasks.size(), [&](u32 i, u32 tid) {
        Task &t = shade_tasks[i];
        shade_range(*jobs[t.job], t.begin, t.end, tid);
    });
    double t1 = g_stats ? now_ms() : 0;
    if (metal_active()) {
        // the Metal backend clips and sets up triangles itself: hand it the shaded vertices
        MBatch &mb = raster_mbatch();
        for (u32 ji = 0; ji < njobs; ji++) {
            const DrawJob &j = *jobs[ji];
            const u32 base = (u32)mb.cverts.size(), first = (u32)mb.idx.size();
            mb.cverts.insert(mb.cverts.end(), j.outv.begin(), j.outv.end());
            for (u32 t : j.tris) mb.idx.push_back((base + t) | MIDX_CPU);
            mb.add_seg(j.ds, first);
        }
        const DrawJob &last = *jobs[njobs - 1];
        for (u32 k = 0; k < 2; k++)
            if (last.pa_buf[k] >= 0) g.pa.buf[k] = last.outv[last.pa_buf[k]];
        njobs = 0;
        pending_indices = 0;
        if (g_stats) g_t_shade += now_ms() - t0;
        return;
    }
    // triangle setup
    u32 ntasks = 0;
    for (u32 ji = 0; ji < njobs; ji++) {
        const u32 nt = (u32)jobs[ji]->tris.size() / 3;
        for (u32 b = 0; b < nt; b += TCHUNK) {
            if (ntasks >= setup_tasks.size()) setup_tasks.emplace_back();
            Task &t = setup_tasks[ntasks++];
            t.job = ji;
            t.begin = b;
            t.end = std::min(nt, b + TCHUNK);
            t.out.clear();
        }
    }
    parallel_for(ntasks, [&](u32 i, u32) {
        Task &t = setup_tasks[i];
        setup_range(*jobs[t.job], t.begin, t.end, t.out);
    });
    // bin in submission order
    for (u32 i = 0; i < ntasks; i++) raster_append(setup_tasks[i].out);
    // vertices left in the primitive assembler by the last draw
    const DrawJob &last = *jobs[njobs - 1];
    for (u32 k = 0; k < 2; k++)
        if (last.pa_buf[k] >= 0) g.pa.buf[k] = last.outv[last.pa_buf[k]];
    njobs = 0;
    pending_indices = 0;
    if (g_stats) {
        double t2 = now_ms();
        g_t_shade += t1 - t0;
        g_t_setup += t2 - t1;
    }
}


// ---------------------------------------------------------------- Metal vertex shading
namespace {
bool gpu_vs = false;  // vertex programs run as Metal compute kernels (GPU_CPU_VS=1: on the CPU)
u64 st_gpu_draws = 0, st_cpu_draws = 0;

// Shades a draw's vertices with the program's compute kernel: copies the vertex arrays it reads and
// appends the draw to the Metal batch. False if the kernel is not ready (the CPU shades the draw).
bool gpu_draw(const Loader &L, const ShaderProgram *prog, u32 entry, u32 outmask, bool indexed, u32 n) {
    const void *kernel = metal_vs_kernel(prog, entry, g.vs);
    if (!kernel) return false;
    vtx_finish_jobs();  // CPU-shaded draws queued before this one go first
    MBatch &mb = raster_mbatch();
    const u32 *regs = g.regs;
    // unique vertices
    const u32 vids_off = (u32)mb.vids.size();
    std::vector<u32> &slots = idx_list;
    u32 minv = ~0u, maxv = 0;
    if (indexed) {
        if (++stamp_cur == 0) {
            std::fill(stamp.begin(), stamp.end(), 0);
            stamp_cur = 1;
        }
        for (u32 i = 0; i < n; i++) {
            const u32 v = idx_list[i] & 0xffff;
            if (stamp[v] != stamp_cur) {
                stamp[v] = stamp_cur;
                slot_of[v] = (u32)mb.vids.size() - vids_off;
                mb.vids.push_back(v);
                minv = std::min(minv, v);
                maxv = std::max(maxv, v);
            }
            slots[i] = slot_of[v];
        }
    } else {
        for (u32 i = 0; i < n; i++) {
            mb.vids.push_back(idx_list[i]);
            slots[i] = i;
        }
        minv = n ? idx_list[0] : 0;
        maxv = n ? idx_list[n - 1] : 0;
    }
    const u32 nverts = (u32)mb.vids.size() - vids_off;
    if (!nverts) return true;
    mb.jobs.emplace_back();
    GpuVsJob &job = mb.jobs.back();
    job.kernel = kernel;
    job.vids_off = vids_off;
    VsParamsG &P = job.p;
    memcpy(P.f, g.vs.f, sizeof(P.f));
    memcpy(P.defaults, g.default_attr, sizeof(P.defaults));
    // vertex arrays: the byte ranges the draw reads, merged where attributes interleave
    struct Range { u32 start, end, attr; };
    Range rs[16];
    u32 nr = 0;
    for (u32 i = 0; i < 16; i++) {
        VsAttrG &A = P.attr[i];
        A = VsAttrG{0, 0, 0, 0};
        if (i >= L.num) continue;
        const LoaderAttr &la = L.a[i];
        if (la.is_default) {
            A.elems = 5;
            continue;
        }
        if (la.elems == 0) continue;
        const u32 es = la.fmt == 3 ? 4 : la.fmt == 2 ? 2 : 1;
        const u32 start = L.base + la.offset + la.stride * minv;
        const u32 end = L.base + la.offset + la.stride * maxv + es * la.elems;
        if (!pptr(start) || !pptr(end - 1) || pptr(end - 1) - pptr(start) != end - 1 - start) continue;
        A.stride = la.stride;
        A.fmt = la.fmt;
        A.elems = la.elems;
        rs[nr++] = {start, end, i};
    }
    std::sort(rs, rs + nr, [](const Range &a, const Range &b) { return a.start < b.start; });
    for (u32 i = 0; i < nr;) {
        u32 j = i + 1, end = rs[i].end;
        while (j < nr && rs[j].start <= end + 64) end = std::max(end, rs[j++].end);
        const u32 at = (u32)mb.vdata.size();
        mb.vdata.resize(at + ((end - rs[i].start + 3) & ~3u));
        memcpy(&mb.vdata[at], pptr(rs[i].start), end - rs[i].start);
        for (u32 k = i; k < j; k++) {
            const LoaderAttr &la = L.a[rs[k].attr];
            P.attr[rs[k].attr].offset = (s32)(at + (rs[k].start - rs[i].start)) - (s32)(la.stride * minv);
        }
        i = j;
    }
    // input registers, output slots, integer and boolean uniforms
    for (auto &r : P.reg_attr) r = -1;
    const u32 in_maxi = regs[R::VS + R::SH_INPUT_CFG] & 0xf;
    const u64 in_map = (u64)regs[R::VS + R::SH_PERM_LO] | ((u64)regs[R::VS + R::SH_PERM_HI] << 32);
    for (u32 a = 0; a <= in_maxi; a++) P.reg_attr[(in_map >> (4 * a)) & 0xf] = a < L.num ? (s32)a : -1;
    OutMap m;
    setup_outmap(m);
    u32 out_reg[16], nout = 0;
    for (u32 r = 0; r < 16; r++)
        if (outmask & (1u << r)) out_reg[nout++] = r;
    for (u32 i = 0; i < 24; i++) {
        const s32 s = m.slot_src[i];
        P.slot_src[i] = s >= 0 && (u32)(s >> 2) < nout ? (s32)(out_reg[s >> 2] * 4 + (s & 3)) : -1;
    }
    for (u32 k = 0; k < 4; k++)
        for (u32 c = 0; c < 4; c++) P.iu[k][c] = g.vs.i[k][c];
    P.b = 0;
    for (u32 k = 0; k < 16; k++) P.b |= (u32)g.vs.b[k] << k;
    P.nverts = nverts;
    P.out_base = mb.gpu_verts;
    P.first = 0;
    mb.gpu_verts += nverts;
    // primitive assembly on slots
    PrimAssembler &pa = g.pa;
    const u32 first = (u32)mb.idx.size(), ob = P.out_base;
    u32 bi[2] = {0, 0}, index = 0;
    bool ready = false;
    if (pa.topology == 1 || pa.topology == 2) {
        const bool strip = pa.topology == 1;
        for (u32 i = 0; i < n; i++) {
            const u32 v = slots[i];
            if (ready) mb.idx.insert(mb.idx.end(), {ob + bi[0], ob + bi[1], ob + v});
            bi[index] = v;
            ready |= index == 1;
            index = strip ? !index : 1;
        }
    } else {
        const u32 nt = (n / 3) * 3;
        for (u32 i = 0; i < nt; i++) mb.idx.push_back(ob + slots[i]);
        for (u32 i = nt; i < n; i++) bi[index++] = slots[i];
    }
    mb.add_seg(cur_ds, first);
    // vertices carried over to a following draw: shade those on the CPU
    pa.index = index;
    pa.strip_ready = ready;
    for (u32 k = 0; k < 2; k++) {
        const bool keep = (pa.topology == 1 || pa.topology == 2) ? (ready || k < index) : k < index;
        if (!keep) continue;
        ShaderUnit &u = unit_for(0);
        V4 in[16], buf[16];
        load_vertex(L, mb.vids[vids_off + bi[k]], in);
        load_inputs(u, R::VS, in);
        shader_run(prog, u, g.vs, entry);
        gather_outputs(u, outmask, buf);
        map_vertex(m, buf, pa.buf[k]);
    }
    if (g_stats) st_gpu_draws++;
    return true;
}
}  // namespace

void vtx_draw(bool indexed) {
    cur_ds = draw_state();
    if (units.empty()) {
        gpu_vs = metal_active() && !getenv("GPU_CPU_VS");
        units.resize(pool_threads());
        unit_draw.assign(pool_threads(), ~0ull);
        stamp.assign(65536, 0);
        slot_of.resize(65536);
    }
    draw_counter++;
    const u32 *regs = g.regs;
    Loader L;
    setup_loader(L);
    u32 n = regs[R::NUM_VERTICES];
    if (n > (1u << 20)) {
        LOG("gpu: draw with %u vertices ignored\n", n);
        return;
    }
    const u32 entry = regs[R::VS + R::SH_ENTRY] & 0xffff;
    const u32 outmask = regs[R::VS + R::SH_OUTMASK] & 0xffff;
    const ShaderProgram *prog = shader_get_program(g.vs);
    gs_reconfigure();

    // vertex ids
    idx_list.resize(n);
    if (indexed) {
        const u32 ia = regs[R::INDEX_ARRAY];
        const u8 *ip = pptr(L.base + (ia & 0x0fffffff));
        if (!ip) return;
        if (ia >> 31) {
            for (u32 i = 0; i < n; i++) {
                u16 v;
                memcpy(&v, ip + 2 * i, 2);
                idx_list[i] = v;
            }
        } else {
            for (u32 i = 0; i < n; i++) idx_list[i] = ip[i];
        }
    } else {
        const u32 off = regs[R::VERTEX_OFFSET];
        for (u32 i = 0; i < n; i++) idx_list[i] = i + off;
    }

    if (gsp.active) {
        // geometry shader path: sequential (GS state persists between invocations)
        vtx_finish_jobs();
        ShaderUnit &u = unit_for(0);
        V4 in[16], buf[16];
        for (u32 i = 0; i < n; i++) {
            const u32 v = idx_list[i];
            if (gsp.mode == 1 && gsp.need_index && indexed) {
                gs_submit_index(v);
                continue;
            }
            load_vertex(L, v, in);
            load_inputs(u, R::VS, in);
            shader_run(prog, u, g.vs, entry);
            gather_outputs(u, outmask, buf);
            gs_submit_vertex(buf);
        }
        return;
    }
    if (g_trace)
        LOG("gpu trace:   skip=%d rt %08x %ux%u fmt %x depth %08x needshade %d cw %d at %d/%d gas %d vp %.1f %.1f\n",
            (int)cur_ds->skip, cur_ds->rt.color_addr, cur_ds->rt.w, cur_ds->rt.h, cur_ds->rt.color_fmt,
            cur_ds->rt.depth_addr, (int)cur_ds->need_shade, (int)cur_ds->color_write, (int)cur_ds->alpha_test,
            cur_ds->alpha_func, (int)cur_ds->gas, cur_ds->vp_hw, cur_ds->vp_hh);
    if (cur_ds->skip) return;

    PrimAssembler &pa = g.pa;
    if (pa.index != 0 || pa.strip_ready) {
        // primitive continues from a previous draw: shade now and use the generic assembler
        vtx_finish_jobs();
        ShaderUnit &u = unit_for(0);
        OutMap m;
        setup_outmap(m);
        V4 in[16], buf[16];
        OutVertex v;
        for (u32 i = 0; i < n; i++) {
            load_vertex(L, idx_list[i], in);
            load_inputs(u, R::VS, in);
            shader_run(prog, u, g.vs, entry);
            gather_outputs(u, outmask, buf);
            map_vertex(m, buf, v);
            pa_submit(v);
        }
        return;
    }

    if (gpu_vs && gpu_draw(L, prog, entry, outmask, indexed, n)) return;

    // record a deferred job
    if (njobs == jobs.size()) jobs.push_back(std::make_unique<DrawJob>());
    DrawJob &j = *jobs[njobs++];
    j.ds = cur_ds;
    j.L = L;
    setup_outmap(j.m);
    j.prog = prog;
    j.entry = entry;
    j.outmask = outmask;
    j.in_maxi = regs[R::VS + R::SH_INPUT_CFG] & 0xf;
    j.in_map = (u64)regs[R::VS + R::SH_PERM_LO] | ((u64)regs[R::VS + R::SH_PERM_HI] << 32);
    memcpy(j.f, g.vs.f, sizeof(j.f));
    memcpy(j.b, g.vs.b, sizeof(j.b));
    memcpy(j.iu, g.vs.i, sizeof(j.iu));
    memcpy(j.defaults, g.default_attr, sizeof(j.defaults));
    // unique vertices; idx_list[i] becomes the slot of vertex i
    j.uniq.clear();
    std::vector<u32> &slots = idx_list;
    if (indexed) {
        if (++stamp_cur == 0) {
            std::fill(stamp.begin(), stamp.end(), 0);
            stamp_cur = 1;
        }
        for (u32 i = 0; i < n; i++) {
            const u32 v = idx_list[i] & 0xffff;
            if (stamp[v] != stamp_cur) {
                stamp[v] = stamp_cur;
                slot_of[v] = (u32)j.uniq.size();
                j.uniq.push_back(v);
            }
            slots[i] = slot_of[v];
        }
    } else {
        j.uniq.resize(n);
        for (u32 i = 0; i < n; i++) {
            j.uniq[i] = idx_list[i];
            slots[i] = i;
        }
    }
    // primitive assembly on slots
    j.tris.clear();
    u32 bi[2] = {0, 0};
    u32 index = 0;
    bool ready = false;
    if (pa.topology == 1 || pa.topology == 2) {
        const bool strip = pa.topology == 1;
        for (u32 i = 0; i < n; i++) {
            const u32 v = slots[i];
            if (ready) j.tris.insert(j.tris.end(), {bi[0], bi[1], v});
            bi[index] = v;
            ready |= index == 1;
            index = strip ? !index : 1;
        }
    } else {
        u32 i = 0;
        j.tris.assign(slots.begin(), slots.begin() + (n / 3) * 3);
        i = (n / 3) * 3;
        for (; i < n; i++) bi[index++] = slots[i];
    }
    // vertices carried over to following draws (filled in when the job is processed)
    pa.index = index;
    pa.strip_ready = ready;
    for (u32 k = 0; k < 2; k++) {
        const bool keep = (pa.topology == 1 || pa.topology == 2) ? (ready || k < index) : k < index;
        j.pa_buf[k] = keep ? (s32)bi[k] : -1;
    }
    pending_indices += n;
    if (njobs >= 1024 || pending_indices >= (1u << 18)) vtx_finish_jobs();
}


}  // namespace gpu
