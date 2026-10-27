// gpu_replay <dumpdir> <out.png> [-t]
//
// Loads <dumpdir>/fcram.bin (guest VA 0x14000000.., physical FCRAM 0x20000000..) and vram.bin
// (physical 0x18000000), replays <dumpdir>/ops.txt through the software GPU and writes the
// presented framebuffers as PNGs (first present -> out.png, further ones -> out_1.png, ...).
// With -t, the linear output of every display transfer is also written (out_xfer<N>.png).
// ops.txt lines (all numbers hex except present width/height which are decimal):
//   cmdlist <paddr> <size>
//   fill <start> <end> <value> <control>
//   transfer <in> <out> <in_dims> <out_dims> <flags>
//   copy <in> <out> <size> <in_wg> <out_wg> <flags>
//   present <paddr> <width> <height> <stride> <format>
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include "gpu/gpu.h"
#include "gpu/pica.h"
#include "png.h"
#include "rt.h"

static bool load_file(const std::string &path, u8 *dst, size_t max) {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) return false;
    size_t n = fread(dst, 1, max, f);
    fclose(f);
    fprintf(stderr, "loaded %s (%zu bytes)\n", path.c_str(), n);
    return true;
}

static std::string out_name(const std::string &base, const std::string &suffix) {
    size_t dot = base.rfind('.');
    if (dot == std::string::npos) return base + suffix + ".png";
    return base.substr(0, dot) + suffix + base.substr(dot);
}

// write a linear framebuffer (lines of `width` pixels, `height` lines) rotated upright
static void write_fb(const std::string &path, u32 paddr, u32 width, u32 height, u32 stride,
                     u32 fmt) {
    gpu::flush();  // asynchronous transfers (Metal) land in guest memory
    const u8 *src = mem::paddr_ptr(paddr);
    if (!src) {
        fprintf(stderr, "present: bad address %08x\n", paddr);
        return;
    }
    fmt &= 7;
    const u32 bpp = gpu::pixel_bytes(fmt);
    std::vector<u32> img((size_t)width * height);
    // image is `height` wide and `width` tall; line l becomes column l, pixel 0 is at the bottom
    for (u32 l = 0; l < height; l++)
        for (u32 p = 0; p < width; p++) {
            u32 c = gpu::decode_pixel(fmt, src + (size_t)l * stride + p * bpp);
            img[(size_t)(width - 1 - p) * height + l] = c | 0xff000000u;
        }
    write_png(path.c_str(), height, width, img.data());
    fprintf(stderr, "wrote %s (%ux%u)\n", path.c_str(), height, width);
}

// the Metal backend's high-resolution image of a framebuffer, if it has one
static void write_hires(const std::string &path, u32 fb_paddr) {
    const u32 *px;
    int w, h;
    if (!gpu::get_display_image_for(fb_paddr, &px, &w, &h)) return;
    write_png(path.c_str(), (u32)w, (u32)h, px);
    fprintf(stderr, "wrote %s (%dx%d)\n", path.c_str(), w, h);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: gpu_replay <dumpdir> <out.png> [-t] [-r n] [--trust] [--inval] [-p presets] [-f addr w h]\n");
        return 1;
    }
    const std::string dir = argv[1], out = argv[2];
    bool dump_transfers = false;
    bool inval = false;  // simulate the OS layer's cache-maintenance invalidations after every op
    int repeat = 1;
    std::string preset;
    u32 final_addr = 0, final_w = 0, final_h = 0;
    for (int i = 3; i < argc; i++) {
        std::string a = argv[i];
        if (a == "-t") dump_transfers = true;
        if (a == "--inval") inval = true;
        if (a == "--trust") gpu::set_trust_invalidations(true);
        if (a == "-r" && i + 1 < argc) repeat = atoi(argv[++i]);
        if (a == "-p" && i + 1 < argc) preset = argv[++i];
        if (a == "-f" && i + 3 < argc) {
            final_addr = strtoul(argv[i + 1], nullptr, 16);
            final_w = atoi(argv[i + 2]);
            final_h = atoi(argv[i + 3]);
            i += 3;
        }
    }
    mem::init();
    if (!load_file(dir + "/fcram.bin", g_mem + mem::LINEAR_BASE, mem::LINEAR_SIZE) ||
        !load_file(dir + "/vram.bin", g_mem + mem::VRAM_BASE, mem::VRAM_SIZE)) {
        fprintf(stderr, "cannot load dump\n");
        return 1;
    }
    gpu::init();
    // optional saved GPU register/shader state taken together with the memory snapshot
    if (gpu::load_state((dir + "/gpustate.bin").c_str())) fprintf(stderr, "loaded gpustate.bin\n");
    // optional register presets: "reg <id> <value>" (hex), executed as a command list
    std::vector<u32> pre_cmds;
    if (!preset.empty()) {
        FILE *pf = fopen(preset.c_str(), "r");
        char pl[256];
        while (pf && fgets(pl, sizeof pl, pf)) {
            u32 id, v;
            if (sscanf(pl, "reg %x %x", &id, &v) == 2) {
                pre_cmds.push_back(v);
                pre_cmds.push_back(id | 0xfu << 16);
            }
        }
        if (pf) fclose(pf);
    }
    auto run_presets = [&]() {
        if (pre_cmds.empty()) return;
        const u32 at = 0x27f00000;  // scratch FCRAM
        memcpy(mem::paddr_ptr(at), pre_cmds.data(), pre_cmds.size() * 4);
        gpu::process_command_list(at, (u32)pre_cmds.size() * 4);
    };
    run_presets();
    std::vector<std::string> lines;
    {
        FILE *f = fopen((dir + "/ops.txt").c_str(), "r");
        if (!f) {
            fprintf(stderr, "cannot open ops.txt\n");
            return 1;
        }
        char buf[512];
        while (fgets(buf, sizeof buf, f)) lines.push_back(buf);
        fclose(f);
    }
    // extra warm-up iterations (without output) to measure steady-state cost
    for (int it = 1; it < repeat; it++) {
        auto t0 = std::chrono::steady_clock::now();
        double per[4] = {0, 0, 0, 0};  // cmdlist, fill, transfer, copy
        for (auto &l : lines) {
            u32 a, b, c, d, e, g2;
            auto ts = std::chrono::steady_clock::now();
            int k = -1;
            if (sscanf(l.c_str(), "cmdlist %x %x", &a, &b) == 2) gpu::process_command_list(a, b), k = 0;
            else if (sscanf(l.c_str(), "fill %x %x %x %x", &a, &b, &c, &d) == 4) gpu::memory_fill(a, b, c, d), k = 1;
            else if (sscanf(l.c_str(), "transfer %x %x %x %x %x", &a, &b, &c, &d, &e) == 5) gpu::display_transfer(a, b, c, d, e), k = 2;
            else if (sscanf(l.c_str(), "copy %x %x %x %x %x %x", &a, &b, &c, &d, &e, &g2) == 6) gpu::texture_copy(a, b, c, d, e, g2), k = 3;
            if (k >= 0) per[k] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ts).count();
        }
        auto t1 = std::chrono::steady_clock::now();
        fprintf(stderr, "iteration %d: %.2f ms (cmdlists %.2f, fills %.2f, transfers %.2f, copies %.2f)\n", it,
                std::chrono::duration<double, std::milli>(t1 - t0).count(), per[0], per[1], per[2], per[3]);
    }
    char line[512];
    int presents = 0, transfers = 0;
    double gpu_ms = 0;
    for (auto &ls : lines) {
        snprintf(line, sizeof line, "%s", ls.c_str());
        char op[32];
        if (sscanf(line, "%31s", op) != 1) continue;
        std::string o = op;
        if (inval) {
            gpu::invalidate_memory(mem::VRAM_PADDR, mem::VRAM_SIZE);
            gpu::invalidate_memory(mem::FCRAM_PADDR, mem::LINEAR_SIZE);
        }
        auto t0 = std::chrono::steady_clock::now();
        if (o == "cmdlist") {
            u32 a, s;
            if (sscanf(line, "%*s %x %x", &a, &s) == 2) gpu::process_command_list(a, s);
        } else if (o == "fill") {
            u32 a, b, v, c;
            if (sscanf(line, "%*s %x %x %x %x", &a, &b, &v, &c) == 4) gpu::memory_fill(a, b, v, c);
        } else if (o == "transfer") {
            u32 a, b, c, d, e;
            if (sscanf(line, "%*s %x %x %x %x %x", &a, &b, &c, &d, &e) == 5) {
                gpu::display_transfer(a, b, c, d, e);
                if (dump_transfers && !(e & 2) && !(e & 0x20)) {
                    u32 hs = ((e >> 24) & 3) ? 1 : 0, vs = ((e >> 24) & 3) == 2 ? 1 : 0;
                    u32 w = (d & 0xffff) >> hs, h = (d >> 16) >> vs;
                    u32 fmt = (e >> 12) & 7;
                    auto t1 = std::chrono::steady_clock::now();
                    gpu_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
                    write_fb(out_name(out, "_xfer" + std::to_string(transfers)), b, w, h,
                             w * gpu::pixel_bytes(fmt), fmt);
                    write_hires(out_name(out, "_xfer" + std::to_string(transfers) + "_hires"), b);
                    transfers++;
                    continue;
                }
                transfers++;
            }
        } else if (o == "copy") {
            u32 a, b, c, d, e, g2;
            if (sscanf(line, "%*s %x %x %x %x %x %x", &a, &b, &c, &d, &e, &g2) == 6)
                gpu::texture_copy(a, b, c, d, e, g2);
        } else if (o == "present") {
            u32 a, w, h, s, fm;
            if (sscanf(line, "%*s %x %u %u %x %x", &a, &w, &h, &s, &fm) == 5) {
                std::string name = presents == 0 ? out : out_name(out, "_" + std::to_string(presents));
                write_fb(name, a, w, h, s, fm);
                write_hires(out_name(name.substr(0, name.rfind('.')), "_hires"), a);
                presents++;
            }
            continue;
        }
        auto t1 = std::chrono::steady_clock::now();
        gpu_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    fprintf(stderr, "gpu time: %.2f ms\n", gpu_ms);
    if (final_addr) {
        // convert the tiled RGBA8 render target to linear and write it upright
        const u32 tmp = 0x27e00000;
        gpu::display_transfer(final_addr, tmp, final_w | final_h << 16, final_w | final_h << 16, 0);
        write_fb(out_name(out, "_final"), tmp, final_w, final_h, final_w * 4, 0);
    }
    return 0;
}
