// shader_golden [--glsl [--samples <n>] [--no-logic-op]] [-o <dir>] <pipeline list>...
// shader_golden --vs [--glsl] [-o <dir>] <vertex program list (.vs.bin)>...
//
// Generates the fragment shader of every key in the pipeline lists (games/*/pipelines.bin) and prints how many
// and one hash of them all, so a change to the generator can be checked against an older build's output (the
// MSL must stay the same byte for byte while only another language is added). -o writes each shader as
// <dir>/<n>.<metal|frag>. Every key must also survive decode_key/fs_key unchanged. With --glsl the Vulkan
// backend's GLSL is generated instead, and each one compiled to SPIR-V with shaderc (when built with it);
// --no-logic-op: as for a device without logic ops in its blender (MoltenVK), which blends those in the shader.
// Each list's count of shaders blending in the shader (reading the target) is printed too.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>

#include "gpu/fsgen.h"
#include "gpu/vsblob.h"
#include "gpu/vk_shaders.h"
#include "rt.h"

#ifdef SHADER_GOLDEN_SHADERC
#include <shaderc/shaderc.h>
#endif

int main(int argc, char **argv) {
    bool glsl = false, vs = false, hw_logic = true;
    unsigned samples = 1;
    const char *outdir = nullptr;
    std::vector<std::string> lists;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--glsl")) glsl = true;
        else if (!strcmp(argv[i], "--vs")) vs = true;
        else if (!strcmp(argv[i], "--no-logic-op")) hw_logic = false;
        else if (!strcmp(argv[i], "-o") && i + 1 < argc) outdir = argv[++i];
        else if (!strcmp(argv[i], "--samples") && i + 1 < argc) samples = (unsigned)atoi(argv[++i]);
        else lists.push_back(argv[i]);
    }
#ifdef SHADER_GOLDEN_SHADERC
    shaderc_compiler_t sc = glsl ? shaderc_compiler_initialize() : nullptr;
    shaderc_compile_options_t opt = glsl ? shaderc_compile_options_initialize() : nullptr;
    if (opt) shaderc_compile_options_set_target_env(opt, shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_3);
#endif
    std::set<std::string> seen;
    if (vs) {
        // vertex kernels: each program's generic kernel, and its specialized one when the blob has a specialization
        u64 hash = 0xcbf29ce484222325ull;
        int n = 0, spec = 0, bad = 0;
        for (const std::string &path : lists)
            for (const std::string &b : gpu::read_blobs(path, gpu::kVsMagic, gpu::kVsVersion)) {
                if (!seen.insert(b).second) continue;
                gpu::VsBlob v;
                if (!gpu::parse_vs_blob(b, v)) continue;
                gpu::ShaderProgram *p = gpu::shader_decode(v.code.data(), v.code_size, v.swz.data());
                std::vector<std::string> srcs = {glsl ? gpu::shader_glsl(p, v.entry) : gpu::shader_msl(p, v.entry, "vk_main")};
                if (v.spec)
                    srcs.push_back(glsl ? gpu::shader_glsl_spec(p, v.entry, v.sp.b, v.sp.iu)
                                        : gpu::shader_msl_spec(p, v.entry, "vk_main", v.sp.b, v.sp.iu));
                gpu::shader_free(p);
                for (const std::string &src : srcs) {
                    for (unsigned char c : src) hash = (hash ^ c) * 0x100000001b3ull;
                    hash = (hash ^ 0xff) * 0x100000001b3ull;
                    if (!src.empty()) n++;
#ifdef SHADER_GOLDEN_SHADERC
                    if (glsl && !src.empty()) {
                        const std::string full = std::string(gpu::kVkVsCommonGLSL) + src;
                        shaderc_compilation_result_t r = shaderc_compile_into_spv(sc, full.data(), full.size(),
                                                                                  shaderc_compute_shader, "vs", "main", opt);
                        if (shaderc_result_get_compilation_status(r) != shaderc_compilation_status_success && bad++ < 3)
                            fprintf(stderr, "vertex kernel %d does not compile:\n%s\n", n, shaderc_result_get_error_message(r));
                        shaderc_result_release(r);
                    }
#endif
                    if (outdir && !src.empty()) {
                        char name[512];
                        snprintf(name, sizeof name, "%s/vs%05d.%s", outdir, n, glsl ? "comp" : "metal");
                        if (FILE *f = fopen(name, "wb")) {
                            fwrite(src.data(), 1, src.size(), f);
                            fclose(f);
                        }
                    }
                }
                spec += v.spec;
            }
        printf("%d vertex kernels (%s, %d programs specialized), hash %016llx, %d not compiling\n", n,
               glsl ? "GLSL" : "MSL", spec, (unsigned long long)hash, bad);
        return bad ? 1 : 0;
    }
    u64 hash = 0xcbf29ce484222325ull;
    int n = 0, bad_keys = 0, bad_compiles = 0;
    for (const std::string &path : lists) {
        int fetch = 0, fetch_nolog = 0;
        for (const std::string &k : gpu::read_keys(path)) {
            const gpu::DrawState ds = gpu::decode_key(k);
            fetch += ds.need_shade && !gpu::fs_hw_blend(ds, {gpu::ShaderLang::GLSL, true, 1});
            fetch_nolog += ds.need_shade && !gpu::fs_hw_blend(ds, {gpu::ShaderLang::GLSL, false, 1});
            if (!seen.insert(k).second) continue;
            if (gpu::fs_key(ds) != k) bad_keys++;
            gpu::FsOptions fo;
            if (glsl) fo = {gpu::ShaderLang::GLSL, hw_logic, samples};  // (as vulkan.cpp: logic ops in the blender)
            const std::string src = gpu::gen_fs(ds, fo);
            for (unsigned char c : src) hash = (hash ^ c) * 0x100000001b3ull;
            hash = (hash ^ 0xff) * 0x100000001b3ull;  // (shader boundary)
            if (outdir) {
                char name[512];
                snprintf(name, sizeof name, "%s/%05d.%s", outdir, n, glsl ? "frag" : "metal");
                if (FILE *f = fopen(name, "wb")) {
                    fwrite(src.data(), 1, src.size(), f);
                    fclose(f);
                }
            }
#ifdef SHADER_GOLDEN_SHADERC
            if (glsl) {
                shaderc_compilation_result_t r = shaderc_compile_into_spv(sc, src.data(), src.size(),
                                                                          shaderc_fragment_shader, "fs", "main", opt);
                if (shaderc_result_get_compilation_status(r) != shaderc_compilation_status_success) {
                    if (bad_compiles++ < 3)
                        fprintf(stderr, "shader %d does not compile:\n%s\n", n, shaderc_result_get_error_message(r));
                }
                shaderc_result_release(r);
            }
#endif
            n++;
        }
        if (fetch_nolog) fprintf(stderr, "%s: %d blending in the shader (%d without logic ops in the blender)\n", path.c_str(), fetch, fetch_nolog);
    }
    printf("%d shaders (%s), hash %016llx, %d keys not round-tripping, %d not compiling\n", n, glsl ? "GLSL" : "MSL",
           (unsigned long long)hash, bad_keys, bad_compiles);
    return bad_keys || bad_compiles ? 1 : 0;
}
