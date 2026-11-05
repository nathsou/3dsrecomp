// shader_golden [--glsl [--samples <n>]] [-o <dir>] <pipeline list>...
//
// Generates the fragment shader of every key in the pipeline lists (games/*/pipelines.bin) and prints how many
// and one hash of them all, so a change to the generator can be checked against an older build's output (the
// MSL must stay the same byte for byte while only another language is added). -o writes each shader as
// <dir>/<n>.<metal|frag>. Every key must also survive decode_key/fs_key unchanged. With --glsl the Vulkan
// backend's GLSL is generated instead, and each one compiled to SPIR-V with shaderc (when built with it).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>

#include "gpu/fsgen.h"
#include "rt.h"

#ifdef SHADER_GOLDEN_SHADERC
#include <shaderc/shaderc.h>
#endif

int main(int argc, char **argv) {
    bool glsl = false;
    unsigned samples = 1;
    const char *outdir = nullptr;
    std::vector<std::string> lists;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--glsl")) glsl = true;
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
    u64 hash = 0xcbf29ce484222325ull;
    int n = 0, bad_keys = 0, bad_compiles = 0;
    for (const std::string &path : lists) {
        for (const std::string &k : gpu::read_keys(path)) {
            if (!seen.insert(k).second) continue;
            const gpu::DrawState ds = gpu::decode_key(k);
            if (gpu::fs_key(ds) != k) bad_keys++;
            gpu::FsOptions fo;
            if (glsl) fo = {gpu::ShaderLang::GLSL, true, samples};  // (as vulkan.cpp: logic ops in the blender)
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
    }
    printf("%d shaders (%s), hash %016llx, %d keys not round-tripping, %d not compiling\n", n, glsl ? "GLSL" : "MSL",
           (unsigned long long)hash, bad_keys, bad_compiles);
    return bad_keys || bad_compiles ? 1 : 0;
}
