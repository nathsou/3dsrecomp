// Vertex programs as the GPU backends record and list them (runtime/gpu/vsblob.cpp): a blob is the program
// (entry, code size, swizzle size, code words, swizzle words), optionally followed by a specialization (SPEC, bool
// uniforms, int uniforms) for which the kernel is shader_msl_spec()'s / shader_glsl_spec()'s. Lists
// (games/<title id>/vsprograms.bin, <saves>/shadercache/<title id>.vs.bin): "VSPR", a version, then blobs, each a
// u32 size and its bytes.
#pragma once
#include <string>
#include <vector>

#include "gpu/pica.h"

namespace gpu {
constexpr u32 kVsMagic = 0x52505356;  // "VSPR"
constexpr u32 kVsVersion = 1;
constexpr u32 kVsSpec = 0x43455053;   // "SPEC"
struct VsSpec {
    u32 b = 0;
    u8 iu[4][4] = {};
};
struct VsBlob {
    u32 entry = 0, code_size = 0, swz_size = 0;
    std::vector<u32> code, swz;
    bool spec = false;
    VsSpec sp;
};
u64 vk_hash(const u32 *code, u32 code_size, const u32 *swz, u32 swz_size);  // a program's hash
u64 vk_key(u64 prog_hash, u32 entry);                                        // ... with its entry point
u64 spec_key(u64 key, const VsSpec &sp);                                     // ... and a specialization
bool parse_vs_blob(const std::string &b, VsBlob &v);
std::string vs_blob(const ShaderSetup &s, u32 entry, const VsSpec *sp);
std::vector<std::string> read_blobs(const std::string &path, u32 magic, u32 version);
}  // namespace gpu
