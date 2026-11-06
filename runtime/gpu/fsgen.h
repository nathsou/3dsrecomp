// Fragment shader generation for the GPU backends (runtime/gpu/fsgen.cpp).
#pragma once
#include <string>
#include <vector>

#include "gpu/pica.h"

namespace gpu {
// pipeline lists (games/<title id>/pipelines.bin, <saves>/shadercache/<title id>.bin): "PKEY", a version, then
// keys, each a u32 size and its bytes
constexpr u32 kKeyMagic = 0x59454B50;  // "PKEY"
constexpr u32 kKeyVersion = 1;         // bump when fs_key() changes
std::vector<std::string> read_keys(const std::string &path);

extern const char *const kCommonMSL;  // the declarations every Metal fragment shader starts with (VOut, Uni)
// can the PICA blend / logic op be done by the fixed-function blender?
bool hw_blend(const DrawState &ds);
// depth computed per fragment (W-buffering)
bool frag_depth(const DrawState &ds);
// key describing everything the generated fragment shader depends on
std::string fs_key(const DrawState &ds);
// the DrawState fields gen_fs() and the pipelines depend on, from a key (inverse of fs_key)
DrawState decode_key(const std::string &k);
enum class ShaderLang { MSL, GLSL };
struct FsOptions {
    ShaderLang lang = ShaderLang::MSL;
    bool hw_logic_op = false;  // the blender does logic ops (Vulkan's logicOp; Metal has none)
    u32 samples = 1;           // (GLSL) multisampled targets: the destination read per sample
};
// the fragment shader: Metal's fs_main, or (GLSL) a Vulkan fragment shader. Its resources in GLSL (set 0): the
// Uni block at binding 0, the LUT and fog LUT buffers at 1 and 2, textures t0-t2 at 3-5 with samplers s0-s2 at
// 6-8, cube faces f1-f5 at 9-13, and (blending in the shader) the target as input attachment 0 at binding 14.
// Outputs at location 0; inputs at locations 0-5 (color, tc01, tc2w, quat, view, zw).
std::string gen_fs(const DrawState &ds, const FsOptions &opt = {});
// the shader blends with the fixed-function blender (else it reads the destination: programmable blending)
bool fs_hw_blend(const DrawState &ds, const FsOptions &opt);
u64 fnv64(const std::string &s);
std::string fs_name(const std::string &src);  // "fs_<hash of the source>"
}  // namespace gpu
