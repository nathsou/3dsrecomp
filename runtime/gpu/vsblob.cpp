// Vertex program blobs and lists (vsblob.h), for the GPU backends' vertex kernels.
#include "gpu/vsblob.h"

#include <cstdio>

namespace gpu {

u64 vk_hash(const u32 *code, u32 code_size, const u32 *swz, u32 swz_size) {
    return hash_bytes((const u8 *)code, code_size * 4) * 31 + hash_bytes((const u8 *)swz, swz_size * 4) + code_size;
}
u64 vk_key(u64 prog_hash, u32 entry) { return prog_hash * 0x9E3779B97F4A7C15ull + entry; }

u64 spec_key(u64 key, const VsSpec &sp) { return (key ^ hash_bytes((const u8 *)&sp, sizeof sp)) * 0x100000001B3ull + 1; }
bool parse_vs_blob(const std::string &b, VsBlob &v) {
    if (b.size() < 12) return false;
    memcpy(&v.entry, b.data(), 4);
    memcpy(&v.code_size, b.data() + 4, 4);
    memcpy(&v.swz_size, b.data() + 8, 4);
    if (v.code_size > 4096 || v.swz_size > 4096) return false;
    const size_t base = 12 + 4 * (size_t)(v.code_size + v.swz_size);
    if (b.size() == base + 4 + sizeof(VsSpec)) {
        u32 m;
        memcpy(&m, b.data() + base, 4);
        if (m != kVsSpec) return false;
        v.spec = true;
        memcpy(&v.sp, b.data() + base + 4, sizeof(VsSpec));
    } else if (b.size() != base) {
        return false;
    }
    v.code.resize(4096);
    v.swz.resize(4096);
    memcpy(v.code.data(), b.data() + 12, 4 * v.code_size);
    memcpy(v.swz.data(), b.data() + 12 + 4 * v.code_size, 4 * v.swz_size);
    return true;
}
std::string vs_blob(const ShaderSetup &s, u32 entry, const VsSpec *sp) {
    const size_t base = 12 + 4 * (size_t)(s.code_size + s.swizzle_size);
    std::string b(base + (sp ? 4 + sizeof(VsSpec) : 0), '\0');
    const u32 hdr[3] = {entry, s.code_size, s.swizzle_size};
    memcpy(&b[0], hdr, 12);
    memcpy(&b[12], s.code, 4 * s.code_size);
    memcpy(&b[12 + 4 * s.code_size], s.swizzle, 4 * s.swizzle_size);
    if (sp) {
        memcpy(&b[base], &kVsSpec, 4);
        memcpy(&b[base + 4], sp, sizeof(VsSpec));
    }
    return b;
}

std::vector<std::string> read_blobs(const std::string &path, u32 magic, u32 version) {
    std::vector<std::string> out;
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) return out;
    u32 hdr[2] = {};
    if (fread(hdr, 4, 2, f) == 2 && hdr[0] == magic && hdr[1] == version) {
        u32 n;
        while (fread(&n, 4, 1, f) == 1 && n > 0 && n < (1u << 16)) {
            std::string k(n, '\0');
            if (fread(&k[0], 1, n, f) != n) break;
            out.push_back(std::move(k));
        }
    }
    fclose(f);
    return out;
}

}  // namespace gpu
