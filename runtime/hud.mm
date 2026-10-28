// The game profile's HUD: icons decoded from the game's own files (BCLIM textures, in SARC archives that may be
// or DARC, Yaz0-compressed or not; or glyph sheets of BCFNT fonts; MTXT/CTPK .bctex) and numbers drawn with Core Text, placed in top-screen pixels and scaled with the screen.
#include "hud.h"

#import <CoreText/CoreText.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "profile.h"

namespace kernel {
std::vector<u8> romfs_file(const std::string &path);  // (kernel.h clashes with MacTypes.h)
}
namespace gpu {
void texture_decode(const u8 *src, u32 w, u32 h, u32 fmt, u32 *out);
}

namespace hud {
namespace {

struct Tex {
    SDL_Texture *t = nullptr;
    float w = 0, h = 0;  // in top-screen pixels
};

u16 le16(const u8 *p) { return (u16)(p[0] | p[1] << 8); }
u32 le32(const u8 *p) { return p[0] | p[1] << 8 | p[2] << 16 | (u32)p[3] << 24; }

std::vector<u8> yaz0(const std::vector<u8> &in) {
    if (in.size() < 16 || memcmp(in.data(), "Yaz0", 4)) return in;
    const u32 n = (u32)in[4] << 24 | in[5] << 16 | in[6] << 8 | in[7];
    std::vector<u8> out;
    out.reserve(n);
    size_t i = 16;
    while (out.size() < n && i < in.size()) {
        const u8 code = in[i++];
        for (int b = 7; b >= 0 && out.size() < n && i < in.size(); b--) {
            if (code >> b & 1) {
                out.push_back(in[i++]);
                continue;
            }
            if (i + 1 >= in.size()) return {};
            const u32 b1 = in[i++], b2 = in[i++];
            const u32 dist = ((b1 & 15) << 8 | b2) + 1;
            u32 len = b1 >> 4;
            if (!len) {
                if (i >= in.size()) return {};
                len = in[i++] + 0x12;
            } else {
                len += 2;
            }
            if (dist > out.size()) return {};
            for (u32 k = 0; k < len && out.size() < n; k++) out.push_back(out[out.size() - dist]);
        }
    }
    return out;
}

// a file inside a SARC archive (little-endian), empty if it is not there
std::vector<u8> sarc_file(const std::vector<u8> &a, const std::string &name) {
    if (a.size() < 0x20 || memcmp(a.data(), "SARC", 4) || memcmp(a.data() + 0x14, "SFAT", 4)) return {};
    const u32 data = le32(&a[0x0C]);
    const u32 n = le16(&a[0x1A]);
    const size_t nodes = 0x20, names = nodes + n * 16 + 8;
    if (names > a.size()) return {};
    for (u32 i = 0; i < n; i++) {
        const u8 *nd = &a[nodes + i * 16];
        const u32 attr = le32(nd + 4);
        if (!(attr >> 24)) continue;
        const size_t at = names + (attr & 0xffffff) * 4;
        if (at >= a.size() || strncmp((const char *)&a[at], name.c_str(), a.size() - at) != 0) continue;
        const u32 s = data + le32(nd + 8), e = data + le32(nd + 12);
        if (s > e || e > a.size()) return {};
        return {a.begin() + s, a.begin() + e};
    }
    return {};
}

// a file inside a NARC archive by its path ("timg/coin.bclim"), empty if it is not there
std::vector<u8> narc_file(const std::vector<u8> &a, const std::string &name) {
    if (a.size() < 0x1C || memcmp(a.data(), "NARC", 4)) return {};
    size_t at = le16(&a[0x0C]);
    size_t fat = 0, fnt = 0, img = 0;
    for (int i = 0; i < 3 && at + 8 <= a.size(); i++) {
        if (!memcmp(&a[at], "BTAF", 4)) fat = at;
        if (!memcmp(&a[at], "BTNF", 4)) fnt = at;
        if (!memcmp(&a[at], "GMIF", 4)) img = at;
        at += le32(&a[at + 4]);
    }
    if (!fat || !fnt || !img) return {};
    const u32 nfiles = le16(&a[fat + 8]);
    const size_t dirs = fnt + 8;
    // walk a directory's name table: files are numbered from the directory's first id, in order
    std::function<int(u32, const std::string &)> walk = [&](u32 dir, const std::string &prefix) -> int {
        const size_t de = dirs + dir * 8;
        if (de + 8 > a.size()) return -1;
        size_t p = dirs + le32(&a[de]);
        u32 id = le16(&a[de + 4]);
        while (p < a.size() && a[p]) {
            const u8 len = a[p] & 0x7f;
            const bool sub = a[p] & 0x80;
            if (p + 1 + len > a.size()) return -1;
            const std::string n = prefix + std::string((const char *)&a[p + 1], len);
            p += 1 + len;
            if (sub) {
                const u32 sd = le16(&a[p]) & 0xfff;
                p += 2;
                if (int r = walk(sd, n + "/"); r >= 0) return r;
            } else {
                if (n == name) return (int)id;
                id++;
            }
        }
        return -1;
    };
    const int id = walk(0, "");
    if (id < 0 || (u32)id >= nfiles) return {};
    const size_t e = fat + 12 + id * 8, base = img + 8;
    const u32 s0 = le32(&a[e]), s1 = le32(&a[e + 4]);
    if (s0 > s1 || base + s1 > a.size()) return {};
    return {a.begin() + base + s0, a.begin() + base + s1};
}

// a file inside a DARC archive (NSMB2's layouts) by its path ("timg/x.bclim"), empty if it is not there
std::vector<u8> darc_file(const std::vector<u8> &a, const std::string &name) {
    if (a.size() < 0x1C || memcmp(a.data(), "darc", 4)) return {};
    const u32 table = le32(&a[0x10]);
    if (table + 12 > a.size()) return {};
    const u32 n = le32(&a[table + 8]);  // the root entry's end index: the number of entries
    const size_t names = table + (size_t)n * 12;
    if (names > a.size()) return {};
    auto entry_name = [&](u32 off) {
        std::string s;
        for (size_t p = names + off; p + 1 < a.size() && (a[p] || a[p + 1]); p += 2) s += (char)a[p];
        return s;
    };
    std::vector<std::pair<std::string, u32>> dirs;  // open directories: path so far, end index
    for (u32 i = 1; i < n; i++) {
        while (!dirs.empty() && i >= dirs.back().second) dirs.pop_back();
        const u8 *e = &a[table + i * 12];
        const u32 no = le32(e);
        const std::string nm = entry_name(no & 0xffffff), at = dirs.empty() ? "" : dirs.back().first;
        if (no >> 24 & 1) {
            dirs.push_back({nm == "." || nm.empty() ? at : at + nm + "/", le32(e + 8)});
        } else if (at + nm == name) {
            const u32 off = le32(e + 4), size = le32(e + 8);
            if ((u64)off + size > a.size()) return {};
            return {a.begin() + off, a.begin() + off + size};
        }
    }
    return {};
}

std::vector<u8> archive_file(const std::vector<u8> &a, const std::string &name) {
    if (a.size() >= 4 && !memcmp(a.data(), "NARC", 4)) return narc_file(a, name);
    if (a.size() >= 4 && !memcmp(a.data(), "darc", 4)) return darc_file(a, name);
    return sarc_file(a, name);
}

// a texture's pixels as top-down rows: layout textures and font sheets store their top row first, while
// texture_decode (made for 3D textures) puts the first stored row at the bottom
void decode_top_down(const u8 *src, int w, int h, int fmt, u32 *out) {
    std::vector<u32> t((size_t)w * h);
    gpu::texture_decode(src, w, h, fmt, t.data());
    for (int y = 0; y < h; y++) memcpy(&out[(size_t)y * w], &t[(size_t)(h - 1 - y) * w], w * 4);
}

// a BCLIM texture as top-down RGBA rows (R in the low byte)
bool bclim(const std::vector<u8> &f, std::vector<u32> &px, int &w, int &h) {
    if (f.size() < 0x28) return false;
    const u8 *ft = &f[f.size() - 0x28];
    if (memcmp(ft, "CLIM", 4) || memcmp(ft + 0x14, "imag", 4)) return false;
    w = le16(ft + 0x1C);
    h = le16(ft + 0x1E);
    const u32 cf = le32(ft + 0x20) & 0xff;
    // BCLIM format numbers -> PICA texture formats
    static const int pica[] = {7, 8, 9, 5, 6, 3, 1, 2, 4, 0, 12, 13, 10, 11};
    static const int bits[] = {32, 24, 16, 16, 16, 16, 16, 8, 8, 8, 4, 4, 4, 8};  // per PICA format
    if (cf >= 14 || !w || !h) return false;
    const int fmt = pica[cf];
    auto pow2 = [](int v) {
        int p = 8;
        while (p < v) p <<= 1;
        return p;
    };
    const int pw = pow2(w), ph = pow2(h);
    if ((size_t)pw * ph * bits[fmt] / 8 > f.size() - 0x28) return false;
    std::vector<u32> full((size_t)pw * ph);
    decode_top_down(f.data(), pw, ph, fmt, full.data());
    px.resize((size_t)w * h);  // (stored padded to powers of two)
    for (int y = 0; y < h; y++) memcpy(&px[(size_t)y * w], &full[(size_t)y * pw], w * 4);
    return true;
}

// a BCFNT font's glyph sheets, stacked top to bottom into one image (icons are rects in it)
bool bcfnt(const std::vector<u8> &f, std::vector<u32> &px, int &w, int &h) {
    if (f.size() < 0x30 || memcmp(f.data(), "CFNT", 4)) return false;
    const u32 finf = le16(&f[6]);  // the FINF block follows the header
    if (finf + 0x18 > f.size() || memcmp(&f[finf], "FINF", 4)) return false;
    const u32 tglp = le32(&f[finf + 0x10]) - 8;  // (offsets point past block headers)
    if (tglp + 0x20 > f.size() || memcmp(&f[tglp], "TGLP", 4)) return false;
    const u32 sheet_size = le32(&f[tglp + 0x0C]);
    const u32 sheets = le16(&f[tglp + 0x10]), fmt = le16(&f[tglp + 0x12]) & 0x7fff;
    const int sw = le16(&f[tglp + 0x18]), sh = le16(&f[tglp + 0x1A]);
    const u32 data = le32(&f[tglp + 0x1C]);
    if (fmt >= 14 || !sheets || !sw || !sh || data + (u64)sheets * sheet_size > f.size()) return false;
    w = sw;
    h = sh * (int)sheets;
    px.resize((size_t)w * h);
    for (u32 i = 0; i < sheets; i++) decode_top_down(&f[data + i * sheet_size], sw, sh, fmt, &px[(size_t)i * sw * sh]);
    return true;
}

// a Metroid: Samus Returns .bctex: an MTXT header around a CTPK package; its first texture's level 0
bool bctex(const std::vector<u8> &f, std::vector<u32> &px, int &w, int &h) {
    if (f.size() < 0x80 || memcmp(f.data(), "MTXT", 4)) return false;
    const size_t c = le32(&f[0x1C]);
    if (c + 0x40 > f.size() || memcmp(&f[c], "CTPK", 4) || !le16(&f[c + 6])) return false;
    const u8 *e = &f[c + 0x20];
    const size_t data = c + le32(&f[c + 8]) + le32(e + 8);
    const u32 fmt = le32(e + 0x0C);
    w = le16(e + 0x10);
    h = le16(e + 0x12);
    static const int bits[] = {32, 24, 16, 16, 16, 16, 16, 8, 8, 8, 4, 4, 4, 8};
    if (fmt >= 14 || w < 8 || h < 8 || (w & 7) || (h & 7) || data + (size_t)w * h * bits[fmt] / 8 > f.size()) return false;
    px.resize((size_t)w * h);
    decode_top_down(&f[data], w, h, fmt, px.data());
    return true;
}

std::map<std::string, Tex> icon_cache;

const Tex *icon(SDL_Renderer *r, const std::string &name) {
    auto it = icon_cache.find(name);
    if (it != icon_cache.end()) return it->second.t ? &it->second : nullptr;
    Tex &t = icon_cache[name];
    const profile::Icon *ic = nullptr;
    for (auto &i : profile::icons())
        if (i.name == name) ic = &i;
    if (!ic) {
        LOG("[hud] no icon named %s\n", name.c_str());
        return nullptr;
    }
    std::vector<u8> f = yaz0(kernel::romfs_file(ic->file));
    if (!ic->inner.empty()) f = archive_file(f, ic->inner);
    std::vector<u32> px;
    int w, h;
    f = yaz0(f);
    if (f.empty() || !(bclim(f, px, w, h) || bcfnt(f, px, w, h) || bctex(f, px, w, h))) {
        LOG("[hud] icon %s: cannot read %s%s%s\n", name.c_str(), ic->file.c_str(), ic->inner.empty() ? "" : ":",
            ic->inner.c_str());
        return nullptr;
    }
    int x0 = 0, y0 = 0, cw = w, ch = h;
    if (ic->rw > 0 && ic->rh > 0) {
        x0 = std::clamp(ic->rx, 0, w - 1);
        y0 = std::clamp(ic->ry, 0, h - 1);
        cw = std::min(ic->rw, w - x0);
        ch = std::min(ic->rh, h - y0);
    }
    std::vector<u32> part((size_t)cw * ch);
    for (int y = 0; y < ch; y++) memcpy(&part[(size_t)y * cw], &px[(size_t)(y0 + y) * w + x0], cw * 4);
    t.t = SDL_CreateTexture(r, SDL_PIXELFORMAT_ABGR8888, SDL_TEXTUREACCESS_STATIC, cw, ch);
    SDL_UpdateTexture(t.t, nullptr, part.data(), cw * 4);
    SDL_SetTextureBlendMode(t.t, SDL_BLENDMODE_BLEND);
    SDL_SetTextureScaleMode(t.t, SDL_SCALEMODE_LINEAR);
    t.w = cw * ic->scale;
    t.h = ch * ic->scale;
    return &t;
}

// white digits with a dark outline, rasterised at the size they are shown (px: render pixels per top-screen pixel)
std::map<std::string, Tex> text_cache;

const Tex &text(SDL_Renderer *r, const std::string &s, float size, float px) {
    char key[48];
    snprintf(key, sizeof key, "|%.1f|%.2f", size, px);
    auto it = text_cache.find(s + key);
    if (it != text_cache.end()) return it->second;
    if (text_cache.size() > 200) {
        for (auto &[k, t] : text_cache) SDL_DestroyTexture(t.t);
        text_cache.clear();
    }
    const float pt = size * px;
    CTFontRef font = CTFontCreateUIFontForLanguage(kCTFontUIFontEmphasizedSystem, pt, nullptr);
    CFStringRef str = CFStringCreateWithCString(nullptr, s.c_str(), kCFStringEncodingUTF8);
    auto make_line = [&](CGColorRef col, float stroke) {
        CFNumberRef sw = CFNumberCreate(nullptr, kCFNumberFloatType, &stroke);
        const void *keys[] = {kCTFontAttributeName, kCTForegroundColorAttributeName, kCTStrokeColorAttributeName,
                              kCTStrokeWidthAttributeName};
        const void *vals[] = {font, col, col, sw};
        CFDictionaryRef attrs = CFDictionaryCreate(nullptr, keys, vals, 4, &kCFTypeDictionaryKeyCallBacks,
                                                   &kCFTypeDictionaryValueCallBacks);
        CFAttributedStringRef as = CFAttributedStringCreate(nullptr, str ? str : CFSTR("?"), attrs);
        CTLineRef line = CTLineCreateWithAttributedString(as);
        CFRelease(as);
        CFRelease(attrs);
        CFRelease(sw);
        return line;
    };
    CGColorRef white = CGColorCreateSRGB(1, 1, 1, 1), dark = CGColorCreateSRGB(0.08, 0.08, 0.1, 1);
    CTLineRef outline = make_line(dark, 22.0f), fill = make_line(white, 0.0f);
    CGFloat ascent, descent, leading;
    const double width = CTLineGetTypographicBounds(outline, &ascent, &descent, &leading);
    const int pad = (int)std::ceil(pt * 0.12f) + 1;
    const int w = std::max(1, (int)std::ceil(width) + 2 * pad), h = std::max(1, (int)std::ceil(ascent + descent) + 2 * pad);
    std::vector<uint32_t> buf((size_t)w * h, 0);
    CGColorSpaceRef cs = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
    CGContextRef ctx = CGBitmapContextCreate(buf.data(), w, h, 8, w * 4, cs,
                                             (uint32_t)kCGImageAlphaPremultipliedLast | (uint32_t)kCGBitmapByteOrder32Big);
    CGContextSetTextPosition(ctx, pad, std::ceil(descent) + pad);
    CTLineDraw(outline, ctx);
    CGContextSetTextPosition(ctx, pad, std::ceil(descent) + pad);
    CTLineDraw(fill, ctx);
    CGContextFlush(ctx);
    Tex t;
    t.t = SDL_CreateTexture(r, SDL_PIXELFORMAT_ABGR8888, SDL_TEXTUREACCESS_STATIC, w, h);
    SDL_UpdateTexture(t.t, nullptr, buf.data(), w * 4);
    SDL_SetTextureBlendMode(t.t, SDL_BLENDMODE_BLEND_PREMULTIPLIED);
    t.w = w / px;
    t.h = h / px;
    CGContextRelease(ctx);
    CGColorSpaceRelease(cs);
    CFRelease(outline);
    CFRelease(fill);
    CGColorRelease(white);
    CGColorRelease(dark);
    if (str) CFRelease(str);
    CFRelease(font);
    return text_cache[s + key] = t;
}

}  // namespace

void draw(SDL_Renderer *r, const SDL_FRect &top) {
    if (!profile::has_hud() || !profile::hud_visible() || top.w <= 0) return;
    const float k = top.w / 400.0f;  // render units per top-screen pixel
    // render pixels per render unit (Retina, logical presentation), so text is rasterised sharp
    float sx = 1, sy = 1;
    SDL_GetRenderScale(r, &sx, &sy);
    int ow = 0, oh = 0, lw = 0, lh = 0;
    SDL_GetCurrentRenderOutputSize(r, &ow, &oh);
    SDL_RendererLogicalPresentation mode;
    if (SDL_GetRenderLogicalPresentation(r, &lw, &lh, &mode) && lw > 0) sx *= std::min((float)ow / lw, (float)oh / lh);
    const float px = k * sx;
    auto put = [&](const Tex &t, float x, float y) {
        const SDL_FRect d{top.x + x * k, top.y + y * k, t.w * k, t.h * k};
        SDL_RenderTexture(r, t.t, nullptr, &d);
    };
    for (const auto &e : profile::elements()) {
        if (!profile::cond(e.cond)) continue;
        // lay the element out at x = 0, then place it (a negative x / y counts from the right / bottom edge)
        std::vector<std::pair<const Tex *, float>> parts;  // texture, x offset
        float w = 0, h = 0;
        auto add = [&](const Tex *t, float gap) {
            if (!t) return;
            if (!parts.empty()) w += gap;
            parts.push_back({t, w});
            w += t->w;
            h = std::max(h, t->h);
        };
        s64 v = 0;
        const bool have = profile::value(e.value, v);
        switch (e.kind) {
        case profile::Element::ICON: add(icon(r, e.icon), 0); break;
        case profile::Element::COUNTER: {
            if (!have) continue;
            const Tex *ic = icon(r, e.icon);
            add(ic, 0);
            char num[32];
            snprintf(num, sizeof num, "×%0*lld", e.digits, (long long)v);
            add(&text(r, num, ic ? std::clamp(ic->h * 0.8f, 12.0f, 18.0f) : 13.0f, px), 1);  // (even sizes across icons)
            break;
        }
        case profile::Element::PIPS:
            if (!have) continue;
            for (int i = 0; i < e.count; i++) add(icon(r, (v >> i & 1) ? e.icon : e.icon_off), 2);
            break;
        case profile::Element::PICK:
            if (!have) continue;
            for (auto &[n, name] : e.picks)
                if (n == v) add(icon(r, name), 0);
            break;
        }
        if (parts.empty()) continue;
        const float x = e.x >= 0 ? e.x : 400 + e.x - w, y = e.y >= 0 ? e.y : 240 + e.y - h;
        for (auto &[t, dx] : parts) put(*t, x + dx, y + (h - t->h) / 2);
    }
}

}  // namespace hud
