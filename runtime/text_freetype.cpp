// Text (text.h) without Core Text: shaped with HarfBuzz and rasterised with FreeType, in the sans-serif font
// Fontconfig picks (the desktop's), falling back through the fonts it sorts after it for characters that font lacks
// (Japanese names, symbols). Images match text.mm's: the line's ascent and descent plus a pixel of margin.
#include "text.h"

#include <fontconfig/fontconfig.h>
#include <ft2build.h>
#include FT_FREETYPE_H
#include FT_GLYPH_H
#include FT_STROKER_H
#include <hb-ft.h>
#include <hb.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

#include "rt.h"

namespace text {
namespace {

struct Face {
    FT_Face ft = nullptr;
    hb_font_t *hb = nullptr;
    float size = 0;  // the size hb was last told about
    void set_size(float px) {
        if (size == px) return;
        FT_Set_Char_Size(ft, 0, (FT_F26Dot6)std::lround(px * 64), 72, 72);
        hb_ft_font_changed(hb);
        size = px;
    }
};

struct Fonts {
    FT_Library lib = nullptr;
    std::unique_ptr<Face> regular, bold;
    FcFontSet *fallback = nullptr;                    // after the regular one, in Fontconfig's order
    std::map<int, std::unique_ptr<Face>> fallbacks;   // (opened when a character needs one)
    std::map<int, FcCharSet *> charsets;
};

std::mutex mx;
Fonts *fonts;

std::unique_ptr<Face> open_face(const char *file, int index) {
    FT_Face ft;
    if (FT_New_Face(fonts->lib, file, index, &ft)) return nullptr;
    auto f = std::make_unique<Face>();
    f->ft = ft;
    f->hb = hb_ft_font_create_referenced(ft);
    return f;
}

std::unique_ptr<Face> match(const char *pattern) {
    FcPattern *p = FcNameParse((const FcChar8 *)pattern);
    FcConfigSubstitute(nullptr, p, FcMatchPattern);
    FcDefaultSubstitute(p);
    FcResult res;
    FcPattern *m = FcFontMatch(nullptr, p, &res);
    std::unique_ptr<Face> f;
    FcChar8 *file = nullptr;
    int index = 0;
    if (m && FcPatternGetString(m, FC_FILE, 0, &file) == FcResultMatch) {
        FcPatternGetInteger(m, FC_INDEX, 0, &index);
        f = open_face((const char *)file, index);
    }
    if (m) FcPatternDestroy(m);
    FcPatternDestroy(p);
    return f;
}

Fonts *get_fonts() {
    if (fonts) return fonts->regular ? fonts : nullptr;
    fonts = new Fonts;
    if (FT_Init_FreeType(&fonts->lib) || !FcInit()) {
        LOG("text: no FreeType or Fontconfig, overlays have no text\n");
        return nullptr;
    }
    fonts->regular = match("sans-serif:weight=regular");
    fonts->bold = match("sans-serif:weight=bold");
    if (!fonts->regular) {
        LOG("text: Fontconfig found no sans-serif font, overlays have no text\n");
        return nullptr;
    }
    if (!fonts->bold) fonts->bold = match("sans-serif:weight=regular");
    FcPattern *p = FcNameParse((const FcChar8 *)"sans-serif");
    FcConfigSubstitute(nullptr, p, FcMatchPattern);
    FcDefaultSubstitute(p);
    FcResult res;
    fonts->fallback = FcFontSort(nullptr, p, FcTrue, nullptr, &res);
    FcPatternDestroy(p);
    LOG("text: %s %s\n", fonts->regular->ft->family_name, fonts->regular->ft->style_name);
    return fonts;
}

// the face that has the character: the line's own, else the first fallback that does (null: none)
Face *face_for(Face *own, uint32_t cp) {
    if (FT_Get_Char_Index(own->ft, cp) || cp < 0x20) return own;
    FcFontSet *fs = fonts->fallback;
    for (int i = 0; fs && i < fs->nfont && i < 64; i++) {
        FcCharSet *cs = nullptr;
        auto ci = fonts->charsets.find(i);
        if (ci != fonts->charsets.end()) cs = ci->second;
        else {
            FcPatternGetCharSet(fs->fonts[i], FC_CHARSET, 0, &cs);
            fonts->charsets[i] = cs;
        }
        if (!cs || !FcCharSetHasChar(cs, cp)) continue;
        auto &f = fonts->fallbacks[i];
        if (!f) {
            FcChar8 *file = nullptr;
            int index = 0;
            if (FcPatternGetString(fs->fonts[i], FC_FILE, 0, &file) != FcResultMatch) continue;
            FcPatternGetInteger(fs->fonts[i], FC_INDEX, 0, &index);
            f = open_face((const char *)file, index);
            if (!f) continue;
        }
        return f.get();
    }
    return own;  // (its "missing" box)
}

struct Glyph {
    Face *face;
    unsigned id;
    float x, y;  // pen position, in pixels from the line's start on the baseline
};
struct Line {
    std::vector<Glyph> glyphs;
    float width = 0, ascent = 0, descent = 0;
};

// decodes UTF-8 (invalid bytes as U+FFFD), with each character's byte offset
void decode(const std::string &s, std::vector<uint32_t> &cps, std::vector<uint32_t> &at) {
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = s[i];
        int n = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 0;
        uint32_t cp = n == 1 ? c : n == 2 ? c & 0x1f : n == 3 ? c & 0x0f : c & 0x07;
        bool ok = n > 0 && i + n <= s.size();
        for (int k = 1; ok && k < n; k++) {
            ok = ((unsigned char)s[i + k] >> 6) == 2;
            cp = cp << 6 | (s[i + k] & 0x3f);
        }
        cps.push_back(ok ? cp : 0xFFFD);
        at.push_back((uint32_t)i);
        i += ok ? n : 1;
    }
}

// the line laid out: runs of characters one face has, each shaped by HarfBuzz
Line layout(const std::string &s, float px, bool bold) {
    Line line;
    Face *own = (bold ? fonts->bold : fonts->regular).get();
    own->set_size(px);
    const FT_Size_Metrics &m = own->ft->size->metrics;
    line.ascent = m.ascender / 64.0f;
    line.descent = -m.descender / 64.0f;
    std::vector<uint32_t> cps, at;
    decode(s, cps, at);
    float pen = 0;
    hb_buffer_t *buf = hb_buffer_create();
    for (size_t i = 0; i < cps.size();) {
        Face *f = face_for(own, cps[i]);
        size_t j = i + 1;
        while (j < cps.size() && face_for(own, cps[j]) == f) j++;
        f->set_size(px);
        hb_buffer_reset(buf);
        const uint32_t end = j < cps.size() ? at[j] : (uint32_t)s.size();
        hb_buffer_add_utf8(buf, s.data(), (int)s.size(), (unsigned)at[i], (int)(end - at[i]));
        hb_buffer_guess_segment_properties(buf);
        hb_shape(f->hb, buf, nullptr, 0);
        unsigned n = 0;
        const hb_glyph_info_t *info = hb_buffer_get_glyph_infos(buf, &n);
        const hb_glyph_position_t *pos = hb_buffer_get_glyph_positions(buf, &n);
        for (unsigned k = 0; k < n; k++) {
            line.glyphs.push_back({f, info[k].codepoint, pen + pos[k].x_offset / 64.0f, -pos[k].y_offset / 64.0f});
            pen += pos[k].x_advance / 64.0f;
        }
        if (f != own) {
            const FT_Size_Metrics &fm = f->ft->size->metrics;
            line.ascent = std::max(line.ascent, fm.ascender / 64.0f);
            line.descent = std::max(line.descent, -fm.descender / 64.0f);
        }
        i = j;
    }
    hb_buffer_destroy(buf);
    line.width = pen;
    return line;
}

// premultiplied RGBA (bytes R, G, B, A), drawn over what is there
void blend(std::vector<uint32_t> &img, int w, int h, const FT_Bitmap &bm, int x0, int y0, SDL_Color c) {
    for (unsigned y = 0; y < bm.rows; y++) {
        const int yy = y0 + (int)y;
        if (yy < 0 || yy >= h) continue;
        for (unsigned x = 0; x < bm.width; x++) {
            const int xx = x0 + (int)x;
            if (xx < 0 || xx >= w) continue;
            const unsigned cov = bm.buffer[y * bm.pitch + x];
            if (!cov) continue;
            const unsigned a = c.a * cov / 255;
            uint32_t &d = img[(size_t)yy * w + xx];
            const unsigned inv = 255 - a;
            auto ch = [&](unsigned src, int shift) {
                const unsigned dc = (d >> shift) & 0xff;
                return std::min(255u, src * a / 255 + dc * inv / 255) << shift;
            };
            d = ch(c.r, 0) | ch(c.g, 8) | ch(c.b, 16) | ch(255, 24);
        }
    }
}

// draws the line with its baseline at (ox, oy) in the image (stroke: the outline's radius in pixels, 0 none)
void draw(std::vector<uint32_t> &img, int w, int h, const Line &line, float px, float ox, float oy, SDL_Color c,
          float stroke) {
    FT_Stroker st = nullptr;
    if (stroke > 0) {
        FT_Stroker_New(fonts->lib, &st);
        FT_Stroker_Set(st, (FT_Fixed)std::lround(stroke * 64), FT_STROKER_LINECAP_ROUND, FT_STROKER_LINEJOIN_ROUND, 0);
    }
    for (const Glyph &g : line.glyphs) {
        g.face->set_size(px);
        if (FT_Load_Glyph(g.face->ft, g.id, FT_LOAD_DEFAULT | FT_LOAD_NO_BITMAP)) continue;
        FT_Glyph gl;
        if (FT_Get_Glyph(g.face->ft->glyph, &gl)) continue;
        if (st) FT_Glyph_Stroke(&gl, st, 1);
        const float gx = ox + g.x, gy = oy + g.y;
        FT_Vector origin = {(FT_Pos)std::lround((gx - std::floor(gx)) * 64), 0};
        if (!FT_Glyph_To_Bitmap(&gl, FT_RENDER_MODE_NORMAL, &origin, 1)) {
            const auto *bg = (FT_BitmapGlyph)gl;
            blend(img, w, h, bg->bitmap, (int)std::floor(gx) + bg->left, (int)std::lround(gy) - bg->top, c);
        }
        FT_Done_Glyph(gl);
    }
    if (st) FT_Stroker_Done(st);
}

Texture upload(SDL_Renderer *r, const std::vector<uint32_t> &img, int w, int h) {
    Texture t{SDL_CreateTexture(r, SDL_PIXELFORMAT_ABGR8888, SDL_TEXTUREACCESS_STATIC, w, h), w, h};
    SDL_UpdateTexture(t.tex, nullptr, img.data(), w * 4);
    SDL_SetTextureBlendMode(t.tex, SDL_BLENDMODE_BLEND_PREMULTIPLIED);
    return t;
}

}  // namespace

float width(const std::string &s, float px, bool bold) {
    std::lock_guard<std::mutex> lk(mx);
    if (!get_fonts()) return 2;
    return layout(s, px, bold).width + 2;
}

Texture render(SDL_Renderer *r, const std::string &s, float px, SDL_Color c, bool bold) {
    std::lock_guard<std::mutex> lk(mx);
    if (!get_fonts()) return upload(r, std::vector<uint32_t>(1, 0), 1, 1);
    const Line line = layout(s, px, bold);
    const int w = std::max(1, (int)std::ceil(line.width) + 2), h = std::max(1, (int)std::ceil(line.ascent + line.descent) + 2);
    std::vector<uint32_t> img((size_t)w * h, 0);
    draw(img, w, h, line, px, 1, (float)h - std::ceil(line.descent) - 1, c, 0);
    return upload(r, img, w, h);
}

Texture render_outlined(SDL_Renderer *r, const std::string &s, float px, int pad) {
    std::lock_guard<std::mutex> lk(mx);
    if (!get_fonts()) return upload(r, std::vector<uint32_t>(1, 0), 1, 1);
    const Line line = layout(s, px, true);
    const int w = std::max(1, (int)std::ceil(line.width) + 2 * pad);
    const int h = std::max(1, (int)std::ceil(line.ascent + line.descent) + 2 * pad);
    std::vector<uint32_t> img((size_t)w * h, 0);
    const float base = (float)h - std::ceil(line.descent) - pad;
    draw(img, w, h, line, px, (float)pad, base, {20, 20, 26, 255}, px * 0.11f);
    draw(img, w, h, line, px, (float)pad, base, {255, 255, 255, 255}, 0);
    return upload(r, img, w, h);
}

}  // namespace text
