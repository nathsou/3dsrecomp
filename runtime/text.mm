// Text through Core Text (text.h).
#include "text.h"

#import <CoreText/CoreText.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace text {
namespace {

// a line of text in the system font, released by the caller
CTLineRef make_line(const std::string &s, float px, SDL_Color c, bool bold) {
    CTFontRef font = CTFontCreateUIFontForLanguage(bold ? kCTFontUIFontEmphasizedSystem : kCTFontUIFontSystem, px, nullptr);
    CGColorRef col = CGColorCreateSRGB(c.r / 255.0, c.g / 255.0, c.b / 255.0, c.a / 255.0);
    CFStringRef str = CFStringCreateWithCString(nullptr, s.c_str(), kCFStringEncodingUTF8);
    const void *keys[] = {kCTFontAttributeName, kCTForegroundColorAttributeName};
    const void *vals[] = {font, col};
    CFDictionaryRef attrs = CFDictionaryCreate(nullptr, keys, vals, 2, &kCFTypeDictionaryKeyCallBacks,
                                               &kCFTypeDictionaryValueCallBacks);
    CFAttributedStringRef as = CFAttributedStringCreate(nullptr, str ? str : CFSTR("?"), attrs);
    CTLineRef line = CTLineCreateWithAttributedString(as);
    CFRelease(as);
    CFRelease(attrs);
    if (str) CFRelease(str);
    CGColorRelease(col);
    CFRelease(font);
    return line;
}

}  // namespace

float width(const std::string &s, float px, bool bold) {
    CTLineRef line = make_line(s, px, {255, 255, 255, 255}, bold);
    const double w = CTLineGetTypographicBounds(line, nullptr, nullptr, nullptr);
    CFRelease(line);
    return (float)w + 2;
}

Texture render(SDL_Renderer *r, const std::string &s, float px, SDL_Color c, bool bold) {
    CTLineRef line = make_line(s, px, c, bold);
    CGFloat ascent, descent, leading;
    const double width = CTLineGetTypographicBounds(line, &ascent, &descent, &leading);
    const int w = std::max(1, (int)std::ceil(width) + 2), h = std::max(1, (int)std::ceil(ascent + descent) + 2);
    std::vector<uint32_t> px_((size_t)w * h, 0);
    CGColorSpaceRef cs = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
    CGContextRef ctx = CGBitmapContextCreate(px_.data(), w, h, 8, w * 4, cs,
                                             (uint32_t)kCGImageAlphaPremultipliedLast | (uint32_t)kCGBitmapByteOrder32Big);
    CGContextSetTextPosition(ctx, 1, std::ceil(descent) + 1);
    CTLineDraw(line, ctx);
    CGContextFlush(ctx);
    Texture t{SDL_CreateTexture(r, SDL_PIXELFORMAT_ABGR8888, SDL_TEXTUREACCESS_STATIC, w, h), w, h};
    SDL_UpdateTexture(t.tex, nullptr, px_.data(), w * 4);
    SDL_SetTextureBlendMode(t.tex, SDL_BLENDMODE_BLEND_PREMULTIPLIED);
    CGContextRelease(ctx);
    CGColorSpaceRelease(cs);
    CFRelease(line);
    return t;
}

Texture render_outlined(SDL_Renderer *r, const std::string &s, float pt, int pad) {
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
    Texture t{SDL_CreateTexture(r, SDL_PIXELFORMAT_ABGR8888, SDL_TEXTUREACCESS_STATIC, w, h), w, h};
    SDL_UpdateTexture(t.tex, nullptr, buf.data(), w * 4);
    SDL_SetTextureBlendMode(t.tex, SDL_BLENDMODE_BLEND_PREMULTIPLIED);
    CGContextRelease(ctx);
    CGColorSpaceRelease(cs);
    CFRelease(outline);
    CFRelease(fill);
    CGColorRelease(white);
    CGColorRelease(dark);
    if (str) CFRelease(str);
    CFRelease(font);
    return t;
}

}  // namespace text
