// Tiny PNG writer (zlib stored blocks, no compression).
#pragma once
#include <cstdint>

// pixels: packed RGBA (r | g<<8 | b<<16 | a<<24), row-major, top row first
bool write_png(const char *path, uint32_t w, uint32_t h, const uint32_t *pixels);
