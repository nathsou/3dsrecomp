// Loads an extracted game (exheader + code.bin) into guest memory and sets up the kernel pages.
#include "loader.h"

#include <ctime>
#include <fstream>
#include <vector>

#include "kernel.h"
#include "settings.h"

static std::vector<u8> read_file(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) FATAL("cannot open %s\n", path.c_str());
    return std::vector<u8>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

GameInfo g_game;

static void init_config_mem() {
    u32 c = mem::CONFIG_MEM;
    memset(gptr(c), 0, 0x1000);
    w8(c + 0x01, 0);       // kernel version revision
    w8(c + 0x02, 0x34);    // kernel version minor
    w8(c + 0x03, 0x02);    // kernel version major
    w64(c + 0x08, 0x0004013000008002ull);  // NS title id
    w32(c + 0x10, 2);      // sys core version
    w8(c + 0x14, 1);       // unit info: retail
    w8(c + 0x16, 1);       // prev firm
    w32(c + 0x18, 0x0000F297);
    w32(c + 0x30, 0);      // app memory type: 64 MB
    w32(c + 0x40, 0x04000000);  // app memory
    w32(c + 0x44, 0x02C00000);  // system memory
    w32(c + 0x48, 0x01400000);  // base memory
    w8(c + 0x61, 0);
    w8(c + 0x62, 0x34);
    w8(c + 0x63, 0x02);
    w32(c + 0x64, 2);
    w32(c + 0x68, 0x0000F297);
}

void update_shared_page_time() {
    u32 sp = mem::SHARED_PAGE;
    // milliseconds since 1900-01-01
    time_t now = time(nullptr) + (time_t)g_settings.clock_offset;  // the menu's Clock page
    struct tm lt;
    localtime_r(&now, &lt);
    s64 local = (s64)now + lt.tm_gmtoff;
    u64 ms = (u64)(local + 2208988800ll) * 1000ull;
    u32 counter = r32(sp);
    u32 which = (counter + 1) & 1;
    u32 dt = sp + 0x20 + which * 0x20;
    w64(dt + 0, ms);
    w64(dt + 8, kernel::system_tick());
    w64(dt + 16, 268111856ull);
    w64(dt + 24, 0);
    w32(sp, counter + 1);
}

static void init_shared_page() {
    u32 sp = mem::SHARED_PAGE;
    memset(gptr(sp), 0, 0x1000);
    w8(sp + 0x04, 1);  // running hw: product
    w8(sp + 0x05, 0);  // mcu hw info
    update_shared_page_time();
    u8 mac[6] = {0x40, 0xF4, 0x07, 0x12, 0x34, 0x56};
    memcpy(gptr(sp + 0x60), mac, 6);
    w8(sp + 0x66, 0);   // wifi link level
    w8(sp + 0x67, 7);   // network state: not connected
    float slider = 0.0f;
    memcpy(gptr(sp + 0x80), &slider, 4);
    w8(sp + 0x84, 0);
    w8(sp + 0x85, 0x1E);  // battery: charging, full, adapter
    w64(sp + 0xA0, 0x0004003000008F02ull);  // home menu
}

// The English (else Japanese) short title of an SMDH icon file, as UTF-8; "" without one.
static std::string smdh_title(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    std::vector<u8> d((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (d.size() < 0x2040 || memcmp(d.data(), "SMDH", 4) != 0) return "";
    for (int lang : {1, 0}) {
        std::string s;
        for (u32 o = 8 + lang * 0x200; o < 8 + lang * 0x200 + 0x80; o += 2) {
            u32 c = d[o] | (d[o + 1] << 8);
            if (!c) break;
            if (c < 0x80) s += (char)c;
            else if (c < 0x800) s += {(char)(0xC0 | c >> 6), (char)(0x80 | (c & 0x3F))};
            else s += {(char)(0xE0 | c >> 12), (char)(0x80 | ((c >> 6) & 0x3F)), (char)(0x80 | (c & 0x3F))};
        }
        if (!s.empty()) return s;
    }
    return "";
}

// The region a game is for, from its SMDH region lockout bits (the lowest one: EUR for EUR+AUS); -1 if region free.
static int smdh_region(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    std::vector<u8> d((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (d.size() < 0x2040 || memcmp(d.data(), "SMDH", 4) != 0) return -1;
    u32 bits = d[0x2018] | d[0x2019] << 8 | d[0x201A] << 16 | (u32)d[0x201B] << 24;
    if ((bits & 0x7F) == 0 || (bits & 0x7F) == 0x7F) return -1;
    return __builtin_ctz(bits & 0x7F);
}

void load_game(const std::string &dir) {
    g_game.dir = dir;
    auto ex = read_file(dir + "/exheader.bin");
    auto code = read_file(dir + "/code.bin");
    auto rd = [&](size_t o) { u32 v; memcpy(&v, &ex[o], 4); return v; };
    g_game.title = std::string((const char *)&ex[0], strnlen((const char *)&ex[0], 8));
    g_game.name = smdh_title(dir + "/exefs/icon.bin");
    if (g_game.name.empty()) g_game.name = g_game.title;
    g_game.region = smdh_region(dir + "/exefs/icon.bin");
    g_game.text_addr = rd(0x10);
    g_game.text_size = rd(0x18);
    g_game.ro_addr = rd(0x20);
    g_game.ro_size = rd(0x28);
    g_game.data_addr = rd(0x30);
    g_game.data_size = rd(0x38);
    g_game.stack_size = rd(0x1C);
    g_game.bss_size = rd(0x3C);
    memcpy(&g_game.program_id, &ex[0x200], 8);
    g_game.priority = ex[0x20F];
    u32 base = g_game.text_addr;
    // code.bin is the three segments at their page-aligned offsets
    auto copy = [&](u32 addr, u32 size) {
        u32 off = addr - base;
        if (off + size > code.size()) size = (u32)code.size() - off;
        memcpy(gptr(addr), &code[off], size);
    };
    copy(g_game.text_addr, g_game.text_size);
    copy(g_game.ro_addr, g_game.ro_size);
    copy(g_game.data_addr, g_game.data_size);
    memset(gptr(g_game.data_addr + g_game.data_size), 0, g_game.bss_size);
    init_config_mem();
    init_shared_page();
    LOG("[loader] %s (%016llx): text %08x+%x ro %08x+%x data %08x+%x bss %x stack %x prio %d\n", g_game.title.c_str(),
        (unsigned long long)g_game.program_id, g_game.text_addr, g_game.text_size, g_game.ro_addr, g_game.ro_size,
        g_game.data_addr, g_game.data_size, g_game.bss_size, g_game.stack_size, g_game.priority);
}
