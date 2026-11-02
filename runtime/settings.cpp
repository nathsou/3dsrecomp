// Per-game settings file and save backups (see settings.h).
#include "settings.h"

#include <sys/stat.h>

#include <algorithm>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <map>

#include "cloud.h"
#include "kernel.h"
#include "profile.h"

namespace fs = std::filesystem;

Settings g_settings;

static std::string settings_path() { return kernel::save_root() + "/settings.ini"; }

void settings_load() {
    std::ifstream f(settings_path());
    std::map<std::string, std::string> kv;
    for (std::string line; std::getline(f, line);) {
        size_t eq = line.find('=');
        if (eq != std::string::npos && line[0] != '#') kv[line.substr(0, eq)] = line.substr(eq + 1);
    }
    auto num = [&](const char *k, auto &v) {
        auto it = kv.find(k);
        if (it != kv.end()) v = (std::remove_reference_t<decltype(v)>)strtoll(it->second.c_str(), nullptr, 10);
    };
    Settings &s = g_settings;
    num("scale", s.scale);
    num("msaa", s.msaa);
    num("aniso", s.aniso);
    num("layout", s.layout);
    num("fullscreen", s.fullscreen);
    num("split", s.split);
    num("fullscreen2", s.fullscreen2);
    auto geom = [&](const char *k, int *g) {
        auto it = kv.find(k);
        if (it != kv.end()) sscanf(it->second.c_str(), "%d,%d,%d,%d", &g[0], &g[1], &g[2], &g[3]);
    };
    geom("window", s.geom);
    geom("window2", s.geom2);
    num("speed", s.speed);
    num("volume", s.volume);
    num("show_fps", s.show_fps);
    num("clock_offset", s.clock_offset);
    num("auto_backup", s.auto_backup);
    num("forgive_unsaved", s.forgive_unsaved);
    num("pip", s.pip);
    num("pip_corner", s.pip_corner);
    num("pip_x", s.pip_x);
    num("pip_y", s.pip_y);
    num("pip_size", s.pip_size);
    num("auto_focus", s.auto_focus);
    num("hud", s.hud);
    num("tex_filter", s.tex_filter);
    s.scale = std::clamp(s.scale, 0, 4);
    s.layout = std::max(s.layout, 0);  // (the frontend clamps it to its layouts)
    s.speed = std::clamp(s.speed, 25, 400);
    s.volume = std::clamp(s.volume, 0, 100);
    s.pip = std::clamp(s.pip, 0, 2);
    s.pip_corner = std::clamp(s.pip_corner, 0, 3);
    s.pip_size = std::clamp(s.pip_size, 15, 80);
    s.hud = std::clamp(s.hud, 0, 2);
    // the GPU backend reads these once at startup; an environment variable given at launch wins
    setenv("GPU_MSAA", std::to_string(s.msaa).c_str(), 0);
    setenv("GPU_ANISO", std::to_string(s.aniso).c_str(), 0);
}

void settings_save() {
    const Settings &s = g_settings;
    std::ofstream f(settings_path());
    f << "# written by the in-game menu (Esc)\n"
      << "scale=" << s.scale << "\nmsaa=" << s.msaa << "\naniso=" << s.aniso << "\nlayout=" << s.layout
      << "\nfullscreen=" << s.fullscreen << "\nsplit=" << s.split << "\nfullscreen2=" << s.fullscreen2
      << "\nwindow=" << s.geom[0] << ',' << s.geom[1] << ',' << s.geom[2] << ',' << s.geom[3]
      << "\nwindow2=" << s.geom2[0] << ',' << s.geom2[1] << ',' << s.geom2[2] << ',' << s.geom2[3] << "\nspeed=" << s.speed << "\nvolume=" << s.volume
      << "\nshow_fps=" << s.show_fps << "\nclock_offset=" << s.clock_offset << "\nauto_backup=" << s.auto_backup
      << "\nforgive_unsaved=" << s.forgive_unsaved
      << "\npip=" << s.pip << "\npip_corner=" << s.pip_corner << "\npip_x=" << s.pip_x << "\npip_y=" << s.pip_y
      << "\npip_size=" << s.pip_size << "\nauto_focus=" << s.auto_focus << "\nhud=" << s.hud
      << "\ntex_filter=" << s.tex_filter
      << "\n";
}

// ---- saves
static const char *kParts[] = {"savedata", "extdata"};

static u64 dir_bytes(const fs::path &p) {
    u64 n = 0;
    std::error_code ec;
    for (auto it = fs::recursive_directory_iterator(p, ec); it != fs::recursive_directory_iterator(); it.increment(ec))
        if (it->is_regular_file(ec)) n += it->file_size(ec);
    return n;
}

static std::string backups_dir() { return kernel::save_root() + "/backups"; }

bool save_exists() {
    std::error_code ec;
    return dir_bytes(kernel::save_root() + "/savedata") > 0 || fs::exists(kernel::save_root() + "/extdata", ec);
}

std::string save_summary() {
    const fs::path p = kernel::save_root() + "/savedata";
    std::error_code ec;
    if (!save_exists()) return "No save yet";
    time_t newest = 0;
    for (auto it = fs::recursive_directory_iterator(p, ec); it != fs::recursive_directory_iterator(); it.increment(ec)) {
        struct stat st;
        if (it->is_regular_file(ec) && stat(it->path().c_str(), &st) == 0) newest = std::max(newest, st.st_mtime);
    }
    u64 bytes = dir_bytes(p) + dir_bytes(kernel::save_root() + "/extdata");
    char when[64] = "";
    if (newest) strftime(when, sizeof when, ", last written %d %b %H:%M", localtime(&newest));
    return std::to_string((bytes + 1023) / 1024) + " KB" + when;
}

std::vector<SaveBackup> save_backups() {
    std::vector<SaveBackup> v;
    std::error_code ec;
    for (auto &e : fs::directory_iterator(backups_dir(), ec))
        if (e.is_directory(ec)) v.push_back({e.path().filename().string(), e.path().string(), dir_bytes(e.path())});
    std::sort(v.begin(), v.end(), [](auto &a, auto &b) { return a.name > b.name; });
    return v;
}

bool save_backup_now(const std::string &suffix) {
    if (!save_exists()) return false;
    char ts[64];
    time_t now = time(nullptr);
    strftime(ts, sizeof ts, "%Y-%m-%d %H.%M.%S", localtime(&now));
    fs::path dst = fs::path(backups_dir()) / (std::string(ts) + suffix);
    std::error_code ec;
    fs::create_directories(dst, ec);
    for (const char *part : kParts) {
        fs::path src = fs::path(kernel::save_root()) / part;
        if (fs::exists(src, ec)) fs::copy(src, dst / part, fs::copy_options::recursive, ec);
        if (ec) {
            LOG("[saves] backup failed: %s\n", ec.message().c_str());
            return false;
        }
    }
    LOG("[saves] backed up to %s\n", dst.c_str());
    return true;
}

static bool remove_current() {
    std::error_code ec;
    for (const char *part : kParts) fs::remove_all(fs::path(kernel::save_root()) / part, ec);
    return !ec;
}

bool save_restore(const SaveBackup &b) {
    save_backup_now(" before restore");
    if (!remove_current()) return false;
    std::error_code ec;
    for (const char *part : kParts) {
        fs::path src = fs::path(b.path) / part;
        if (fs::exists(src, ec)) fs::copy(src, fs::path(kernel::save_root()) / part, fs::copy_options::recursive, ec);
    }
    LOG("[saves] restored %s%s\n", b.name.c_str(), ec ? (" (error: " + ec.message() + ")").c_str() : "");
    if (!ec) cloud_restored();
    return !ec;
}

bool save_delete_current() {
    save_backup_now(" before delete");
    LOG("[saves] deleted the current save (a copy is in backups/)\n");
    if (!remove_current()) return false;
    cloud_deleted();
    return true;
}

bool save_delete_backup(const SaveBackup &b) {
    std::error_code ec;
    fs::remove_all(b.path, ec);
    return !ec;
}

void save_auto_backup() {
    if (!g_settings.auto_backup || !save_backup_now(" auto")) return;
    int kept = 0;
    for (auto &b : save_backups())
        if (b.name.size() > 5 && b.name.compare(b.name.size() - 5, 5, " auto") == 0 && ++kept > 5)
            save_delete_backup(b);
}

// ---- a session that ended without a save: the game finds its own mark still set and scolds the player for it
// (Animal Crossing's Resetti). What the game last wrote is kept; only the mark goes.
static u32 crc32c(const u8 *p, size_t n) {
    static u32 table[256];
    if (!table[1])
        for (u32 i = 0; i < 256; i++) {
            u32 c = i;
            for (int k = 0; k < 8; k++) c = c & 1 ? (c >> 1) ^ 0x82F63B78 : c >> 1;  // (the Castagnoli polynomial)
            table[i] = c;
        }
    u32 c = 0xffffffff;
    while (n--) c = table[(c ^ *p++) & 0xff] ^ (c >> 8);
    return ~c;
}

void save_forgive_unsaved() {
    if (!g_settings.forgive_unsaved) return;
    std::map<std::string, std::vector<const profile::UnsavedMark *>> by_file;
    for (const auto &m : profile::unsaved_marks()) by_file[m.file].push_back(&m);
    for (const auto &[name, marks] : by_file) {
        const std::string path = kernel::save_root() + "/savedata/" + name;
        std::ifstream in(path, std::ios::binary);
        std::vector<u8> d((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        in.close();
        bool changed = false;
        for (const profile::UnsavedMark *m : marks) {
            if ((u64)m->block + 4 + m->size > d.size() || !(d[m->offset] & m->bits)) continue;
            u32 stored;
            memcpy(&stored, &d[m->block], 4);
            if (stored != crc32c(&d[m->block + 4], m->size)) {
                LOG("[save] %s: the block at %x does not match its CRC; its unsaved mark is left\n", name.c_str(), m->block);
                continue;
            }
            d[m->offset] &= (u8)~m->bits;
            const u32 crc = crc32c(&d[m->block + 4], m->size);
            memcpy(&d[m->block], &crc, 4);
            changed = true;
            LOG("[save] %s: the last session ended without a save; cleared its mark at %x\n", name.c_str(), m->offset);
        }
        if (!changed) continue;
        const std::string tmp = path + ".tmp";  // (written whole, then put in place)
        std::ofstream out(tmp, std::ios::binary);
        out.write((const char *)d.data(), (std::streamsize)d.size());
        out.close();
        std::error_code ec;
        if (out) fs::rename(tmp, path, ec);
        if (!out || ec) {
            fs::remove(tmp, ec);
            LOG("[save] could not write %s\n", path.c_str());
        }
    }
}
