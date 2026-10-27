// Per-game settings file and save backups (see settings.h).
#include "settings.h"

#include <sys/stat.h>

#include <algorithm>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <map>

#include "kernel.h"

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
    s.scale = std::clamp(s.scale, 0, 4);
    s.layout = std::max(s.layout, 0);  // (the frontend clamps it to its layouts)
    s.speed = std::clamp(s.speed, 25, 400);
    s.volume = std::clamp(s.volume, 0, 100);
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
    return !ec;
}

bool save_delete_current() {
    save_backup_now(" before delete");
    LOG("[saves] deleted the current save (a copy is in backups/)\n");
    return remove_current();
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
