// Per-game user settings (<save dir>/<title id>/settings.ini), changed from the in-game menu (runtime/menu.mm).
// Environment variables given at launch (GPU_MSAA, GPU_ANISO, ...) take precedence over the file.
#pragma once
#include <string>
#include <vector>

#include "rt.h"

struct Settings {
    int scale = 0;          // internal resolution: 0 auto (from the window), 1-4
    int msaa = 4;           // samples per pixel: 1 (off), 2, 4, 8 (applies at launch)
    int aniso = 16;         // max anisotropy: 1 (off), 2, 4, 8, 16 (applies at launch)
    int layout = 0;         // see the layout table in frontend.cpp (0 stacked, 1 top screen only, ...)
    bool fullscreen = false;
    bool split = false;        // separate windows: the top screen in the main one, the bottom screen in a second
    bool fullscreen2 = false;  // the bottom screen's window
    int geom[4] = {}, geom2[4] = {};  // window positions and sizes (x, y, w, h; w 0: not saved yet)
    int speed = 100;        // game speed in percent of the 3DS's (VBlank and audio clocks)
    int volume = 100;       // percent
    bool show_fps = false;  // frame rate counter in the corner
    s64 clock_offset = 0;   // seconds added to the Mac's clock for the 3DS clock
    bool auto_backup = true;  // copy the save into backups/ at each launch (the last 5 are kept)
};

extern Settings g_settings;

void settings_load();  // also exports the launch-time settings to the GPU backend's environment variables
void settings_save();

// Save management (the menu's Saves page): the game's save data is <save root>/savedata and extdata/.
struct SaveBackup {
    std::string name, path;  // name: "2026-10-27 19.42.03" (+ " auto" / " before restore" / " before delete")
    u64 bytes;
};
std::vector<SaveBackup> save_backups();  // newest first
bool save_exists();
std::string save_summary();               // "18 KB, written 27 Oct 19:04" or "no save yet"
bool save_backup_now(const std::string &suffix = "");
bool save_restore(const SaveBackup &b);   // backs up the current save first; the game must be restarted
bool save_delete_current();               // backs it up first; the game must be restarted
bool save_delete_backup(const SaveBackup &b);
void save_auto_backup();                  // at launch, if enabled
