// Cloud saves (runtime/cloud.cpp): each device's save of each game in an S3-compatible bucket (Backblaze B2, ...),
// the newest one used when the game starts.
#pragma once
#include <ctime>
#include <string>
#include <vector>

// At launch (main.cpp), after the settings and before the game runs: a newer save of this game (same game files)
// from any device replaces the local one, which is backed up first.
void cloud_boot();
// The game wrote to its save (services/fs.cpp): an upload follows, unless more writes come.
void cloud_save_written();
// Upload the save now if it changed (the menu opened).
void cloud_flush();
// Quitting or restarting (frontend.cpp): upload the save if it changed and wait for it, for a few seconds at most.
void cloud_quit();
// The save service's lock: held while it writes, and while cloud saves read the save (services/fs.cpp)
void save_lock();
void save_unlock();
// settings.cpp: a backup was restored (or a save picked from the cloud): it's the newest save from now on, uploaded
// at the next launch for the other devices to take. Or the save was deleted: older saves in the bucket aren't
// brought back.
void cloud_restored();
void cloud_deleted();

// The menu's Saves page (runtime/menu.mm)
struct CloudSlot {
    std::string key, device;
    time_t saved = 0;
    unsigned long long size = 0;
    bool mine = false;       // this device's own upload
    bool same_game = false;  // the same game files as here (another version's save isn't used)
    bool current = false;    // the save here now
};
struct CloudView {
    bool configured = false, busy = false, sync = true;
    std::string where, device, status, error;  // where: "<bucket> at <endpoint>"
    float progress = -1;                        // 0..1 during an upload / download, else -1
    time_t listed = 0;
    std::vector<CloudSlot> slots;
    std::string fetched;  // a save downloaded for the menu to restore (cloud_fetch), once
};
CloudView cloud_view();
// "s3://<key ID>:<application key>@<endpoint>/<bucket>[/<folder>]": stored for every game on this device and
// tested. false: why it can't be used.
bool cloud_connect(const std::string &connection, std::string *why);
void cloud_disconnect();
void cloud_set_sync(bool on);
void cloud_refresh();     // the slots listed again (in the background)
void cloud_upload_now();  // the save uploaded (in the background), changed or not
// That slot's save downloaded into backups/ (in the background): CloudView::fetched when it's there
void cloud_fetch(const std::string &key);
