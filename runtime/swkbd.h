// The 3DS software keyboard applet, typed on the Mac's keyboard: services/apt.cpp holds the game's request
// until the player confirms or cancels; menu.mm draws the text field and takes the keys (no kernel types here,
// so Objective-C++ files can include it).
#pragma once
#include <string>

namespace swkbd {
struct Request {
    std::string hint, initial;  // UTF-8
    std::string ok, cancel;     // button labels the game set ("" for the defaults)
    int max_len = 0;            // in UTF-16 units (0: no limit)
    int max_digits = 0;         // with `digits_limited`
    bool numpad = false, password = false, multiline = false, can_cancel = false;
    bool digits_limited = false, no_at = false, no_percent = false, no_backslash = false;
    int valid = 0;              // 0 anything, 1 not empty, 2 not empty or blank, 3 not blank, 4 fixed length
    unsigned serial = 0;        // changes with every request
};
bool pending(Request *out);                       // (any thread) a request waits for the player
void finish(const std::string &text, bool ok);    // (host thread, kernel lock not held) confirm or cancel
}  // namespace swkbd
