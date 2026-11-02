#pragma once
#include <string>
#include <vector>

#include "rt.h"

struct GameInfo {
    std::string dir, title;
    std::string name;  // display name: the SMDH's English short title, else the exheader title
    u64 program_id = 0;
    u32 text_addr = 0, text_size = 0, ro_addr = 0, ro_size = 0, data_addr = 0, data_size = 0, bss_size = 0;
    u32 stack_size = 0;
    int priority = 0x30;
    int region = -1;  // from the SMDH region lockout: 0 JPN, 1 USA, 2 EUR, 3 AUS, 4 CHN, 5 KOR, 6 TWN; -1 region free
};
extern GameInfo g_game;

void load_game(const std::string &dir);
std::string smdh_title(const std::string &path);  // an SMDH's English (else Japanese) short title, as UTF-8
std::vector<u32> smdh_icon(const std::string &path);  // its 48x48 icon as RGBA (empty without one)
void update_shared_page_time();
