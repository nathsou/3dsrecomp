// The game list of a multi-game app (RECOMP_MODULES): each game's icon and name from its SMDH; a tap or click,
// Return or a gamepad's A starts one (runtime/launcher.cpp). It has its own window, closed before the game opens.
#pragma once
#include <string>
#include <vector>

#include "rt.h"

struct LauncherGame {
    std::string id, dir, name;
    std::vector<u32> icon;  // 48x48 RGBA (empty: none)
};
int launcher_choose(const std::vector<LauncherGame> &games, int selected);  // the chosen game's index
