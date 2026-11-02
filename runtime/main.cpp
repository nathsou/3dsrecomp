// Entry point: loads the game files, boots the HLE kernel and runs the host frontend.
#include <unistd.h>

#include "frontend.h"
#include "gpu_iface.h"
#include "kernel.h"
#include "loader.h"
#include "service.h"
#include "settings.h"

#include <SDL3/SDL.h>
#include <TargetConditionals.h>
#include <dlfcn.h>
#include <mach-o/dyld.h>
#include <sys/stat.h>
#if TARGET_OS_IOS
#include <SDL3/SDL_main.h>  // UIKit owns the process: SDL starts the app and calls main() from it
#endif

#include <algorithm>
#include <filesystem>
#include <fstream>

#include "launcher.h"

std::string g_save_base = "save";
char **g_argv;  // for restarting the game from the menu
std::string g_start_game;  // (multi-game app) the game playing, which a restart goes straight back to

#ifdef RECOMP_MODULES
// A multi-game app: games/<name>/ holds each game's files and <name>.dylib its code. The player picks one from the
// list, the last one played selected (RECOMP_START_GAME=<name> skips the list: restarts, tests).
static std::string choose_game(const std::filesystem::path &games_dir, const std::filesystem::path &code_dir) {
    std::vector<LauncherGame> games;
    std::error_code ec;
    for (const auto &e : std::filesystem::directory_iterator(games_dir, ec)) {
        if (!std::filesystem::exists(e.path() / "exheader.bin")) continue;
        LauncherGame g{e.path().filename().string(), e.path().string()};
        g.name = smdh_title(g.dir + "/exefs/icon.bin");
        if (g.name.empty()) g.name = g.id;
        g.icon = smdh_icon(g.dir + "/exefs/icon.bin");
        games.push_back(std::move(g));
    }
    if (games.empty()) FATAL("no games in %s\n", games_dir.c_str());
    std::sort(games.begin(), games.end(), [](const auto &a, const auto &b) { return a.name < b.name; });
    const std::string last_file = g_save_base + "/last_game.txt";
    std::string want;
    const char *start = getenv("RECOMP_START_GAME");
    if (start) want = start;
    else std::getline(std::ifstream(last_file) >> std::ws, want);
    auto it = std::find_if(games.begin(), games.end(), [&](const auto &g) { return g.id == want; });
    const int found = it == games.end() ? -1 : (int)(it - games.begin());
    const int i = start && found >= 0 ? found : games.size() == 1 ? 0 : launcher_choose(games, std::max(found, 0));
    std::filesystem::create_directories(g_save_base, ec);
    std::ofstream(last_file) << games[i].id << "\n";
    const std::string lib = (code_dir / (games[i].id + ".dylib")).string();
    void *h = dlopen(lib.c_str(), RTLD_NOW);
    if (!h) FATAL("cannot load %s: %s\n", lib.c_str(), dlerror());
    const auto entries = (const GuestEntry *)dlsym(h, "guest_entries");
    const auto count = (const u32 *)dlsym(h, "guest_entry_count");
    if (!entries || !count) FATAL("%s has no game code\n", lib.c_str());
    set_game_code(entries, *count);
    const auto cros = (const GuestCro *)dlsym(h, "guest_cros");  // (games with CRO modules)
    const auto cro_count = (const u32 *)dlsym(h, "guest_cro_count");
    if (cros && cro_count) set_game_cros(cros, *cro_count);
    g_start_game = games[i].id;
    return games[i].dir;
}
#endif

// In an app bundle nobody sees stderr: show fatal errors in a dialog that says where the log is.
static void fatal_dialog(const char *msg) {
    std::string text = std::string(msg) + "\nLog: " + kernel::save_root() + "/log.txt";
    SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, g_game.title.c_str(), text.c_str(), nullptr);
}

int main(int argc, char **argv) {
    g_argv = argv;
    if (argc > 1 && std::string(argv[1]) == "--emit-shaders") return gpu::emit_shaders(argc - 2, argv + 2);
    std::string dir = "data/" RECOMP_GAME;
    bool bundle = false;
    // inside an app bundle: game files in Contents/Resources/game (macOS) or game/ next to the executable (iOS)
    // (games/<name>/ for a multi-game app, with the code in Frameworks/), saves in Application Support (the app's
    // own container on iOS)
    char exe[4096];
    uint32_t len = sizeof exe;
    if (_NSGetExecutablePath(exe, &len) == 0) {
        const auto exe_dir = std::filesystem::path(exe).parent_path();
        const auto res = TARGET_OS_IOS ? exe_dir : exe_dir.parent_path() / "Resources";
        const char *home = getenv("HOME");
        const std::string app_support = std::string(home ? home : ".") + "/Library/Application Support/3dsrecomp";
#ifdef RECOMP_MODULES
        if (std::filesystem::exists(res / "games")) {
            bundle = true;
            g_save_base = app_support;
            dir = choose_game(res / "games", TARGET_OS_IOS ? exe_dir / "Frameworks" : exe_dir.parent_path() / "Frameworks");
        } else {
            dir = choose_game(exe_dir / "games", exe_dir);  // a build folder (games/<name>: links to data/<name>)
        }
#else
        if (std::filesystem::exists(res / "game" / "exheader.bin")) {
            dir = (res / "game").string();
            bundle = true;
            g_save_base = app_support;
        }
#endif
    }
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "-v") g_verbose = 1;
        else if (a == "-vv") g_verbose = 2;
        else dir = a;
    }
    if (const char *v = getenv("RECOMP_VERBOSE")) g_verbose = atoi(v);
    mem::init();
    load_game(dir);
    std::filesystem::create_directories(kernel::save_root());
    rt_log_open(kernel::save_root() + "/log.txt");
    if (bundle) g_fatal_hook = fatal_dialog;
    settings_load();
    save_auto_backup();
    kernel::init_services();
    frontend_init();
    save_forgive_unsaved();  // (after the backup, with the profile loaded, before the game reads its save)
    kernel::BootInfo bi{g_game.text_addr, g_game.stack_size, g_game.priority, g_game.program_id};
    kernel::start(bi);
    frontend_run();
    return 0;
}
