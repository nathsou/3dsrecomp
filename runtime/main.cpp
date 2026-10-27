// Entry point: loads the game files, boots the HLE kernel and runs the host frontend.
#include <unistd.h>

#include "frontend.h"
#include "gpu_iface.h"
#include "kernel.h"
#include "loader.h"
#include "service.h"
#include "settings.h"

#include <SDL3/SDL.h>
#include <mach-o/dyld.h>
#include <sys/stat.h>

#include <filesystem>

std::string g_save_base = "save";
char **g_argv;  // for restarting the game from the menu

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
    // inside an app bundle: game files in Contents/Resources/game, saves in Application Support
    char exe[4096];
    uint32_t len = sizeof exe;
    if (_NSGetExecutablePath(exe, &len) == 0) {
        auto res = std::filesystem::path(exe).parent_path().parent_path() / "Resources" / "game";
        if (std::filesystem::exists(res / "exheader.bin")) {
            dir = res.string();
            bundle = true;
            const char *home = getenv("HOME");
            g_save_base = std::string(home ? home : ".") + "/Library/Application Support/3dsrecomp";
        }
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
    kernel::BootInfo bi{g_game.text_addr, g_game.stack_size, g_game.priority, g_game.program_id};
    kernel::start(bi);
    frontend_run();
    return 0;
}
