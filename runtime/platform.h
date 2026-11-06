// What differs between the host systems, behind one interface: runtime/platform_apple.mm (macOS, iOS) and
// runtime/platform_posix.cpp (Linux and other POSIX systems). New system-specific code goes here, not into the
// portable files.
// Still POSIX outside this layer, for a Windows port to take over: the guest address space (mmap in core.cpp, mprotect
// in watch.cpp), host threads (pthreads in kernel.cpp), files and folders (unistd/fcntl/dirent in services/fs.cpp,
// cloud.cpp, settings.cpp, gsp.cpp's dumps), loadable code (dlopen/dladdr in core.cpp, main.cpp, kernel.cpp), signals
// (watch.cpp), restarting the process (execv in frontend.cpp) and realpath (core.cpp).
#pragma once
#if defined(__has_include)
#if __has_include(<TargetConditionals.h>)
#include <TargetConditionals.h>
#endif
#endif
#ifndef TARGET_OS_IOS
#define TARGET_OS_IOS 0
#endif
#ifndef TARGET_OS_IPHONE
#define TARGET_OS_IPHONE 0
#endif
#ifndef TARGET_OS_OSX
#define TARGET_OS_OSX 0
#endif

#include <pthread.h>

#include <cstdint>
#include <string>

namespace platform {
// the running executable's path ("" if unknown)
std::string exe_path();
// where an app bundle's saves go: ~/Library/Application Support/3dsrecomp (Apple),
// $XDG_DATA_HOME/3dsrecomp, by default ~/.local/share/3dsrecomp (elsewhere)
std::string app_data_dir();
// file name suffix of loadable code: game modules of a multi-game app, lazily recompiled code
const char *module_suffix();
// shows a folder in the system's file manager
void open_folder(const std::string &path);
// this device's name, as the user knows it (cloud saves name devices with it)
std::string device_name();
// the calling thread serves the player directly (GPU commands, input, audio): scheduled ahead of others
void set_thread_interactive();
// names the calling thread for debuggers, profilers and top (at most 15 characters are kept)
void set_thread_name(const char *name);
// Sampling another thread's stack (thread dumps show the guest functions a thread is in): a handle for the thread
// (0: none), whether it is the calling one, and the return addresses on its stack, innermost first (pc first)
uint64_t thread_handle(pthread_t t);
bool is_current_thread(uint64_t handle);
int sample_thread_pcs(uint64_t handle, uint64_t *pcs, int max);
// the program counter of a signal's ucontext_t (SIGSEGV handlers)
uint64_t fault_pc(void *ucontext);
}  // namespace platform
