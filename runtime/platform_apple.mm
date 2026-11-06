// runtime/platform.h on macOS and iOS.
#include "platform.h"

#import <Foundation/Foundation.h>
#if TARGET_OS_IPHONE
#import <UIKit/UIKit.h>
#else
#import <AppKit/AppKit.h>
#endif
#include <mach-o/dyld.h>
#include <mach/mach.h>
#include <pthread/qos.h>
#include <sys/ucontext.h>

namespace platform {

std::string exe_path() {
    char exe[4096];
    uint32_t len = sizeof exe;
    return _NSGetExecutablePath(exe, &len) == 0 ? exe : "";
}

std::string app_data_dir() {
    const char *home = getenv("HOME");  // (on iOS, the app's own container)
    return std::string(home ? home : ".") + "/Library/Application Support/3dsrecomp";
}

const char *module_suffix() { return ".dylib"; }

void open_folder(const std::string &path) {
#if !TARGET_OS_IPHONE
    NSString *p = [NSString stringWithUTF8String:path.c_str()];
    [[NSWorkspace sharedWorkspace] openURL:[NSURL fileURLWithPath:p]];
#else
    (void)path;
#endif
}

std::string device_name() {
    std::string name;
#if TARGET_OS_IPHONE
    name = UIDevice.currentDevice.model.UTF8String;  // ("iPhone", "iPad": iOS doesn't tell apps the device's own name)
#else
    name = NSHost.currentHost.localizedName.UTF8String ?: "Mac";
#endif
    return name;
}

void set_thread_interactive() { pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0); }

void set_thread_name(const char *name) { pthread_setname_np(name); }

uint64_t thread_handle(pthread_t t) { return pthread_mach_thread_np(t); }

bool is_current_thread(uint64_t handle) { return handle && pthread_mach_thread_np(pthread_self()) == handle; }

// the thread suspended, its registers read, the frame records walked from its frame pointer
int sample_thread_pcs(uint64_t handle, uint64_t *pcs, int max) {
    if (!handle || max < 2 || is_current_thread(handle)) return 0;
    const thread_act_t th = (thread_act_t)handle;
    int n = 0;
    if (thread_suspend(th) != KERN_SUCCESS) return 0;
    arm_thread_state64_t st;
    mach_msg_type_number_t cnt = ARM_THREAD_STATE64_COUNT;
    if (thread_get_state(th, ARM_THREAD_STATE64, (thread_state_t)&st, &cnt) == KERN_SUCCESS) {
        pcs[n++] = arm_thread_state64_get_pc(st);
        pcs[n++] = arm_thread_state64_get_lr(st);
        u_int64_t fp = arm_thread_state64_get_fp(st);
        // frame records are {previous fp, return address}; stop at anything that does not look like one
        for (int i = 0; i < 60 && fp && !(fp & 7) && n < max; i++) {
            u_int64_t *rec = (u_int64_t *)fp;
            pcs[n++] = rec[1];
            if (rec[0] <= fp) break;
            fp = rec[0];
        }
    }
    thread_resume(th);
    return n;
}

uint64_t fault_pc(void *ucontext) { return ((ucontext_t *)ucontext)->uc_mcontext->__ss.__pc; }

}  // namespace platform
