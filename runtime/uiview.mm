// SDL's view kept filling its window (runtime/uiview.h).
#include "uiview.h"

#include <TargetConditionals.h>

#include "rt.h"

#if TARGET_OS_IOS
#import <QuartzCore/CAMetalLayer.h>
#import <UIKit/UIKit.h>

namespace uiview {

bool keep_filling(SDL_Renderer *renderer) {
    // (a rotation or the keyboard can take a few frames to settle: only what stays off is put back)
    constexpr int kSettleFrames = 10;
    constexpr int kMaxLogs = 20;
    static int off_for, logs;
    if (UIApplication.sharedApplication.applicationState != UIApplicationStateActive) {
        off_for = 0;  // (in the background iOS lays the app out in other shapes for its snapshots)
        return false;
    }
    CAMetalLayer *layer = (__bridge CAMetalLayer *)SDL_GetRenderMetalLayer(renderer);
    UIView *view = [layer.delegate isKindOfClass:UIView.class] ? (UIView *)layer.delegate : nil;  // SDL's Metal view
    UIView *root = view.superview;                                                               // its view controller's
    UIWindow *window = view.window;
    if (!root || !window || window.bounds.size.width <= 0) return false;
    const CGSize ws = window.bounds.size;
    const CGFloat scale = window.screen.nativeScale;
    // (SDL moves the root view up while the keyboard covers a text field, but keeps its size)
    const bool root_ok = CGSizeEqualToSize(root.bounds.size, ws);
    const bool view_ok = CGSizeEqualToSize(view.bounds.size, root.bounds.size);
    const bool scale_ok = layer.contentsScale == scale;
    const CGSize want = {view.bounds.size.width * layer.contentsScale, view.bounds.size.height * layer.contentsScale};
    const bool drawable_ok = CGSizeEqualToSize(layer.drawableSize, want);
    if (root_ok && view_ok && scale_ok && drawable_ok) {
        off_for = 0;
        return false;
    }
    if (++off_for < kSettleFrames) return false;
    off_for = 0;
    if (logs < kMaxLogs) {
        logs++;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        const long status_orientation = (long)UIApplication.sharedApplication.statusBarOrientation;
#pragma clang diagnostic pop
        LOG("uiview: the view no longer fills its window, put back (window %.0fx%.0f, root view %.0fx%.0f at %.0f,%.0f, "
            "view %.0fx%.0f at %.0f,%.0f, scale %.2f of %.2f, drawable %.0fx%.0f, orientation %ld, status bar %ld)\n",
            ws.width, ws.height, root.frame.size.width, root.frame.size.height, root.frame.origin.x, root.frame.origin.y,
            view.frame.size.width, view.frame.size.height, view.frame.origin.x, view.frame.origin.y, layer.contentsScale,
            scale, layer.drawableSize.width, layer.drawableSize.height, (long)window.windowScene.interfaceOrientation,
            status_orientation);
    }
    if (!root_ok) root.frame = window.bounds;
    if (!CGSizeEqualToSize(view.bounds.size, root.bounds.size)) view.frame = root.bounds;
    if (!scale_ok) view.contentScaleFactor = scale;
    // SDL's views resize the drawable and tell SDL the window's new size when they lay out
    [root setNeedsLayout];
    [view setNeedsLayout];
    [root layoutIfNeeded];
    return true;
}

}  // namespace uiview

#else
namespace uiview {
bool keep_filling(SDL_Renderer *) { return false; }
}  // namespace uiview
#endif
