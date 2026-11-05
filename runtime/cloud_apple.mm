// runtime/cloud_http.h on macOS and iOS: requests through NSURLSession, the iOS Keychain.
#include "cloud_http.h"

#include <TargetConditionals.h>
#import <Foundation/Foundation.h>
#if TARGET_OS_IPHONE
#import <Security/Security.h>
#import <UIKit/UIKit.h>
#endif

#include <mutex>

#if TARGET_OS_IPHONE
static NSMutableDictionary *kc_query(const std::string &kid) {
    return [@{(id)kSecClass: (id)kSecClassGenericPassword, (id)kSecAttrService: @"3dsrecomp cloud saves",
              (id)kSecAttrAccount: @(kid.c_str())} mutableCopy];
}
std::string cloud_keychain_get(const std::string &kid) {
    NSMutableDictionary *q = kc_query(kid);
    q[(id)kSecReturnData] = @YES;
    q[(id)kSecMatchLimit] = (id)kSecMatchLimitOne;
    CFTypeRef r = NULL;
    if (SecItemCopyMatching((__bridge CFDictionaryRef)q, &r) != errSecSuccess || !r) return "";
    NSData *d = CFBridgingRelease(r);
    return std::string((const char *)d.bytes, d.length);
}
void cloud_keychain_clear(const std::string &kid) { SecItemDelete((__bridge CFDictionaryRef)kc_query(kid)); }
bool cloud_keychain_set(const std::string &kid, const std::string &secret) {
    cloud_keychain_clear(kid);
    NSMutableDictionary *q = kc_query(kid);
    q[(id)kSecValueData] = [NSData dataWithBytes:secret.data() length:secret.size()];
    q[(id)kSecAttrAccessible] = (id)kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly;
    return SecItemAdd((__bridge CFDictionaryRef)q, NULL) == errSecSuccess;
}
#else
std::string cloud_keychain_get(const std::string &) { return ""; }
void cloud_keychain_clear(const std::string &) {}
bool cloud_keychain_set(const std::string &, const std::string &) { return false; }
#endif

// ------------------------------------------------------------------ HTTPS (NSURLSession)
CloudResp cloud_http_send(const std::string &method, const std::string &url, const s3::Headers &headers,
                          const std::string &body, double timeout, bool cellular, const CloudProgress &progress) {
    CloudResp out;
    @autoreleasepool {
        static NSURLSession *sessions[2];  // ([1]: mobile data allowed)
        static std::mutex m;
        NSURLSession *session;
        {
            std::lock_guard<std::mutex> l(m);
            int k = cellular ? 1 : 0;
            if (!sessions[k]) {
                NSURLSessionConfiguration *c = [NSURLSessionConfiguration ephemeralSessionConfiguration];
                c.allowsCellularAccess = k;
                c.timeoutIntervalForResource = 600;
                c.URLCache = nil;
                sessions[k] = [NSURLSession sessionWithConfiguration:c];
            }
            session = sessions[k];
        }
        NSMutableURLRequest *r = [NSMutableURLRequest requestWithURL:[NSURL URLWithString:@(url.c_str())]];
        r.HTTPMethod = @(method.c_str());
        r.timeoutInterval = timeout;  // (with nothing received for that long)
        for (auto &h : headers)
            if (s3::lower(h.first) != "host") [r setValue:@(h.second.c_str()) forHTTPHeaderField:@(h.first.c_str())];
        dispatch_semaphore_t done = dispatch_semaphore_create(0);
        CloudResp *o = &out;
        void (^handler)(NSData *, NSURLResponse *, NSError *) = ^(NSData *data, NSURLResponse *resp, NSError *e) {
            if (e) o->error = e.localizedDescription.UTF8String ?: "network error";
            else {
                NSHTTPURLResponse *h = (NSHTTPURLResponse *)resp;
                o->status = (int)h.statusCode;
                if (data) o->body.assign((const char *)data.bytes, data.length);
                [h.allHeaderFields enumerateKeysAndObjectsUsingBlock:^(id key, id val, BOOL *stop) {
                    (void)stop;
                    o->headers[s3::lower([key description].UTF8String)] = [val description].UTF8String;
                }];
            }
            dispatch_semaphore_signal(done);
        };
        bool up = method == "PUT";
        NSURLSessionTask *t = up ? [session uploadTaskWithRequest:r fromData:[NSData dataWithBytesNoCopy:(void *)body.data()
                                                                                                      length:body.size() freeWhenDone:NO]
                                                completionHandler:handler]
                                 : [session dataTaskWithRequest:r completionHandler:handler];
#if TARGET_OS_IPHONE
        // (an upload carries on for a while when the app goes to the background)
        __block UIBackgroundTaskIdentifier bg = UIBackgroundTaskInvalid;
        UIApplication *app = UIApplication.sharedApplication;
        bg = [app beginBackgroundTaskWithName:@"cloud save" expirationHandler:^{
            [t cancel];
            [app endBackgroundTask:bg];
            bg = UIBackgroundTaskInvalid;
        }];
#endif
        [t resume];
        while (dispatch_semaphore_wait(done, dispatch_time(DISPATCH_TIME_NOW, 100 * NSEC_PER_MSEC)))
            if (progress) {
                int64_t n = up ? t.countOfBytesSent : t.countOfBytesReceived;
                int64_t of = up ? t.countOfBytesExpectedToSend : t.countOfBytesExpectedToReceive;
                if (of > 0) progress((double)n / of);
            }
#if TARGET_OS_IPHONE
        if (bg != UIBackgroundTaskInvalid) [app endBackgroundTask:bg];
#endif
    }
    return out;
}
