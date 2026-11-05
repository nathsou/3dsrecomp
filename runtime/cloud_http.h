// The system's side of cloud saves (runtime/cloud.cpp): HTTPS requests, and on iOS the Keychain.
// runtime/cloud_apple.mm (NSURLSession, Keychain), runtime/cloud_curl.cpp (libcurl) elsewhere.
#pragma once
#include <functional>
#include <map>
#include <string>

#include "s3.h"

struct CloudResp {
    int status = 0;
    std::string body, error;                     // error: the network's (empty when the server answered)
    std::map<std::string, std::string> headers;  // (lower-case names)
};
using CloudProgress = std::function<void(double)>;

// one request, waited for; timeout: seconds with nothing sent or received; cellular: mobile data allowed (iOS);
// progress: the share of the body sent (PUT) or received so far, now and then
CloudResp cloud_http_send(const std::string &method, const std::string &url, const s3::Headers &headers,
                          const std::string &body, double timeout, bool cellular, const CloudProgress &progress);

// (iOS) the application key's secret in this app's Keychain, not synced to iCloud
std::string cloud_keychain_get(const std::string &kid);
void cloud_keychain_clear(const std::string &kid);
bool cloud_keychain_set(const std::string &kid, const std::string &secret);
