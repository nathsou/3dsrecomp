// runtime/cloud_http.h without Apple's frameworks: requests through libcurl. The application key's secret stays in
// <save dir>/cloud.secret (runtime/cloud.cpp), as on a Mac.
#include <curl/curl.h>

#include <algorithm>
#include <cstring>
#include <mutex>

#include "cloud_http.h"

namespace {
struct Xfer {
    const std::string *body;
    size_t sent = 0;
    CloudResp *out;
    const CloudProgress *progress;
    bool up;
};

size_t on_read(char *buf, size_t size, size_t n, void *p) {
    auto *x = (Xfer *)p;
    const size_t k = std::min(size * n, x->body->size() - x->sent);
    memcpy(buf, x->body->data() + x->sent, k);
    x->sent += k;
    return k;
}

size_t on_write(char *buf, size_t size, size_t n, void *p) {
    ((Xfer *)p)->out->body.append(buf, size * n);
    return size * n;
}

size_t on_header(char *buf, size_t size, size_t n, void *p) {
    auto *x = (Xfer *)p;
    std::string line(buf, size * n);
    if (!line.compare(0, 5, "HTTP/")) x->out->headers.clear();  // (a new response: after a 100 Continue)
    const size_t colon = line.find(':');
    if (colon != std::string::npos) {
        std::string v = line.substr(colon + 1);
        while (!v.empty() && (v.back() == '\r' || v.back() == '\n')) v.pop_back();
        x->out->headers[s3::lower(line.substr(0, colon))] = s3::trim(v);
    }
    return size * n;
}

int on_progress(void *p, curl_off_t dl_total, curl_off_t dl_now, curl_off_t ul_total, curl_off_t ul_now) {
    auto *x = (Xfer *)p;
    const curl_off_t of = x->up ? ul_total : dl_total, now = x->up ? ul_now : dl_now;
    if (*x->progress && of > 0) (*x->progress)((double)now / (double)of);
    return 0;
}
}  // namespace

CloudResp cloud_http_send(const std::string &method, const std::string &url, const s3::Headers &headers,
                          const std::string &body, double timeout, bool cellular, const CloudProgress &progress) {
    (void)cellular;
    static std::once_flag once;
    std::call_once(once, [] { curl_global_init(CURL_GLOBAL_DEFAULT); });
    CloudResp out;
    CURL *c = curl_easy_init();
    if (!c) {
        out.error = "cannot start a request (libcurl)";
        return out;
    }
    Xfer x{&body, 0, &out, &progress, method == "PUT"};
    curl_slist *hl = nullptr;
    for (auto &h : headers)
        if (s3::lower(h.first) != "host") hl = curl_slist_append(hl, (h.first + ": " + h.second).c_str());
    hl = curl_slist_append(hl, "Expect:");  // (no 100 Continue round trip before an upload)
    curl_easy_setopt(c, CURLOPT_URL, url.c_str());
    curl_easy_setopt(c, CURLOPT_HTTPHEADER, hl);
    if (method == "PUT") {
        curl_easy_setopt(c, CURLOPT_UPLOAD, 1L);
        curl_easy_setopt(c, CURLOPT_READFUNCTION, on_read);
        curl_easy_setopt(c, CURLOPT_READDATA, &x);
        curl_easy_setopt(c, CURLOPT_INFILESIZE_LARGE, (curl_off_t)body.size());
    } else if (method == "HEAD") {
        curl_easy_setopt(c, CURLOPT_NOBODY, 1L);
    } else if (method != "GET") {
        curl_easy_setopt(c, CURLOPT_CUSTOMREQUEST, method.c_str());
        if (!body.empty()) {
            curl_easy_setopt(c, CURLOPT_POSTFIELDS, body.data());
            curl_easy_setopt(c, CURLOPT_POSTFIELDSIZE_LARGE, (curl_off_t)body.size());
        }
    }
    curl_easy_setopt(c, CURLOPT_WRITEFUNCTION, on_write);
    curl_easy_setopt(c, CURLOPT_WRITEDATA, &x);
    curl_easy_setopt(c, CURLOPT_HEADERFUNCTION, on_header);
    curl_easy_setopt(c, CURLOPT_HEADERDATA, &x);
    curl_easy_setopt(c, CURLOPT_NOPROGRESS, 0L);
    curl_easy_setopt(c, CURLOPT_XFERINFOFUNCTION, on_progress);
    curl_easy_setopt(c, CURLOPT_XFERINFODATA, &x);
    curl_easy_setopt(c, CURLOPT_NOSIGNAL, 1L);
    // as NSURLSession's: `timeout` with nothing sent or received, at most 10 minutes in all
    const long idle = std::max(1L, (long)timeout);
    curl_easy_setopt(c, CURLOPT_CONNECTTIMEOUT, idle);
    curl_easy_setopt(c, CURLOPT_LOW_SPEED_LIMIT, 1L);
    curl_easy_setopt(c, CURLOPT_LOW_SPEED_TIME, idle);
    curl_easy_setopt(c, CURLOPT_TIMEOUT, 600L);
    const CURLcode rc = curl_easy_perform(c);
    if (rc != CURLE_OK) {
        out.error = curl_easy_strerror(rc);
        out.headers.clear();
        out.body.clear();
    } else {
        long status = 0;
        curl_easy_getinfo(c, CURLINFO_RESPONSE_CODE, &status);
        out.status = (int)status;
    }
    curl_slist_free_all(hl);
    curl_easy_cleanup(c);
    return out;
}

// (the Keychain is iOS's)
std::string cloud_keychain_get(const std::string &) { return ""; }
void cloud_keychain_clear(const std::string &) {}
bool cloud_keychain_set(const std::string &, const std::string &) { return false; }
