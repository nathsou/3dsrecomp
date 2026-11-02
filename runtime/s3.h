// S3 requests: AWS Signature Version 4 and the few replies cloud saves read (runtime/cloud.mm). Any S3-compatible
// storage takes them: Backblaze B2, Cloudflare R2, MinIO, AWS. Path-style URLs (https://<host>/<bucket>/<key>).
#pragma once
#include <ctime>
#include <string>
#include <utility>
#include <vector>
#include <algorithm>

#include <cctype>
#include <cstdint>
#include <cstring>

namespace s3 {

// SHA-256 (FIPS 180-4)
class Sha256 {
    uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    unsigned char block[64]{};
    uint64_t bytes = 0;
    size_t used = 0;
    static uint32_t rotr(uint32_t x, unsigned n) { return (x >> n) | (x << (32 - n)); }
    void compress() {
        static const uint32_t k[64] = {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
            0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
            0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
        uint32_t w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = (uint32_t(block[i * 4]) << 24) | (uint32_t(block[i * 4 + 1]) << 16) |
                   (uint32_t(block[i * 4 + 2]) << 8) | block[i * 4 + 3];
        for (int i = 16; i < 64; ++i) {
            uint32_t a = w[i - 15], b = w[i - 2];
            w[i] =
                w[i - 16] + (rotr(a, 7) ^ rotr(a, 18) ^ (a >> 3)) + w[i - 7] + (rotr(b, 17) ^ rotr(b, 19) ^ (b >> 10));
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], v = h[7];
        for (int i = 0; i < 64; ++i) {
            uint32_t t1 = v + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + k[i] + w[i];
            uint32_t t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
            v = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }
        h[0] += a;
        h[1] += b;
        h[2] += c;
        h[3] += d;
        h[4] += e;
        h[5] += f;
        h[6] += g;
        h[7] += v;
    }

  public:
    void add(const void *data, size_t n) {
        auto p = static_cast<const unsigned char *>(data);
        bytes += n;
        while (n) {
            size_t take = n < 64 - used ? n : 64 - used;
            memcpy(block + used, p, take);
            used += take;
            p += take;
            n -= take;
            if (used == 64) {
                compress();
                used = 0;
            }
        }
    }
    std::string finish() {
        uint64_t bits = bytes * 8;
        block[used++] = 0x80;
        if (used > 56) {
            memset(block + used, 0, 64 - used);
            compress();
            used = 0;
        }
        memset(block + used, 0, 56 - used);
        for (int i = 0; i < 8; ++i)
            block[63 - i] = (unsigned char)(bits >> (i * 8));
        compress();
        static const char hex[] = "0123456789abcdef";
        std::string result;
        for (uint32_t word : h)
            for (int shift = 28; shift >= 0; shift -= 4)
                result += hex[(word >> shift) & 15];
        return result;
    }
};


using Headers = std::vector<std::pair<std::string, std::string>>;

inline std::string sha256_hex(const void *p, size_t n) {
    Sha256 h;
    h.add(p, n);
    return h.finish();
}
inline std::string sha256_hex(const std::string &s) { return sha256_hex(s.data(), s.size()); }
inline std::string unhex(const std::string &h) {
    std::string out;
    auto v = [](char c) { return c <= '9' ? c - '0' : (c | 32) - 'a' + 10; };
    for (size_t i = 0; i + 1 < h.size(); i += 2) out += (char)(v(h[i]) << 4 | v(h[i + 1]));
    return out;
}
inline std::string hmac(const std::string &key, const std::string &msg) {
    std::string k = key.size() > 64 ? unhex(sha256_hex(key)) : key;
    k.resize(64, '\0');
    std::string ipad(64, '\0'), opad(64, '\0');
    for (int i = 0; i < 64; i++) ipad[i] = k[i] ^ 0x36, opad[i] = k[i] ^ 0x5c;
    return unhex(sha256_hex(opad + unhex(sha256_hex(ipad + msg))));
}
// RFC 3986 percent-encoding, as SigV4 wants it: all but A-Z a-z 0-9 - _ . ~ (and '/' in a path)
inline std::string encode(const std::string &s, bool path = false) {
    static const char hex[] = "0123456789ABCDEF";
    std::string out;
    for (unsigned char c : s) {
        if (isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~' || (path && c == '/')) out += (char)c;
        else out += '%', out += hex[c >> 4], out += hex[c & 15];
    }
    return out;
}
inline std::string lower(std::string s) {
    for (char &c : s) c = (char)tolower((unsigned char)c);
    return s;
}
inline std::string trim(const std::string &s) {
    size_t a = s.find_first_not_of(" \t"), b = s.find_last_not_of(" \t");
    return a == std::string::npos ? "" : s.substr(a, b - a + 1);
}

// Signs a request: adds x-amz-date (and, for S3, x-amz-content-sha256) and Authorization to headers, which must
// already hold every other header to send (host first among them is not required). path: the URL's path, not yet
// encoded ("/bucket/key"); query: its parameters, not yet encoded. Returns the encoded query string for the URL.
inline std::string sign(const std::string &method, const std::string &path, std::vector<std::pair<std::string, std::string>> query,
                        Headers &headers, const std::string &payload_hash, const std::string &region, const std::string &key_id,
                        const std::string &secret, time_t now, const std::string &service = "s3") {
    char date[32], day[16];
    struct tm tm;
    gmtime_r(&now, &tm);
    strftime(date, sizeof date, "%Y%m%dT%H%M%SZ", &tm);
    strftime(day, sizeof day, "%Y%m%d", &tm);
    headers.push_back({"x-amz-date", date});
    if (service == "s3") headers.push_back({"x-amz-content-sha256", payload_hash});

    for (auto &q : query) q = {encode(q.first), encode(q.second)};
    std::sort(query.begin(), query.end());
    std::string qs;
    for (auto &q : query) qs += (qs.empty() ? "" : "&") + q.first + "=" + q.second;
    Headers canon;
    for (auto &h : headers) canon.push_back({lower(h.first), trim(h.second)});
    std::sort(canon.begin(), canon.end());
    std::string ch, signed_headers;
    for (auto &h : canon) {
        ch += h.first + ":" + h.second + "\n";
        signed_headers += (signed_headers.empty() ? "" : ";") + h.first;
    }
    std::string request = method + "\n" + encode(path, true) + "\n" + qs + "\n" + ch + "\n" + signed_headers + "\n" + payload_hash;
    std::string scope = std::string(day) + "/" + region + "/" + service + "/aws4_request";
    std::string to_sign = "AWS4-HMAC-SHA256\n" + std::string(date) + "\n" + scope + "\n" + sha256_hex(request);
    std::string k = hmac(hmac(hmac(hmac("AWS4" + secret, day), region), service), "aws4_request");
    static const char hex[] = "0123456789abcdef";
    std::string sig;
    for (unsigned char c : hmac(k, to_sign)) sig += hex[c >> 4], sig += hex[c & 15];
    headers.push_back({"Authorization", "AWS4-HMAC-SHA256 Credential=" + key_id + "/" + scope +
                                            ", SignedHeaders=" + signed_headers + ", Signature=" + sig});
    return qs;
}

// The text of each <tag>...</tag> in xml, in order (S3's replies are flat enough for this), entities decoded
inline std::vector<std::string> xml_all(const std::string &xml, const std::string &tag) {
    std::vector<std::string> out;
    std::string open = "<" + tag + ">", close = "</" + tag + ">";
    for (size_t at = 0; (at = xml.find(open, at)) != std::string::npos;) {
        size_t b = at + open.size(), e = xml.find(close, b);
        if (e == std::string::npos) break;
        std::string v = xml.substr(b, e - b), d;
        for (size_t i = 0; i < v.size(); i++) {
            static const char *const ent[][2] = {{"&amp;", "&"}, {"&lt;", "<"}, {"&gt;", ">"}, {"&quot;", "\""}, {"&apos;", "'"}};
            bool hit = false;
            for (auto &x : ent)
                if (!v.compare(i, strlen(x[0]), x[0])) { d += x[1], i += strlen(x[0]) - 1, hit = true; break; }
            if (!hit) d += v[i];
        }
        out.push_back(d);
        at = e + close.size();
    }
    return out;
}
inline std::string xml_one(const std::string &xml, const std::string &tag) {
    auto v = xml_all(xml, tag);
    return v.empty() ? "" : v[0];
}

}  // namespace s3
