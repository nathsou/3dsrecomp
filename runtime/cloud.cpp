// Cloud saves: each device uploads its save of each game to an S3-compatible bucket (Backblaze B2, Cloudflare R2,
// MinIO, AWS), and a game's launch uses the newest save of that game any device uploaded.
//
// - One object per device and game: [<folder>/]<title id>/<device id>.3dssave, the save (savedata/ and extdata/ of
//   <save dir>/<title id>/) as a tar. A device only writes its own, so two devices never overwrite each other's
//   save. Its metadata (x-amz-meta-nx-*) says which device it is from, when it was saved, which game files (a hash
//   of code.bin) and the tar's SHA-256, checked when it's downloaded.
// - Uploads: 30 s after the game's last write (a save is many writes), at most one every 10 minutes while playing,
//   and at once when the menu opens, the game restarts or quits.
// - At launch: the newest save made with the same game files, if it is newer than the local one, replaces it after a
//   backup of the local one (backups/). A save from other game files (another version) is never used.
// - <save dir>/<title id>/.cloud: the local save's hash and when it was saved, as last uploaded or downloaded. A
//   local save that differs from it changed since (played offline): its time is that of its newest file.
// Settings, for every game on this device: <save dir>/cloud.ini (the endpoint, the bucket, the application key's ID,
// this device's name); the key's secret in <save dir>/cloud.secret (owner-only), the Keychain on iOS.
// RECOMP_CLOUD=<connection> (tests): that bucket, nothing stored; RECOMP_CLOUD_DEVICE=<name>: another device's
// name and slot; RECOMP_CLOUD_EVERY=<s>: uploads that often at most.
#include "cloud.h"

#include "cloud_http.h"
#include "loader.h"
#include "platform.h"
#include "rt.h"
#include "s3.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <mutex>
#include <random>
#include <thread>

namespace fs = std::filesystem;
extern std::string g_save_base;
namespace kernel {
std::string save_root();  // (kernel.h clashes with MacTypes.h)
}
bool save_backup_now(const std::string &suffix);  // settings.cpp

// ------------------------------------------------------------------ settings (<save dir>/cloud.ini)
struct Config {
    bool sync = true, cellular = true;
    std::string endpoint, bucket, folder, key_id, device, device_id;
};
static Config g_cfg;
static std::mutex g_cfg_mx;
static std::string cfg_path() { return g_save_base + "/cloud.ini"; }
static std::string secret_path() { return g_save_base + "/cloud.secret"; }
static void cfg_load() {
    std::ifstream f(cfg_path());
    Config c;
    for (std::string line; std::getline(f, line);) {
        size_t eq = line.find('=');
        if (eq == std::string::npos || line[0] == '#') continue;
        std::string k = line.substr(0, eq), v = line.substr(eq + 1);
        if (k == "sync") c.sync = v != "0";
        else if (k == "mobile_data") c.cellular = v != "0";
        else if (k == "endpoint") c.endpoint = v;
        else if (k == "bucket") c.bucket = v;
        else if (k == "folder") c.folder = v;
        else if (k == "key_id") c.key_id = v;
        else if (k == "device") c.device = v;
        else if (k == "device_id") c.device_id = v;
    }
    std::lock_guard<std::mutex> l(g_cfg_mx);
    g_cfg = c;
}
static void cfg_save() {
    Config c;
    {
        std::lock_guard<std::mutex> l(g_cfg_mx);
        c = g_cfg;
    }
    std::error_code ec;
    fs::create_directories(g_save_base, ec);
    std::ofstream f(cfg_path());
    f << "# cloud saves, for every game on this device (the in-game menu's Saves page)\n"
      << "sync=" << c.sync << "\nmobile_data=" << c.cellular << "\nendpoint=" << c.endpoint << "\nbucket=" << c.bucket
      << "\nfolder=" << c.folder << "\nkey_id=" << c.key_id << "\ndevice=" << c.device << "\ndevice_id=" << c.device_id << "\n";
}
static Config cfg() {
    std::lock_guard<std::mutex> l(g_cfg_mx);
    return g_cfg;
}

// ------------------------------------------------------------------ the bucket
struct Conn { std::string scheme = "https", host, region, bucket, prefix, key_id, secret; };

static std::string pct_decode(const std::string &s) {
    std::string out;
    for (size_t i = 0; i < s.size(); i++) {
        if (s[i] == '%' && i + 2 < s.size() && isxdigit((unsigned char)s[i + 1]) && isxdigit((unsigned char)s[i + 2]))
            out += (char)strtol(s.substr(i + 1, 2).c_str(), nullptr, 16), i += 2;
        else out += s[i];
    }
    return out;
}
// "https://s3.eu-central-003.backblazeb2.com" (or without https://): the host, and the region in its name
static void parse_endpoint(std::string e, Conn *c) {
    c->scheme = "https";
    if (!e.compare(0, 8, "https://")) e = e.substr(8);
    else if (!e.compare(0, 7, "http://")) e = e.substr(7), c->scheme = "http";
    while (!e.empty() && e.back() == '/') e.pop_back();
    c->host = e;
    c->region = "us-east-1";  // (MinIO's; Cloudflare R2 takes any)
    if (!e.compare(0, 3, "s3.")) {  // s3.<region>.backblazeb2.com, s3.<region>.amazonaws.com
        size_t dot = e.find('.', 3);
        if (dot != std::string::npos && e.find('.', dot + 1) != std::string::npos) c->region = e.substr(3, dot - 3);
    }
}
static bool parse_conn(std::string s, Conn *c, std::string *why) {
    s = s3::trim(s);
    while (!s.empty() && isspace((unsigned char)s.back())) s.pop_back();
    std::string scheme = "https";
    if (!s.compare(0, 5, "s3://")) s = s.substr(5);
    else if (!s.compare(0, 10, "s3+http://")) s = s.substr(10), scheme = "http";  // (a server on the local network)
    else { *why = "it should start with s3://"; return false; }
    size_t at = s.rfind('@'), colon = s.find(':');
    if (at == std::string::npos || colon == std::string::npos || colon > at) {
        *why = "it should be s3://<key ID>:<application key>@<endpoint>/<bucket>";
        return false;
    }
    std::string where = s.substr(at + 1);
    size_t slash = where.find('/');
    if (slash == std::string::npos || slash + 1 >= where.size()) { *why = "the bucket's name is missing after the endpoint"; return false; }
    parse_endpoint(scheme + "://" + where.substr(0, slash), c);
    std::string path = where.substr(slash + 1);
    while (!path.empty() && path.back() == '/') path.pop_back();
    size_t s2 = path.find('/');
    c->bucket = path.substr(0, s2);
    c->prefix = s2 == std::string::npos ? "" : path.substr(s2 + 1);
    c->key_id = pct_decode(s.substr(0, colon));
    c->secret = pct_decode(s.substr(colon + 1, at - colon - 1));
    if (c->host.empty() || c->bucket.empty() || c->key_id.empty() || c->secret.empty()) { *why = "a part of it is empty"; return false; }
    return true;
}

// The application key's secret: the Keychain on iOS (this app's own, not synced to iCloud), else an owner-only file
#if TARGET_OS_IPHONE
static std::string secret_get(const std::string &kid) { return cloud_keychain_get(kid); }
static void secret_clear(const std::string &kid) { cloud_keychain_clear(kid); }
static bool secret_set(const std::string &kid, const std::string &secret) { return cloud_keychain_set(kid, secret); }
#else
static std::string secret_get(const std::string &kid) {  // "<key ID> <secret>"
    std::ifstream f(secret_path());
    std::string id, secret;
    f >> id >> secret;
    return id == kid ? secret : "";
}
static void secret_clear(const std::string &kid) { (void)kid; unlink(secret_path().c_str()); }
static bool secret_set(const std::string &kid, const std::string &secret) {
    std::string tmp = secret_path() + ".tmp";
    unlink(tmp.c_str());
    int fd = open(tmp.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);  // (only this user can read it)
    FILE *f = fd >= 0 ? fdopen(fd, "w") : nullptr;
    if (!f) return false;
    bool ok = fprintf(f, "%s %s\n", kid.c_str(), secret.c_str()) > 0;
    ok = !fclose(f) && ok;
    return ok && !rename(tmp.c_str(), secret_path().c_str());
}
#endif

static std::mutex g_secret_mx;  // (the secret, read once per key: forgotten when the key changes)
static std::string g_secret_id, g_secret;
static void secret_forget() {
    std::lock_guard<std::mutex> l(g_secret_mx);
    g_secret_id.clear(), g_secret.clear();
}
static bool conn(Conn *c) {
    const char *e = getenv("RECOMP_CLOUD");
    std::string why;
    if (e && *e) return parse_conn(e, c, &why);
    Config k = cfg();
    if (k.endpoint.empty() || k.bucket.empty() || k.key_id.empty()) return false;
    parse_endpoint(k.endpoint, c);
    c->bucket = k.bucket, c->prefix = k.folder, c->key_id = k.key_id;
    std::lock_guard<std::mutex> l(g_secret_mx);
    if (g_secret_id != c->key_id) g_secret = secret_get(c->key_id), g_secret_id = c->key_id;
    c->secret = g_secret;
    return !c->secret.empty();
}

// ------------------------------------------------------------------ HTTPS (cloud_http.h)
using Resp = CloudResp;
using Progress = CloudProgress;

static Resp http_send(const std::string &method, const std::string &url, const s3::Headers &headers, const std::string &body,
                      double timeout, const Progress &progress) {
    return cloud_http_send(method, url, headers, body, timeout, cfg().cellular, progress);
}

// A request to the bucket (key: an object in it, "" the bucket itself), signed. Its error: the network's, or S3's.
static Resp request(const Conn &c, const std::string &method, const std::string &key,
                    const std::vector<std::pair<std::string, std::string>> &query, const std::string &body, s3::Headers extra,
                    double timeout, const Progress &progress = nullptr) {
    std::string path = "/" + c.bucket + (key.empty() ? "" : "/" + key);
    s3::Headers h = {{"host", c.host}};
    h.insert(h.end(), extra.begin(), extra.end());
    std::string qs = s3::sign(method, path, query, h, s3::sha256_hex(body), c.region, c.key_id, c.secret, time(nullptr));
    std::string url = c.scheme + "://" + c.host + s3::encode(path, true) + (qs.empty() ? "" : "?" + qs);
    Resp r = http_send(method, url, h, body, timeout, progress);
    if (r.error.empty() && r.status / 100 != 2) {
        std::string code = s3::xml_one(r.body, "Code"), msg = s3::xml_one(r.body, "Message");
        r.error = code.empty() ? "the server answered " + std::to_string(r.status) : code + (msg.empty() ? "" : ": " + msg);
        if (r.status == 403 && code.empty()) r.error = "access denied (403): check the key and its bucket";
        if (r.status == 404 && method == "GET" && key.empty()) r.error = "there is no bucket named " + c.bucket;
    }
    return r;
}

// ------------------------------------------------------------------ the save as a tar
static const char *const kParts[] = {"savedata", "extdata"};  // (as settings.cpp's)
struct Snap {
    std::string root, game, build, tar, hash;
    time_t newest = 0;  // its newest file's time
    bool any = false;   // a save at all (a file in it)
};
struct Entry { std::string name; fs::path path; bool dir; };

static bool is_part(const std::string &n) { return n == kParts[0] || n == kParts[1]; }
static void collect(const fs::path &dir, const std::string &rel, std::vector<Entry> &out, Snap *s) {
    std::vector<std::string> names;
    std::error_code ec;
    for (auto &d : fs::directory_iterator(dir, ec)) names.push_back(d.path().filename().string());
    std::sort(names.begin(), names.end());
    for (auto &n : names) {
        fs::path p = dir / n;
        struct stat st;
        if (stat(p.c_str(), &st)) continue;
        if (S_ISDIR(st.st_mode)) {
            out.push_back({rel + n, p, true});
            collect(p, rel + n + "/", out, s);
        } else if (S_ISREG(st.st_mode)) {
            out.push_back({rel + n, p, false});
            s->any = true;
            if (st.st_mtime > s->newest) s->newest = st.st_mtime;
        }
    }
}
static void tar_add(std::string &t, const std::string &name, const std::string &data, bool dir) {
    char h[512] = {0};
    std::string n = dir ? name + "/" : name, prefix;
    if (n.size() > 100) {  // (ustar: a prefix of up to 155 characters, cut at a '/')
        size_t cut = n.rfind('/', n.size() - 2);
        while (cut != std::string::npos && (cut > 155 || n.size() - cut - 1 > 100)) cut = cut ? n.rfind('/', cut - 1) : std::string::npos;
        if (cut != std::string::npos) prefix = n.substr(0, cut), n = n.substr(cut + 1);
    }
    memcpy(h, n.data(), std::min<size_t>(n.size(), 100));
    snprintf(h + 100, 8, "%07o", dir ? 0755 : 0644);
    snprintf(h + 108, 8, "%07o", 0);
    snprintf(h + 116, 8, "%07o", 0);
    snprintf(h + 124, 12, "%011llo", (unsigned long long)(dir ? 0 : data.size()));
    snprintf(h + 136, 12, "%011o", 0);  // (no times: the same save is the same tar, byte for byte)
    h[156] = dir ? '5' : '0';
    memcpy(h + 257, "ustar", 6);
    memcpy(h + 263, "00", 2);
    memcpy(h + 345, prefix.data(), std::min<size_t>(prefix.size(), 155));
    memset(h + 148, ' ', 8);
    unsigned sum = 0;
    for (int i = 0; i < 512; i++) sum += (unsigned char)h[i];
    snprintf(h + 148, 8, "%06o", sum);
    t.append(h, 512);
    if (dir) return;
    t += data;
    t.append((512 - data.size() % 512) % 512, '\0');
}
static bool read_file(const fs::path &p, std::string *out) {
    FILE *f = fopen(p.c_str(), "rb");
    if (!f) return false;
    out->clear();
    char buf[1 << 16];
    size_t n;
    while ((n = fread(buf, 1, sizeof buf, f)) > 0) out->append(buf, n);
    bool ok = !ferror(f);
    fclose(f);
    return ok;
}
// The save as a tar, read under the save service's lock (never half a write)
static bool snapshot(Snap *s, std::string *why) {
    std::vector<Entry> entries;
    save_lock();
    for (const char *part : kParts) {
        fs::path p = fs::path(s->root) / part;
        std::error_code ec;
        if (!fs::is_directory(p, ec)) continue;
        entries.push_back({part, p, true});
        collect(p, std::string(part) + "/", entries, s);
    }
    s->tar.clear();
    bool ok = true;
    std::string data;
    for (auto &e : entries) {
        if (!e.dir && !read_file(e.path, &data)) { *why = "could not read " + e.path.string() + ": " + strerror(errno); ok = false; break; }
        tar_add(s->tar, e.name, e.dir ? "" : data, e.dir);
    }
    save_unlock();
    if (!ok) return false;
    s->tar.append(1024, '\0');
    s->hash = s3::sha256_hex(s->tar);
    return true;
}
struct TarEntry { std::string name; bool dir; size_t off, size; };
// The entries of a downloaded tar: false (and why) for anything but plain files and folders under savedata/ and
// extdata/
static bool tar_list(const std::string &t, std::vector<TarEntry> *out, std::string *why) {
    for (size_t at = 0; at + 512 <= t.size();) {
        const unsigned char *h = (const unsigned char *)t.data() + at;
        bool zero = true;
        for (int i = 0; i < 512 && zero; i++) zero = !h[i];
        if (zero) return true;
        unsigned sum = 0;
        for (int i = 0; i < 512; i++) sum += i >= 148 && i < 156 ? ' ' : h[i];
        if (strtoul(std::string((const char *)h + 148, 8).c_str(), nullptr, 8) != sum) { *why = "the save is damaged (a bad tar header)"; return false; }
        size_t size = strtoull(std::string((const char *)h + 124, 12).c_str(), nullptr, 8);
        std::string name((const char *)h, strnlen((const char *)h, 100)), prefix((const char *)h + 345, strnlen((const char *)h + 345, 155));
        if (!prefix.empty()) name = prefix + "/" + name;
        char type = (char)h[156];
        bool dir = type == '5';
        while (!name.empty() && name.back() == '/') name.pop_back();
        if (at + 512 + size > t.size()) { *why = "the save is damaged (cut short)"; return false; }
        bool safe = !name.empty() && name[0] != '/' && name.find('\\') == std::string::npos && name.find('\0') == std::string::npos &&
                    is_part(name.substr(0, name.find('/')));
        for (size_t b = 0; safe && b <= name.size();) {
            size_t e = name.find('/', b);
            if (e == std::string::npos) e = name.size();
            std::string part = name.substr(b, e - b);
            safe = !part.empty() && part != "." && part != "..";
            b = e + 1;
        }
        if (!safe || (type != '0' && type != '\0' && !dir)) { *why = "the save has an entry this doesn't take: " + name; return false; }
        out->push_back({name, dir, at + 512, size});
        at += 512 + (size + 511) / 512 * 512;
    }
    return true;
}
static bool tar_extract(const std::string &t, const fs::path &dst, std::string *why) {
    std::vector<TarEntry> list;
    if (!tar_list(t, &list, why)) return false;
    std::error_code ec;
    fs::create_directories(dst, ec);
    for (auto &e : list) {
        fs::path p = dst / e.name;
        if (e.dir) { fs::create_directories(p, ec); continue; }
        fs::create_directories(p.parent_path(), ec);
        FILE *f = fopen(p.c_str(), "wb");
        bool ok = f && fwrite(t.data() + e.off, 1, e.size, f) == e.size && !fflush(f) && !fsync(fileno(f));
        if (f) ok = !fclose(f) && ok;
        if (!ok) { *why = "could not write " + p.string() + ": " + strerror(errno); return false; }
    }
    return true;
}

// ------------------------------------------------------------------ what was last synced (<save root>/.cloud)
struct State { std::string hash; time_t saved = 0; };
static State state_read(const fs::path &dir) {
    State s;
    std::ifstream f(dir / ".cloud");
    for (std::string line; std::getline(f, line);) {
        size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        if (line.substr(0, eq) == "hash") s.hash = line.substr(eq + 1);
        else if (line.substr(0, eq) == "saved") s.saved = (time_t)atoll(line.substr(eq + 1).c_str());
    }
    return s;
}
static void state_write(const fs::path &dir, const State &s) {
    fs::path tmp = dir / ".cloud.tmp";
    FILE *f = fopen(tmp.c_str(), "w");
    if (!f) return;
    fprintf(f, "hash=%s\nsaved=%lld\n", s.hash.c_str(), (long long)s.saved);
    if (!fclose(f)) rename(tmp.c_str(), (dir / ".cloud").c_str());
}
// When the save here was saved: as synced, or (changed since) its newest file's time. No save: when it was deleted
// (cloud_deleted), else never.
static time_t local_saved(const Snap &s) {
    State st = state_read(s.root);
    if (!s.any) return st.hash == "deleted" ? st.saved : 0;
    return st.hash == s.hash ? st.saved : s.newest;
}

// ------------------------------------------------------------------ slots
struct Slot { std::string key, device, device_id, build, hash; time_t saved = 0; unsigned long long size = 0; };

static std::string game_prefix(const Conn &c, const std::string &game) { return (c.prefix.empty() ? "" : c.prefix + "/") + game + "/"; }
static std::string device_id() {
    const char *e = getenv("RECOMP_CLOUD_DEVICE");
    return e && *e ? std::string(e) : cfg().device_id;
}
static std::string device_name() {
    const char *e = getenv("RECOMP_CLOUD_DEVICE");
    return e && *e ? std::string(e) : cfg().device;
}
static std::string slot_key(const Conn &c, const std::string &game) { return game_prefix(c, game) + device_id() + ".3dssave"; }

// The game's slots, each with its metadata (one HEAD each: a few devices)
static bool list_slots(const Conn &c, const std::string &game, std::vector<Slot> *out, std::string *why) {
    Resp r = request(c, "GET", "", {{"list-type", "2"}, {"prefix", game_prefix(c, game)}}, "", {}, 10);
    if (!r.error.empty()) { *why = r.error; return false; }
    out->clear();
    for (auto &item : s3::xml_all(r.body, "Contents")) {
        Slot s;
        s.key = s3::xml_one(item, "Key");
        s.size = strtoull(s3::xml_one(item, "Size").c_str(), nullptr, 10);
        if (s.key.size() < 8 || s.key.compare(s.key.size() - 8, 8, ".3dssave")) continue;
        Resp h = request(c, "HEAD", s.key, {}, "", {}, 10);
        if (!h.error.empty()) { *why = h.error; return false; }
        size_t slash = s.key.rfind('/');
        s.device_id = s.key.substr(slash + 1, s.key.size() - slash - 9);
        s.device = pct_decode(h.headers["x-amz-meta-nx-device"]);
        s.build = h.headers["x-amz-meta-nx-build"];
        s.hash = h.headers["x-amz-meta-nx-hash"];
        s.saved = (time_t)atoll(h.headers["x-amz-meta-nx-saved"].c_str());
        if (s.device.empty()) s.device = s.device_id;
        if (s.hash.size() == 64 && s.saved > 0) out->push_back(s);
    }
    std::sort(out->begin(), out->end(), [](const Slot &a, const Slot &b) { return a.saved > b.saved; });
    return true;
}
static bool download(const Conn &c, const Slot &s, std::string *tar, std::string *why, const Progress &p) {
    Resp r = request(c, "GET", s.key, {}, "", {}, 30, p);
    if (!r.error.empty()) { *why = r.error; return false; }
    *tar = std::move(r.body);
    if (s3::sha256_hex(*tar) != s.hash) { *why = "the save is damaged (its checksum doesn't match)"; return false; }
    std::vector<TarEntry> list;
    return tar_list(*tar, &list, why);
}

// ------------------------------------------------------------------ state shared with the menu and the worker
struct Game { std::string root, key, build; };
static std::mutex g_lock;  // all below
static std::condition_variable g_wake, g_idle;
static Game g_game_now;
static bool g_dirty, g_flush, g_working, g_started;
static double g_dirty_at, g_uploaded_at = -1e9;
static std::deque<std::function<void()>> g_jobs;
static CloudView g_view;
static std::vector<Slot> g_slots;  // (g_view.slots, with what the menu doesn't show)

static double mono() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static std::string fmt_time(time_t t, const char *f = "%a %d %b, %H:%M") {
    char s[64];
    struct tm tm;
    localtime_r(&t, &tm);
    strftime(s, sizeof s, f, &tm);
    return s;
}
static std::string kb(unsigned long long n) {
    char s[32];
    snprintf(s, sizeof s, n >= (1u << 20) ? "%.1f MB" : "%.0f KB", n >= (1u << 20) ? n / 1048576.0 : (n + 1023) / 1024.0);
    return s;
}
static void status(const std::string &s, bool error = false) {  // (g_lock held)
    if (error) g_view.error = s;
    else g_view.status = s, g_view.error.clear();
    LOG("[cloud] %s\n", s.c_str());
}
static void publish_slots(const std::vector<Slot> &slots, const Game &g, const std::string &local_hash) {  // (g_lock held)
    g_slots = slots;
    g_view.slots.clear();
    std::string me = device_id();
    for (auto &s : slots) {
        CloudSlot v;
        v.key = s.key, v.device = s.device, v.saved = s.saved, v.size = s.size;
        v.mine = s.device_id == me;
        v.same_game = s.build == g.build;
        v.current = s.hash == local_hash;
        g_view.slots.push_back(v);
    }
    g_view.listed = time(nullptr);
}

static void upload(const Snap &s) {
    Conn c;
    if (!conn(&c)) return;
    {
        std::lock_guard<std::mutex> l(g_lock);
        g_view.busy = true, g_view.progress = 0;
        status("Uploading the save (" + kb(s.tar.size()) + ")");
    }
    time_t saved = local_saved(s);
    s3::Headers meta = {{"x-amz-meta-nx-device", s3::encode(device_name())},
                        {"x-amz-meta-nx-saved", std::to_string((long long)saved)},
                        {"x-amz-meta-nx-build", s.build},
                        {"x-amz-meta-nx-hash", s.hash}};
    Resp r = request(c, "PUT", slot_key(c, s.game), {}, s.tar, meta, 30, [](double p) {
        std::lock_guard<std::mutex> l(g_lock);
        g_view.progress = (float)p;
    });
    std::lock_guard<std::mutex> l(g_lock);
    g_view.busy = false, g_view.progress = -1;
    if (!r.error.empty()) {
        status("Upload failed: " + r.error + " (trying again in 2 minutes)", true);
        if (s.root == g_game_now.root) g_dirty = true, g_dirty_at = mono() + 120 - 30;
        return;
    }
    state_write(s.root, {s.hash, saved});
    g_uploaded_at = mono();
    status("Uploaded at " + fmt_time(time(nullptr), "%H:%M") + " (" + kb(s.tar.size()) + ")");
    if (s.root != g_game_now.root) return;
    std::vector<Slot> slots = g_slots;  // (this device's slot, as it is now)
    std::string me = device_id();
    slots.erase(std::remove_if(slots.begin(), slots.end(), [&](const Slot &x) { return x.device_id == me; }), slots.end());
    Slot mine;
    mine.key = slot_key(c, s.game), mine.device = device_name(), mine.device_id = me, mine.build = s.build;
    mine.hash = s.hash, mine.saved = saved, mine.size = s.tar.size();
    slots.push_back(mine);
    std::sort(slots.begin(), slots.end(), [](const Slot &a, const Slot &b) { return a.saved > b.saved; });
    publish_slots(slots, g_game_now, s.hash);
}
static Snap snap_of(const Game &g) {
    Snap s;
    s.root = g.root, s.game = g.key, s.build = g.build;
    return s;
}
// The game's save uploaded if it changed since it was last synced (force: changed or not)
static void upload_current(const Game &g, bool force) {
    Snap s = snap_of(g);
    std::string why;
    if (!snapshot(&s, &why)) { std::lock_guard<std::mutex> l(g_lock); status("Upload failed: " + why, true); return; }
    if (!s.any || (!force && state_read(s.root).hash == s.hash)) {
        LOG("[cloud] the save is as last synced: not uploaded\n");
        return;
    }
    upload(s);
}

static void worker() {
    std::unique_lock<std::mutex> l(g_lock);
    for (;;) {
        if (!g_jobs.empty()) {
            auto job = g_jobs.front();
            g_jobs.pop_front();
            g_working = true;
            l.unlock();
            job();
            l.lock();
            g_working = false;
            g_idle.notify_all();
            continue;
        }
        static const double every = getenv("RECOMP_CLOUD_EVERY") ? atof(getenv("RECOMP_CLOUD_EVERY")) : 600;  // (tests)
        double now = mono(), due = std::max(g_dirty_at + 30, g_uploaded_at + every);
        if (g_dirty && (g_flush || now >= due)) {
            g_dirty = g_flush = false;
            Game g = g_game_now;
            g_working = true;
            l.unlock();
            Conn c;
            if (cfg().sync && conn(&c)) upload_current(g, false);
            l.lock();
            g_working = false;
            g_idle.notify_all();
            continue;
        }
        g_flush = false;
        if (g_dirty) g_wake.wait_for(l, std::chrono::duration<double>(due - now));
        else g_wake.wait(l);
    }
}
static void start_worker() {  // (g_lock held)
    if (!g_started) { g_started = true; std::thread(worker).detach(); }
}
static void enqueue(std::function<void()> job) {
    std::lock_guard<std::mutex> l(g_lock);
    start_worker();
    g_jobs.push_back(std::move(job));
    g_wake.notify_all();
}

// ------------------------------------------------------------------ the runtime's calls
static void ensure_device() {
    if (!cfg().device_id.empty()) return;
    std::string name = platform::device_name();
    if (name.empty()) name = "device";
    std::string id;
    for (char ch : name)
        if (isalnum((unsigned char)ch) && id.size() < 24) id += ch;
        else if ((ch == ' ' || ch == '-') && !id.empty() && id.back() != '-') id += '-';
    while (!id.empty() && id.back() == '-') id.pop_back();
    char tail[8];
    snprintf(tail, sizeof tail, "-%04x", (unsigned)(std::random_device{}() & 0xffff));
    {
        std::lock_guard<std::mutex> l(g_cfg_mx);
        g_cfg.device_id = (id.empty() ? "device" : id) + tail;
        if (g_cfg.device.empty()) g_cfg.device = name;
    }
    cfg_save();
}
// Which game files: a hash of code.bin (another version of the game has another)
static std::string game_build() {
    std::string code;
    if (!read_file(g_game.dir + "/code.bin", &code)) return "";
    return s3::sha256_hex(code).substr(0, 16);
}

void cloud_boot() {
    cfg_load();
    Game g;
    g.root = kernel::save_root();
    char tid[32];
    snprintf(tid, sizeof tid, "%016llx", (unsigned long long)g_game.program_id);
    g.key = tid;
    g.build = game_build();
    {
        std::lock_guard<std::mutex> l(g_lock);
        g_game_now = g, g_dirty = g_flush = false;
    }
    Conn c;
    if (!conn(&c)) return;
    ensure_device();
    if (!cfg().sync) return;
    Snap local = snap_of(g);
    std::string why;
    if (!snapshot(&local, &why)) { std::lock_guard<std::mutex> l(g_lock); status(why, true); return; }
    time_t mine = local_saved(local);
    std::vector<Slot> slots;
    if (!list_slots(c, g.key, &slots, &why)) {
        std::lock_guard<std::mutex> l(g_lock);
        status("Couldn't check the cloud saves: " + why, true);
        return;
    }
    const Slot *best = nullptr, *other = nullptr;
    for (auto &s : slots) {
        if (s.hash == local.hash || s.saved <= mine) continue;
        if (s.build == g.build) { if (!best) best = &s; }
        else if (!other) other = &s;
    }
    std::string note;
    if (best) {
        std::string what = "the save from " + best->device + " (" + fmt_time(best->saved) + ")";
        LOG("[cloud] downloading %s\n", what.c_str());
        std::string tar;
        bool ok = download(c, *best, &tar, &why, nullptr);
        if (ok) {  // (into place: the local save backed up first; nothing replaced if that fails)
            fs::path in = fs::path(g.root) / ".cloud-incoming";
            std::error_code ec;
            fs::remove_all(in, ec);
            ok = tar_extract(tar, in, &why);
            if (ok && local.any && !save_backup_now(" before the save from " + best->device))
                ok = false, why = "couldn't back up the save here first";
            if (ok) {
                for (const char *part : kParts) {
                    fs::remove_all(fs::path(g.root) / part, ec);
                    if (fs::exists(in / part, ec)) fs::rename(in / part, fs::path(g.root) / part, ec);
                }
                state_write(g.root, {best->hash, best->saved});
                local.hash = best->hash, local.any = true;
                note = "Using " + what;
            }
            fs::remove_all(in, ec);
        }
        if (!ok) note = "Couldn't use " + what + ": " + why;
    } else if (other)
        note = "The save from " + other->device + " (" + fmt_time(other->saved) + ") is from other game files (another "
               "version of the game): not used";
    bool up = local.any;  // (unless the bucket has it already: this device's upload, or the save just used)
    if (best && note.compare(0, 6, "Using ") == 0) mine = best->saved;
    for (auto &s : slots) if (s.hash == local.hash && s.saved >= mine) up = false;
    {
        std::lock_guard<std::mutex> l(g_lock);
        publish_slots(slots, g, local.hash);
        if (!note.empty()) status(note, note.compare(0, 6, "Using ") != 0);
        else if (!up) status("Up to date (" + fmt_time(time(nullptr), "%H:%M") + ")");
    }
    if (up) enqueue([g] { upload_current(g, true); });  // (a save here that the bucket doesn't have)
}

void cloud_save_written() {
    std::lock_guard<std::mutex> l(g_lock);
    g_dirty = true, g_dirty_at = mono();
    start_worker();
    g_wake.notify_all();
}
void cloud_flush() {
    std::lock_guard<std::mutex> l(g_lock);
    if (!g_dirty) return;
    g_flush = true;
    g_wake.notify_all();
}
void cloud_quit() {
    Game g;
    {
        std::lock_guard<std::mutex> l(g_lock);
        if (g_dirty) g = g_game_now;
        g_dirty = false;
    }
    Conn c;
    if (!g.root.empty() && cfg().sync && conn(&c)) {  // (the save as the game left it)
        Snap s = snap_of(g);
        std::string why;
        if (snapshot(&s, &why) && s.any && state_read(s.root).hash != s.hash) enqueue([s] { upload(s); });
    }
    std::unique_lock<std::mutex> l(g_lock);
    g_idle.wait_for(l, std::chrono::seconds(20), [] { return g_jobs.empty() && !g_working; });
}
void cloud_restored() {
    Game g;
    {
        std::lock_guard<std::mutex> l(g_lock);
        g = g_game_now, g_dirty = false;  // (not the save that was there: the one restored, uploaded at the next launch)
    }
    Snap s = snap_of(g);
    std::string why;
    if (snapshot(&s, &why) && s.any) state_write(g.root, {s.hash, time(nullptr)});
}
void cloud_deleted() {
    std::lock_guard<std::mutex> l(g_lock);
    g_dirty = false;
    if (!g_game_now.root.empty()) state_write(g_game_now.root, {"deleted", time(nullptr)});
}

// ------------------------------------------------------------------ the menu's
CloudView cloud_view() {
    std::lock_guard<std::mutex> l(g_lock);
    CloudView v = g_view;
    Conn c;
    Config k = cfg();
    v.configured = conn(&c);
    v.where = c.bucket + (c.prefix.empty() ? "" : "/" + c.prefix) + " at " + c.host;
    v.device = getenv("RECOMP_CLOUD_DEVICE") ? getenv("RECOMP_CLOUD_DEVICE") : k.device;
    v.sync = k.sync;
    v.busy = v.busy || g_working || !g_jobs.empty();
    g_view.fetched.clear();
    return v;
}
static void refresh_job(bool upload_missing) {
    Conn c;
    Game g;
    {
        std::lock_guard<std::mutex> l(g_lock);
        g = g_game_now;
        if (!g.key.empty()) status("Checking the cloud saves");
    }
    if (g.key.empty() || !conn(&c)) return;
    Snap local = snap_of(g);
    std::string why;
    std::vector<Slot> slots;
    bool ok = snapshot(&local, &why) && list_slots(c, g.key, &slots, &why);
    bool up = upload_missing && local.any;
    for (auto &s : slots) if (s.hash == local.hash && s.saved >= local_saved(local)) up = false;
    {
        std::lock_guard<std::mutex> l(g_lock);
        if (!ok) { status("Couldn't check the cloud saves: " + why, true); return; }
        publish_slots(slots, g, local.hash);
        status("Checked at " + fmt_time(time(nullptr), "%H:%M"));
    }
    if (up) upload(local);
}
bool cloud_connect(const std::string &connection, std::string *why) {
    Conn c;
    if (!parse_conn(connection, &c, why)) return false;
    std::error_code ec;
    fs::create_directories(g_save_base, ec);
    if (!secret_set(c.key_id, c.secret)) { *why = "couldn't store the key on this device"; return false; }
    secret_forget();
    {
        std::lock_guard<std::mutex> l(g_cfg_mx);
        g_cfg.endpoint = c.scheme + "://" + c.host, g_cfg.bucket = c.bucket, g_cfg.folder = c.prefix, g_cfg.key_id = c.key_id;
        g_cfg.sync = true;
    }
    cfg_save();
    ensure_device();
    enqueue([] { refresh_job(true); });
    return true;
}
void cloud_disconnect() {
    Config k = cfg();
    if (!k.key_id.empty()) secret_clear(k.key_id);
    secret_forget();
    {
        std::lock_guard<std::mutex> l(g_cfg_mx);
        g_cfg.endpoint.clear(), g_cfg.bucket.clear(), g_cfg.folder.clear(), g_cfg.key_id.clear();
    }
    cfg_save();
    std::lock_guard<std::mutex> l(g_lock);
    g_slots.clear();
    g_view = CloudView();
}
void cloud_set_sync(bool on) {
    {
        std::lock_guard<std::mutex> l(g_cfg_mx);
        g_cfg.sync = on;
    }
    cfg_save();
}
void cloud_refresh() { enqueue([] { refresh_job(false); }); }
void cloud_upload_now() {
    enqueue([] {
        Game g;
        {
            std::lock_guard<std::mutex> l(g_lock);
            g = g_game_now, g_dirty = false;
        }
        if (!g.key.empty()) upload_current(g, true);
    });
}
void cloud_fetch(const std::string &key) {
    enqueue([key] {
        Conn c;
        Slot s;
        Game g;
        {
            std::lock_guard<std::mutex> l(g_lock);
            g = g_game_now;
            for (auto &x : g_slots) if (x.key == key) s = x;
            g_view.busy = true, g_view.progress = 0;
            status("Downloading the save from " + s.device);
        }
        std::string tar, why;
        bool ok = !s.key.empty() && conn(&c) && download(c, s, &tar, &why, [](double p) {
            std::lock_guard<std::mutex> l(g_lock);
            g_view.progress = (float)p;
        });
        fs::path dst;
        if (ok) {  // (a backup: the menu restores it, as any other, with a restart)
            std::string name = fmt_time(time(nullptr), "%Y-%m-%d %H.%M.%S") + " from " + s.device;
            for (char &ch : name) if (ch == '/' || ch == ':' || ch == '\\') ch = '-';
            dst = fs::path(g.root) / "backups" / name;
            ok = tar_extract(tar, dst, &why);
            if (!ok) { std::error_code ec; fs::remove_all(dst, ec); }
        }
        std::lock_guard<std::mutex> l(g_lock);
        g_view.busy = false, g_view.progress = -1;
        if (!ok) { status("Download failed: " + (why.empty() ? std::string("the save is gone") : why), true); return; }
        status("Downloaded the save from " + s.device);
        g_view.fetched = dst.string();
    });
}
