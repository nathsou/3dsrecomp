// fs:USER: RomFS from romfs.bin, save data / extdata / sdmc in host folders.
#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <filesystem>
#include <functional>
#include <map>
#include <mutex>

#include "../loader.h"
#include "../service.h"
#include "services.h"

namespace fsys = std::filesystem;

namespace kernel {

static u32 fs_err(u32 desc, u32 summary, u32 level = 25) { return (level << 27) | (summary << 21) | (17 << 10) | desc; }
static const u32 ERR_FILE_NOT_FOUND = fs_err(112, 4);
static const u32 ERR_PATH_NOT_FOUND = fs_err(113, 4);
static const u32 ERR_NOT_FOUND = fs_err(120, 4);
static const u32 ERR_NOT_FORMATTED = fs_err(340, 5);
static const u32 ERR_FILE_EXISTS = fs_err(180, 1);
static const u32 ERR_DIR_EXISTS = fs_err(185, 1);
static const u32 ERR_ARCHIVE = fs_err(101, 4);
static const u32 ERR_UNSUPPORTED = fs_err(760, 6, 28);

}  // namespace kernel
extern std::string g_save_base;
namespace kernel {

std::string save_root() {
    char tid[32];
    snprintf(tid, sizeof tid, "%016llx", (unsigned long long)g_game.program_id);
    return g_save_base + "/" + std::string(tid);
}

// ---- paths
struct LowPath {
    u32 type = 1;
    std::vector<u8> data;
    std::string str() const {
        if (type == 3) return std::string(data.begin(), std::find(data.begin(), data.end(), 0));
        if (type == 4) {
            std::string s;
            for (size_t i = 0; i + 1 < data.size(); i += 2) {
                u16 ch = data[i] | (data[i + 1] << 8);
                if (!ch) break;
                if (ch < 0x80) s += (char)ch;
                else if (ch < 0x800) {
                    s += (char)(0xC0 | (ch >> 6));
                    s += (char)(0x80 | (ch & 0x3F));
                } else {
                    s += (char)(0xE0 | (ch >> 12));
                    s += (char)(0x80 | ((ch >> 6) & 0x3F));
                    s += (char)(0x80 | (ch & 0x3F));
                }
            }
            return s;
        }
        if (type == 1) return "";
        std::string s;  // binary: hex
        char b[4];
        for (u8 c : data) {
            snprintf(b, sizeof b, "%02x", c);
            s += b;
        }
        return s;
    }
    u32 word(int i) const {
        u32 v = 0;
        if ((size_t)(4 * i + 4) <= data.size()) memcpy(&v, &data[4 * i], 4);
        return v;
    }
};

static LowPath read_path(u32 type, u32 size, u32 ptr) {
    LowPath p;
    p.type = type;
    if (size && ptr) p.data.assign(gptr(ptr), gptr(ptr) + size);
    return p;
}

static std::string sanitize(const std::string &s) {
    // resolve . and .. (homebrew passes paths like /3ds/../rom.nes), never leaving the archive folder
    std::vector<std::string> parts;
    for (const auto &part : fsys::path(s)) {
        std::string q = part.string();
        if (q == "/" || q == "." || q.empty()) continue;
        if (q == "..") {
            if (!parts.empty()) parts.pop_back();
            continue;
        }
        parts.push_back(q);
    }
    std::string out;
    for (auto &q : parts) out += "/" + q;
    return out;
}

// ---- archives
struct Archive {
    u32 id;
    std::string host;  // host folder for directory-backed archives, or the RomFS image of a system archive
    bool romfs = false;
};

// System archives (NCCH archive 0x2345678A) come from system_archives/<title id>.romfs, next to the game files
// in the working directory or in the source tree (tools/import_system_archives.py makes them).
static std::string system_archive_path(u64 tid) {
    char name[64];
    snprintf(name, sizeof name, "%016llx.romfs", (unsigned long long)tid);
    for (std::string dir : {g_game.dir + "/../system_archives/", g_game.dir + "/system_archives/", std::string("system_archives/")
#ifdef RECOMP_SOURCE_DIR
                            , std::string(RECOMP_SOURCE_DIR "/system_archives/")
#endif
         }) {
        if (fsys::exists(dir + name)) return dir + name;
    }
    return "";
}

std::vector<u8> system_archive_file(u64 tid, const std::string &want) {
    std::string host = system_archive_path(tid);
    FILE *f = host.empty() ? nullptr : fopen(host.c_str(), "rb");
    if (!f) return {};
    std::vector<u8> d;
    u8 buf[65536];
    for (size_t n; (n = fread(buf, 1, sizeof buf, f)) > 0;) d.insert(d.end(), buf, buf + n);
    fclose(f);
    size_t base = d.size() >= 4 && !memcmp(d.data(), "IVFC", 4) ? 0x1000 : 0;  // level 3 image
    auto u = [&](size_t o) { u32 v = 0; if (base + o + 4 <= d.size()) memcpy(&v, &d[base + o], 4); return v; };
    u32 dmo = u(0x0C), fmo = u(0x1C), dato = u(0x24);
    auto name = [&](size_t o, u32 len) {
        std::string s;
        for (u32 i = 0; i + 1 < len && base + o + i < d.size(); i += 2) s += (char)d[base + o + i];
        return s;
    };
    std::vector<u8> out;
    std::function<void(u32, std::string)> walk = [&](u32 dir, std::string path) {
        for (u32 fe = u(dmo + dir + 12); fe != 0xFFFFFFFF && out.empty(); fe = u(fmo + fe + 4)) {
            if (path + "/" + name(fmo + fe + 32, u(fmo + fe + 28)) != want) continue;
            u64 off = base + dato + ((u64)u(fmo + fe + 8) | (u64)u(fmo + fe + 12) << 32);
            u64 size = (u64)u(fmo + fe + 16) | (u64)u(fmo + fe + 20) << 32;
            if (off + size <= d.size()) out.assign(d.begin() + off, d.begin() + off + size);
        }
        for (u32 c = u(dmo + dir + 8); c != 0xFFFFFFFF && out.empty(); c = u(dmo + c + 4))
            walk(c, path + "/" + name(dmo + c + 24, u(dmo + c + 20)));
    };
    walk(0, "");
    return out;
}

static std::map<u64, Archive> archives;
static u64 next_archive = 1;

static int romfs_fd = -1;
static u64 romfs_size;

// RomFS index, for logging which files the game reads
struct RomfsFile {
    u64 off, size;
    std::string path;
};
static std::vector<RomfsFile> romfs_index;
// RECOMP_ROMFS_SWAP="A=B,...": every RomFS file whose name contains A gets the data of the file named with B
// instead (by rewriting its metadata entry as the game reads it), e.g. FirstStage=CaveInkStage to load another
// level in place of World 1-1 when testing.
struct RomfsPatch { u64 at; u8 bytes[16]; };
static std::vector<RomfsPatch> romfs_patches;
// RECOMP_ROMFS_REPLACE="<path>=<host file>,...": serve a host file in place of a RomFS file (for mods and tests).
// Its data is placed past the end of the image, where reads are redirected to the host file.
struct RomfsExtra { u64 off, size; int fd; };
static std::vector<RomfsExtra> romfs_extra;

static void build_romfs_index() {
    std::vector<u8> hdr(0x28);
    if (pread(romfs_fd, hdr.data(), 0x28, 0) != 0x28) return;
    auto u32at = [](const std::vector<u8> &b, size_t o) { u32 v; memcpy(&v, &b[o], 4); return v; };
    u32 dmeta_off = u32at(hdr, 0x0C), dmeta_len = u32at(hdr, 0x10);
    u32 fmeta_off = u32at(hdr, 0x1C), fmeta_len = u32at(hdr, 0x20);
    u32 data_off = u32at(hdr, 0x24);
    std::vector<u8> dm(dmeta_len), fm(fmeta_len);
    if (pread(romfs_fd, dm.data(), dmeta_len, dmeta_off) != (ssize_t)dmeta_len) return;
    if (pread(romfs_fd, fm.data(), fmeta_len, fmeta_off) != (ssize_t)fmeta_len) return;
    auto name = [](const std::vector<u8> &b, size_t o, u32 len) {
        std::string s;
        for (u32 i = 0; i + 1 < len; i += 2) s += (char)b[o + i];
        return s;
    };
    std::function<void(u32, std::string)> walk = [&](u32 d, std::string path) {
        u32 child = u32at(dm, d + 8), file = u32at(dm, d + 12);
        while (file != 0xFFFFFFFF) {
            u64 fo, fs;
            memcpy(&fo, &fm[file + 8], 8);
            memcpy(&fs, &fm[file + 16], 8);
            romfs_index.push_back({data_off + fo, fs, path + "/" + name(fm, file + 32, u32at(fm, file + 28))});
            file = u32at(fm, file + 4);
        }
        while (child != 0xFFFFFFFF) {
            walk(child, path + "/" + name(dm, child + 24, u32at(dm, child + 20)));
            child = u32at(dm, child + 4);
        }
    };
    walk(0, "");
    if (const char *rp = getenv("RECOMP_ROMFS_REPLACE")) {
        std::map<std::string, u32> entry;
        std::function<void(u32, std::string)> walk2 = [&](u32 d, std::string path) {
            for (u32 f = u32at(dm, d + 12); f != 0xFFFFFFFF; f = u32at(fm, f + 4))
                entry[path + "/" + name(fm, f + 32, u32at(fm, f + 28))] = f;
            for (u32 c = u32at(dm, d + 8); c != 0xFFFFFFFF; c = u32at(dm, c + 4))
                walk2(c, path + "/" + name(dm, c + 24, u32at(dm, c + 20)));
        };
        walk2(0, "");
        u64 end = (romfs_size + 0xFFF) & ~0xFFFull;
        std::string spec = rp;
        for (size_t pos = 0; pos < spec.size();) {
            size_t e = spec.find(',', pos);
            std::string item = spec.substr(pos, e == std::string::npos ? std::string::npos : e - pos);
            pos = e == std::string::npos ? spec.size() : e + 1;
            size_t eq = item.find('=');
            auto it = eq == std::string::npos ? entry.end() : entry.find(item.substr(0, eq));
            int fd = eq == std::string::npos ? -1 : open(item.substr(eq + 1).c_str(), O_RDONLY);
            if (it == entry.end() || fd < 0) {
                LOG("[fs] romfs replace: cannot use '%s'\n", item.c_str());
                continue;
            }
            struct stat st;
            fstat(fd, &st);
            romfs_extra.push_back({end, (u64)st.st_size, fd});
            RomfsPatch p;
            p.at = fmeta_off + it->second + 8;
            u64 fo = end - data_off, sz = (u64)st.st_size;
            memcpy(p.bytes, &fo, 8);
            memcpy(p.bytes + 8, &sz, 8);
            romfs_patches.push_back(p);
            LOG("[fs] romfs replace: %s <- %s\n", it->first.c_str(), item.substr(eq + 1).c_str());
            end = (end + sz + 0xFFF) & ~0xFFFull;
        }
        romfs_size = end;
    }
    if (const char *sw = getenv("RECOMP_ROMFS_SWAP")) {
        std::map<std::string, u32> entry;  // path -> offset of its file metadata entry
        std::function<void(u32, std::string)> walk2 = [&](u32 d, std::string path) {
            for (u32 f = u32at(dm, d + 12); f != 0xFFFFFFFF; f = u32at(fm, f + 4))
                entry[path + "/" + name(fm, f + 32, u32at(fm, f + 28))] = f;
            for (u32 c = u32at(dm, d + 8); c != 0xFFFFFFFF; c = u32at(dm, c + 4))
                walk2(c, path + "/" + name(dm, c + 24, u32at(dm, c + 20)));
        };
        walk2(0, "");
        std::string spec = sw;
        for (size_t pos = 0; pos < spec.size();) {
            size_t e = spec.find(',', pos);
            std::string item = spec.substr(pos, e == std::string::npos ? std::string::npos : e - pos);
            pos = e == std::string::npos ? spec.size() : e + 1;
            size_t eq = item.find('=');
            if (eq == std::string::npos) continue;
            std::string from = item.substr(0, eq), to = item.substr(eq + 1);
            for (auto &[path, f] : entry) {
                size_t at = path.find(from);
                if (at == std::string::npos) continue;
                auto it = entry.find(path.substr(0, at) + to + path.substr(at + from.size()));
                if (it == entry.end()) continue;
                RomfsPatch p;
                p.at = fmeta_off + f + 8;
                memcpy(p.bytes, &fm[it->second + 8], 16);
                romfs_patches.push_back(p);
                LOG("[fs] romfs swap: %s -> %s\n", path.c_str(), it->first.c_str());
            }
        }
    }
    std::sort(romfs_index.begin(), romfs_index.end(), [](auto &a, auto &b) { return a.off < b.off; });
}

// a RomFS file by path ("/LayoutData/x.szs"), empty if missing (host side: HUD icons)
static std::mutex romfs_mtx;
static bool open_romfs() {
    std::lock_guard<std::mutex> lk(romfs_mtx);
    if (romfs_fd >= 0) return true;
    const int fd = open((g_game.dir + "/romfs.bin").c_str(), O_RDONLY);
    if (fd < 0) return false;
    struct stat st;
    fstat(fd, &st);
    romfs_size = (u64)st.st_size;
    romfs_fd = fd;
    build_romfs_index();
    return true;
}

std::vector<u8> romfs_file(const std::string &path) {
    if (!open_romfs()) return {};
    std::lock_guard<std::mutex> lk(romfs_mtx);
    for (const auto &f : romfs_index)
        if (f.path == path) {
            std::vector<u8> d(f.size);
            if (pread(romfs_fd, d.data(), f.size, (off_t)f.off) != (ssize_t)f.size) return {};
            return d;
        }
    return {};
}

static void log_romfs_read(u64 off) {
    if (!g_verbose) return;
    auto it = std::upper_bound(romfs_index.begin(), romfs_index.end(), off,
                               [](u64 v, const RomfsFile &f) { return v < f.off; });
    if (it == romfs_index.begin()) return;
    --it;
    static RomfsFile *last;
    if (off < it->off + it->size && &*it != last) {
        LOG("[fs] romfs: %s%s\n", it->path.c_str(), off == it->off ? "" : " (inside)");
        last = &*it;
    }
}

static std::string archive_dir(u32 id, const LowPath &p) {
    std::string root = save_root();
    switch (id) {
    case 4: return root + "/savedata";
    case 6:
    case 7: {
        char b[64];
        snprintf(b, sizeof b, "/extdata/%08x%08x", p.word(2), p.word(1));
        return (id == 7 ? g_save_base + "/shared" : root) + b;
    }
    case 8: {
        char b[64];
        snprintf(b, sizeof b, "/system/%08x%08x", p.word(1), p.word(0));
        return g_save_base + b;
    }
    case 9: return g_save_base + "/sdmc";
    }
    return "";
}

// ---- file and directory sessions
struct FileSession : Service {
    int fd = -1;
    bool romfs = false;
    u64 base = 0, size = 0;
    std::string path;
    FileSession() : Service("fs file") {}
    ~FileSession() override {
        if (fd >= 0 && !romfs) close(fd);
    }
    u64 file_size() {
        if (romfs) return size;
        struct stat st;
        fstat(fd, &st);
        return (u64)st.st_size;
    }
    void handle(IPC &ipc) override {
        switch (ipc.cmd()) {
        case 0x0802: {  // Read(offset u64, size, mapped buffer)
            u64 off = (u64)ipc[1] | ((u64)ipc[2] << 32);
            u32 len = ipc[3];
            u32 dst = ipc[5];
            s64 n;
            if (romfs) {
                log_romfs_read(base + off);
                if (off >= size) n = 0;
                else n = pread(romfs_fd, gptr(dst), (size_t)std::min<u64>(len, size - off), (off_t)(base + off));
                for (auto &x : romfs_extra)
                    if (base + off >= x.off && base + off < x.off + x.size)
                        n = pread(x.fd, gptr(dst), (size_t)std::min<u64>(len, x.off + x.size - base - off),
                                  (off_t)(base + off - x.off));
                for (auto &p : romfs_patches)
                    for (u32 i = 0; i < 16; i++)
                        if (p.at + i >= base + off && p.at + i < base + off + (u64)std::max<s64>(n, 0))
                            w8(dst + (u32)(p.at + i - base - off), p.bytes[i]);
            } else {
                n = pread(fd, gptr(dst), len, (off_t)off);
            }
            if (n < 0) n = 0;
            if (g_verbose >= 2) LOG("[fs] read %s off %llx len %x -> %lld\n", romfs ? "romfs" : path.c_str(), (unsigned long long)off, len, (long long)n);
            ipc.ok(2, 2);
            ipc.set(2, (u32)n);
            ipc.set(3, (len << 4) | 0xC);
            ipc.set(4, dst);
            break;
        }
        case 0x0803: {  // Write(offset u64, size, flush, mapped buffer)
            u64 off = (u64)ipc[1] | ((u64)ipc[2] << 32);
            u32 len = ipc[3];
            u32 src = ipc[6];
            s64 n = romfs ? 0 : pwrite(fd, gptr(src), len, (off_t)off);
            if (n < 0) n = 0;
            ipc.ok(2, 2);
            ipc.set(2, (u32)n);
            ipc.set(3, (len << 4) | 0xA);
            ipc.set(4, src);
            break;
        }
        case 0x0804:  // GetSize
            ipc.ok(3);
            ipc.set64(2, file_size());
            break;
        case 0x0805:  // SetSize
            if (!romfs) {
                if (ftruncate(fd, (off_t)((u64)ipc[1] | ((u64)ipc[2] << 32)))) {
                }
            }
            ipc.ok();
            break;
        case 0x0808:  // Close
        case 0x0809:  // Flush
        case 0x080A:  // SetPriority
            ipc.ok();
            break;
        case 0x080B:
            ipc.ok(2);
            ipc.set(2, 0);
            break;
        case 0x080C:  // OpenLinkFile
            ipc.ok(1, 2);
            ipc.handles(2, {get_object_for(shared_from_this())}, true);
            break;
        default:
            unimplemented(ipc);
        }
    }
    static ObjPtr get_object_for(std::shared_ptr<Service> s) {
        auto ses = std::make_shared<Session>();
        ses->service = s;
        return ses;
    }
};

struct DirSession : Service {
    std::vector<fsys::directory_entry> entries;
    size_t pos = 0;
    DirSession() : Service("fs dir") {}
    void handle(IPC &ipc) override {
        switch (ipc.cmd()) {
        case 0x0801: {  // Read(count, mapped buffer)
            u32 count = ipc[1];
            u32 dst = ipc[3];
            u32 n = 0;
            while (n < count && pos < entries.size()) {
                auto &e = entries[pos++];
                u32 a = dst + n * 0x228;
                memset(gptr(a), 0, 0x228);
                std::string name = e.path().filename().string();
                for (size_t i = 0; i < name.size() && i < 0x105; i++) w16(a + 2 * (u32)i, (u8)name[i]);
                std::string stem = e.path().stem().string(), ext = e.path().extension().string();
                for (size_t i = 0; i < 8 && i < stem.size(); i++) w8(a + 0x20C + (u32)i, (u8)toupper(stem[i]));
                for (size_t i = 1; i < 4 && i < ext.size(); i++) w8(a + 0x216 + (u32)i - 1, (u8)toupper(ext[i]));
                bool dir = e.is_directory();
                w8(a + 0x21C, dir);
                w8(a + 0x21E, !dir);
                w64(a + 0x220, dir ? 0 : (u64)e.file_size());
                n++;
            }
            ipc.ok(2, 2);
            ipc.set(2, n);
            ipc.set(3, ((count * 0x228) << 4) | 0xC);
            ipc.set(4, dst);
            break;
        }
        case 0x0802:
            ipc.ok();
            break;
        default:
            unimplemented(ipc);
        }
    }
};

// Shared extdata F000000B always exists on a console (made at system setup): it holds gamecoin.dat, the
// Play Coin counter, which some games read (or fail) without creating it.
static void create_gamecoin(const std::string &dir) {
    fsys::create_directories(dir);
    struct { u32 magic; u16 coins, coins_today; u32 steps, last_steps; u16 year; u8 month, day; } gc{0x4F00, 10, 0, 0, 0, 2014, 1, 1};
    if (FILE *f = fopen((dir + "/gamecoin.dat").c_str(), "wb")) {
        fwrite(&gc, sizeof gc, 1, f);
        fclose(f);
    }
    LOG("[fs] created shared extdata F000000B with gamecoin.dat (10 Play Coins)\n");
}

// ---- fs:USER
struct FsUser : Service {
    using Service::Service;

    u32 open_archive(u32 id, const LowPath &p, u64 *out) {
        Archive a{id, "", false};
        if (id == 3) {
            a.romfs = true;
        } else if (id == 0x2345678A) {
            u64 tid = (u64)p.word(0) | ((u64)p.word(1) << 32);
            a.host = system_archive_path(tid);
            if (a.host.empty()) {
                VLOG("[fs] system archive %016llx not available\n", (unsigned long long)tid);
                return ERR_NOT_FOUND;
            }
        } else {
            a.host = archive_dir(id, p);
            if (a.host.empty()) {
                LOG("[fs] OpenArchive: unsupported archive %x path %s\n", id, p.str().c_str());
                return ERR_NOT_FOUND;
            }
            if (id == 9) fsys::create_directories(a.host);
            if (id == 7 && p.word(1) == 0xF000000B && !fsys::exists(a.host)) create_gamecoin(a.host);
            if (!fsys::exists(a.host)) {
                VLOG("[fs] archive %x (%s) not formatted\n", id, a.host.c_str());
                return id == 4 ? ERR_NOT_FORMATTED : ERR_NOT_FOUND;
            }
        }
        *out = next_archive++;
        archives[*out] = a;
        VLOG("[fs] OpenArchive %x -> %llu (%s)\n", id, (unsigned long long)*out, a.host.c_str());
        return 0;
    }

    Archive *arch(u32 lo, u32 hi) {
        auto it = archives.find((u64)lo | ((u64)hi << 32));
        return it == archives.end() ? nullptr : &it->second;
    }

    u32 open_file(Archive *a, const LowPath &p, u32 flags, ObjPtr *out) {
        auto fs = std::make_shared<FileSession>();
        if (a->id == 0x2345678A) {
            fs->fd = open(a->host.c_str(), O_RDONLY);
            if (fs->fd < 0) return ERR_NOT_FOUND;
            VLOG("[fs] system archive file %s\n", a->host.c_str());
        } else if (a->romfs) {
            // binary path: [type, ...]; type 0 = the RomFS
            open_romfs();
            fs->romfs = true;
            fs->base = 0;
            fs->size = romfs_size;
            if (p.data.size() >= 4 && p.word(0) == 2) {
                // an ExeFS file by name ("icon", "banner", "logo")
                char nm[9] = {};
                memcpy(nm, p.data.data() + 4, std::min<size_t>(8, p.data.size() - 4));
                std::string name = nm;
                if (!name.empty() && name[0] == '.') name = name.substr(1);
                fs->romfs = false;
                fs->fd = open((g_game.dir + "/exefs/" + name + ".bin").c_str(), O_RDONLY);
                VLOG("[fs] SelfNCCH ExeFS %s -> %s\n", nm, fs->fd >= 0 ? "ok" : "missing");
                if (fs->fd < 0) return ERR_NOT_FOUND;
            } else if (p.data.size() >= 4 && p.word(0) != 0) {
                LOG("[fs] SelfNCCH open of non-RomFS section %u\n", p.word(0));
                return ERR_NOT_FOUND;
            }
        } else {
            std::string path = a->host + sanitize(p.str());
            fs->path = path;
            int oflags = (flags & 2) ? O_RDWR : O_RDONLY;
            if (flags & 4) oflags |= O_CREAT;
            if (fsys::is_directory(path)) return fs_err(250, 7, 28);
            fs->fd = open(path.c_str(), oflags, 0644);
            if (fs->fd < 0) {
                VLOG("[fs] open %s failed\n", path.c_str());
                return fsys::exists(fsys::path(path).parent_path()) ? ERR_FILE_NOT_FOUND : ERR_PATH_NOT_FOUND;
            }
            VLOG("[fs] open %s\n", path.c_str());
        }
        auto ses = std::make_shared<Session>();
        ses->service = fs;
        *out = ses;
        return 0;
    }

    void handle(IPC &ipc) override {
        switch (ipc.cmd()) {
        case 0x0801:  // Initialize
        case 0x0861:  // InitializeWithSdkVersion
        case 0x0862:  // SetPriority
            ipc.ok();
            break;
        case 0x0863:
            ipc.ok(2);
            ipc.set(2, 0);
            break;
        case 0x080C: {  // OpenArchive(id, path type, path size, static desc, ptr)
            LowPath p = read_path(ipc[2], ipc[3], ipc[5]);
            u64 h = 0;
            u32 r = open_archive(ipc[1], p, &h);
            ipc.result(r, 3);
            ipc.set64(2, h);
            break;
        }
        case 0x080D:  // ControlArchive
            ipc.ok();
            break;
        case 0x080E:  // CloseArchive
            archives.erase((u64)ipc[1] | ((u64)ipc[2] << 32));
            ipc.ok();
            break;
        case 0x0802: {  // OpenFile(transaction, archive u64, path type, size, flags, attrs, desc, ptr)
            Archive *a = arch(ipc[2], ipc[3]);
            if (!a) { ipc.result(ERR_ARCHIVE); break; }
            LowPath p = read_path(ipc[4], ipc[5], ipc[9]);
            ObjPtr f;
            u32 r = open_file(a, p, ipc[6], &f);
            ipc.result(r, 1, 2);
            ipc.handles(2, {f}, true);
            break;
        }
        case 0x0803: {  // OpenFileDirectly
            LowPath ap = read_path(ipc[3], ipc[4], ipc[10]);
            LowPath fp = read_path(ipc[5], ipc[6], ipc[12]);
            u64 h;
            u32 r = open_archive(ipc[2], ap, &h);
            ObjPtr f;
            if (!r) {
                r = open_file(&archives[h], fp, ipc[7], &f);
                archives.erase(h);
            }
            ipc.result(r, 1, 2);
            ipc.handles(2, {f}, true);
            break;
        }
        case 0x0804: {  // DeleteFile(transaction, archive, path type, size, desc, ptr)
            Archive *a = arch(ipc[2], ipc[3]);
            if (!a) { ipc.result(ERR_ARCHIVE); break; }
            std::string path = a->host + sanitize(read_path(ipc[4], ipc[5], ipc[7]).str());
            ipc.result(fsys::remove(path) ? 0 : ERR_FILE_NOT_FOUND);
            break;
        }
        case 0x0805: {  // RenameFile(transaction, src archive, src type, src size, dst archive, dst type, dst size, ...)
            Archive *a = arch(ipc[2], ipc[3]);
            Archive *b = arch(ipc[6], ipc[7]);
            if (!a || !b) { ipc.result(ERR_ARCHIVE); break; }
            std::string s = a->host + sanitize(read_path(ipc[4], ipc[5], ipc[11]).str());
            std::string d = b->host + sanitize(read_path(ipc[8], ipc[9], ipc[13]).str());
            std::error_code ec;
            fsys::rename(s, d, ec);
            ipc.result(ec ? ERR_FILE_NOT_FOUND : 0);
            break;
        }
        case 0x0806:    // DeleteDirectory
        case 0x0807: {  // DeleteDirectoryRecursively
            Archive *a = arch(ipc[2], ipc[3]);
            if (!a) { ipc.result(ERR_ARCHIVE); break; }
            std::string path = a->host + sanitize(read_path(ipc[4], ipc[5], ipc[7]).str());
            std::error_code ec;
            if (ipc.cmd() == 0x0807) fsys::remove_all(path, ec);
            else fsys::remove(path, ec);
            ipc.result(ec ? ERR_PATH_NOT_FOUND : 0);
            break;
        }
        case 0x0808: {  // CreateFile(transaction, archive, path type, size, attrs, filesize u64, desc, ptr)
            Archive *a = arch(ipc[2], ipc[3]);
            if (!a) { ipc.result(ERR_ARCHIVE); break; }
            std::string path = a->host + sanitize(read_path(ipc[4], ipc[5], ipc[10]).str());
            u64 size = (u64)ipc[7] | ((u64)ipc[8] << 32);
            if (fsys::exists(path)) { ipc.result(ERR_FILE_EXISTS); break; }
            if (!fsys::exists(fsys::path(path).parent_path())) { ipc.result(ERR_PATH_NOT_FOUND); break; }
            int fd = open(path.c_str(), O_CREAT | O_RDWR, 0644);
            if (fd >= 0) {
                if (ftruncate(fd, (off_t)size)) {
                }
                close(fd);
            }
            VLOG("[fs] CreateFile %s size %llx\n", path.c_str(), (unsigned long long)size);
            ipc.result(fd >= 0 ? 0 : ERR_PATH_NOT_FOUND);
            break;
        }
        case 0x0809: {  // CreateDirectory(transaction, archive, type, size, attrs, desc, ptr)
            Archive *a = arch(ipc[2], ipc[3]);
            if (!a) { ipc.result(ERR_ARCHIVE); break; }
            std::string path = a->host + sanitize(read_path(ipc[4], ipc[5], ipc[8]).str());
            if (fsys::exists(path)) { ipc.result(ERR_DIR_EXISTS); break; }
            std::error_code ec;
            fsys::create_directory(path, ec);
            ipc.result(ec ? ERR_PATH_NOT_FOUND : 0);
            break;
        }
        case 0x080A: {  // RenameDirectory
            Archive *a = arch(ipc[2], ipc[3]);
            Archive *b = arch(ipc[6], ipc[7]);
            if (!a || !b) { ipc.result(ERR_ARCHIVE); break; }
            std::string s = a->host + sanitize(read_path(ipc[4], ipc[5], ipc[11]).str());
            std::string d = b->host + sanitize(read_path(ipc[8], ipc[9], ipc[13]).str());
            std::error_code ec;
            fsys::rename(s, d, ec);
            ipc.result(ec ? ERR_PATH_NOT_FOUND : 0);
            break;
        }
        case 0x080B: {  // OpenDirectory(archive u64, type, size, desc, ptr)
            Archive *a = arch(ipc[1], ipc[2]);
            if (!a || a->romfs) { ipc.result(ERR_ARCHIVE); break; }
            std::string path = a->host + sanitize(read_path(ipc[3], ipc[4], ipc[6]).str());
            if (!fsys::is_directory(path)) { ipc.result(ERR_PATH_NOT_FOUND, 1, 2); ipc.handles(2, {nullptr}, true); break; }
            auto ds = std::make_shared<DirSession>();
            for (auto &e : fsys::directory_iterator(path)) ds->entries.push_back(e);
            auto ses = std::make_shared<Session>();
            ses->service = ds;
            ipc.ok(1, 2);
            ipc.handles(2, {ses}, true);
            break;
        }
        case 0x080F:    // FormatThisUserSaveData
        case 0x084C: {  // FormatSaveData
            std::string d = save_root() + "/savedata";
            std::error_code ec;
            fsys::remove_all(d, ec);
            fsys::create_directories(d);
            LOG("[fs] formatted save data at %s\n", d.c_str());
            ipc.ok();
            break;
        }
        case 0x0845: {  // GetFormatInfo(archive id, path type, size, desc, ptr)
            std::string d = archive_dir(ipc[1], read_path(ipc[2], ipc[3], ipc[5]));
            if (d.empty() || !fsys::exists(d)) {
                ipc.result(ERR_NOT_FORMATTED, 5);
                break;
            }
            ipc.ok(5);
            ipc.set(2, 0x20000);  // total size
            ipc.set(3, 0x10);     // dirs
            ipc.set(4, 0x10);     // files
            ipc.set(5, 0);        // duplicate data
            break;
        }
        case 0x0851: {  // CreateExtSaveData(info[0..5 words], ...)
            // ExtSaveDataInfo: media type (u8), unknown, save id (u64)...
            u32 lo = ipc[2], hi = ipc[3];
            char b[64];
            snprintf(b, sizeof b, "/extdata/%08x%08x", hi, lo);
            fsys::create_directories(save_root() + b);
            ipc.ok();
            break;
        }
        case 0x0812:  // GetFreeBytes
            ipc.ok(3);
            ipc.set64(2, 0x10000000ull);
            break;
        case 0x0813:  // GetCardType
            ipc.ok(2);
            ipc.set(2, 0);
            break;
        case 0x0814:    // GetSdmcArchiveResource
        case 0x0815:    // GetNandArchiveResource
        case 0x0849: {  // GetArchiveResource
            ipc.ok(5);
            ipc.set(2, 0x200);      // sector size
            ipc.set(3, 0x20000);    // cluster size
            ipc.set(4, 0x80000);    // partition capacity in clusters
            ipc.set(5, 0x80000);    // free clusters
            break;
        }
        case 0x0817:  // IsSdmcDetected
        case 0x0818:  // IsSdmcWritable
        case 0x0821:  // CardSlotIsInserted
            ipc.ok(2);
            ipc.set(2, 1);
            break;
        case 0x082E:  // GetProductInfo(process id)
            ipc.ok(6);
            memset(gptr(ipc.buf + 8), 0, 16);
            break;
        case 0x082F:  // GetProgramLaunchInfo
            ipc.ok(5);
            ipc.set64(2, g_game.program_id);
            ipc.set(4, 2);  // game card
            ipc.set(5, 0);
            break;
        case 0x086E:  // SetThisSaveDataSecureValue
        case 0x0867:  // ControlSecureSave
            ipc.ok();
            break;
        case 0x086F:  // GetThisSaveDataSecureValue
            ipc.ok(5);
            ipc.set(2, 0);  // exists
            ipc.set(3, 0);
            ipc.set64(4, 0);
            break;
        default:
            unimplemented(ipc);
        }
    }
};

void register_fs() { register_service(std::make_shared<FsUser>("fs:USER")); }

}  // namespace kernel
