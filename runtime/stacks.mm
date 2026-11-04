// Stacks in the pockets (stacks.h). The game's thread tidies the pockets as it dispatches to the profile's address
// while the player is free and the pockets are closed: a second pocket with an item that stacks joins the stack (its
// pocket empties), and a stack's item that left the pockets (dropped, sold, given, planted, eaten) comes back from
// the stack into the pocket it left. With the pockets full, the stack of the item the player stands on gives up its
// pocket so the game can pick the item up into it. As a conversation starts (a shop's counter, a villager), the extra
// items are laid out in free pockets so the game's screens can take several, and they gather back afterwards. With
// the option off, the stacks empty into free pockets as room allows.
#include "stacks.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <vector>

#include "onescreen.h"
#include "profile.h"
#include "rt.h"
#include "settings.h"
#include "text.h"

namespace kernel {
std::vector<u8> romfs_file(const std::string &path);  // (kernel.h clashes with MacTypes.h)
std::string save_root();
}  // namespace kernel

namespace stacks {
namespace {

constexpr int kSlots = 16;  // pockets
constexpr const char *kFile = "recomp_stacks.txt";

std::mutex m;
bool loaded;
std::vector<u8> records;  // the item records ("tools items"), for the items' categories
// the extra items of each stack (beyond the one in the pocket), by owner and item (its 4 bytes)
std::map<std::string, std::map<u32, int>> extra;
std::map<u32, int> where;  // the pocket each stack was last seen in (its item comes back there)
bool dirty;                // changed since the stacks were last written

std::string file_path() { return kernel::save_root() + "/savedata/" + kFile; }

void load_locked() {
    if (loaded) return;
    loaded = true;
    const profile::Tools &t = profile::tools();
    if (!t.items.empty()) records = kernel::romfs_file(t.items);
    std::ifstream f(file_path());
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream in(line);
        std::string owner;
        u32 item;
        int n;
        if (in >> owner >> std::hex >> item >> std::dec >> n && n > 0) extra[owner][item] = n;
    }
}

// how many of this item a pocket holds (1: it does not stack)
int max_of(u32 item) {
    const profile::Tools &t = profile::tools();
    const u32 id = item & 0xffff;
    if (id == t.empty || !t.record || id < t.first) return 1;
    const size_t o = (size_t)(id - t.first) * t.record + t.category_off;
    if (o >= records.size()) return 1;
    for (const profile::StackKind &k : profile::stacks().kinds)
        if (k.kind == records[o]) return k.max;
    return 1;
}

// whose pockets these are ("" while unknown)
std::string owner() {
    std::string s;
    for (int v : profile::stacks().owner) {
        s64 x;
        if (!profile::value_now(v, x)) return "";
        char b[24];
        snprintf(b, sizeof b, "%s%llx", s.empty() ? "" : "-", (unsigned long long)x);
        s += b;
    }
    return s;
}

bool pockets_at(u32 &a) {
    s64 v;
    return profile::value_now(profile::tools().pockets, v, &a);
}

u8 state_of(CPU *c) { return r8(c->r[0] + profile::stacks().free_off); }

bool in(const std::vector<u8> &v, u8 x) { return std::find(v.begin(), v.end(), x) != v.end(); }

// the player can move (no dialog, menu or shop screen, which show the pockets as they read them)
bool player_free(CPU *c) {
    const profile::Stacks &s = profile::stacks();
    if (s.free_states.empty()) return profile::cond_now(profile::tools().ready_cond);
    return in(s.free_states, state_of(c));
}

bool laid_out;  // this conversation has laid the stacks out in free pockets

void tidy(CPU *c) {
    const profile::Tools &t = profile::tools();
    const profile::Stacks &s = profile::stacks();
    u32 pa;
    // never while the pockets are open (they show what they read as they opened: a pocket changed behind them
    // would not match)
    if (profile::cond_now(t.open_cond) || !pockets_at(pa)) return;
    const bool free = player_free(c);
    // only while the player is free, or once as a conversation starts (before its screens read the pockets)
    const bool lay_out = !free && !laid_out && g_settings.stack_items && in(s.spread_states, state_of(c));
    if (free) laid_out = false;
    if (!free && !lay_out) return;
    const std::string who = owner();
    if (who.empty()) return;
    std::lock_guard<std::mutex> g(m);
    load_locked();
    std::map<u32, int> &mine = extra[who];
    u32 p[kSlots];
    for (int i = 0; i < kSlots; i++) p[i] = r32(pa + 4 * i);
    auto put = [&](int i, u32 item) {
        w32(pa + 4 * i, item);
        p[i] = item;
        dirty = true;
    };
    auto first_free = [&] {
        for (int i = 0; i < kSlots; i++)
            if ((p[i] & 0xffff) == t.empty) return i;
        return -1;
    };
    if (lay_out) {
        // a conversation starts: the extra items go to free pockets (but the last few, for what it may give), so a
        // shop's screen can take a whole stack; they gather back once the player is free
        laid_out = true;
        int room = -s.spread_keep;
        for (int i = 0; i < kSlots; i++) room += (p[i] & 0xffff) == t.empty;
        for (auto it = mine.begin(); it != mine.end();) {
            for (; it->second > 0 && room > 0; it->second--, room--) put(first_free(), it->first);
            it = it->second > 0 ? std::next(it) : mine.erase(it);
        }
        return;
    }
    // the item the player stands on, when its stack (in the pockets, with room) could take it
    const u32 under = s.underfoot_off && g_settings.stack_items ? r32(c->r[0] + s.underfoot_off) : t.empty;
    auto has_room = [&](u32 item) {
        const auto e = mine.find(item);
        return max_of(item) > 1 + (e == mine.end() ? 0 : e->second);
    };
    // with the pockets full, its stack gives up its pocket while the player stands there: the game picks the item
    // up into it, and it joins the stack (back in its pocket, one more)
    if ((under & 0xffff) != t.empty && first_free() < 0 && has_room(under)) {
        for (int i = 0; i < kSlots; i++) {
            if (p[i] != under) continue;
            where[under] = i;
            mine[under]++;
            put(i, t.empty);
            break;
        }
    }
    // stacks whose item left the pockets: the next one comes back where it was (or to the first free pocket);
    // with the option off, the extra ones go to free pockets as room allows
    for (auto it = mine.begin(); it != mine.end();) {
        const u32 item = it->first;
        int at = -1;
        for (int i = 0; i < kSlots && at < 0; i++)
            if (p[i] == item) at = i;
        int frees = 0;
        for (int i = 0; i < kSlots; i++) frees += (p[i] & 0xffff) == t.empty;
        if (at < 0 && item == under && frees == 1) {  // (the room kept for the item stood on)
            ++it;
            continue;
        }
        if (at < 0 || !g_settings.stack_items) {
            const auto w = where.find(item);
            int slot = at < 0 && w != where.end() && (p[w->second] & 0xffff) == t.empty ? w->second : first_free();
            if (slot >= 0) {
                put(slot, item);
                if (at < 0) where[item] = slot;
                it->second--;
            }
        }
        it = it->second > 0 ? std::next(it) : mine.erase(it);
    }
    if (!g_settings.stack_items) return;
    // a second pocket with an item that stacks joins the stack (in the pocket the stack was in, or the first one)
    for (int i = 0; i < kSlots; i++) {
        const u32 item = p[i];
        const int mx = max_of(item);
        if (mx < 2) continue;
        auto w = where.find(item);
        int home = w != where.end() && p[w->second] == item ? w->second : i;
        where[item] = home;
        for (int j = 0; j < kSlots; j++) {
            if (j == home || p[j] != item) continue;
            int &n = mine[item];
            if (1 + n >= mx) break;
            n++;
            put(j, t.empty);
        }
    }
}

// counts on the pockets: the bottom screen's pixels of each pocket's centre, and the count there (0: none)
void counts(int out[kSlots]) {
    for (int i = 0; i < kSlots; i++) out[i] = 0;
    u32 pa;
    if (!pockets_at(pa)) return;
    const std::string who = owner();
    if (who.empty()) return;
    std::lock_guard<std::mutex> g(m);
    const auto e = extra.find(who);
    if (e == extra.end()) return;
    u32 p[kSlots];
    for (int i = 0; i < kSlots; i++) p[i] = r32(pa + 4 * i);
    for (const auto &[item, n] : e->second) {
        const auto w = where.find(item);
        int at = w != where.end() && p[w->second] == item ? w->second : -1;
        for (int i = 0; i < kSlots && at < 0; i++)
            if (p[i] == item) at = i;
        if (at >= 0) out[at] = 1 + n;
    }
}

}  // namespace

void init() {
    const profile::Stacks &s = profile::stacks();
    if (s.kinds.empty() || !s.on) return;
    rt_add_dispatch_hook(s.on, tidy);
}

void save_written(const std::string &path) {
    if (profile::stacks().kinds.empty() || path.find(kFile) != std::string::npos) return;
    std::lock_guard<std::mutex> g(m);
    load_locked();
    if (!dirty) return;
    dirty = false;
    const std::string f = file_path(), tmp = f + ".tmp";
    {
        std::ofstream o(tmp);
        o << "# stacks in the pockets (3dsrecomp): owner, item, extra items beyond the pocket's own\n";
        for (const auto &[who, items] : extra)
            for (const auto &[item, n] : items)
                if (n > 0) o << who << ' ' << std::hex << item << std::dec << ' ' << n << '\n';
    }
    std::rename(tmp.c_str(), f.c_str());
}

void draw(SDL_Renderer *r, const SDL_FRect &d, float px, float alpha) {
    const profile::Tools &t = profile::tools();
    const profile::Stacks &st = profile::stacks();
    if (st.kinds.empty() || !d.w || alpha <= 0 || (int)t.slots.size() < kSlots ||
        !(profile::cond(t.open_cond) || (st.shown_cond >= 0 && profile::cond(st.shown_cond))))
        return;
    int n[kSlots];
    counts(n);
    const float k = d.w / 320;  // render units per bottom-screen pixel
    for (int i = 0; i < kSlots; i++) {
        if (n[i] < 2) continue;
        const std::string s = std::to_string(n[i]);
        static std::map<std::string, std::pair<text::Texture, float>> cache;  // (by text: one size at a time)
        auto &c = cache[s];
        if (!c.first.tex || c.second != k * px) {
            if (c.first.tex) SDL_DestroyTexture(c.first.tex);
            c = {text::render(r, s, 9 * k * px, {255, 255, 255, 255}, true), k * px};
        }
        const float tw = c.first.w / px, th = c.first.h / px;
        const float h = 13 * k, w = std::max(h, tw + 7 * k);
        const float cx = d.x + (t.slots[i].first + 12) * k, cy = d.y + (t.slots[i].second + 12) * k;  // lower right
        onescreen::rounded_rect(r, nullptr, {cx - w / 2, cy - h / 2, w, h}, h / 2, {0.18f, 0.16f, 0.13f, 0.9f * alpha});
        SDL_SetTextureAlphaModFloat(c.first.tex, alpha);
        const SDL_FRect td{cx - tw / 2, cy - th / 2, tw, th};
        SDL_RenderTexture(r, c.first.tex, nullptr, &td);
    }
}

}  // namespace stacks
