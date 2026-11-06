// The game list (launcher.h): a grid of cards sized to fit every game in the window, drawn in window pixels.
#include "launcher.h"

#include <SDL3/SDL.h>
#include "platform.h"

#include <algorithm>
#include <cmath>
#include <map>

#include "text.h"
#include "uiview.h"

namespace {

SDL_Renderer *ren;
std::map<std::string, text::Texture> cache;

const text::Texture &label(const std::string &s, float px, SDL_Color c, bool bold) {
    char key[64];
    snprintf(key, sizeof key, "|%.0f|%02x%02x%02x|%d", px, c.r, c.g, c.b, bold);
    auto it = cache.find(s + key);
    if (it != cache.end()) return it->second;
    return cache[s + key] = text::render(ren, s, px, c, bold);
}

// a name in at most two lines of width w (the rest cut with an ellipsis)
std::vector<std::string> wrap(const std::string &s, float px, float w) {
    std::vector<std::string> words, lines;
    for (size_t i = 0; i < s.size();) {
        const size_t j = s.find(' ', i);
        words.push_back(s.substr(i, j == std::string::npos ? std::string::npos : j - i));
        i = j == std::string::npos ? s.size() : j + 1;
    }
    std::string line;
    size_t k = 0;
    for (; k < words.size(); k++) {
        const std::string next = line.empty() ? words[k] : line + " " + words[k];
        if (!line.empty() && text::width(next, px, false) > w) {
            lines.push_back(line);
            line = words[k];
            if (lines.size() == 1) continue;
            break;
        }
        line = next;
    }
    if (lines.size() < 2) {
        lines.push_back(line);
        return lines;
    }
    // a third line would be needed: the second one ends with an ellipsis
    std::string last = lines[1] + " " + line;
    for (size_t r = k + 1; r < words.size(); r++) last += " " + words[r];
    while (last.size() > 1 && text::width(last + "…", px, false) > w) {
        last.pop_back();
        while (!last.empty() && ((unsigned char)last.back() & 0xC0) == 0x80) last.pop_back();  // (whole characters)
    }
    lines[1] = last + "…";
    return lines;
}

}  // namespace

int launcher_choose(const std::vector<LauncherGame> &games, int selected) {
#if TARGET_OS_IOS
    SDL_SetHint(SDL_HINT_ORIENTATIONS, "LandscapeLeft LandscapeRight Portrait PortraitUpsideDown");
#endif
    if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_GAMEPAD)) FATAL("SDL_Init: %s\n", SDL_GetError());
    SDL_Window *win = SDL_CreateWindow("Games", 760, 540, SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY |
                                                              (TARGET_OS_IOS ? SDL_WINDOW_FULLSCREEN : 0));
    if (!win) FATAL("SDL_CreateWindow: %s\n", SDL_GetError());
    ren = SDL_CreateRenderer(win, nullptr);
    SDL_SetRenderVSync(ren, 1);
    std::vector<SDL_Texture *> icons;
    for (const auto &g : games) {
        SDL_Texture *t = nullptr;
        if (g.icon.size() == 48 * 48) {
            t = SDL_CreateTexture(ren, SDL_PIXELFORMAT_ABGR8888, SDL_TEXTUREACCESS_STATIC, 48, 48);
            SDL_UpdateTexture(t, nullptr, g.icon.data(), 48 * 4);
            SDL_SetTextureScaleMode(t, SDL_SCALEMODE_PIXELART);
        }
        icons.push_back(t);
    }
    const int n = (int)games.size();
    int sel = std::clamp(selected, 0, n - 1), chosen = -1, pressed = -1, cols = 1;
    bool stick_moved = false;
    SDL_Gamepad *pad = nullptr;
    std::vector<SDL_FRect> cards(n);
    auto card_at = [&](float x, float y) {
        for (int i = 0; i < n; i++)
            if (x >= cards[i].x && x < cards[i].x + cards[i].w && y >= cards[i].y && y < cards[i].y + cards[i].h) return i;
        return -1;
    };
    auto move = [&](int d) { sel = std::clamp(sel + d, 0, n - 1); };
    int last_w = 0, last_h = 0;
    while (chosen < 0) {
        uiview::keep_filling(ren);
        const float d = SDL_GetWindowPixelDensity(win);
        SDL_Event ev;
        while (SDL_PollEvent(&ev)) {
            switch (ev.type) {
            case SDL_EVENT_QUIT:
            case SDL_EVENT_WINDOW_CLOSE_REQUESTED: exit(0);
            case SDL_EVENT_KEY_DOWN:
                switch (ev.key.scancode) {
                case SDL_SCANCODE_LEFT: move(-1); break;
                case SDL_SCANCODE_RIGHT: move(1); break;
                case SDL_SCANCODE_UP: move(-cols); break;
                case SDL_SCANCODE_DOWN: move(cols); break;
                case SDL_SCANCODE_RETURN:
                case SDL_SCANCODE_KP_ENTER:
                case SDL_SCANCODE_SPACE:
                case SDL_SCANCODE_X: chosen = sel; break;
                default: break;
                }
                break;
            case SDL_EVENT_GAMEPAD_ADDED:
                if (!pad) pad = SDL_OpenGamepad(ev.gdevice.which);
                break;
            case SDL_EVENT_GAMEPAD_BUTTON_DOWN:
                switch (ev.gbutton.button) {
                case SDL_GAMEPAD_BUTTON_DPAD_LEFT: move(-1); break;
                case SDL_GAMEPAD_BUTTON_DPAD_RIGHT: move(1); break;
                case SDL_GAMEPAD_BUTTON_DPAD_UP: move(-cols); break;
                case SDL_GAMEPAD_BUTTON_DPAD_DOWN: move(cols); break;
                case SDL_GAMEPAD_BUTTON_EAST:  // the 3DS's A position
                case SDL_GAMEPAD_BUTTON_START: chosen = sel; break;
                default: break;
                }
                break;
            case SDL_EVENT_GAMEPAD_AXIS_MOTION:  // the left stick: one step per push
                if (ev.gaxis.axis == SDL_GAMEPAD_AXIS_LEFTX || ev.gaxis.axis == SDL_GAMEPAD_AXIS_LEFTY) {
                    const bool far = std::abs(ev.gaxis.value) > 20000;
                    if (far && !stick_moved) {
                        const int dir = ev.gaxis.value > 0 ? 1 : -1;
                        move(ev.gaxis.axis == SDL_GAMEPAD_AXIS_LEFTX ? dir : dir * cols);
                    }
                    if (far) stick_moved = true;
                    else if (std::abs(ev.gaxis.value) < 8000) stick_moved = false;
                }
                break;
            case SDL_EVENT_MOUSE_MOTION:  // (a real pointer: fingers only tap)
                if (ev.motion.which != SDL_TOUCH_MOUSEID)
                    if (const int i = card_at(ev.motion.x * d, ev.motion.y * d); i >= 0) sel = i;
                break;
            case SDL_EVENT_MOUSE_BUTTON_DOWN:
                pressed = card_at(ev.button.x * d, ev.button.y * d);
                if (pressed >= 0) sel = pressed;
                break;
            case SDL_EVENT_MOUSE_BUTTON_UP:  // a tap: down and up on the same card
                if (pressed >= 0 && card_at(ev.button.x * d, ev.button.y * d) == pressed) chosen = pressed;
                pressed = -1;
                break;
            default: break;
            }
        }
        int ow, oh;
        SDL_GetCurrentRenderOutputSize(ren, &ow, &oh);
        if (ow != last_w || oh != last_h) {  // new sizes for the text
            for (auto &[k, t] : cache) SDL_DestroyTexture(t.tex);
            cache.clear();
            last_w = ow;
            last_h = oh;
        }
        SDL_Rect sa;
        if (!SDL_GetWindowSafeArea(win, &sa) || sa.w <= 0) SDL_GetWindowSize(win, &sa.w, &sa.h), sa.x = sa.y = 0;
        const float m = 20 * d, title_h = 64 * d, gap = 14 * d;
        const SDL_FRect area{sa.x * d + m, sa.y * d + m + title_h, sa.w * d - 2 * m, sa.h * d - 2 * m - title_h};
        // the largest cards (up to 220 points wide) with every game in sight
        float cw = 0;
        for (int c = 1; c <= n; c++) {
            const int rows = (n + c - 1) / c;
            float w = std::min((area.w - (c - 1) * gap) / c, 220 * d);
            w = std::min(w, (area.h - (rows - 1) * gap) / rows / 1.2f);
            if (w > cw) {
                cw = w;
                cols = c;
            }
        }
        const float ch = cw * 1.2f;
        const int rows = (n + cols - 1) / cols;
        const float gx = area.x + (area.w - (cols * cw + (cols - 1) * gap)) / 2,
                    gy = area.y + std::max(0.0f, (area.h - (rows * ch + (rows - 1) * gap)) / 2);
        SDL_SetRenderDrawColor(ren, 16, 18, 26, 255);
        SDL_RenderClear(ren);
        SDL_SetRenderDrawBlendMode(ren, SDL_BLENDMODE_BLEND);
        const text::Texture &tt = label("Choose a game", 28 * d, {255, 255, 255, 255}, true);
        const SDL_FRect tr{area.x, sa.y * d + m + (title_h - tt.h) / 2, (float)tt.w, (float)tt.h};
        SDL_RenderTexture(ren, tt.tex, nullptr, &tr);
        const float name_px = std::clamp(cw * 0.085f, 12 * d, 18 * d);
        for (int i = 0; i < n; i++) {
            const SDL_FRect c{gx + (i % cols) * (cw + gap), gy + (i / cols) * (ch + gap), cw, ch};
            cards[i] = c;
            const bool on = i == sel;
            SDL_SetRenderDrawColor(ren, 255, 255, 255, on ? 34 : 14);
            SDL_RenderFillRect(ren, &c);
            if (on) {  // an accent frame
                const float b = std::max(2.0f, 2.5f * d);
                SDL_SetRenderDrawColor(ren, 255, 205, 64, 255);
                const SDL_FRect edges[4] = {{c.x, c.y, c.w, b}, {c.x, c.y + c.h - b, c.w, b}, {c.x, c.y, b, c.h}, {c.x + c.w - b, c.y, b, c.h}};
                SDL_RenderFillRects(ren, edges, 4);
            }
            const float is = cw * 0.6f;
            const SDL_FRect ir{c.x + (cw - is) / 2, c.y + cw * 0.1f, is, is};
            if (icons[i]) SDL_RenderTexture(ren, icons[i], nullptr, &ir);
            float y = ir.y + is + cw * 0.06f;
            for (const auto &line : wrap(games[i].name, name_px, cw * 0.9f)) {
                const text::Texture &t = label(line, name_px, {235, 236, 240, 255}, false);
                const SDL_FRect r{std::round(c.x + (cw - t.w) / 2), std::round(y), (float)t.w, (float)t.h};
                SDL_RenderTexture(ren, t.tex, nullptr, &r);
                y += t.h * 0.95f;
            }
        }
        SDL_RenderPresent(ren);
    }
    for (SDL_Texture *t : icons)
        if (t) SDL_DestroyTexture(t);
    for (auto &[k, t] : cache) SDL_DestroyTexture(t.tex);
    cache.clear();
    if (pad) SDL_CloseGamepad(pad);
    SDL_DestroyRenderer(ren);
    SDL_DestroyWindow(win);
    return chosen;
}
