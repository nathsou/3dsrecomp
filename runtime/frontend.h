#pragma once
#include <cstdint>

void frontend_init();
void frontend_run();  // the host main thread's loop (window, input, GPU, presentation)

// 3DS buttons (the HID pad state's bits)
enum : uint32_t {
    BTN_A = 1 << 0, BTN_B = 1 << 1, BTN_SELECT = 1 << 2, BTN_START = 1 << 3, BTN_RIGHT = 1 << 4, BTN_LEFT = 1 << 5,
    BTN_UP = 1 << 6, BTN_DOWN = 1 << 7, BTN_R = 1 << 8, BTN_L = 1 << 9, BTN_X = 1 << 10, BTN_Y = 1 << 11,
    BTN_ZL = 1 << 14, BTN_ZR = 1 << 15,
};
