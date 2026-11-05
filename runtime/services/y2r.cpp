// y2r:u: the YUV to RGB conversion hardware (video players such as Mobiclip's hand their decoded frames to it).
// The conversion runs at once when started: the source planes are read and the RGB image written to guest memory
// with the transfer units and gaps the game gave, in strips of 8 lines, linear or in 8x8 tiles as textures want
// it, with the hardware's fixed-point arithmetic.
#include <algorithm>

#include "../service.h"
#include "services.h"

namespace kernel {

namespace {

enum InputFormat { YUV422_8, YUV420_8, YUV422_16, YUV420_16, YUYV422 };
enum OutputFormat { RGB32, RGB24, RGB5A1, RGB565 };

// one DMA side: a buffer read or written in units of `unit` bytes with `gap` bytes skipped after each
struct Transfer {
    u32 addr = 0, size = 0;
    s32 unit = 0, gap = 0;
};

// Y, R-V, G-V, G-U, B-U and the offsets, as the hardware takes them (ITU-R BT.601 / 709, full or scaled range)
using Coefficients = s16[8];
constexpr Coefficients kStandard[4] = {
    {0x100, 0x166, 0xB6, 0x58, 0x1C5, -0x166F, 0x10EE, -0x1C5B},
    {0x100, 0x193, 0x77, 0x2F, 0x1DB, -0x1933, 0xA7C, -0x1D51},
    {0x12A, 0x198, 0xD0, 0x64, 0x204, -0x1BDE, 0x10F2, -0x229B},
    {0x12A, 0x1CA, 0x88, 0x36, 0x21C, -0x1F04, 0x99C, -0x2421},
};

struct Y2rState {
    u32 input = YUV422_8, output = RGB32, rotation = 0, block = 0;  // block: 0 linear, 1 8x8 tiles
    u32 line_width = 0, lines = 0;
    s16 coef[8] = {0x100, 0x166, 0xB6, 0x58, 0x1C5, -0x166F, 0x10EE, -0x1C5B};
    u32 standard = 0;
    u16 alpha = 0xFF;
    bool end_interrupt = false;
    Transfer y, u, v, yuyv, dst;
} st;

// reads n bytes of one plane (every `step`th byte of 16-bit samples: their low byte)
void receive(Transfer &t, u8 *out, u32 n, int step) {
    const u32 per_unit = t.unit > 0 ? (u32)t.unit / step : n;
    while (n > 0) {
        const u32 k = std::min(per_unit, n);
        for (u32 i = 0; i < k; i++) out[i] = r8(t.addr + i * step);
        out += k;
        n -= k;
        t.addr += (t.unit > 0 ? (u32)t.unit : k * step) + (u32)t.gap;
    }
}

void send_pixel(u32 &addr, u32 &left_in_unit, u32 rgb) {
    const u8 r = rgb >> 16, g = rgb >> 8, b = rgb;
    const u8 a = (u8)st.alpha;
    auto unit_done = [&](u32 bytes) {
        addr += bytes;
        if (st.dst.unit > 0 && (left_in_unit -= bytes) == 0) {
            addr += (u32)st.dst.gap;
            left_in_unit = (u32)st.dst.unit;
        }
    };
    switch (st.output) {
    case RGB32:  // (the PICA's RGBA8: a, b, g, r in memory)
        w8(addr, a), w8(addr + 1, b), w8(addr + 2, g), w8(addr + 3, r);
        unit_done(4);
        break;
    case RGB24:
        w8(addr, b), w8(addr + 1, g), w8(addr + 2, r);
        unit_done(3);
        break;
    case RGB5A1:
        w16(addr, (u16)((r >> 3) << 11 | (g >> 3) << 6 | (b >> 3) << 1 | (a >> 7)));
        unit_done(2);
        break;
    default:  // RGB565
        w16(addr, (u16)((r >> 3) << 11 | (g >> 2) << 5 | (b >> 3)));
        unit_done(2);
        break;
    }
}

// the position of (x, y) in an 8x8 tile (Morton order)
u32 morton8(u32 x, u32 y) {
    return (x & 1) | (y & 1) << 1 | (x & 2) << 1 | (y & 2) << 2 | (x & 4) << 2 | (y & 4) << 3;
}

void convert() {
    const u32 w = st.line_width, h = st.lines;
    if (!w || !h || w > 1024) return;
    const bool interleaved = st.input == YUYV422;
    const bool half_lines = st.input == YUV420_8 || st.input == YUV420_16;  // 4:2:0: one chroma line per two
    const int step = st.input == YUV422_16 || st.input == YUV420_16 ? 2 : 1;
    std::vector<u8> ybuf(w * 8 * 2), ubuf(w * 8 / 2), vbuf(w * 8 / 2);
    std::vector<u32> strip(w * 8);
    u32 out = st.dst.addr, left = st.dst.unit > 0 ? (u32)st.dst.unit : 0;
    const s16 *c = st.coef;
    for (u32 y0 = 0; y0 < h; y0 += 8) {
        const u32 rows = std::min(8u, h - y0), n = rows * w;
        if (interleaved) {
            receive(st.yuyv, ybuf.data(), n * 2, 1);
        } else {
            receive(st.y, ybuf.data(), n, step);
            receive(st.u, ubuf.data(), half_lines ? n / 4 : n / 2, step);
            receive(st.v, vbuf.data(), half_lines ? n / 4 : n / 2, step);
        }
        for (u32 y = 0; y < rows; y++) {
            for (u32 x = 0; x < w; x++) {
                s32 Y, U, V;
                if (interleaved) {
                    Y = ybuf[(y * w + x) * 2];
                    U = ybuf[(y * w + (x & ~1u)) * 2 + 1];
                    V = ybuf[(y * w + (x & ~1u)) * 2 + 3];
                } else {
                    Y = ybuf[y * w + x];
                    const u32 ci = ((half_lines ? y / 2 : y) * w + x) / 2;
                    U = ubuf[ci];
                    V = vbuf[ci];
                }
                const s32 cy = c[0] * Y;
                constexpr s32 kRounding = 0x18;
                const s32 r = ((cy + c[1] * V) >> 3) + c[5] + kRounding;
                const s32 g = ((cy - c[2] * V - c[3] * U) >> 3) + c[6] + kRounding;
                const s32 b = ((cy + c[4] * U) >> 3) + c[7] + kRounding;
                const u32 rgb = (u32)std::clamp(r >> 5, 0, 255) << 16 | (u32)std::clamp(g >> 5, 0, 255) << 8 |
                                (u32)std::clamp(b >> 5, 0, 255);
                const u32 at = st.block ? (x / 8) * 64 + morton8(x % 8, y) : y * w + x;
                strip[at] = rgb;
            }
        }
        // (a partial last strip in tiles keeps whole tiles' worth of pixels: the hardware needs 8-line images there)
        const u32 count = st.block ? w * 8 : n;
        for (u32 i = 0; i < count; i++) send_pixel(out, left, strip[i]);
    }
}

Transfer parse_transfer(IPC &ipc) {
    Transfer t;
    t.addr = ipc[1];
    t.size = ipc[2];
    t.unit = (s16)ipc[3];
    t.gap = (s16)ipc[4];
    return t;
}

struct Y2r : Service {
    using Service::Service;
    std::shared_ptr<Event> done = make_event(ONESHOT, "y2r");
    void get(IPC &ipc, u32 v) {
        ipc.ok(2);
        ipc.set(2, v);
    }
    void handle(IPC &ipc) override {
        switch (ipc.cmd()) {
        case 0x0001: st.input = ipc[1]; ipc.ok(); break;        // SetInputFormat
        case 0x0002: get(ipc, st.input); break;
        case 0x0003: st.output = ipc[1]; ipc.ok(); break;       // SetOutputFormat
        case 0x0004: get(ipc, st.output); break;
        case 0x0005: st.rotation = ipc[1]; ipc.ok(); break;     // SetRotation (only none is done)
        case 0x0006: get(ipc, st.rotation); break;
        case 0x0007: st.block = ipc[1]; ipc.ok(); break;        // SetBlockAlignment
        case 0x0008: get(ipc, st.block); break;
        case 0x000D: st.end_interrupt = ipc[1]; ipc.ok(); break;  // SetTransferEndInterrupt
        case 0x000E: get(ipc, st.end_interrupt); break;
        case 0x000F:  // GetTransferEndEvent
            ipc.ok(1, 2);
            ipc.handles(2, {done});
            break;
        case 0x0010: st.y = parse_transfer(ipc); ipc.ok(); break;     // SetSendingY
        case 0x0011: st.u = parse_transfer(ipc); ipc.ok(); break;     // SetSendingU
        case 0x0012: st.v = parse_transfer(ipc); ipc.ok(); break;     // SetSendingV
        case 0x0013: st.yuyv = parse_transfer(ipc); ipc.ok(); break;  // SetSendingYUYV
        case 0x0014: case 0x0015: case 0x0016: case 0x0017:  // IsFinishedSending*
        case 0x0019:                                          // IsFinishedReceiving
            get(ipc, 1);
            break;
        case 0x0018: st.dst = parse_transfer(ipc); ipc.ok(); break;  // SetReceiving
        case 0x001A: st.line_width = ipc[1]; ipc.ok(); break;       // SetInputLineWidth
        case 0x001B: get(ipc, st.line_width); break;
        case 0x001C: st.lines = ipc[1]; ipc.ok(); break;            // SetInputLines
        case 0x001D: get(ipc, st.lines); break;
        case 0x001E:  // SetCoefficient
            for (int i = 0; i < 8; i++) st.coef[i] = (s16)(ipc[1 + i / 2] >> (16 * (i & 1)));
            ipc.ok();
            break;
        case 0x001F:
            ipc.ok(5);
            for (int i = 0; i < 4; i++) ipc.set(2 + i, (u16)st.coef[2 * i] | (u32)(u16)st.coef[2 * i + 1] << 16);
            break;
        case 0x0020:  // SetStandardCoefficient
            if (ipc[1] < 4) {
                st.standard = ipc[1];
                std::copy(kStandard[st.standard], kStandard[st.standard] + 8, st.coef);
            }
            ipc.ok();
            break;
        case 0x0021:  // GetStandardCoefficient(index)
            ipc.ok(5);
            for (int i = 0; i < 4; i++) {
                const s16 *k = kStandard[std::min(ipc[1], 3u)];
                ipc.set(2 + i, (u16)k[2 * i] | (u32)(u16)k[2 * i + 1] << 16);
            }
            break;
        case 0x0022: st.alpha = (u16)ipc[1]; ipc.ok(); break;  // SetAlpha
        case 0x0023: get(ipc, st.alpha); break;
        case 0x0026:  // StartConversion
            if (g_verbose) {
                LOG("[y2r] %ux%u in %u out %u rot %u block %u y %08x/%d/%d u %08x v %08x yuyv %08x dst %08x/%d/%d\n",
                    st.line_width, st.lines, st.input, st.output, st.rotation, st.block, st.y.addr, st.y.unit, st.y.gap,
                    st.u.addr, st.v.addr, st.yuyv.addr, st.dst.addr, st.dst.unit, st.dst.gap);
            }
            convert();
            done->signal();
            ipc.ok();
            break;
        case 0x0028: get(ipc, 0); break;  // IsBusyConversion
        case 0x0029: {  // SetPackageParameter: formats, rotation, alignment (bytes), line width, lines (s16),
                        // standard coefficient (byte), padding, alpha (u16)
            const u32 a = ipc[1], b = ipc[2], d = ipc[3];
            st.input = a & 0xff, st.output = a >> 8 & 0xff, st.rotation = a >> 16 & 0xff, st.block = a >> 24;
            st.line_width = b & 0xffff, st.lines = b >> 16;
            if ((d & 0xff) < 4) std::copy(kStandard[d & 0xff], kStandard[d & 0xff] + 8, st.coef);
            st.alpha = (u16)(d >> 16);
            ipc.ok();
            break;
        }
        case 0x002A: get(ipc, 0); break;  // PingProcess
        default:
            ipc.ok();  // dithering, driver initialisation, ...
        }
    }
};

}  // namespace

void register_y2r() { register_service(std::make_shared<Y2r>("y2r:u")); }

}  // namespace kernel
