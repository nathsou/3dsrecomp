// The AAC decoder behind the DSP's (runtime/audio/aac.cpp): runtime/audio/aac_audiotoolbox.cpp (Apple),
// runtime/audio/aac_ffmpeg.cpp (FFmpeg's libavcodec), or none (aac_none.cpp: the music is silent).
#pragma once
#include <vector>

#include "../rt.h"

namespace audio::hle {
struct AacFrame {
    const u8 *adts;  // an ADTS frame: its header, then raw AAC
    u32 size, head;  // the frame's bytes, the header's
};
bool aac_open(int rate, int channels);  // (again with the same: keeps the decoder's state); false: unavailable
std::vector<s16> aac_decode(const std::vector<AacFrame> &frames);  // consecutive frames: interleaved samples
void aac_close();
}  // namespace audio::hle
