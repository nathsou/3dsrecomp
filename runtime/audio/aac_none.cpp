// aac_decoder.h without a decoder (a build without FFmpeg): AAC music is silent.
#include "aac_decoder.h"

namespace audio::hle {
bool aac_open(int rate, int channels) {
    static bool said;
    if (!said) LOG("dsp: no AAC decoder in this build (FFmpeg's libavcodec): AAC music is silent (%d Hz, %d channels)\n",
                   rate, channels);
    said = true;
    return false;
}
std::vector<s16> aac_decode(const std::vector<AacFrame> &) { return {}; }
void aac_close() {}
}  // namespace audio::hle
