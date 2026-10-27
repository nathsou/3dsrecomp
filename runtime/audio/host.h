// Host audio output (host_sdl.cpp).
#pragma once
#include "dsp_shared.h"

namespace audio::hle {
// Queues `n` stereo samples at 32728 Hz for playback (called from the DSP tick thread; never blocks
// for long). Does nothing until audio::init() has opened a device (except the optional WAV dump).
void host_output(const Sample *samples, size_t n);
// Number of samples currently waiting in the host stream (diagnostics/tests), -1 if no device.
int host_queued();
}  // namespace audio::hle
