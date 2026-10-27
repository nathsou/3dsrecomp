// The DSP (audio) HLE entry points the dsp::DSP service drives (implemented in runtime/audio/).
// All calls happen with the kernel lock held, except tick() which runs on a host timer thread.
#pragma once
#include <vector>

#include "rt.h"

namespace audio {
void init();                                        // host audio output
void pipe_write(u32 pipe, const u8 *data, u32 size);  // WriteProcessPipe
std::vector<u8> pipe_read(u32 pipe, u32 size);       // ReadPipe / ReadPipeIfPossible (consumes)
u32 pipe_readable(u32 pipe);                         // GetPipeReadableSize
u16 recv_data(u32 reg);                              // RecvData: 0 when on, 1 when off/asleep
void set_volume(float v);  // host output gain (0..1), from the in-game menu
bool tick();  // one audio frame (160 samples at 32728 Hz); true if the DSP is running (-> interrupt)
}  // namespace audio
