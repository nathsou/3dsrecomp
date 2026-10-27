// Host audio output through an SDL3 audio stream (32728 Hz stereo s16; SDL resamples to the device).
//
// The DSP tick thread pushes one 160-sample frame every ~4.89 ms. The producer (host timer) and the
// consumer (sound card) run on different clocks, so the stream keeps a small cushion (~40 ms) and
// the playback rate is nudged by up to +/-1% (inaudible) to keep the queue around that level,
// instead of letting it underrun (crackles) or grow without bound (latency).
//
// Environment: RECOMP_NOAUDIO=1 disables output; RECOMP_AUDIO_DUMP=out.wav records the DSP output.
#include <SDL3/SDL.h>

#include <algorithm>
#include <cmath>
#include <mutex>

#include "../audio_iface.h"
#include "host.h"

namespace audio::hle {

namespace {
constexpr int kTargetQueued = kSampleRate * 40 / 1000;  // samples kept in the stream
constexpr int kMaxQueued = kSampleRate * 150 / 1000;    // beyond this frames are dropped
constexpr float kMaxRatioDelta = 0.01f;

struct Host {
    std::mutex lock;
    SDL_AudioStream *stream = nullptr;
    bool playing = false;
    double avg_queued = 0;
    float ratio = 1.0f;
    float volume = 1.0f;
    FILE *dump = nullptr;
    u32 dump_bytes = 0;
    u32 dump_since_header = 0;
};

Host &host() {
    static Host *h = new Host;  // never destroyed: the tick thread may outlive static destructors
    return *h;
}

void put_le32(u8 *p, u32 v) { memcpy(p, &v, 4); }

void write_wav_header(FILE *f, u32 data_bytes) {
    u8 h[44] = {'R', 'I', 'F', 'F', 0, 0, 0, 0, 'W', 'A', 'V', 'E', 'f', 'm', 't', ' ',
                16, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 16, 0,
                'd', 'a', 't', 'a', 0, 0, 0, 0};
    put_le32(h + 4, 36 + data_bytes);
    put_le32(h + 24, kSampleRate);
    put_le32(h + 28, kSampleRate * 4);
    put_le32(h + 40, data_bytes);
    fseek(f, 0, SEEK_SET);
    fwrite(h, 1, sizeof(h), f);
    fseek(f, 0, SEEK_END);
    fflush(f);
}
}  // namespace

int host_queued() {
    Host &h = host();
    std::lock_guard lk(h.lock);
    return h.stream ? SDL_GetAudioStreamQueued(h.stream) / 4 : -1;
}

void host_output(const Sample *samples, size_t n) {
    Host &h = host();
    std::lock_guard lk(h.lock);

    if (h.dump) {
        fwrite(samples, 4, n, h.dump);
        h.dump_bytes += (u32)(n * 4);
        h.dump_since_header += (u32)n;
        if (h.dump_since_header >= (u32)kSampleRate) {  // keep the file valid about once a second
            h.dump_since_header = 0;
            write_wav_header(h.dump, h.dump_bytes);
        }
    }
    if (!h.stream) return;

    int queued = SDL_GetAudioStreamQueued(h.stream) / 4;
    if (queued < 0) return;
    if (h.playing && queued == 0) {
        // Underrun (the DSP stopped, or the producer stalled): pause and rebuild the cushion.
        SDL_PauseAudioStreamDevice(h.stream);
        h.playing = false;
    }
    if (queued > kMaxQueued) return;  // the device stalled; drop rather than add latency
    SDL_PutAudioStreamData(h.stream, samples, (int)(n * 4));
    queued += (int)n;

    if (!h.playing) {
        if (queued >= kTargetQueued) {
            h.ratio = 1.0f;
            h.avg_queued = queued;
            SDL_SetAudioStreamFrequencyRatio(h.stream, 1.0f);
            SDL_ResumeAudioStreamDevice(h.stream);
            h.playing = true;
        }
        return;
    }

    // The device drains in bursts, so steer on a slow average of the queue level.
    h.avg_queued += (queued - h.avg_queued) * 0.01;
    const double err = (h.avg_queued - kTargetQueued) / kTargetQueued;
    const float want = 1.0f + std::clamp((float)(err * 0.02), -kMaxRatioDelta, kMaxRatioDelta);
    if (std::fabs(want - h.ratio) > 0.0005f) {
        h.ratio = want;
        SDL_SetAudioStreamFrequencyRatio(h.stream, h.ratio);
    }
}

}  // namespace audio::hle

namespace audio {

void set_volume(float v) {
    using namespace hle;
    Host &h = host();
    std::lock_guard lk(h.lock);
    h.volume = v;
    if (h.stream) SDL_SetAudioStreamGain(h.stream, v);
}

void init() {
    using namespace hle;
    Host &h = host();
    std::lock_guard lk(h.lock);
    if (h.stream || h.dump) return;

    if (const char *path = getenv("RECOMP_AUDIO_DUMP"); path && *path) {
        h.dump = fopen(path, "wb");
        if (h.dump) write_wav_header(h.dump, 0);
        else LOG("audio: cannot open dump file %s\n", path);
    }
    if (const char *off = getenv("RECOMP_NOAUDIO"); off && *off && *off != '0') return;

    if (!SDL_Init(SDL_INIT_AUDIO)) {
        LOG("audio: SDL_Init(SDL_INIT_AUDIO) failed: %s\n", SDL_GetError());
        return;
    }
    SDL_AudioSpec spec{};
    spec.format = SDL_AUDIO_S16;
    spec.channels = 2;
    spec.freq = kSampleRate;
    h.stream = SDL_OpenAudioDeviceStream(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &spec, nullptr, nullptr);
    if (!h.stream) {
        LOG("audio: cannot open playback device: %s\n", SDL_GetError());
        return;
    }
    SDL_SetAudioStreamGain(h.stream, h.volume);
    // The device starts paused; host_output resumes it once the initial cushion is queued.
    VLOG("audio: output opened (%d Hz stereo s16)\n", kSampleRate);
}

}  // namespace audio
