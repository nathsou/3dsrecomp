# Unit tests for the DSP firmware HLE in runtime/audio/.  `make run` builds and runs them.
ROOT := ../..
CXX := clang++
SDL_CFLAGS := $(shell pkg-config --cflags sdl3 2>/dev/null || echo -I/opt/homebrew/include)
SDL_LIBS := $(shell pkg-config --libs sdl3 2>/dev/null || echo -L/opt/homebrew/lib -lSDL3)
# same flags the main executable uses for runtime/audio, plus warnings
CXXFLAGS := -std=c++20 -O2 -g -Wall -Wextra -Wno-unused-parameter -I$(ROOT)/runtime $(SDL_CFLAGS)

AUDIO_SRC := $(wildcard $(ROOT)/runtime/audio/*.cpp)
AUDIO_HDR := $(wildcard $(ROOT)/runtime/audio/*.h) $(ROOT)/runtime/audio_iface.h $(ROOT)/runtime/rt.h
OBJ := $(patsubst $(ROOT)/runtime/audio/%.cpp,build/%.o,$(AUDIO_SRC)) build/audio_test.o

audio_test: $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJ) $(SDL_LIBS)

build/%.o: $(ROOT)/runtime/audio/%.cpp $(AUDIO_HDR) | build
	$(CXX) $(CXXFLAGS) -c $< -o $@

build/audio_test.o: audio_test.cpp $(AUDIO_HDR) | build
	$(CXX) $(CXXFLAGS) -c $< -o $@

build:
	mkdir -p build

run: audio_test
	./audio_test

clean:
	rm -rf build audio_test

.PHONY: run clean
