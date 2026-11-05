# Standalone build of the software PICA200 GPU plus its unit tests and the dump replay tool.
#   make            -> build/gpu_test, build/gpu_replay
#   make test       -> run unit tests
ROOT := ../..
ifeq ($(origin CXX),default)
CXX := clang++
endif
CXXFLAGS ?= -std=c++20 -O2 -g -Wall -Wno-unused-function -Wno-unused-variable -I$(ROOT)/runtime -I. -DRECOMP_SOURCE_DIR=\"$(abspath $(ROOT))\"
GPU_SRC := $(wildcard $(ROOT)/runtime/gpu/*.cpp)
ifeq ($(shell uname),Darwin)
LDFLAGS ?= -lpthread -framework Metal -framework Foundation
GPU_MM := $(wildcard $(ROOT)/runtime/gpu/*.mm)
GPU_SRC := $(filter-out $(wildcard $(ROOT)/runtime/gpu/vulkan*.cpp $(ROOT)/runtime/gpu/vk_*.cpp),$(GPU_SRC))
else
# the Vulkan backend (runtime/gpu/vulkan.cpp) with shaderc
LDFLAGS ?= -lpthread -ldl $(shell pkg-config --libs vulkan shaderc 2>/dev/null)
CXXFLAGS += -I$(ROOT)/third_party
GPU_MM :=
endif
GPU_OBJ := $(patsubst $(ROOT)/runtime/gpu/%.cpp,build/gpu_%.o,$(GPU_SRC)) $(patsubst $(ROOT)/runtime/gpu/%.mm,build/gpumm_%.o,$(GPU_MM))
COMMON_OBJ := build/stub.o build/png.o
HDR := $(wildcard $(ROOT)/runtime/gpu/*.h) $(ROOT)/runtime/rt.h $(ROOT)/runtime/cpu.h $(wildcard *.h)

all: build/gpu_test build/gpu_replay

build:
	mkdir -p build

build/gpu_%.o: $(ROOT)/runtime/gpu/%.cpp $(HDR) | build
	$(CXX) $(CXXFLAGS) -c $< -o $@

build/gpumm_%.o: $(ROOT)/runtime/gpu/%.mm $(HDR) | build
	$(CXX) $(CXXFLAGS) -fobjc-arc -c $< -o $@

build/%.o: %.cpp $(HDR) | build
	$(CXX) $(CXXFLAGS) -c $< -o $@

build/gpu_test: build/test.o $(COMMON_OBJ) $(GPU_OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

build/gpu_replay: build/replay.o $(COMMON_OBJ) $(GPU_OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

test: build/gpu_test
	./build/gpu_test

clean:
	rm -rf build

.PHONY: all test clean
