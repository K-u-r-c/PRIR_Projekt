APP          ?= prir
STD          ?= c++20
BUILD_DIR    ?= build
OBJ_DIR      := $(BUILD_DIR)/obj
BIN_DIR      := $(BUILD_DIR)/bin

USE_MPI      ?= 1
USE_OPENMP   ?= 1
USE_CUDA     ?= 1

HOST_CXX     ?= g++
MPI_CXX      ?= mpicxx
CXX          := $(if $(filter 1,$(USE_MPI)),$(MPI_CXX),$(HOST_CXX))
NVCC         ?= nvcc

SRC_CPP      := $(wildcard *.cpp) $(wildcard src/*.cpp)
SRC_CPP      := $(filter-out %/gpu_histogram.cu,$(SRC_CPP))
OBJ_CPP      := $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(SRC_CPP))

ifeq ($(USE_CUDA),1)
SRC_CU       := $(wildcard src/*.cu)
OBJ_CU       := $(patsubst %.cu,$(OBJ_DIR)/%.cu.o,$(SRC_CU))
else
SRC_CU       :=
OBJ_CU       :=
endif

OBJS         := $(OBJ_CPP) $(OBJ_CU)
DEPS         := $(OBJ_CPP:.o=.d)

WARNINGS     := -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wformat=2
COMMON       := -std=$(STD) $(WARNINGS) -MMD -MP

ifeq ($(USE_MPI),1)
COMMON += -DUSE_MPI
endif
ifeq ($(USE_CUDA),1)
COMMON += -DHAVE_CUDA_RUNTIME
endif

ifeq ($(USE_OPENMP),1)
OPENMP_CXX   := -fopenmp
else
OPENMP_CXX   :=
endif

CXXFLAGS_RELEASE := $(COMMON) -O3 -march=native $(OPENMP_CXX)
LDFLAGS_RELEASE  := $(OPENMP_CXX)

CXXFLAGS_DEBUG   := $(COMMON) -O0 -g3 -fno-omit-frame-pointer $(OPENMP_CXX)
LDFLAGS_DEBUG    := $(OPENMP_CXX)

CUDA_HOME    ?= /usr/local/cuda
CUDA_INC_DIR ?= $(CUDA_HOME)/include
CUDA_LIB_DIR ?= $(CUDA_HOME)/lib64
NVCCFLAGS    ?= -O3 -std=c++17 -I$(CUDA_INC_DIR)
ifeq ($(USE_CUDA),1)
NVCCFLAGS    += -DHAVE_CUDA_RUNTIME
endif
CUDA_LIBS     = -L$(CUDA_LIB_DIR) -lcudart

.DEFAULT_GOAL := release

.PHONY: all release debug clean run info

all: release

release: CXXFLAGS := $(CXXFLAGS_RELEASE) $(CXXFLAGS_EXTRA)
release: LDFLAGS  := $(LDFLAGS_RELEASE)  $(LDFLAGS_EXTRA)
release: $(BIN_DIR)/$(APP)

debug:   CXXFLAGS := $(CXXFLAGS_DEBUG)   $(CXXFLAGS_EXTRA)
debug:   LDFLAGS  := $(LDFLAGS_DEBUG)    $(LDFLAGS_EXTRA)
debug:   $(BIN_DIR)/$(APP)

$(BIN_DIR)/$(APP): $(OBJS) | $(BIN_DIR)
	$(CXX) $(OBJS) -o $@ $(LDFLAGS) $(if $(filter 1,$(USE_CUDA)),$(CUDA_LIBS))
	@echo "Built $@"

$(OBJ_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/%.cu.o: %.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BIN_DIR):
	@mkdir -p $@

run: release
	$(BIN_DIR)/$(APP) $(ARGS)

info:
	@echo "CXX          = $(CXX)"
	@echo "NVCC         = $(NVCC)"
	@echo "USE_MPI      = $(USE_MPI)"
	@echo "USE_OPENMP   = $(USE_OPENMP)"
	@echo "USE_CUDA     = $(USE_CUDA)"
	@echo "SOURCES CPP  = $(SRC_CPP)"
	@echo "SOURCES CU   = $(SRC_CU)"

clean:
	rm -rf $(BUILD_DIR)

-include $(DEPS)
