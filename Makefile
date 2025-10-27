
APP       ?= prir
CXX       ?= g++
STD       ?= c++20

# Sources: works with single main.cpp now; scales to many files later
SRC       := $(wildcard main.cpp) $(wildcard *.cpp) $(wildcard src/*.cpp)
OBJDIR    := build/obj
BINDIR    := build/bin
# Preserve subpaths to avoid name clashes; auto-create dirs for each object
OBJ       := $(patsubst %.cpp,$(OBJDIR)/%.o,$(SRC))
DEPS      := $(OBJ:.o=.d)

# Flags
WARNINGS  := -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wformat=2
COMMON    := -std=$(STD) $(WARNINGS) -MMD -MP
THREADS   := -pthread

# Build types
CXXFLAGS_RELEASE := $(COMMON) -O3 -march=native
LDFLAGS_RELEASE  := $(THREADS)

CXXFLAGS_DEBUG   := $(COMMON) -O0 -g3 -fno-omit-frame-pointer
LDFLAGS_DEBUG    := $(THREADS)

# Default
.DEFAULT_GOAL := release

.PHONY: all release debug clean run install uninstall

all: release

release: CXXFLAGS := $(CXXFLAGS_RELEASE) $(CXXFLAGS_EXTRA)
release: LDFLAGS  := $(LDFLAGS_RELEASE)  $(LDFLAGS_EXTRA)
release: $(BINDIR)/$(APP)

debug:   CXXFLAGS := $(CXXFLAGS_DEBUG)   $(CXXFLAGS_EXTRA)
debug:   LDFLAGS  := $(LDFLAGS_DEBUG)    $(LDFLAGS_EXTRA)
debug:   $(BINDIR)/$(APP)

# Link
$(BINDIR)/$(APP): $(OBJ) | $(BINDIR)
	$(CXX) $(OBJ) -o $@ $(LDFLAGS)
	@echo "Built $@"

# Compile (auto-create object subdirs)
$(OBJDIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Ensure bin dir exists
$(BINDIR):
	@mkdir -p $@

# Convenience
run: release
	$(BINDIR)/$(APP) $(ARGS)

# Install (override PREFIX/DESTDIR as needed)
PREFIX ?= /usr/local
install: release
	install -d $(DESTDIR)$(PREFIX)/bin
	install -m 0755 $(BINDIR)/$(APP) $(DESTDIR)$(PREFIX)/bin/$(APP)

uninstall:
	rm -f $(DESTDIR)$(PREFIX)/bin/$(APP)

clean:
	rm -rf $(OBJDIR) $(BINDIR)

# Include auto-generated deps
-include $(DEPS)
