# Makefile for lamina-c examples
#
# Targets:
#   make              build Rust lib + all stable examples
#   make test         build lib + run all stable examples
#   make nightly      build nightly Rust lib + nightly examples
#   make test-nightly build nightly lib + run nightly examples
#   make clean        remove compiled examples
#
# Variables:
#   PROFILE=release   build in release mode (default: debug)

PROFILE   ?= debug
CARGO     ?= cargo
CC        ?= cc
CFLAGS    ?= -Wall -Wextra -std=c11

# Paths relative to this Makefile (lamina-c/)
EXAMPLES_DIR  := examples
INCLUDE_DIR   := include
CARGO_TARGET  := ../target/$(PROFILE)

# Platform-specific library name and linker flags
UNAME := $(shell uname -s)
ifeq ($(UNAME), Darwin)
  LIB_FILE  := $(CARGO_TARGET)/liblamina_c.dylib
  RPATH_FLAG := -Wl,-rpath,$(abspath $(CARGO_TARGET))
else ifeq ($(OS), Windows_NT)
  LIB_FILE  := $(CARGO_TARGET)/lamina_c.dll
  RPATH_FLAG :=
else
  LIB_FILE  := $(CARGO_TARGET)/liblamina_c.so
  RPATH_FLAG := -Wl,-rpath,$(abspath $(CARGO_TARGET))
endif

LDFLAGS := -L$(CARGO_TARGET) -llamina_c $(RPATH_FLAG)
INCLUDE := -I$(INCLUDE_DIR)

# Cargo profile flag
ifeq ($(PROFILE), release)
  CARGO_PROFILE_FLAG := --release
else
  CARGO_PROFILE_FLAG :=
endif

# Stable examples
STABLE_SRCS   := $(EXAMPLES_DIR)/builder_aot.c $(EXAMPLES_DIR)/raw_ir.c
STABLE_BINS   := $(EXAMPLES_DIR)/builder_aot   $(EXAMPLES_DIR)/raw_ir

# Nightly examples
NIGHTLY_SRCS  := $(EXAMPLES_DIR)/builder_jit.c
NIGHTLY_BINS  := $(EXAMPLES_DIR)/builder_jit

# -------------------------------------------------------------------------

.PHONY: all lib nightly-lib nightly test test-nightly clean

all: lib $(STABLE_BINS)

# lib is phony — cargo decides what needs rebuilding.
# $(LIB_FILE) is a byproduct; examples depend on it for timestamp checks.
lib:
	$(CARGO) build $(CARGO_PROFILE_FLAG)

$(LIB_FILE): lib

# Pattern rule for stable examples
$(EXAMPLES_DIR)/%: $(EXAMPLES_DIR)/%.c $(LIB_FILE)
	$(CC) $(CFLAGS) $< $(INCLUDE) $(LDFLAGS) -o $@

# -------------------------------------------------------------------------
# Nightly
# -------------------------------------------------------------------------

nightly: nightly-lib $(NIGHTLY_BINS)

nightly-lib:
	$(CARGO) build --features nightly $(CARGO_PROFILE_FLAG)

$(EXAMPLES_DIR)/builder_jit: $(EXAMPLES_DIR)/builder_jit.c nightly-lib
	$(CC) $(CFLAGS) $< $(INCLUDE) $(LDFLAGS) -o $@

# -------------------------------------------------------------------------
# Test targets
# -------------------------------------------------------------------------

test: all
	@echo "--- builder_aot ---"
	$(EXAMPLES_DIR)/builder_aot
	@echo ""
	@echo "--- raw_ir ---"
	$(EXAMPLES_DIR)/raw_ir
	@echo ""
	@echo "All stable tests passed."

test-nightly: nightly
	@echo "--- builder_jit ---"
	$(EXAMPLES_DIR)/builder_jit
	@echo ""
	@echo "All nightly tests passed."

# -------------------------------------------------------------------------

clean:
	rm -f $(STABLE_BINS) $(NIGHTLY_BINS)
