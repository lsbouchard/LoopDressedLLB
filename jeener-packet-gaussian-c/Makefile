CC ?= cc
CFLAGS ?= -O3 -std=c11 -Wall -Wextra -pedantic
CPPFLAGS += -Iinclude
OBJCFLAGS ?= -O3 -fobjc-arc
PYTHON ?= python3
METAL ?= 1
LDFLAGS ?=
LDLIBS ?= -lm -lpthread -llapack -lblas

UNAME_S := $(shell uname -s)

SRC_COMMON = src/main.c src/jpg.c
OBJ_COMMON = $(SRC_COMMON:.c=.o)

ifeq ($(UNAME_S),Darwin)
  MACOS_SDK ?= $(shell xcrun --sdk macosx --show-sdk-path)
  CPPFLAGS += -isysroot $(MACOS_SDK)
  LDFLAGS += -isysroot $(MACOS_SDK)
ifeq ($(METAL),1)
  METAL_OBJ = src/metal_backend.o
  METAL_LDLIBS = -framework Foundation -framework Metal
else
  METAL_OBJ = src/metal_backend_stub.o
  METAL_LDLIBS =
endif
else
  METAL_OBJ = src/metal_backend_stub.o
  METAL_LDLIBS =
endif

.PHONY: all clean test test-numerics metal-shaders

all: bin/jeener_packet_c

bin/jeener_packet_c: $(OBJ_COMMON) $(METAL_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS) $(METAL_LDLIBS)

bin:
	mkdir -p bin

build:
	mkdir -p build

%.o: %.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -MMD -MP -c -o $@ $<

src/metal_backend.o: src/metal_backend.m
	$(CC) $(CPPFLAGS) $(OBJCFLAGS) -MMD -MP -c -o $@ $<

bin/test_numerics: tests/test_numerics.c src/jpg.c include/jpg.h | bin
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) -o $@ $< $(LDLIBS)

test-numerics: bin/test_numerics
	./bin/test_numerics

test: all test-numerics
	$(PYTHON) tests/compare_python.py

-include $(OBJ_COMMON:.o=.d) $(METAL_OBJ:.o=.d)

metal-shaders:
	@echo "No offline shader build is required. Use --backend metal for optional runtime Metal shader compilation."

clean:
	rm -rf bin build src/*.o src/*.d tests/__pycache__ tests/out
