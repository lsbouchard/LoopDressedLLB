CC ?= cc
CFLAGS ?= -O3 -std=c11 -Wall -Wextra -pedantic
CPPFLAGS += -Iinclude
LDFLAGS ?=
LDLIBS ?= -lm -lpthread -llapack -lblas

UNAME_S := $(shell uname -s)

SRC_COMMON = src/main.c src/jpg.c
OBJ_COMMON = $(SRC_COMMON:.c=.o)

ifeq ($(UNAME_S),Darwin)
  METAL_OBJ = src/metal_backend.o
  METAL_LDLIBS = -framework Foundation -framework Metal
else
  METAL_OBJ = src/metal_backend_stub.o
  METAL_LDLIBS =
endif

.PHONY: all clean test metal-shaders

all: bin/jeener_packet_c

bin/jeener_packet_c: $(OBJ_COMMON) $(METAL_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS) $(METAL_LDLIBS)

bin:
	mkdir -p bin

build:
	mkdir -p build

%.o: %.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -c -o $@ $<

src/metal_backend.o: src/metal_backend.m
	clang $(CPPFLAGS) -O3 -fobjc-arc -c -o $@ $<

test: all
	PYTHONPATH=tests python3 tests/compare_python.py

metal-shaders:
	@echo "No offline shader build is required. Use --backend metal for optional runtime Metal shader compilation."

clean:
	rm -rf bin build src/*.o tests/__pycache__ tests/out
