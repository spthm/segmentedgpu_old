
include ../common.mk

all: build

build: benchmarksort

benchmarksort.o: benchmarksort.cu
	nvcc $(NVCCFLAGS) -o $@ -c $<

sgpuutil.o: ../src/sgpuutil.cpp
	nvcc $(NVCCFLAGS) -o $@ -c $<

benchmarksort: benchmarksort.o sgpuutil.o
	nvcc $(NVCCFLAGS) -o $@ $+

run: build
	./benchmarksort

clean:
	rm -f benchmarksort.o sgpuutil.o benchmarksort

clobber:
	clean

