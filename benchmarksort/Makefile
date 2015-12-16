
include ../common.mk

all: build

build: benchmarksort

benchmarksort.o: benchmarksort.cu
	nvcc $(NVCCFLAGS) -o $@ -c $<

sgpucontext.o: ../src/sgpucontext.cu
	nvcc $(NVCCFLAGS) -o $@ -c $<

sgpuutil.o: ../src/sgpuutil.cpp
	nvcc $(NVCCFLAGS) -o $@ -c $<

benchmarksort: benchmarksort.o sgpucontext.o sgpuutil.o
	nvcc $(NVCCFLAGS) -o $@ $+

run: build
	./benchmarksort

clean:
	rm -f benchmarksort.o sgpucontext.o sgpuutil.o benchmarksort

clobber:
	clean

