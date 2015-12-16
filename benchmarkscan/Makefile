
include ../common.mk

all: build

build: benchmarkscan

benchmarkscan.o: benchmarkscan.cu
	nvcc $(NVCCFLAGS) -o $@ -c $<

sgpucontext.o: ../src/sgpucontext.cu
	nvcc $(NVCCFLAGS) -o $@ -c $<

sgpuutil.o: ../src/sgpuutil.cpp
	nvcc $(NVCCFLAGS) -o $@ -c $<

benchmarkscan: benchmarkscan.o sgpucontext.o sgpuutil.o
	nvcc $(NVCCFLAGS) -o $@ $+

run: build
	./benchmarkscan

clean:
	rm -f benchmarkscan.o sgpucontext.o sgpuutil.o benchmarkscan

clobber:
	clean

