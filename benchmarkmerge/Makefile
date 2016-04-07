
include ../common.mk

all: build

build: benchmarkmerge

benchmarkmerge.o: benchmarkmerge.cu
	nvcc $(NVCCFLAGS) -o $@ -c $<

sgpuutil.o: ../src/sgpuutil.cpp
	nvcc $(NVCCFLAGS) -o $@ -c $<

benchmarkmerge: benchmarkmerge.o sgpuutil.o
	nvcc $(NVCCFLAGS) -o $@ $+

run: build
	./benchmarkmerge

clean:
	rm -f benchmarkmerge.o sgpuutil.o benchmarkmerge

clobber:
	clean

