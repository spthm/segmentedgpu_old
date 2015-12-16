
include ../common.mk

all: build

build: benchmarksegreduce

benchmarksegreduce.o: benchmarksegreduce.cu
	nvcc $(NVCCFLAGS) -o $@ -c $<

sgpucontext.o: ../src/sgpucontext.cu
	nvcc $(NVCCFLAGS) -o $@ -c $<

sgpuutil.o: ../src/sgpuutil.cpp
	nvcc $(NVCCFLAGS) -o $@ -c $<

benchmarksegreduce: benchmarksegreduce.o sgpucontext.o sgpuutil.o
	nvcc $(NVCCFLAGS) -o $@ $+

run: build
	./benchmarksegreduce

clean:
	rm -f benchmarksegreduce.o sgpucontext.o sgpuutil.o benchmarksegreduce

clobber:
	clean

