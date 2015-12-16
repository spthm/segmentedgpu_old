
include ../common.mk

all: build

build: benchmarklaunchbox

benchmarklaunchbox.o: benchmarklaunchbox.cu
	nvcc $(NVCCFLAGS) -o $@ -c $<

sgpucontext.o: ../src/sgpucontext.cu
	nvcc $(NVCCFLAGS) -o $@ -c $<

sgpuutil.o: ../src/sgpuutil.cpp
	nvcc $(NVCCFLAGS) -o $@ -c $<

benchmarklaunchbox: benchmarklaunchbox.o sgpucontext.o sgpuutil.o
	nvcc $(NVCCFLAGS) -o $@ $+

run: build
	./benchmarklaunchbox

clean:
	rm -f benchmarklaunchbox.o sgpucontext.o sgpuutil.o benchmarklaunchbox

clobber:
	clean

