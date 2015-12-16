
include ../common.mk

all: build

build: testlaunchbox

testlaunchbox.o: testlaunchbox.cu
	nvcc $(NVCCFLAGS) -o $@ -c $<

sgpucontext.o: ../src/sgpucontext.cu
	nvcc $(NVCCFLAGS) -o $@ -c $<

sgpuutil.o: ../src/sgpuutil.cpp
	nvcc $(NVCCFLAGS) -o $@ -c $<

testlaunchbox: testlaunchbox.o sgpucontext.o sgpuutil.o
	nvcc $(NVCCFLAGS) -o $@ $+

run: build
	./testlaunchbox

clean:
	rm -f testlaunchbox.o sgpucontext.o sgpuutil.o testlaunchbox

clobber:
	clean

