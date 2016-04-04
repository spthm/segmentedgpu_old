
include ../common.mk

all: build

build: benchmarksegscan

benchmarksegscan.o: benchmarksegscan.cu
	nvcc $(NVCCFLAGS) -o $@ -c $<

sgpucontext.o: ../src/sgpucontext.cu
	nvcc $(NVCCFLAGS) -o $@ -c $<

sgpuutil.o: ../src/sgpuutil.cpp
	nvcc $(NVCCFLAGS) -o $@ -c $<

benchmarksegscan: benchmarksegscan.o sgpucontext.o sgpuutil.o
	nvcc $(NVCCFLAGS) -o $@ $+

run: build
	./benchmarksegscan

clean:
	rm -f benchmarksegscan.o sgpucontext.o sgpuutil.o benchmarksegscan

clobber:
	clean

