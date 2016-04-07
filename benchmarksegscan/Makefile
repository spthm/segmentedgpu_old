
include ../common.mk

all: build

build: benchmarksegscan

benchmarksegscan.o: benchmarksegscan.cu
	nvcc $(NVCCFLAGS) -o $@ -c $<

sgpuutil.o: ../src/sgpuutil.cpp
	nvcc $(NVCCFLAGS) -o $@ -c $<

benchmarksegscan: benchmarksegscan.o sgpuutil.o
	nvcc $(NVCCFLAGS) -o $@ $+

run: build
	./benchmarksegscan

clean:
	rm -f benchmarksegscan.o sgpuutil.o benchmarksegscan

clobber:
	clean

