
include ../common.mk

all: build

build: benchmarksegsort

benchmarksegsort.o: benchmarksegsort.cu
	nvcc $(NVCCFLAGS) -o $@ -c $<

sgpuutil.o: ../src/sgpuutil.cpp
	nvcc $(NVCCFLAGS) -o $@ -c $<

benchmarksegsort: benchmarksegsort.o sgpuutil.o
	nvcc $(NVCCFLAGS) -o $@ $+

run: build
	./benchmarksegsort

clean:
	rm -f benchmarksegsort.o sgpuutil.o benchmarksegsort

clobber:
	clean

