
include ../common.mk

all: build

build: benchmarksegsort

benchmarksegsort.o: benchmarksegsort.cu
	nvcc $(NVCCFLAGS) -o $@ -c $<

sgpucontext.o: ../src/sgpucontext.cu
	nvcc $(NVCCFLAGS) -o $@ -c $<

sgpuutil.o: ../src/sgpuutil.cpp
	nvcc $(NVCCFLAGS) -o $@ -c $<

benchmarksegsort: benchmarksegsort.o sgpucontext.o sgpuutil.o
	nvcc $(NVCCFLAGS) -o $@ $+

run: build
	./benchmarksegsort

clean:
	rm -f benchmarksegsort.o sgpucontext.o sgpuutil.o benchmarksegsort

clobber:
	clean

