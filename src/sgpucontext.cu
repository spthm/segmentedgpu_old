/******************************************************************************
 * Copyright (c) 2013, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/******************************************************************************
 *
 * Code and text by Sean Baxter, NVIDIA Research
 * See http://nvlabs.github.io/moderngpu for repository and documentation.
 *
 ******************************************************************************/

#include "util/sgpucontext.h"
#include "util/format.h"

namespace sgpu {

////////////////////////////////////////////////////////////////////////////////
// CudaTimer

void CudaTimer::Start() {
	cudaEventRecord(start);
	cudaDeviceSynchronize();
}
double CudaTimer::Split() {
	cudaEventRecord(end);
	cudaDeviceSynchronize();
	float t;
	cudaEventElapsedTime(&t, start, end);
	start.Swap(end);
	return (t / 1000.0);
}
double CudaTimer::Throughput(int count, int numIterations) {
	double elapsed = Split();
	return (double)numIterations * count / elapsed;
}

////////////////////////////////////////////////////////////////////////////////
// CudaDevice

__global__ void KernelVersionShim() { }

struct DeviceGroup {
	int numCudaDevices;
	CudaDevice** cudaDevices;

	DeviceGroup() {
		numCudaDevices = -1;
		cudaDevices = 0;
	}

	int GetDeviceCount() {
		if(-1 == numCudaDevices) {
			cudaError_t error = cudaGetDeviceCount(&numCudaDevices);
			if(cudaSuccess != error || numCudaDevices <= 0) {
				fprintf(stderr, "ERROR ENUMERATING CUDA DEVICES.\nExiting.\n");
				exit(0);
			}
			cudaDevices = new CudaDevice*[numCudaDevices];
			memset(cudaDevices, 0, sizeof(CudaDevice*) * numCudaDevices);
		}
		return numCudaDevices;
	}

	CudaDevice* GetByOrdinal(int ordinal) {
		if(ordinal >= GetDeviceCount()) return 0;

		if(!cudaDevices[ordinal]) {
			// Retrieve the device properties.
			CudaDevice* device = cudaDevices[ordinal] = new CudaDevice;
			device->_ordinal = ordinal;
			cudaError_t error = cudaGetDeviceProperties(&device->_prop,
				ordinal);
			if(cudaSuccess != error) {
				fprintf(stderr, "FAILURE TO CREATE CUDA DEVICE %d\n", ordinal);
				exit(0);
			}

			// Get the compiler version for this device.
			cudaSetDevice(ordinal);
			cudaFuncAttributes attr;
			error = cudaFuncGetAttributes(&attr, KernelVersionShim);
			if(cudaSuccess == error)
				device->_ptxVersion = 10 * attr.ptxVersion;
			else {
				printf("NOT COMPILED WITH COMPATIBLE PTX VERSION FOR DEVICE"
					" %d\n", ordinal);
				// The module wasn't compiled with support for this device.
				device->_ptxVersion = 0;
			}
		}
		return cudaDevices[ordinal];
	}

	~DeviceGroup() {
		if(cudaDevices) {
			for(int i = 0; i < numCudaDevices; ++i)
				delete cudaDevices[i];
			delete [] cudaDevices;
		}
		cudaDeviceReset();
	}
};

std::auto_ptr<DeviceGroup> deviceGroup;


int CudaDevice::DeviceCount() {
	if(!deviceGroup.get())
		deviceGroup.reset(new DeviceGroup);
	return deviceGroup->GetDeviceCount();
}

CudaDevice& CudaDevice::ByOrdinal(int ordinal) {
	if(ordinal < 0 || ordinal >= DeviceCount()) {
		fprintf(stderr, "CODE REQUESTED INVALID CUDA DEVICE %d\n", ordinal);
		exit(0);
	}
	return *deviceGroup->GetByOrdinal(ordinal);
}

CudaDevice& CudaDevice::Selected() {
	int ordinal;
	cudaError_t error = cudaGetDevice(&ordinal);
	if(cudaSuccess != error) {
		fprintf(stderr, "ERROR RETRIEVING CUDA DEVICE ORDINAL\n");
		exit(0);
	}
	return ByOrdinal(ordinal);
}

void CudaDevice::SetActive() {
	cudaError_t error = cudaSetDevice(_ordinal);
	if(cudaSuccess != error) {
		fprintf(stderr, "ERROR SETTING CUDA DEVICE TO ORDINAL %d\n", _ordinal);
		exit(0);
	}
}

std::string CudaDevice::DeviceString() const {
	size_t freeMem, totalMem;
	cudaError_t error = cudaMemGetInfo(&freeMem, &totalMem);
	if(cudaSuccess != error) {
		fprintf(stderr, "ERROR RETRIEVING MEM INFO FOR CUDA DEVICE %d\n",
			_ordinal);
		exit(0);
	}

	double memBandwidth = (_prop.memoryClockRate * 1000.0) *
		(_prop.memoryBusWidth / 8 * 2) / 1.0e9;

	std::string s = stringprintf(
		"%s : %8.3lf Mhz   (Ordinal %d)\n"
		"%d SMs enabled. Compute Capability sm_%d%d\n"
		"FreeMem: %6dMB   TotalMem: %6dMB   %2d-bit pointers.\n"
		"Mem Clock: %8.3lf Mhz x %d bits   (%5.1lf GB/s)\n"
		"ECC %s\n\n",
		_prop.name, _prop.clockRate / 1000.0, _ordinal,
		_prop.multiProcessorCount, _prop.major, _prop.minor,
		(int)(freeMem / (1<< 20)), (int)(totalMem / (1<< 20)), 8 * sizeof(int*),
		_prop.memoryClockRate / 1000.0, _prop.memoryBusWidth, memBandwidth,
		_prop.ECCEnabled ? "Enabled" : "Disabled");
	return s;
}

////////////////////////////////////////////////////////////////////////////////
// CudaContext

struct ContextCache {
	CudaContext** standardContexts;
	int numDevices;

	ContextCache() {
		numDevices = CudaDevice::DeviceCount();
		standardContexts = new CudaContext*[numDevices];
		memset(standardContexts, 0, sizeof(CudaContext*) * numDevices);
	}

	CudaContext* GetByOrdinal(int ordinal) {
		if(!standardContexts[ordinal]) {
			CudaDevice& device = CudaDevice::ByOrdinal(ordinal);
			standardContexts[ordinal] = new CudaContext(device, false);
		}
		return standardContexts[ordinal];
	}

	~ContextCache() {
		if(standardContexts) {
			for(int i = 0; i < numDevices; ++i)
				delete standardContexts[i];
			delete [] standardContexts;
		}
	}
};
std::auto_ptr<ContextCache> contextCache;

CudaContext::CudaContext(CudaDevice& device, bool newStream) :
	_event(cudaEventDisableTiming /*| cudaEventBlockingSync */),
	_stream(0), _noRefCount(false), _pageLocked(0) {

	// Create an allocator.
	_alloc.reset(new CudaAllocSimple(device));

	if(newStream) cudaStreamCreate(&_stream);
	_ownStream = newStream;

	// Allocate 4KB of page-locked memory.
	cudaError_t error = cudaMallocHost((void**)&_pageLocked, 4096);

	// Allocate an auxiliary stream.
	error = cudaStreamCreate(&_auxStream);
}

CudaContext::~CudaContext() {
	if(_pageLocked)
		cudaFreeHost(_pageLocked);
	if(_ownStream && _stream)
		cudaStreamDestroy(_stream);
	if(_auxStream)
		cudaStreamDestroy(_auxStream);
}

CudaContext& CudaContext::CachedContext(int ordinal) {
	bool setActive = -1 != ordinal;
	if(-1 == ordinal) {
		cudaError_t error = cudaGetDevice(&ordinal);
		if(cudaSuccess != error) {
			fprintf(stderr, "ERROR RETRIEVING CUDA DEVICE ORDINAL\n");
			exit(0);
		}
	}
	int numDevices = CudaDevice::DeviceCount();

	if(ordinal < 0 || ordinal >= numDevices) {
		fprintf(stderr, "CODE REQUESTED INVALID CUDA DEVICE %d\n", ordinal);
		exit(0);
	}

	CudaContext& context = *contextCache->GetByOrdinal(ordinal);
	if(!context.PTXVersion()) {
		fprintf(stderr, "This CUDA executable was not compiled with support"
			" for device %d (sm_%2d)\n", ordinal, context.ArchVersion() / 10);
		exit(0);
	}

	if(setActive) context.SetActive();
	return context;
}

ContextPtr CreateCudaDevice(int ordinal) {
	CudaDevice& device = CudaDevice::ByOrdinal(ordinal);
	ContextPtr context(new CudaContext(device, false));
	return context;
}

ContextPtr CreateCudaDeviceStream(int ordinal) {
	ContextPtr context(new CudaContext(CudaDevice::ByOrdinal(ordinal), true));
	return context;
}

ContextPtr CreateCudaDeviceAttachStream(int ordinal, cudaStream_t stream) {
	ContextPtr context(new CudaContext(CudaDevice::ByOrdinal(ordinal), false));
	context->_stream = stream;
	return context;
}

ContextPtr CreateCudaDeviceAttachStream(cudaStream_t stream) {
	int ordinal;
	cudaGetDevice(&ordinal);
	return CreateCudaDeviceAttachStream(ordinal, stream);
}

////////////////////////////////////////////////////////////////////////////////
// CudaAllocSimple

cudaError_t CudaAllocSimple::Malloc(size_t size, void** p) {
	cudaError_t error = cudaSuccess;
	*p = 0;
	if(size) error = cudaMalloc(p, size);

	if(cudaSuccess != error) {
		printf("CUDA MALLOC ERROR %d\n", error);
		exit(0);
	}

	return error;
}

bool CudaAllocSimple::Free(void* p) {
	cudaError_t error = cudaSuccess;
	if(p) error = cudaFree(p);
	return cudaSuccess == error;
}

} // namespace sgpu
