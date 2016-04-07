/******************************************************************************
 * Copyright (c) 2013, NVIDIA CORPORATION; 2016, Sam Thomson.
 * All rights reserved.
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
 * Original code and text by Sean Baxter, NVIDIA Research
 * Modified code and text by Sam Thomson.
 * Segmented GPU is a derivative of Modern GPU.
 * See http://nvlabs.github.io/moderngpu for original repository and
 * documentation.
 *
 ******************************************************************************/

#include "singleton.h"
#include "util/format.h"
#include <cstring>

namespace sgpu {

////////////////////////////////////////////////////////////////////////////////
// CudaEvent and CudaTimer method implementations.

inline CudaEvent::CudaEvent() {
	cudaEventCreate(&_event);
}
inline CudaEvent::CudaEvent(int flags) {
	cudaEventCreateWithFlags(&_event, flags);
}
inline CudaEvent::operator cudaEvent_t() {
	return _event;
}
inline CudaEvent::~CudaEvent() {
	cudaEventDestroy(_event);
}
inline void CudaEvent::Swap(CudaEvent& rhs) {
	std::swap(_event, rhs._event);
}

inline void CudaTimer::Start() {
	cudaEventRecord(start);
	cudaDeviceSynchronize();
}
inline double CudaTimer::Split() {
	cudaEventRecord(end);
	cudaDeviceSynchronize();
	float t;
	cudaEventElapsedTime(&t, start, end);
	start.Swap(end);
	return (t / 1000.0);
}
inline double CudaTimer::Throughput(int count, int numIterations) {
	double elapsed = Split();
	return (double)numIterations * count / elapsed;
}

////////////////////////////////////////////////////////////////////////////////
// CudaDeviceMem method implementations

template<typename T>
cudaError_t CudaDeviceMem<T>::ToDevice(T* data, size_t count) const {
	return ToDevice(0, sizeof(T) * count, data);
}
template<typename T>
cudaError_t CudaDeviceMem<T>::ToDevice(size_t srcOffset, size_t bytes,
	void* data) const {
	cudaError_t error = cudaMemcpy(data, (char*)_p + srcOffset, bytes,
		cudaMemcpyDeviceToDevice);
	if(cudaSuccess != error) {
		printf("CudaDeviceMem::ToDevice copy error %d\n", error);
		exit(0);
	}
	return error;
}

template<typename T>
cudaError_t CudaDeviceMem<T>::ToHost(T* data, size_t count) const {
	return ToHost(0, sizeof(T) * count, data);
}
template<typename T>
cudaError_t CudaDeviceMem<T>::ToHost(std::vector<T>& data, size_t count) const {
	data.resize(count);
	cudaError_t error = cudaSuccess;
	if(_size) error = ToHost(&data[0], count);
	return error;
}
template<typename T>
cudaError_t CudaDeviceMem<T>::ToHost(std::vector<T>& data) const {
	return ToHost(data, _size);
}
template<typename T>
cudaError_t CudaDeviceMem<T>::ToHost(size_t srcOffset, size_t bytes,
	void* data) const {

	cudaError_t error = cudaMemcpy(data, (char*)_p + srcOffset, bytes,
		cudaMemcpyDeviceToHost);
	if(cudaSuccess != error) {
		printf("CudaDeviceMem::ToHost copy error %d\n", error);
		exit(0);
	}
	return error;
}

template<typename T>
cudaError_t CudaDeviceMem<T>::FromDevice(const T* data, size_t count) {
	return FromDevice(0, sizeof(T) * count, data);
}
template<typename T>
cudaError_t CudaDeviceMem<T>::FromDevice(size_t dstOffset, size_t bytes,
	const void* data) {
	if(dstOffset + bytes > sizeof(T) * _size)
		return cudaErrorInvalidValue;
	cudaMemcpy(_p + dstOffset, data, bytes, cudaMemcpyDeviceToDevice);
	return cudaSuccess;
}
template<typename T>
cudaError_t CudaDeviceMem<T>::FromHost(const std::vector<T>& data,
	size_t count) {
	cudaError_t error = cudaSuccess;
	if(data.size()) error = FromHost(&data[0], count);
	return error;
}
template<typename T>
cudaError_t CudaDeviceMem<T>::FromHost(const std::vector<T>& data) {
	return FromHost(data, data.size());
}
template<typename T>
cudaError_t CudaDeviceMem<T>::FromHost(const T* data, size_t count) {
	return FromHost(0, sizeof(T) * count, data);
}
template<typename T>
cudaError_t CudaDeviceMem<T>::FromHost(size_t dstOffset, size_t bytes,
	const void* data) {
	if(dstOffset + bytes > sizeof(T) * _size)
		return cudaErrorInvalidValue;
	cudaMemcpy(_p + dstOffset, data, bytes, cudaMemcpyHostToDevice);
	return cudaSuccess;
}
template<typename T>
CudaDeviceMem<T>::~CudaDeviceMem() {
	_alloc->Free(_p);
}

////////////////////////////////////////////////////////////////////////////////
// CudaMemSupport method implementations

template<typename T>
SGPU_MEM(T) CudaMemSupport::Malloc(size_t count) {
	SGPU_MEM(T) mem(new CudaDeviceMem<T>(_alloc.get()));
	mem->_size = count;
	cudaError_t error = _alloc->Malloc(sizeof(T) * count, (void**)&mem->_p);
	if(cudaSuccess != error) {
		printf("cudaMalloc error %d\n", error);
		exit(0);
		throw CudaException(cudaErrorMemoryAllocation);
	}
#ifdef DEBUG
	// Initialize the memory to -1 in debug mode.
//	cudaMemset(mem->get(), -1, count);
#endif

	return mem;
}

template<typename T>
SGPU_MEM(T) CudaMemSupport::Malloc(const T* data, size_t count) {
	SGPU_MEM(T) mem = Malloc<T>(count);
	mem->FromHost(data, count);
	return mem;
}

template<typename T>
SGPU_MEM(T) CudaMemSupport::Malloc(const std::vector<T>& data) {
	SGPU_MEM(T) mem = Malloc<T>(data.size());
	if(data.size()) mem->FromHost(&data[0], data.size());
	return mem;
}

template<typename T>
SGPU_MEM(T) CudaMemSupport::Fill(size_t count, T fill) {
	std::vector<T> data(count, fill);
	return Malloc(data);
}

template<typename T>
SGPU_MEM(T) CudaMemSupport::FillAscending(size_t count, T first, T step) {
	std::vector<T> data(count);
	for(size_t i = 0; i < count; ++i)
		data[i] = first + i * step;
	return Malloc(data);
}

template<typename T>
SGPU_MEM(T) CudaMemSupport::GenRandom(size_t count, T min, T max) {
	std::vector<T> data(count);
	for(size_t i = 0; i < count; ++i)
		data[i] = Rand(min, max);
	return Malloc(data);
}

template<typename T>
SGPU_MEM(T) CudaMemSupport::SortRandom(size_t count, T min, T max) {
	std::vector<T> data(count);
	for(size_t i = 0; i < count; ++i)
		data[i] = Rand(min, max);
	std::sort(data.begin(), data.end());
	return Malloc(data);
}

template<typename T, typename Func>
SGPU_MEM(T) CudaMemSupport::GenFunc(size_t count, Func f) {
	std::vector<T> data(count);
	for(size_t i = 0; i < count; ++i)
		data[i] = f(i);

	SGPU_MEM(T) mem = Malloc<T>(count);
	mem->FromHost(data, count);
	return mem;
}

////////////////////////////////////////////////////////////////////////////////
// CudaDevice

// Must be template to avoid multiple definition error.
// (Cannot inline a __global__).
template<int dummy>
__global__ void KernelVersionShim() { }

struct DeviceCache {
	int numCudaDevices;
	CudaDevice** cudaDevices;

	DeviceCache() {
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
			error = cudaFuncGetAttributes(&attr, KernelVersionShim<0>);
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

	~DeviceCache() {
		if(cudaDevices) {
			for(int i = 0; i < numCudaDevices; ++i)
				delete cudaDevices[i];
			delete [] cudaDevices;
		}
		cudaDeviceReset();
	}
};
typedef Singleton<DeviceCache> deviceCache;

inline int CudaDevice::DeviceCount() {
	return deviceCache::Instance().GetDeviceCount();
}

inline CudaDevice& CudaDevice::ByOrdinal(int ordinal) {
	if(ordinal < 0 || ordinal >= DeviceCount()) {
		fprintf(stderr, "CODE REQUESTED INVALID CUDA DEVICE %d\n", ordinal);
		exit(0);
	}
	return *(deviceCache::Instance().GetByOrdinal(ordinal));
}

inline CudaDevice& CudaDevice::Selected() {
	int ordinal;
	cudaError_t error = cudaGetDevice(&ordinal);
	if(cudaSuccess != error) {
		fprintf(stderr, "ERROR RETRIEVING CUDA DEVICE ORDINAL\n");
		exit(0);
	}
	return ByOrdinal(ordinal);
}

inline void CudaDevice::SetActive() {
	cudaError_t error = cudaSetDevice(_ordinal);
	if(cudaSuccess != error) {
		fprintf(stderr, "ERROR SETTING CUDA DEVICE TO ORDINAL %d\n", _ordinal);
		exit(0);
	}
}

inline std::string CudaDevice::DeviceString() const {
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

template<typename T>
int CudaDevice::MaxActiveBlocks(T kernel, int blockSize,
	size_t dynamicSMemSize) const {
	int maxBlocksPerSM;

#if CUDA_VERSION < 6050
	// Play it safe-ish. Though in reality the true value may be zero!
	maxBlocksPerSM = 1;

#else
	cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerSM, kernel,
		blockSize, dynamicSMemSize);
#endif

	return maxBlocksPerSM * NumSMs();
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
typedef Singleton<ContextCache> contextCache;

inline CudaContext::CudaContext(CudaDevice& device, bool newStream) :
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

inline CudaContext::~CudaContext() {
	if(_pageLocked)
		cudaFreeHost(_pageLocked);
	if(_ownStream && _stream)
		cudaStreamDestroy(_stream);
	if(_auxStream)
		cudaStreamDestroy(_auxStream);
}

inline CudaContext& CudaContext::CachedContext(int ordinal) {
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

	CudaContext& context = *(contextCache::Instance().GetByOrdinal(ordinal));
	if(!context.PTXVersion()) {
		fprintf(stderr, "This CUDA executable was not compiled with support"
			" for device %d (sm_%2d)\n", ordinal, context.ArchVersion() / 10);
		exit(0);
	}

	if(setActive) context.SetActive();
	return context;
}

inline ContextPtr CreateCudaDevice(int ordinal) {
	CudaDevice& device = CudaDevice::ByOrdinal(ordinal);
	ContextPtr context(new CudaContext(device, false));
	return context;
}

inline ContextPtr CreateCudaDeviceStream(int ordinal) {
	ContextPtr context(new CudaContext(CudaDevice::ByOrdinal(ordinal), true));
	return context;
}

inline ContextPtr CreateCudaDeviceAttachStream(int ordinal, cudaStream_t stream) {
	ContextPtr context(new CudaContext(CudaDevice::ByOrdinal(ordinal), false));
	context->_stream = stream;
	return context;
}

inline ContextPtr CreateCudaDeviceAttachStream(cudaStream_t stream) {
	int ordinal;
	cudaGetDevice(&ordinal);
	return CreateCudaDeviceAttachStream(ordinal, stream);
}

inline ContextPtr CreateCudaDeviceFromArgv(int argc, char** argv,
	bool printInfo) {
	int ordinal = 0;
	if(argc >= 2 && !sscanf(argv[1], "%d", &ordinal)) {
		fprintf(stderr, "INVALID COMMAND LINE ARGUMENT - NOT A CUDA ORDINAL\n");
		exit(0);
	}
	ContextPtr context = CreateCudaDevice(ordinal);
	if(!context->PTXVersion()) {
		fprintf(stderr, "This CUDA executable was not compiled with support"
			" for device %d (sm_%2d)\n", ordinal, context->ArchVersion() / 10);
		exit(0);
	}

	context->SetActive();
	if(printInfo)
		printf("%s\n", context->Device().DeviceString().c_str());
	return context;
}

inline ContextPtr CreateCudaDeviceStreamFromArgv(int argc, char** argv,
	bool printInfo) {
	int ordinal = 0;
	if(argc >= 2 && !sscanf(argv[1], "%d", &ordinal)) {
		fprintf(stderr, "INVALID COMMAND LINE ARGUMENT - NOT A CUDA ORDINAL\n");
		exit(0);
	}
	ContextPtr context = CreateCudaDeviceStream(ordinal);
	if(!context->PTXVersion()) {
		fprintf(stderr, "This CUDA executable was not compiled with support"
			" for device %d (sm_%2d)\n", ordinal, context->ArchVersion() / 10);
		exit(0);
	}

	context->SetActive();
	if(printInfo)
		printf("%s\n", context->Device().DeviceString().c_str());
	return context;
}

} // namespace sgpu
