
#pragma once

#include "util/util.h"
#include <cuda.h>
#include <cuda_runtime_api.h>

namespace sgpu {

class CudaDevice;

class CudaContext;
typedef intrusive_ptr<CudaContext> ContextPtr;

////////////////////////////////////////////////////////////////////////////////
// Customizable allocator.

// CudaAlloc is the interface class all allocator accesses. Users may derive
// this, implement custom allocators, and set it to the device with
// CudaDevice::SetAllocator.

class CudaAlloc : public CudaBase {
public:
	virtual cudaError_t Malloc(size_t size, void** p) = 0;
	virtual bool Free(void* p) = 0;
	virtual void Clear() = 0;

	virtual ~CudaAlloc() { }

	CudaDevice& Device() { return _device; }

protected:
	CudaAlloc(CudaDevice& device) : _device(device) { }
	CudaDevice& _device;
};

// A concrete class allocator that simply calls cudaMalloc and cudaFree.
class CudaAllocSimple : public CudaAlloc {
public:
	CudaAllocSimple(CudaDevice& device) : CudaAlloc(device) { }

	virtual cudaError_t Malloc(size_t size, void** p);
	virtual bool Free(void* p);
	virtual void Clear() { }
	virtual ~CudaAllocSimple() { }
};

} // namespace sgpu
