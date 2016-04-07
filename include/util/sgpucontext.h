#pragma once

#include "util/util.h"
#include "sgpualloc.h"
#include <cuda.h>
#include <cuda_runtime_api.h>

namespace sgpu {


#ifdef _DEBUG
#define SGPU_SYNC_CHECK(s) {												\
	cudaError_t error = cudaDeviceSynchronize();							\
	if(cudaSuccess != error) {												\
		printf("CUDA ERROR %d %s\n%s:%d.\n%s\n",							\
			error, cudaGetErrorString(error), __FILE__, __LINE__, s);		\
		exit(0);															\
	}																		\
}
#else
#define SGPU_SYNC_CHECK(s)
#endif

template<typename T>
void copyDtoH(T* dest, const T* source, int count) {
	cudaMemcpy(dest, source, sizeof(T) * count, cudaMemcpyDeviceToHost);
}
template<typename T>
void copyDtoD(T* dest, const T* source, int count, cudaStream_t stream = 0) {
	cudaMemcpyAsync(dest, source, sizeof(T) * count, cudaMemcpyDeviceToDevice,
		stream);
}
template<typename T>
void copyDtoH(std::vector<T>& dest, const T* source, int count) {
	dest.resize(count);
	if(count)
		copyDtoH(&dest[0], source, count);
}

template<typename T>
void copyHtoD(T* dest, const T* source, int count) {
	cudaMemcpy(dest, source, sizeof(T) * count, cudaMemcpyHostToDevice);
}
template<typename T>
void copyHtoD(T* dest, const std::vector<T>& source) {
	if(source.size())
		copyHtoD(dest, &source[0], source.size());
}


////////////////////////////////////////////////////////////////////////////////

class CudaContext;
typedef intrusive_ptr<CudaContext> ContextPtr;
typedef intrusive_ptr<CudaAlloc> AllocPtr;

class CudaException : public std::exception {
public:
	cudaError_t error;

	CudaException() throw() { }
	CudaException(cudaError_t e) throw() : error(e) { }
	CudaException(const CudaException& e) throw() : error(e.error) { }

	virtual const char* what() const throw() {
		return "CUDA runtime error";
	}
};


////////////////////////////////////////////////////////////////////////////////
// CudaEvent and CudaTimer.
// Exception-safe wrappers around cudaEvent_t.

class CudaEvent : public noncopyable {
public:
	CudaEvent() {
		cudaEventCreate(&_event);
	}
	explicit CudaEvent(int flags) {
		cudaEventCreateWithFlags(&_event, flags);
	}
	~CudaEvent() {
		cudaEventDestroy(_event);
	}
	operator cudaEvent_t() { return _event; }
	void Swap(CudaEvent& rhs) {
		std::swap(_event, rhs._event);
	}
private:
	cudaEvent_t _event;
};

class CudaTimer : noncopyable {
	CudaEvent start, end;
public:
	void Start();
	double Split();
	double Throughput(int count, int numIterations);
};


////////////////////////////////////////////////////////////////////////////////

struct DeviceCache;

class CudaDevice : public noncopyable {
	friend struct DeviceCache;
public:
	static int DeviceCount();
	static CudaDevice& ByOrdinal(int ordinal);
	static CudaDevice& Selected();

	// Device properties.
	const cudaDeviceProp& Prop() const { return _prop; }
	int Ordinal() const { return _ordinal; }
	int NumSMs() const { return _prop.multiProcessorCount; }
	int ArchVersion() const { return 100 * _prop.major + 10 * _prop.minor; }

	// LaunchBox properties.
	int PTXVersion() const { return _ptxVersion; }

	template<typename T>
	int MaxActiveBlocks(T kernel, int blockSize, size_t dynamicSMemSize = 0) const;

	std::string DeviceString() const;

	// Set this device as the active device on the thread.
	void SetActive();

private:
	CudaDevice() { }		// hide the destructor.
	int _ordinal;
	int _ptxVersion;
	cudaDeviceProp _prop;
};

////////////////////////////////////////////////////////////////////////////////
// CudaDeviceMem
// Exception-safe CUDA device memory container. Use the SGPU_MEM(T) macro for
// the type of the reference-counting container.
// CudaDeviceMem AddRefs the allocator that returned the memory, releasing the
// pointer when the object is destroyed.

template<typename T>
class CudaDeviceMem : public CudaBase {
	friend class CudaMemSupport;
public:
	~CudaDeviceMem();

	const T* get() const { return _p; }
	T* get() { return _p; }

	operator const T*() const { return get(); }
	operator T*() { return get(); }

	// Size is in units of T, not bytes.
	size_t Size() const { return _size; }

	// Copy from this to the argument array.
	cudaError_t ToDevice(T* data, size_t count) const;
	cudaError_t ToDevice(size_t srcOffest, size_t bytes, void* data) const;
	cudaError_t ToHost(T* data, size_t count) const;
	cudaError_t ToHost(std::vector<T>& data) const;
	cudaError_t ToHost(std::vector<T>& data, size_t count) const;
	cudaError_t ToHost(size_t srcOffset, size_t bytes, void* data) const;

	// Copy from the argument array to this.
	cudaError_t FromDevice(const T* data, size_t count);
	cudaError_t FromDevice(size_t dstOffset, size_t bytes, const void* data);
	cudaError_t FromHost(const std::vector<T>& data);
	cudaError_t FromHost(const std::vector<T>& data, size_t count);
	cudaError_t FromHost(const T* data, size_t count);
	cudaError_t FromHost(size_t destOffset, size_t bytes, const void* data);

private:
	friend class CudaContext;
	CudaDeviceMem(CudaAlloc* alloc) : _p(0), _size(0), _alloc(alloc) { }

	AllocPtr _alloc;
	T* _p;
	size_t _size;
};

typedef intrusive_ptr<CudaAlloc> AllocPtr;
#define SGPU_MEM(type) sgpu::intrusive_ptr< sgpu::CudaDeviceMem< type > >

////////////////////////////////////////////////////////////////////////////////
// CudaMemSupport
// Convenience functions for allocating device memory and copying to it from
// the host. These functions are factored into their own class for clarity.
// The class is derived by CudaContext.

class CudaMemSupport : public CudaBase {
	friend class CudaDevice;
	friend class CudaContext;
public:
	CudaDevice& Device() { return _alloc->Device(); }

	// Swap out the associated allocator.
	void SetAllocator(CudaAlloc* alloc) {
		assert(alloc->Device().Ordinal() == _alloc->Device().Ordinal());
		_alloc.reset(alloc);
	}

	// Access the associated allocator.
	CudaAlloc* GetAllocator() { return _alloc.get(); }

	// Support for creating arrays.
	template<typename T>
	SGPU_MEM(T) Malloc(size_t count);

	template<typename T>
	SGPU_MEM(T) Malloc(const T* data, size_t count);

	template<typename T>
	SGPU_MEM(T) Malloc(const std::vector<T>& data);

	template<typename T>
	SGPU_MEM(T) Fill(size_t count, T fill);

	template<typename T>
	SGPU_MEM(T) FillAscending(size_t count, T first, T step);

	template<typename T>
	SGPU_MEM(T) GenRandom(size_t count, T min, T max);

	template<typename T>
	SGPU_MEM(T) SortRandom(size_t count, T min, T max);

	template<typename T, typename Func>
	SGPU_MEM(T) GenFunc(size_t count, Func f);

protected:
	CudaMemSupport() { }
	AllocPtr _alloc;
};

////////////////////////////////////////////////////////////////////////////////

class CudaContext;
typedef sgpu::intrusive_ptr<CudaContext> ContextPtr;

// Create a context on the default stream (0).
ContextPtr CreateCudaDevice(int ordinal);

// Create a context on a new stream.
ContextPtr CreateCudaDeviceStream(int ordinal);

// Create a context and attach to an existing stream.
ContextPtr CreateCudaDeviceAttachStream(cudaStream_t stream);
ContextPtr CreateCudaDeviceAttachStream(int ordinal, cudaStream_t stream);

struct ContextCache;

class CudaContext : public CudaMemSupport {
	friend struct ContextCache;

	friend ContextPtr CreateCudaDevice(int ordinal);
	friend ContextPtr CreateCudaDeviceStream(int ordinal);
	friend ContextPtr CreateCudaDeviceAttachStream(int ordinal,
		cudaStream_t stream);
public:
	static CudaContext& CachedContext(int ordinal = -1);

	// 4KB of page-locked memory per context.
	int* PageLocked() { return _pageLocked; }
	cudaStream_t AuxStream() const { return _auxStream; }

	int NumSMs() { return Device().NumSMs(); }
	int ArchVersion() { return Device().ArchVersion(); }
	int PTXVersion() { return Device().PTXVersion(); }

	template<typename T>
	int MaxActiveBlocks(T kernel, int blockSize, size_t dynamicSMemSize = 0) {
		return Device().MaxActiveBlocks(kernel, blockSize, dynamicSMemSize);
	}

	std::string DeviceString() { return Device().DeviceString(); }

	cudaStream_t Stream() const { return _stream; }

	// Set this device as the active device on the thread.
	void SetActive() { Device().SetActive(); }

	// Access the included event.
	CudaEvent& Event() { return _event; }

	// Use the included timer.
	CudaTimer& Timer() { return _timer; }
	void Start() { _timer.Start(); }
	double Split() { return _timer.Split(); }
	double Throughput(int count, int numIterations) {
		return _timer.Throughput(count, numIterations);
	}

	virtual long AddRef() {
		return _noRefCount ? 1 : CudaMemSupport::AddRef();
	}
	virtual void Release() {
		if(!_noRefCount) CudaMemSupport::Release();
	}
private:
	CudaContext(CudaDevice& device, bool newStream);
	~CudaContext();

	bool _ownStream;
	cudaStream_t _stream;
	cudaStream_t _auxStream;
	CudaEvent _event;
	CudaTimer _timer;
	bool _noRefCount;
	int* _pageLocked;
};


////////////////////////////////////////////////////////////////////////////////


} // namespace sgpu

#include "sgpucontext.inl"
