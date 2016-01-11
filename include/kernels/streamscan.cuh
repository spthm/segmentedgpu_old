/******************************************************************************
 * Copyright (c) 2016, Sam Thomson.  All rights reserved.
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
 * Code and text by Sam Thomson.
 * Segmented GPU is a derivative of Modern GPU.
 * See http://nvlabs.github.io/moderngpu for original repository and
 * documentation.
 *
 ******************************************************************************/

#pragma once

#include "../sgpudevice.cuh"
#include "../device/ctascan.cuh"
#include "../kernels/reduce.cuh"

namespace sgpu {


////////////////////////////////////////////////////////////////////////////////
// KernelStreamScan

template<int TSize>
struct StreamScanTuningSelector;

template<>
struct StreamScanTuningSelector<4> {
	typedef LaunchBoxVT<
		512, 34, 0,
		512, 22, 0
	> Tuning;
};

template<>
struct StreamScanTuningSelector<8> {
	typedef LaunchBoxVT<
		512, 18, 0,
		256, 18, 0
	> Tuning;
};

// The below StreamScan implementation differs slightly from the reference, and
// does not fully conform to the CUDA programming model.
//
// The thread with tid == 0 should call __threadfence() between writing its
// tile's reduction,
//     DeviceL2Store(total, reductions_global + tile),
// and writing the syncronization flag,
//     DeviceL2Store(1, flags_global + tile)
// (c.f. commented-out __threadfence() in kernel.)
//
// However, use of __threadfence() here greatly degrades performance. Instead,
// we use inline PTX asm, marked volatile and with a memory clobber, to
// read/write directly from/to the globally-coherent L2 cache. This appears to
// work as desired on at least SM20/30 devices for at least CUDA <= 7.5.
//
// The reference StreamScan algorithm avoids such issues, as there, the previous
// tile's reduction is also the sychronization flag.
// That is, all reduction values are initialized to some value 'f'. Upon reading
// a value not equal to 'f', we assume the value read is the corresponding
// tile's reduction.
// Unforunately, this is not practical: there is _in general_ no value 'f' we
// can initialize each tile's reduction to which is _guaranteed_ not to be a
// possible result of a reduction.
// That is, if we find reductions_global[tile] == 'f', it is possible that the
// tile's reduction has been computed and written, but that it was equal to 'f'.
// The kernel would then never return, as it is waiting for
//     reductions_global[tile] != 'f'.

template<typename Tuning, SgpuScanType Type, typename DataIt, typename OutputIt,
	typename T, typename Op>
SGPU_LAUNCH_BOUNDS void KernelStreamScan(DataIt data_global, int count,
	T identity, Op op, OutputIt dest_global, volatile T* reductions_global,
	volatile int* flags_global) {

	typedef SGPU_LAUNCH_PARAMS Params;
	const int NT = Params::NT;
	const int VT = Params::VT;
	// To avoid bank conflicts, both rVT and sVT should be odd if possible.
	// Failing that, sVT should be odd.
	const int rVT = (VT % 4 == 1 || VT % 4 == 2) ? VT / 2 : (VT / 2 + 1);
	const int sVT = VT - rVT;
	const int NV = NT * VT;
	const int numTiles = SGPU_DIV_UP(count, NV);

	typedef CTAScan<NT, Op> Scan;
	union Shared {
		typename Scan::Storage scanStorage;
	};
	__shared__ Shared shared;
	__shared__ T sm_data[SGPU_MAX(rVT, sVT) * NT];
	__shared__ T sm_start;

	int tid = threadIdx.x;

	// Persistent threads.
	for(int tile = blockIdx.x; tile < numTiles; tile += gridDim.x) {

		int gid = NV * tile;
		// Number of items in registers.
		int rCount = min(rVT * NT, count - gid);
		// Number of items in shared memory.
		int sCount = min(sVT * NT, count - gid - rCount);

		// Coalesced load of rCount items into thread-order registers.
		T data[rVT];
		DeviceGlobalToSharedDefault<NT, rVT>(rCount, data_global + gid, tid,
            sm_data, identity);
		DeviceSharedToThread<rVT>(sm_data, tid, data); // syncs

		// Coalesced load of sCount items into shared memory.
		DeviceGlobalToSharedDefault<NT, sVT>(sCount,
        	data_global + gid + rCount, tid, sm_data, identity); // syncs

		T* shared_data = sm_data + sVT * tid;

		// Reduce elements within each thread for thread totals.
		T r;
		#pragma unroll
		for(int i = 0; i < rVT; ++i)
			r = i ? op(r, data[i]) : data[i];

		T s;
		#pragma unroll
		for(int i = 0; i < sVT; ++i)
			s = i ? op(s, shared_data[i]) : shared_data[i];

		// Scan thread-totals over the CTA and get tile's reduction.
		T rTotal;
		T rScan = Scan::Scan(tid, r, shared.scanStorage, &rTotal,
			SgpuScanTypeExc, identity, op);
		T sTotal;
		T sScan = Scan::Scan(tid, s, shared.scanStorage, &sTotal,
	    	SgpuScanTypeExc, identity, op);

		// Wait until the block processing the preceeding tile's reduction has
		// completed.
		if(tid == 0 && tile > 0) {
			int flag = 0;
			do {
				DeviceL2Load(flags_global + tile - 1, &flag);
			} while(flag == 0);
		}

		if(tid == 0) {
			// Get the previous tile's reduction.
			T start = identity;
			if(tile > 0) DeviceL2Load(reductions_global + tile - 1, &start);
			sm_start = start;

			// Write this tile's reduction for the next block to use.
			T total = op(op(start, rTotal), sTotal);
			DeviceL2Store(total, reductions_global + tile);

			// Technically necessary; flushes this tile's 'total' before we
			// write its syncronization flag.
			// __threadfence();

			// Write the tile's synchronization flag.
			DeviceL2Store(1, flags_global + tile);
		}
		__syncthreads();

		// Add the thread's CTA scan and previous blocks' reductions as carry-in
		// and re-scan the thread's items.
		T localScan[VT];

		// Thread's in-register items.
		rScan = op(sm_start, rScan);
		#pragma unroll
		for(int i = 0; i < rVT; ++i) {
			if(SgpuScanTypeExc == Type)
				localScan[i] = rScan;
			rScan = op(rScan, data[i]);
			if(SgpuScanTypeInc == Type)
				localScan[i] = rScan;
		}

		// Thread's in-smem items.
		sScan = op(op(sm_start, rTotal), sScan);
		#pragma unroll
		for(int i = 0; i < sVT; ++i) {
			if(SgpuScanTypeExc == Type)
				localScan[i + rVT] = sScan;
			sScan = op(sScan, shared_data[i]);
			if(SgpuScanTypeInc == Type)
				localScan[i + rVT] = sScan;
		}
		__syncthreads(); // shared_data ~ sm_data will next be used for writing.

		// Store tile's in-register scan to dest_global.
		DeviceThreadToShared<rVT>(localScan, tid, sm_data);
		DeviceSharedToGlobal<NT, rVT>(rCount, sm_data, tid, dest_global + gid);

		// Store tile's shared-data scan to dest_global.
		DeviceThreadToShared<sVT>(localScan + rVT, tid, sm_data);
		DeviceSharedToGlobal<NT, sVT>(sCount, sm_data, tid,
			dest_global + gid + rCount);
	}
}

////////////////////////////////////////////////////////////////////////////////

template<SgpuScanType Type, typename DataIt, typename T, typename Op,
	typename DestIt>
SGPU_HOST void StreamScan(DataIt data_global, int count, T identity, Op op,
	T* reduce_global, T* reduce_host, DestIt dest_global,
	CudaContext& context) {

	cudaSharedMemConfig smConfig;
	// Increasing the bank size for 8-byte data types achieves a performance
	// increase of a few per cent where available.
	if (sizeof(T) >= 8) {
		cudaDeviceGetSharedMemConfig(&smConfig);
		cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
	}

	typedef typename StreamScanTuningSelector<sizeof(T)>::Tuning Tuning;
	int2 launch = Tuning::GetLaunchParams(context);
	int NV = launch.x * launch.y;

	int numTiles = SGPU_DIV_UP(count, NV);
	int targetBlocks = context.MaxActiveBlocks(KernelStreamScan<Tuning, Type, DataIt, DestIt, T, Op>,
											   launch.x);
	int numBlocks = std::min(numTiles, targetBlocks);
	SGPU_MEM(T) reduceDevice = context.Malloc<T>(numTiles);
	SGPU_MEM(int) syncFlagsDevice = context.Fill<int>(numTiles, 0);

	KernelStreamScan<Tuning, Type><<<numBlocks, launch.x, 0, context.Stream()>>>(
		data_global, count, identity, op, dest_global, reduceDevice->get(),
		syncFlagsDevice->get());
	SGPU_SYNC_CHECK("KernelStreamScan");

	if(reduce_global)
		copyDtoD(reduce_global, reduceDevice->get() + numTiles - 1, 1);

	if(reduce_host)
		copyDtoH(reduce_host, reduceDevice->get() + numTiles - 1, 1);

	if(sizeof(T) >= 8) {
		cudaDeviceSetSharedMemConfig(smConfig);
	}
}

template<typename InputIt, typename T>
SGPU_HOST void StreamScanExc(InputIt data_global, int count, T* total,
	CudaContext& context) {

	StreamScan<SgpuScanTypeExc>(data_global, count, (T)0, sgpu::plus<T>(),
		(T*)0, total, data_global, context);
}

template<typename InputIt>
SGPU_HOST void StreamScanExc(InputIt data_global, int count,
							 CudaContext& context) {

	typedef typename std::iterator_traits<InputIt>::value_type T;
	StreamScan<SgpuScanTypeExc>(data_global, count, (T)0, sgpu::plus<T>(),
		(T*)0, (T*)0, data_global, context);
}

} // namespace sgpu
