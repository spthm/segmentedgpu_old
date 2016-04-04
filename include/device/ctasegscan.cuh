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

#pragma once

#include "../device/ctascan.cuh"

namespace sgpu {

////////////////////////////////////////////////////////////////////////////////
// DeviceFindSegScanDelta
// Runs an inclusive max-index scan over binary inputs.

template<int NT>
SGPU_DEVICE int DeviceFindSegScanDelta(int tid, bool flag, int* delta_shared) {
	const int NumWarps = NT / 32;

	int warp = tid / 32;
	int lane = 31 & tid;
	uint warpMask = 0xffffffff>> (31 - lane);		// inclusive search
	uint ctaMask = 0x7fffffff>> (31 - lane);		// exclusive search

	uint warpBits = __ballot(flag);
	delta_shared[warp] = warpBits;
	__syncthreads();

	if(tid < NumWarps) {
		uint ctaBits = __ballot(0 != delta_shared[tid]);
		int warpSegment = 31 - clz(ctaMask & ctaBits);
		int start = (-1 != warpSegment) ?
			(31 - clz(delta_shared[warpSegment]) + 32 * warpSegment) : 0;
		delta_shared[NumWarps + tid] = start;
	}
	__syncthreads();

	// Find the closest flag to the left of this thread within the warp.
	// Include the flag for this thread.
	int start = 31 - clz(warpMask & warpBits);
	if(-1 != start) start += ~31 & tid;
	else start = delta_shared[NumWarps + warp];
	__syncthreads();

	return tid - start;
}

////////////////////////////////////////////////////////////////////////////////
// CTASegScan

template<int NT, typename _Op = sgpu::plus<int> >
struct CTASegScan {
	typedef _Op Op;
	typedef typename Op::result_type T;
	enum { NumWarps = NT / 32, Size = NT, Capacity = 2 * NT };
	union Storage {
		int delta[NumWarps];
		T values[Capacity];
	};

	// Each thread passes the reduction of the LAST SEGMENT that it covers.
	// flag is set to true if there's at least one segment flag in the thread.
	// SegScan returns the reduction of values for the first segment in this
	// thread over the preceding threads.
	// Return the value init for the first thread.

	// When scanning single elements per thread, interpret the flag as a BEGIN
	// FLAG. If tid's flag is set, its value belongs to thread tid + 1, not
	// thread tid.

	// The function returns the reduction of the last segment in the CTA.

	SGPU_DEVICE static T SegScanDelta(int tid, int tidDelta, T x,
		Storage& storage, T* carryOut, T identity = (T)0, Op op = Op()) {

		// Run an inclusive scan
		int first = 0;
		storage.values[first + tid] = x;
		__syncthreads();

		#pragma unroll
		for(int offset = 1; offset < NT; offset += offset) {
			if(tidDelta >= offset)
				x = op(storage.values[first + tid - offset], x);
			first = NT - first;
			storage.values[first + tid] = x;
			__syncthreads();
		}

		// Get the exclusive scan.
		x = tid ? storage.values[first + tid - 1] : identity;
		*carryOut = storage.values[first + NT - 1];
		__syncthreads();
		return x;
	}

	SGPU_DEVICE static T SegScan(int tid, T x, bool flag, Storage& storage,
		T* carryOut, T identity = (T)0, Op op = Op()) {

		// Find the left-most thread that covers the first segment of this
		// thread.
		int tidDelta = DeviceFindSegScanDelta<NT>(tid, flag, storage.delta);

		return SegScanDelta(tid, tidDelta, x, storage, carryOut, identity, op);
	}
};

////////////////////////////////////////////////////////////////////////////////
// CTASegScanStore
// Store logic for segmented scan. Essentially the inverse of CTASegReduceLoad.
// Supports direct and indirect stores.
// Pass the scan data in thread order.

template<int NT, int VT, bool HalfCapacity, typename T>
struct CTASegScanStore {
	enum {
		NV = NT * VT,
		Capacity = HalfCapacity ? (NV / 2) : NV
	};

	union Storage {
		int sources[NV];
		T data[Capacity];
	};

	// Store thread-order elements from multiple segments.
	template<typename DestIt>
	SGPU_DEVICE static void StoreDirect(int count2, int tid, int gid,
		T data[VT], DestIt dest_global, Storage& storage) {

		// Transpose data through shared memory.
		T stridedData[VT];
		if(HalfCapacity)
			HalfSmemThreadToReg<NT, VT>(data, tid, storage.data, stridedData);
		else {
			DeviceThreadToShared<VT>(data, tid, storage.data);
			DeviceSharedToReg<NT, VT>(storage.data, tid, stridedData);
		}

		DeviceRegToGlobal<NT, VT>(count2, stridedData, tid, dest_global + gid);
	}

	// Store thread-order elements from multiple segments.
	template<typename DestIt, typename SourcesIt>
	SGPU_DEVICE static void StoreIndirect(int count2, int tid, int gid,
		int numSegments, int startSeg, const int segs[VT + 1],
		const int segStarts[VT], T data[VT], SourcesIt sources_global,
		DestIt dest_global, Storage& storage) {

		// Load source offsets from sources_global into smem.
		DeviceGlobalToSharedLoop<NT, VT>(numSegments, sources_global + startSeg,
			tid, storage.sources);

		// Compute the offset of each element within its segment.
		int indices[VT];
		#pragma unroll
		for(int i = 0; i < VT; ++i) {
			int index = VT * tid + i;
			int segOffset = gid + index - segStarts[i];
			int source = storage.sources[segs[i]];
			indices[i] = (index < count2) ? (source + segOffset) : 0;
		}
		__syncthreads();

		// Transpose indices through shared memory.
		DeviceThreadToShared<VT>(indices, tid, storage.sources);
		DeviceSharedToReg<NT, VT>(storage.sources, tid, indices);

		// Transpose data through shared memory.
		T stridedData[VT];
		if(HalfCapacity)
			HalfSmemThreadToReg<NT, VT>(data, tid, storage.data, stridedData);
		else {
			DeviceThreadToShared<VT>(data, tid, storage.data);
			DeviceSharedToReg<NT, VT>(storage.data, tid, stridedData);
		}

		// Cooperatively store all data elements.
		#pragma unroll
		for(int i = 0; i < VT; ++i) {
			int index = NT * i + tid;
			if (index < count2) dest_global[indices[i]] = stridedData[i];
		}
	}
};

} // namespace sgpu
