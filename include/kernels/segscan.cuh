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

#include "../kernels/csrtools.cuh"

namespace sgpu {

////////////////////////////////////////////////////////////////////////////////
// KernelSegScanSpine1
// Almost identical to KernelSegReduceSpine1, but does not write a row's
// reduction.
// Compute each tile's carry in in-place. Return the carry out for groups of NT
// tiles.
// A final pair of spine-reducers scan carryOut_global and add it into
// carryIn_global.

template<int NT, typename T, typename Op>
__global__ void KernelSegScanSpine1(const int* limits_global, int count,
	T identity, Op op, T* carryIn_global, T* carryOut_global) {

	typedef CTASegScan<NT, Op> SegScan;
	union Shared {
		typename SegScan::Storage segScanStorage;
	};
	__shared__ Shared shared;

	int tid = threadIdx.x;
	int block = blockIdx.x;
	int gid = NT * block + tid;

	// Load the current carry-in and the current and next row indices.
	int row = (gid < count) ?
		(0x7fffffff & limits_global[gid]) :
		INT_MAX;
	int row2 = (gid + 1 < count) ?
		(0x7fffffff & limits_global[gid + 1]) :
		INT_MAX;

	T carryIn = (gid < count) ? carryIn_global[gid] : identity;

	// Run a segmented scan of the carry-in values.
	bool endFlag = row != row2;

	T carryOut;
	T x = SegScan::SegScan(tid, carryIn, endFlag, shared.segScanStorage,
		&carryOut, identity, op);

	carryIn_global[gid] = x;

	// Store the CTA carry-out.
	if(!tid) carryOut_global[block] = carryOut;
}

////////////////////////////////////////////////////////////////////////////////
// KernelSegScanSpine2a
// Loop over the carry outs in carryIn2_global, and convert it to a carry in
// in-place.

template<int NT, typename T, typename Op>
__global__ void KernelSegScanSpine2a(const int* limits_global, int numBlocks,
	int count, int spine1NV, T identity, Op op,	T* carryIn2_global) {

	typedef CTASegScan<NT, Op> SegScan;
	struct Shared {
		typename SegScan::Storage segScanStorage;
		int carryInRow;
		T carryIn;
	};
	__shared__ Shared shared;

	int tid = threadIdx.x;

	for(int i = 0; i < numBlocks; i += NT) {
		int bid = (i + tid);
		int gid = bid * spine1NV;

		// Load the current carry-in and the current and next row indices.
		int row = (gid < count) ?
			(0x7fffffff & limits_global[gid]) : INT_MAX;
		int row2 = (gid + spine1NV < count) ?
			(0x7fffffff & limits_global[gid + spine1NV]) : INT_MAX;
		T carryIn2 = (bid < numBlocks) ? carryIn2_global[bid] : identity;

		// Run a segmented scan of the carry-in values.
		bool endFlag = row != row2;

		T carryOut;
		T x = SegScan::SegScan(tid, carryIn2, endFlag, shared.segScanStorage,
			&carryOut, identity, op);

		if(i && row == shared.carryInRow)
			x = op(shared.carryIn, x);
		carryIn2_global[bid] = x;

		// Set the carry-in for the next loop iteration.
		if(i + NT < numBlocks) {
			__syncthreads();
			if(i > 0) {
				// Add in the previous carry-in.
				if(NT - 1 == tid) {
					shared.carryIn = (shared.carryInRow == row2) ?
						op(shared.carryIn, carryOut) : carryOut;
					shared.carryInRow = row2;
				}
			} else {
				if(NT - 1 == tid) {
					shared.carryIn = carryOut;
					shared.carryInRow = row2;
				}
			}
			__syncthreads();
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
// KernelSegScanSpine2b
// For each per-tile carry in in carryIn_global, add-in the corresponding
// carry in from carryIn2_global, if it corresponds to the same row, or segment.

template<int NT, typename T, typename Op>
__global__ void KernelSegScanSpine2b(const int* limits_global, int count,
	int spine1NV, const T* carryIn2_global, Op op, T* carryIn_global) {

	int tid = threadIdx.x;
	int block = blockIdx.x;
	int gid = NT * block + tid;

	// Load the current row index.
	int row = (gid < count) ? (0x7fffffff & limits_global[gid]) : INT_MAX;

	// Load the row index to which our carryIn2 our corresponds.
	int index = gid / spine1NV;
	int gid2 = index * spine1NV;
	int row2 = (gid2 < count) ? (0x7fffffff & limits_global[gid2]) : INT_MIN;

	// Compute the carry-in.
	// carryIn2 should be added into the current carry in only if they
	// are carry ins for the same row.
	if (row == row2) {
		T carryIn = carryIn_global[gid];
		T carryIn2 = carryIn2_global[index];

		carryIn = op(carryIn2, carryIn);

		carryIn_global[gid] = carryIn;
	}
}

////////////////////////////////////////////////////////////////////////////////
// SegScanSpine
// Take the per-tile carry out from a segmented scan, and convert it in-place to
// a carry in for each tile.
template<typename T, typename Op>
SGPU_HOST void SegScanSpine(const int* limits_global, int count,
	T* carryIn_global, T identity, Op op, CudaContext& context) {

	const int NT = 128;
	int numBlocks = SGPU_DIV_UP(count, NT);

	// Fix-up the segment outputs between the original tiles by performing a
	// segmented, exclusive in-place scan of the original tiles' carry-out.
	SGPU_MEM(T) carryOutDevice = context.Malloc<T>(numBlocks);
	KernelSegScanSpine1<NT><<<numBlocks, NT, 0, context.Stream()>>>(
		limits_global, count, identity, op, carryIn_global,
		carryOutDevice->get());
	SGPU_SYNC_CHECK("KernelSegScanSpine1");

	// Loop over the segments that span the blocks of KernelSegScanSpine1 and
	// similarly fix those.
	if(numBlocks > 1) {
		// convert carryOutDevice, in-place, to a carry-In.
		KernelSegScanSpine2a<NT><<<1, NT, 0, context.Stream()>>>(
			limits_global, numBlocks, count, NT, identity, op,
			carryOutDevice->get());
		SGPU_SYNC_CHECK("KernelSegScanSpine2a");

		// Apply elements of carryOutDevice to their corresponding per-tile
		// elements in carryIn_global.
		KernelSegScanSpine2b<NT><<<numBlocks, NT, 0, context.Stream()>>>(
			limits_global, count, NT, carryOutDevice->get(), op,
			carryIn_global);
		SGPU_SYNC_CHECK("KernelSegScanSpine2b");
	}
}

////////////////////////////////////////////////////////////////////////////////
// Common LaunchBox structure for segmented scans.

template<int NT_, int VT_, int OCC_, bool HalfCapacity_, bool LdgTranspose_>
struct SegScanTuning {
	enum {
		NT = NT_,
		VT = VT_,
		OCC = OCC_,
		HalfCapacity = HalfCapacity_,
		LdgTranspose = LdgTranspose_
	};
};

} // namespace sgpu
