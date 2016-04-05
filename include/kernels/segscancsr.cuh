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

#include "../sgpuhost.cuh"
#include "../kernels/segscan.cuh"
#include "../kernels/bulkinsert.cuh"

namespace sgpu {

// SegScanCSR - Normal
template<size_t size>
struct SegScanNormalTuning {
	typedef LaunchBox<
		SegScanTuning<128, 11, 0, false, false>,
		SegScanTuning<128, 7, 0, true, false>,
		SegScanTuning<128, (size > sizeof(int)) ? 11 : 7, 0, true, true>
	> Tuning;
};

// SegScanCSR - Preprocess
template<size_t size>
struct SegScanPreprocessTuning {
	typedef LaunchBox<
		SegScanTuning<128, 11, 0, false, false>,
		SegScanTuning<128, 11, 0, true, false>,
		SegScanTuning<128, 11, 0, true, (size > 4) ? false : true>
	> Tuning;
};

// SegScanCSR - Indirect
template<size_t size>
struct SegScanIndirectTuning {
	typedef LaunchBox<
		SegScanTuning<128, 11, 0, false, false>,
		SegScanTuning<128, 7, 0, true, false>,
		SegScanTuning<128, 7, 0, true, true>
	> Tuning;
};


////////////////////////////////////////////////////////////////////////////////
// KernelSegScanCsrUpsweep
// Almost identical to KernelSegReduceCsr, but it does not write a tile's
// reduction, only its carry-out.

template<typename Tuning, bool Indirect, typename CsrIt, typename SourcesIt,
	typename InputIt, typename T, typename Op>
SGPU_LAUNCH_BOUNDS void KernelSegScanCsrUpsweep(int numTiles, CsrIt csr_global,
	SourcesIt sources_global, int count, const int* limits_global,
	InputIt data_global, T identity, Op op, T* carryOut_global) {

	typedef SGPU_LAUNCH_PARAMS Params;
	typedef typename Op::first_argument_type OpT;
	const int NT = Params::NT;
	const int VT = Params::VT;
	const int NV = NT * VT;
	const bool HalfCapacity = (sizeof(T) > sizeof(int)) && Params::HalfCapacity;
	const bool LdgTranspose = Params::LdgTranspose;

	typedef CTASegReduceLoad<NT, VT, HalfCapacity, LdgTranspose, T>
		SegReduceLoad;
	typedef CTAReduce<NT, OpT, Op> FastReduce;
	typedef CTASegScan<NT, Op> SegScan;

	union Shared {
		int csr[NV + 1];
		typename SegReduceLoad::Storage loadStorage;
		typename FastReduce::Storage reduceStorage;
		typename SegScan::Storage segScanStorage;
	};
	__shared__ Shared shared;

	int tid = threadIdx.x;

	for(int block = blockIdx.x; block < numTiles; block += gridDim.x) {
		__syncthreads();

		int gid = NV * block;
		int count2 = min(NV, count - gid);

		int limit0 = limits_global[block];
		int limit1 = limits_global[block + 1];

		SegReduceRange range;
		SegReduceTerms terms;
		int segs[VT + 1], segStarts[VT];
		T data[VT];
		if(Indirect) {
			// Indirect load. We need to load the CSR terms before loading any
			// data.
			range = DeviceShiftRange(limit0, limit1);
			int numSegments = range.end - range.begin;

			// Load the CSR interval.
			DeviceGlobalToSharedLoop<NT, VT>(numSegments,
				csr_global + range.begin, tid, shared.csr);

			// Compute the segmented scan terms.
			terms = DeviceSegReducePrepare<NT, VT>(shared.csr, numSegments,
				tid, gid, range.flushLast, segs, segStarts);

			// Load tile of data in thread order from segment IDs.
			SegReduceLoad::LoadIndirect(count2, tid, gid, numSegments,
				range.begin, segs, segStarts, data_global, sources_global,
				identity, data, shared.loadStorage);

			// SegReduceLoad::LoadIndirectFast loads in register order, not
			// thread order, so is not appropriate for a scan.

		} else {
			// Direct load. It is more efficient to load the full tile before
			// dealing with data dependencies.
			SegReduceLoad::LoadDirect(count2, tid, gid, data_global, identity,
				data, shared.loadStorage);

			range = DeviceShiftRange(limit0, limit1);
			int numSegments = range.end - range.begin;

			if(range.total) {
				// Load the CSR interval.
				DeviceGlobalToSharedLoop<NT, VT>(numSegments,
					csr_global + range.begin, tid, shared.csr);

				// Compute the segmented scan terms.
				terms = DeviceSegReducePrepare<NT, VT>(shared.csr, numSegments,
					tid, gid, range.flushLast, segs, segStarts);
			}
		}

		if(range.total) {
			// Reduce tile data and store tile's carry-out term to
			// carryOut_global.

			// Compute thread's carry out.
			T x;
			#pragma unroll
			for(int i = 0; i < VT; ++i) {
				x = i ? op(x, data[i]) : data[i];
				if(segs[i] != segs[i + 1]) x = identity;
			}

			// Run a parallel segmented scan over each thread's carry-out values
			// to compute the CTA's carry-out.
			T carryOut;
			SegScan::SegScanDelta(tid, terms.tidDelta, x, shared.segScanStorage,
				&carryOut, identity, op);

			// Store the carry-out for the entire CTA to global memory.
			if(!tid)
				carryOut_global[block] = carryOut;
		} else {
			T x;
			#pragma unroll
			for(int i = 0; i < VT; ++i)
				x = i ? op(x, data[i]) : data[i];
			x = FastReduce::Reduce(tid, x, shared.reduceStorage, op);
			if(!tid)
				carryOut_global[block] = x;
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
// KernelSegScanCsrDownsweep
// Similar to KernelSegScanCsrUpsweep, but stores the final scanned values to
// the output, and does not store any carry-out.

template<typename Tuning, bool Indirect, SgpuScanType Type, typename CsrIt,
	typename SourcesIt,	typename InputIt, typename DestIt, typename T,
	typename Op>
SGPU_LAUNCH_BOUNDS void KernelSegScanCsrDownsweep(int numTiles,
	CsrIt csr_global, SourcesIt sources_global, int count,
	const int* limits_global, InputIt data_global, const T* carryIn_global,
	T identity, Op op, DestIt dest_global) {

	typedef SGPU_LAUNCH_PARAMS Params;
	typedef typename Op::first_argument_type OpT;
	const int NT = Params::NT;
	const int VT = Params::VT;
	const int NV = NT * VT;
	const bool HalfCapacity = (sizeof(T) > sizeof(int)) && Params::HalfCapacity;
	const bool LdgTranspose = Params::LdgTranspose;

	typedef CTASegReduceLoad<NT, VT, HalfCapacity, LdgTranspose, T>
		SegReduceLoad;
	typedef CTAScan<NT, Op> Scan;
	typedef CTASegScan<NT, Op> SegScan;
	typedef CTASegScanStore<NT, VT, HalfCapacity, T> SegScanStore;

	union Shared {
		int csr[NV + 1];
		typename SegReduceLoad::Storage loadStorage;
		typename Scan::Storage scanStorage;
		typename SegScan::Storage segScanStorage;
		typename SegScanStore::Storage storeStorage;
	};
	__shared__ Shared shared;

	int tid = threadIdx.x;

	for(int block = blockIdx.x; block < numTiles; block += gridDim.x) {
		__syncthreads();

		int gid = NV * block;
		int count2 = min(NV, count - gid);

		int limit0 = limits_global[block];
		int limit1 = limits_global[block + 1];

		SegReduceRange range;
		SegReduceTerms terms;
		int segs[VT + 1], segStarts[VT];
		T data[VT];
		if(Indirect) {
			// Indirect load. We need to load the CSR terms before loading any
			// data.
			range = DeviceShiftRange(limit0, limit1);
			int numSegments = range.end - range.begin;

			// Load the CSR interval.
			DeviceGlobalToSharedLoop<NT, VT>(numSegments,
				csr_global + range.begin, tid, shared.csr);

			// Compute the segmented scan terms.
			terms = DeviceSegReducePrepare<NT, VT>(shared.csr, numSegments,
				tid, gid, range.flushLast, segs, segStarts);

			// Load tile of data in thread order from segment IDs.
			SegReduceLoad::LoadIndirect(count2, tid, gid, numSegments,
				range.begin, segs, segStarts, data_global, sources_global,
				identity, data, shared.loadStorage);

			// SegReduceLoad::LoadIndirectFast loads in register order, not
			// thread order, so is not appropriate for a scan.

		} else {
			// Direct load. It is more efficient to load the full tile before
			// dealing with data dependencies.
			SegReduceLoad::LoadDirect(count2, tid, gid, data_global, identity,
				data, shared.loadStorage);

			range = DeviceShiftRange(limit0, limit1);
			int numSegments = range.end - range.begin;

			if(range.total) {
				// Load the CSR interval.
				DeviceGlobalToSharedLoop<NT, VT>(numSegments,
					csr_global + range.begin, tid, shared.csr);

				// Compute the segmented scan terms.
				terms = DeviceSegReducePrepare<NT, VT>(shared.csr, numSegments,
					tid, gid, range.flushLast, segs, segStarts);
			}
		}

		T carryIn = carryIn_global[block];
		T localScan[VT];

		// Scan tile data, incorporating carry in, and store to localScan.
		if(range.total) {
			// Run a segmented scan over the tile's data.

			// Compute thread's carry out.
			T x;
			#pragma unroll
			for(int i = 0; i < VT; ++i) {
				x = i ? op(x, data[i]) : data[i];
				if(segs[i] != segs[i + 1]) x = identity;
			}

			// Run a parallel exclusive segmented scan over each thread's scan.
			T carryOut;
			x = SegScan::SegScanDelta(tid, terms.tidDelta, x,
				shared.segScanStorage, &carryOut, identity, op);

			// Add carry-in to the exclusive segmented scan of the tile data if
			// it applies to this thread's first segment.
			// Note that values in segs[] are CTA-local. Hence segs[i] = 0 means
			// thread-local value 'i' is part of the first segment in this tile.
			if(segs[0] == 0)
				x = op(carryIn, x);

			// Perform the desired segmented scan type over this thread's data.
			#pragma unroll
			for(int i = 0; i < VT; ++i) {
				if(SgpuScanTypeExc == Type)
					localScan[i] = x;
				x = op(x, data[i]);
				if(SgpuScanTypeInc == Type)
					localScan[i] = x;

				if(segs[i] != segs[i + 1])
					x = identity;
			}
		} else {
			// Run a (non-segmented) scan over the tile's data.

			// Exclusive scan over thread's data.
			T x;
			#pragma unroll
			for(int i = 0; i < VT; ++i) {
				x = i ? op(x, data[i]) : data[i];
			}

			// Run a parallel exclusive scan over each thread's scan.
			T total;
			x = Scan::Scan(tid, x, shared.scanStorage, &total, SgpuScanTypeExc,
				identity, op);

			// All items in this tile are from the same segment, hence all
			// require the carry in.
			x = op(carryIn, x);

			// Perform the desired scan type over this thread's data.
			#pragma unroll
			for(int i = 0; i < VT; ++i) {
				if(SgpuScanTypeExc == Type)
					localScan[i] = x;
				x = op(x, data[i]);
				if(SgpuScanTypeInc == Type)
					localScan[i] = x;
			}
		}

		if(Indirect) {
			int numSegments = range.end - range.begin;

			SegScanStore::StoreIndirect(count2, tid, gid, numSegments,
				range.begin, segs, segStarts, localScan, sources_global,
				dest_global, shared.storeStorage);
		} else {
			SegScanStore::StoreDirect(count2, tid, gid, localScan, dest_global,
				shared.storeStorage);
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
// SegScanHost
// Generic host implementation for seg-scan and interval-scan.

template<typename Tuning, bool Indirect, SgpuScanType Type, typename InputIt,
	typename CsrIt, typename SourcesIt, typename DestIt, typename T,
	typename Op>
SGPU_HOST void SegScanInner(InputIt data_global, CsrIt csr_global,
	SourcesIt sources_global, int count, int numSegments,
	const int* numSegments2_global, DestIt dest_global, T identity, Op op,
	CudaContext& context) {

	int2 launch = Tuning::GetLaunchParams(context);
	int NV = launch.x * launch.y;

	int numTiles = SGPU_DIV_UP(count, NV);
	int maxBlocks = context.MaxGridSize();
	int numBlocks = min(numTiles, maxBlocks);

	// Use upper-bound binary search to partition the CSR structure into tiles.
	SGPU_MEM(int) limitsDevice = PartitionCsrPlus(count, NV, csr_global,
		numSegments, numSegments2_global, numTiles + 1, context);

	// Segmented scan without source intervals.
	SGPU_MEM(T) carryOutDevice = context.Malloc<T>(numTiles);
	KernelSegScanCsrUpsweep<Tuning, Indirect>
		<<<numBlocks, launch.x, 0, context.Stream()>>>(numTiles, csr_global,
		sources_global, count, limitsDevice->get(), data_global, identity, op,
		carryOutDevice->get());
	SGPU_SYNC_CHECK("KernelSegScanCsr");

	// Fix-up carry-in values which span tiles in KernelSegScanCsr.
	SegScanSpine(limitsDevice->get(), numTiles, carryOutDevice->get(),
		identity, op, context);

	// Segmented scan, with carry-ins from the spines.
	KernelSegScanCsrDownsweep<Tuning, Indirect, Type>
		<<<numBlocks, launch.x, 0, context.Stream()>>>(numTiles, csr_global,
		sources_global, count, limitsDevice->get(), data_global,
		carryOutDevice->get(), identity, op, dest_global);
	SGPU_SYNC_CHECK("KernelSegScanCsrDownsweep");
}

template<typename Tuning, bool Indirect, SgpuScanType Type, typename InputIt,
	typename CsrIt, typename SourcesIt, typename DestIt, typename T,
	typename Op>
SGPU_HOST void SegScanHost(InputIt data_global, CsrIt csr_global,
	SourcesIt sources_global, int count, int numSegments, bool supportEmpty,
	DestIt dest_global, T identity, Op op, CudaContext& context) {

	if(supportEmpty) {
		// Allocate space for Csr2 and Sources2.
		SGPU_MEM(int) csr2Device = context.Malloc<int>(numSegments + 1);
		SGPU_MEM(int) sources2Device;
		if(Indirect) sources2Device = context.Malloc<int>(numSegments);

		// Strip the empties from Csr and store in Csr2.
		CsrStripEmpties<Indirect>(count, csr_global, sources_global,
			numSegments, csr2Device->get(),
			Indirect ? sources2Device->get() : (int*)0,	(int*)0, context);

		// Run the segmented scan in the Csr2 coordinate space; empty segments
		// are not present in the input data_global, and will similarly not be
		// present in the output dest_global.
		SegScanInner<Tuning, Indirect, Type>(data_global, csr2Device->get(),
			Indirect ? sources2Device->get() : (const int*)0, count, -1,
			csr2Device->get() + numSegments, dest_global, identity, op,
			context);

	} else {
		// Evaluate the scan directly into dest_global.
		SegScanInner<Tuning, Indirect, Type>(data_global, csr_global,
			sources_global,	count, numSegments, (const int*)0, dest_global,
			identity, op, context);
	}
}

template<SgpuScanType Type, typename InputIt, typename CsrIt, typename OutputIt,
	typename T,	typename Op>
SGPU_HOST void SegScanCsr(InputIt data_global, int count, CsrIt csr_global,
	int numSegments, bool supportEmpty, OutputIt dest_global, T identity, Op op,
	CudaContext& context) {

	typedef typename SegScanNormalTuning<sizeof(T)>::Tuning Tuning;

	SegScanHost<Tuning, false, Type>(data_global, csr_global, (const int*)0,
		count, numSegments, supportEmpty, dest_global, identity, op, context);
}

template<typename InputIt, typename CsrIt>
SGPU_HOST void SegScanCsrExc(InputIt data_global, int count, CsrIt csr_global,
	int numSegments, bool supportEmpty, CudaContext& context) {

	typedef typename std::iterator_traits<InputIt>::value_type T;
	SegScanCsr<SgpuScanTypeExc>(data_global, count, csr_global, numSegments,
		supportEmpty, data_global, (T)0, sgpu::plus<T>(), context);
}

template<SgpuScanType Type, typename InputIt, typename CsrIt,
	typename SourcesIt,	typename OutputIt, typename T, typename Op>
SGPU_HOST void IndirectScanCsr(InputIt data_global, int count, CsrIt csr_global,
	SourcesIt sources_global, int numSegments, bool supportEmpty,
	OutputIt dest_global, T identity, Op op, CudaContext& context) {

	typedef typename SegScanIndirectTuning<sizeof(T)>::Tuning Tuning;

	SegScanHost<Tuning, true, Type>(data_global, csr_global, sources_global,
		count, numSegments, supportEmpty, dest_global, identity, op, context);
}

template<typename InputIt, typename CsrIt, typename SourcesIt>
SGPU_HOST void IndirectScanCsrExc(InputIt data_global, int count,
	CsrIt csr_global, SourcesIt sources_global, int numSegments,
	bool supportEmpty, CudaContext& context) {

	typedef typename std::iterator_traits<InputIt>::value_type T;
	IndirectScanCsr<SgpuScanTypeExc>(data_global, count, csr_global,
		sources_global, numSegments, supportEmpty, data_global, (T)0,
		sgpu::plus<T>(), context);
}

////////////////////////////////////////////////////////////////////////////////
// seg-csr preprocessed format.

template<typename T, typename CsrIt>
SGPU_HOST void SegScanCsrPreprocess(int count, CsrIt csr_global,
	int numSegments, bool supportEmpty,
	std::auto_ptr<SegCsrPreprocessData>* ppData, CudaContext& context) {

	typedef typename SegScanPreprocessTuning<sizeof(T)>::Tuning Tuning;
	SegCsrPreprocess<Tuning>(count, csr_global, numSegments, supportEmpty,
		ppData, context);
}

// Like KernelSegScanCsr but for pre-processed CSR data.
template<typename Tuning, typename InputIt, typename T,	typename Op>
SGPU_LAUNCH_BOUNDS void KernelSegScanApplyUpsweep(int numTiles,
	const int* threadCodes_global, int count, const int* limits_global,
	InputIt data_global, T identity, Op op, T* carryOut_global) {

	typedef SGPU_LAUNCH_PARAMS Params;
	typedef typename Op::first_argument_type OpT;
	const int NT = Params::NT;
	const int VT = Params::VT;
	const int NV = NT * VT;
	const bool HalfCapacity = (sizeof(T) > sizeof(int)) && Params::HalfCapacity;
	const bool LdgTranspose = Params::LdgTranspose;

	typedef CTASegReduceLoad<NT, VT, HalfCapacity, LdgTranspose, T>
		SegReduceLoad;
	typedef CTAReduce<NT, OpT, Op> FastReduce;
	typedef CTASegScan<NT, Op> SegScan;

	union Shared {
		typename SegReduceLoad::Storage loadStorage;
		typename FastReduce::Storage reduceStorage;
		typename SegScan::Storage segScanStorage;
	};
	__shared__ Shared shared;

	int tid = threadIdx.x;

	for(int block = blockIdx.x; block < numTiles; block += gridDim.x) {
		__syncthreads();

		int gid = NV * block;
		int count2 = min(NV, count - gid);

		int limit0 = limits_global[block];
		int limit1 = limits_global[block + 1];
		int threadCodes = threadCodes_global[NT * block + tid];

		// Load the data and transpose into thread order.
		T data[VT];
		SegReduceLoad::LoadDirect(count2, tid, gid, data_global, identity, data,
			shared.loadStorage);

		// Compute the range.
		SegReduceRange range = DeviceShiftRange(limit0, limit1);

		if(range.total) {
			// Expand the segment indices.
			int segs[VT + 1];
			DeviceExpandFlagsToRows<VT>(threadCodes>> 20, threadCodes, segs);
			int tidDelta = 0x7f & (threadCodes>> 13);

			// Reduce tile data and store tile's carry-out term to
			// carryOut_global.

			// Compute thread's carry out.
			T x;
			#pragma unroll
			for(int i = 0; i < VT; ++i) {
				x = i ? op(x, data[i]) : data[i];
				if(segs[i] != segs[i + 1]) x = identity;
			}

			// Run a parallel segmented scan over each thread's carry-out values
			// to compute the CTA's carry-out.
			T carryOut;
			SegScan::SegScanDelta(tid, tidDelta, x, shared.segScanStorage,
				&carryOut, identity, op);

			// Store the carry-out for the entire CTA to global memory.
			if(!tid)
				carryOut_global[block] = carryOut;
		} else {
			// If there are no end flags in this CTA, use a fast reduction.
			T x;
			#pragma unroll
			for(int i = 0; i < VT; ++i)
				x = i ? op(x, data[i]) : data[i];
			x = FastReduce::Reduce(tid, x, shared.reduceStorage, op);
			if(!tid)
				carryOut_global[block] = x;
		}
	}
}

// Like KernelSegScanCsrDownsweep but for pre-processed CSR data.
template<typename Tuning, SgpuScanType Type, typename InputIt, typename DestIt,
	typename T,	typename Op>
SGPU_LAUNCH_BOUNDS void KernelSegScanApplyDownsweep(int numTiles,
	const int* threadCodes_global, int count, const int* limits_global,
	InputIt data_global, const T* carryIn_global, T identity, Op op,
	DestIt dest_global) {

	typedef SGPU_LAUNCH_PARAMS Params;
	typedef typename Op::first_argument_type OpT;
	const int NT = Params::NT;
	const int VT = Params::VT;
	const int NV = NT * VT;
	const bool HalfCapacity = (sizeof(T) > sizeof(int)) && Params::HalfCapacity;
	const bool LdgTranspose = Params::LdgTranspose;

	typedef CTASegReduceLoad<NT, VT, HalfCapacity, LdgTranspose, T>
		SegReduceLoad;
	typedef CTAScan<NT, Op> Scan;
	typedef CTASegScan<NT, Op> SegScan;
	typedef CTASegScanStore<NT, VT, HalfCapacity, T> SegScanStore;

	union Shared {
		typename SegReduceLoad::Storage loadStorage;
		typename Scan::Storage scanStorage;
		typename SegScan::Storage segScanStorage;
		typename SegScanStore::Storage storeStorage;
	};
	__shared__ Shared shared;

	int tid = threadIdx.x;

	for(int block = blockIdx.x; block < numTiles; block += gridDim.x) {
		__syncthreads();

		int gid = NV * block;
		int count2 = min(NV, count - gid);

		int limit0 = limits_global[block];
		int limit1 = limits_global[block + 1];
		int threadCodes = threadCodes_global[NT * block + tid];

		// Load the data and transpose into thread order.
		T data[VT];
		SegReduceLoad::LoadDirect(count2, tid, gid, data_global, identity, data,
			shared.loadStorage);

		// Compute the range.
		SegReduceRange range = DeviceShiftRange(limit0, limit1);

		T carryIn = carryIn_global[block];
		T localScan[VT];

		// Scan tile data, incorporating carry in, and store to localScan.
		if(range.total) {
			// Run a segmented scan over the tile's data.

			// Expand the segment indices.
			int segs[VT + 1];
			DeviceExpandFlagsToRows<VT>(threadCodes>> 20, threadCodes, segs);
			int tidDelta = 0x7f & (threadCodes>> 13);

			// Compute thread's carry out.
			T x;
			#pragma unroll
			for(int i = 0; i < VT; ++i) {
				x = i ? op(x, data[i]) : data[i];
				if(segs[i] != segs[i + 1]) x = identity;
			}

			// Run a parallel exclusive segmented scan over each thread's scan.
			T carryOut;
			x = SegScan::SegScanDelta(tid, tidDelta, x, shared.segScanStorage,
				&carryOut, identity, op);

			// Add carry-in to the exclusive segmented scan of the tile data if it
			// applies to this thread's first segment.
			// Note that values in segs[] are CTA-local. Hence segs[i] = 0 means
			// thread-local value 'i' is part of the first segment in this tile.
			if(segs[0] == 0)
				x = op(carryIn, x);

			// Perform the desired segmented scan type over this thread's data.
			#pragma unroll
			for(int i = 0; i < VT; ++i) {
				if(SgpuScanTypeExc == Type)
					localScan[i] = x;
				x = op(x, data[i]);
				if(SgpuScanTypeInc == Type)
					localScan[i] = x;

				if(segs[i] != segs[i + 1])
					x = identity;
			}
		} else {
			// Run a (non-segmented) scan over the tile's data.

			// Compute thread's carry out.
			T x;
			#pragma unroll
			for(int i = 0; i < VT; ++i) {
				x = i ? op(x, data[i]) : data[i];
			}

			// Run a parallel exclusive scan over each thread's scan.
			T total;
			x = Scan::Scan(tid, x, shared.scanStorage, &total, SgpuScanTypeExc,
				identity, op);

			// All items in this tile are from the same segment, hence all require
			// the carry in.
			x = op(carryIn, x);

			// Perform the desired scan type over this thread's data.
			#pragma unroll
			for(int i = 0; i < VT; ++i) {
				if(SgpuScanTypeExc == Type)
					localScan[i] = x;
				x = op(x, data[i]);
				if(SgpuScanTypeInc == Type)
					localScan[i] = x;
			}
		}

		SegScanStore::StoreDirect(count2, tid, gid, localScan, dest_global,
			shared.storeStorage);
	}
}

template<SgpuScanType Type, typename InputIt, typename DestIt, typename T,
	typename Op>
SGPU_HOST void SegScanApply(const SegCsrPreprocessData& preprocess,
	InputIt data_global, T identity, Op op, DestIt dest_global,
	CudaContext& context) {

	typedef typename SegScanPreprocessTuning<sizeof(T)>::Tuning Tuning;
	int2 launch = Tuning::GetLaunchParams(context);

	int maxBlocks = context.MaxGridSize();
	int numBlocks = min(preprocess.numTiles, maxBlocks);

	if(preprocess.csr2Device.get()) {
		// Support empties.
		SGPU_MEM(T) carryOutDevice = context.Malloc<T>(preprocess.numTiles);
		KernelSegScanApplyUpsweep<Tuning>
			<<<numBlocks, launch.x, 0, context.Stream()>>>(preprocess.numTiles,
			preprocess.threadCodesDevice->get(), preprocess.count,
			preprocess.limitsDevice->get(), data_global, identity, op,
			carryOutDevice->get());
		SGPU_SYNC_CHECK("KernelSegScanApply");

		// Fix-up carry-in values which span tiles in KernelSegScanApply.
		SegScanSpine(preprocess.limitsDevice->get(), preprocess.numTiles,
			carryOutDevice->get(), identity, op, context);

		KernelSegScanApplyDownsweep<Tuning, Type>
			<<<numBlocks, launch.x, 0, context.Stream()>>>(preprocess.numTiles,
			preprocess.threadCodesDevice->get(), preprocess.count,
			preprocess.limitsDevice->get(), data_global, carryOutDevice->get(),
			identity, op, dest_global);
		SGPU_SYNC_CHECK("KernelSegScanDownsweepApply");
	} else {
		// No empties.
		SGPU_MEM(T) carryOutDevice = context.Malloc<T>(preprocess.numTiles);
		KernelSegScanApplyUpsweep<Tuning>
			<<<numBlocks, launch.x, 0, context.Stream()>>>(preprocess.numTiles,
			preprocess.threadCodesDevice->get(), preprocess.count,
			preprocess.limitsDevice->get(), data_global, identity, op,
			carryOutDevice->get());
		SGPU_SYNC_CHECK("KernelSegReduceApply");

		// Fix-up carry-in values which span tiles in KernelSegScanApply.
		SegScanSpine(preprocess.limitsDevice->get(), preprocess.numTiles,
			carryOutDevice->get(), identity, op, context);

		KernelSegScanApplyDownsweep<Tuning, Type>
			<<<numBlocks, launch.x, 0, context.Stream()>>>(preprocess.numTiles,
			preprocess.threadCodesDevice->get(), preprocess.count,
			preprocess.limitsDevice->get(), data_global, carryOutDevice->get(),
			identity, op, dest_global);
		SGPU_SYNC_CHECK("KernelSegScanDownsweepApply");
	}
}

template<typename InputIt>
SGPU_HOST void SegScanApplyExc(const SegCsrPreprocessData& preprocess,
	InputIt data_global, CudaContext& context) {

	typedef typename std::iterator_traits<InputIt>::value_type T;
	SegScanApply<SgpuScanTypeExc>(preprocess, data_global, (T)0,
		sgpu::plus<T>(), data_global, context);
}

} // namespace sgpu
