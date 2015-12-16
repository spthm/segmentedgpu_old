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

#pragma once

#include "sgpudevice.cuh"
#include "util/sgpucontext.h"

namespace sgpu {

////////////////////////////////////////////////////////////////////////////////
// kernels/reduce.cuh

// Reduce input and return variable in device memory or host memory, or both.
// Provide a non-null pointer to retrieve data.
template<typename InputIt, typename T, typename Op>
SGPU_HOST void Reduce(InputIt data_global, int count, T identity, Op op,
	T* reduce_global, T* reduce_host, CudaContext& context);

// T = std::iterator_traits<InputIt>::value_type.
// Reduce with identity = 0 and op = sgpu::plus<T>.
// Returns the value in host memory.
template<typename InputIt>
SGPU_HOST typename std::iterator_traits<InputIt>::value_type
Reduce(InputIt data_global, int count, CudaContext& context);


////////////////////////////////////////////////////////////////////////////////
// kernels/scan.cuh

// Scan inputs in device memory.
// SgpuScanType may be:
//		SgpuScanTypeExc (exclusive) or
//		SgpuScanTypeInc (inclusive).
// Return the total in device memory, host memory, or both.
template<SgpuScanType Type, typename DataIt, typename T, typename Op,
	typename DestIt>
SGPU_HOST void Scan(DataIt data_global, int count, T identity, Op op,
	T* reduce_global, T* reduce_host, DestIt dest_global,
	CudaContext& context);

// Exclusive scan with identity = 0 and op = sgpu::plus<T>.
// Returns the total in host memory.
template<typename InputIt, typename TotalType>
SGPU_HOST void ScanExc(InputIt data_global, int count, TotalType* total,
	CudaContext& context);

// Like above, but don't return the total.
template<typename InputIt>
SGPU_HOST void ScanExc(InputIt data_global, int count, CudaContext& context);


////////////////////////////////////////////////////////////////////////////////
// kernels/bulkinsert.cuh

// Combine aCount elements in a_global with bCount elements in b_global.
// Each element a_global[i] is inserted before position indices_global[i] and
// stored to dest_global. The insertion indices are relative to the B array,
// not the output. Indices must be sorted but not necessarily unique.

// If aCount = 5, bCount = 3, and indices = (1, 1, 2, 3, 3), the output is:
// B0 A0 A1 B1 A2 B2 A3 A4.
template<typename InputIt1, typename IndicesIt, typename InputIt2,
	typename OutputIt>
SGPU_HOST void BulkInsert(InputIt1 a_global, IndicesIt indices_global,
	int aCount, InputIt2 b_global, int bCount, OutputIt dest_global,
	CudaContext& context);


////////////////////////////////////////////////////////////////////////////////
// kernels/merge.cuh

// MergeKeys merges two arrays of sorted inputs with C++-comparison semantics.
// aCount items from aKeys_global and bCount items from bKeys_global are merged
// into aCount + bCount items in keys_global.
// Comp is a comparator type supporting strict weak ordering.
// If !comp(b, a), then a is placed before b in the output.
template<typename KeysIt1, typename KeysIt2, typename KeysIt3, typename Comp>
SGPU_HOST void MergeKeys(KeysIt1 aKeys_global, int aCount, KeysIt2 bKeys_global,
	int bCount, KeysIt3 keys_global, Comp comp, CudaContext& context);

// MergeKeys specialized with Comp = sgpu::less<T>.
template<typename KeysIt1, typename KeysIt2, typename KeysIt3>
SGPU_HOST void MergeKeys(KeysIt1 aKeys_global, int aCount, KeysIt2 bKeys_global,
	int bCount, KeysIt3 keys_global, CudaContext& context);

// MergePairs merges two arrays of sorted inputs by key and copies values.
// If !comp(bKey, aKey), then aKey is placed before bKey in the output, and
// the corresponding aData is placed before bData. This corresponds to *_by_key
// functions in Thrust.
template<typename KeysIt1, typename KeysIt2, typename KeysIt3, typename ValsIt1,
	typename ValsIt2, typename ValsIt3, typename Comp>
SGPU_HOST void MergePairs(KeysIt1 aKeys_global, ValsIt1 aVals_global,
	int aCount, KeysIt2 bKeys_global, ValsIt2 bVals_global, int bCount,
	KeysIt3 keys_global, ValsIt3 vals_global, Comp comp, CudaContext& context);

// MergePairs specialized with Comp = sgpu::less<T>.
template<typename KeysIt1, typename KeysIt2, typename KeysIt3, typename ValsIt1,
	typename ValsIt2, typename ValsIt3>
SGPU_HOST void MergePairs(KeysIt1 aKeys_global, ValsIt1 aVals_global,
	int aCount, KeysIt2 bKeys_global, ValsIt2 bVals_global, int bCount,
	KeysIt3 keys_global, ValsIt3 vals_global, CudaContext& context);


////////////////////////////////////////////////////////////////////////////////
// kernels/mergesort.cuh

// MergesortKeys sorts data_global using comparator Comp.
// If !comp(b, a), then a comes before b in the output. The data is sorted
// in-place.
template<typename T, typename Comp>
SGPU_HOST void MergesortKeys(T* data_global, int count, Comp comp,
	CudaContext& context);

// MergesortKeys specialized with Comp = sgpu::less<T>.
template<typename T>
SGPU_HOST void MergesortKeys(T* data_global, int count, CudaContext& context);

// MergesortPairs sorts data by key, copying data. This corresponds to
// sort_by_key in Thrust.
template<typename KeyType, typename ValType, typename Comp>
SGPU_HOST void MergesortPairs(KeyType* keys_global, ValType* values_global,
	int count, Comp comp, CudaContext& context);

// MergesortPairs specialized with Comp = sgpu::less<KeyType>.
template<typename KeyType, typename ValType>
SGPU_HOST void MergesortPairs(KeyType* keys_global, ValType* values_global,
	int count, CudaContext& context);

// MergesortIndices is like MergesortPairs where values_global is treated as
// if initialized with integers (0 ... count - 1).
template<typename KeyType, typename Comp>
SGPU_HOST void MergesortIndices(KeyType* keys_global, int* values_global,
	int count, Comp comp, CudaContext& context);

// MergesortIndices specialized with Comp = sgpu::less<KeyType>.
template<typename KeyType>
SGPU_HOST void MergesortIndices(KeyType* keys_global, int* values_global,
	int count, CudaContext& context);


////////////////////////////////////////////////////////////////////////////////
// kernels/segmentedsort.cuh

// Mergesort count items in-place in data_global. Keys are compared with Comp
// (as they are in MergesortKeys), however keys remain inside the segments
// defined by flags_global.

// flags_global is a bitfield cast to uint*. Each bit in flags_global is a
// segment head flag. Only keys between segment head flags (inclusive on the
// left and exclusive on the right) may be exchanged. The first element is
// assumed to start a segment, regardless of the value of bit 0.

// Passing verbose=true causes the function to print mergepass statistics to the
// console. This may be helpful for developers to understand the performance
// characteristics of the function and how effectively it early-exits merge
// operations.
template<typename T, typename Comp>
SGPU_HOST void SegSortKeysFromFlags(T* data_global, int count,
	const uint* flags_global, CudaContext& context, Comp comp,
	bool verbose = false);

// SegSortKeysFromFlags specialized with Comp = sgpu::less<T>.
template<typename T>
SGPU_HOST void SegSortKeysFromFlags(T* data_global, int count,
	const uint* flags_global, CudaContext& context, bool verbose = false);

// Segmented sort using head flags and supporting value exchange.
template<typename KeyType, typename ValType, typename Comp>
SGPU_HOST void SegSortPairsFromFlags(KeyType* keys_global,
	ValType* values_global, int count, const uint* flags_global,
	CudaContext& context, Comp comp, bool verbose = false);

// SegSortPairsFromFlags specialized with Comp = sgpu::less<T>.
template<typename KeyType, typename ValType>
SGPU_HOST void SegSortPairsFromFlags(KeyType* keys_global,
	ValType* values_global, int count, const uint* flags_global,
	CudaContext& context, bool verbose = false);

// Segmented sort using segment indices rather than head flags. indices_global
// is a sorted and unique list of indicesCount segment start locations. These
// indices correspond to the set bits in the flags_global field. A segment
// head index for position 0 may be omitted.
template<typename T, typename Comp>
SGPU_HOST void SegSortKeysFromIndices(T* data_global, int count,
	const int* indices_global, int indicesCount, CudaContext& context,
	Comp comp, bool verbose = false);

// SegSortKeysFromIndices specialized with Comp = sgpu::less<T>.
template<typename T>
SGPU_HOST void SegSortKeysFromIndices(T* data_global, int count,
	const int* indices_global, int indicesCount, CudaContext& context,
	bool verbose = false);

// Segmented sort using segment indices and supporting value exchange.
template<typename KeyType, typename ValType, typename Comp>
SGPU_HOST void SegSortPairsFromIndices(KeyType* keys_global,
	ValType* values_global, int count, const int* indices_global,
	int indicesCount, CudaContext& context, Comp comp, bool verbose = false);

// SegSortPairsFromIndices specialized with Comp = sgpu::less<KeyType>.
template<typename KeyType, typename ValType>
SGPU_HOST void SegSortPairsFromIndices(KeyType* keys_global,
	ValType* values_global, int count, const int* indices_global,
	int indicesCount, CudaContext& context, bool verbose = false);


////////////////////////////////////////////////////////////////////////////////
// kernels/segreducecsr.cuh

// SegReducePreprocessData is defined in segreduce.cuh. It includes:
// -	limits for CSR->tiles
// -	packed thread codes for each thread in the reduction
// -	(optional) CSR2 array of filtered segment offsets
struct SegReducePreprocessData;

// SegReduceCsr runs a segmented reduction given an input and a sorted list of
// segment start offsets. This implementation requires operators support
// commutative (a + b = b + a) and associative (a + (b + c) = (a + b) + c)
// evaluation.

// In the segmented reduction, reduce-by-key, and Spmv documentation, "segment"
// and "row" are used interchangably. A
//

// InputIt data_global		- Data value input.
// int count				- Size of input array data_global.
// CsrIt csr_global			- List of integers for start of each segment.
//							  The first entry must be 0 (indicating that the
//							  first segment starts at offset 0).
//							  Equivalent to exc-scan of segment sizes.
//							  If supportEmpty is false: must be ascending.
//							  If supportEmpty is true: must be non-descending.
// int numSegments			- Size of segment list csr_global. Must be >= 1.
// bool supportEmpty		- Basic seg-reduce code does not support empty
//							  segments.
//							  Set supportEmpty = true to add pre- and post-
//							  processing to support empty segments.
// OutputIt dest_global		- Output array for segmented reduction. Allocate
//							  numSegments elements. Should be same data type as
//							  InputIt and identity.
// T identity				- Identity for reduction operation. Eg, use 0 for
//							  addition or 1 for multiplication.
// Op op					- Reduction operator. Model on std::plus<>. SGPU
//							  provides operators sgpu::plus<>, minus<>,
//							  multiplies<>, modulus<>, bit_or<> bit_and<>,
//							  bit_xor<>, maximum<>, and minimum<>.
// CudaContext& context		- SGPU context support object. All kernels are
//							  launched on the associated stream.
template<typename InputIt, typename CsrIt, typename OutputIt, typename T,
	typename Op>
SGPU_HOST void SegReduceCsr(InputIt data_global, int count, CsrIt csr_global,
	int numSegments, bool supportEmpty, OutputIt dest_global, T identity, Op op,
	CudaContext& context);

// IndirectReduceCsr is like SegReduceCsr but with one level of source
// indirection. The start of each segment/row i in data_global starts at
// sources_global[i].
// SourcesIt sources_global	- List of integers for source data of each segment.
//							  Must be numSegments in size.
template<typename InputIt, typename CsrIt, typename SourcesIt,
	typename OutputIt, typename T, typename Op>
SGPU_HOST void IndirectReduceCsr(InputIt data_global, int count,
	CsrIt csr_global, SourcesIt sources_global, int numSegments,
	bool supportEmpty, OutputIt dest_global, T identity, Op op,
	CudaContext& context);

// SegReduceCsrPreprocess accelerates multiple seg-reduce calls on different
// data with the same segment geometry. Partitioning and CSR->CSR2 transform is
// off-loaded to a preprocessing pass. The actual reduction is evaluated by
// SegReduceApply.
template<typename T, typename CsrIt>
SGPU_HOST void SegReduceCsrPreprocess(int count, CsrIt csr_global,
	int numSegments, bool supportEmpty,
	std::auto_ptr<SegReducePreprocessData>* ppData, CudaContext& context);

template<typename InputIt, typename DestIt, typename T, typename Op>
SGPU_HOST void SegReduceApply(const SegReducePreprocessData& preprocess,
	InputIt data_global, T identity, Op op, DestIt dest_global,
	CudaContext& context);

} // namespace sgpu

