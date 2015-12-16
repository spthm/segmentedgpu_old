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

#include "../device/intrinsics.cuh"

namespace sgpu {

////////////////////////////////////////////////////////////////////////////////
// CTAReduce

template<int NT, typename T, typename Op = sgpu::plus<T> >
struct CTAReduce {
    enum { Size = NT, Capacity = NT };
    struct Storage { T shared[Capacity]; };

    SGPU_DEVICE static T Reduce(int tid, T x, Storage& storage, Op op = Op()) {
        storage.shared[tid] = x;
        __syncthreads();

        // Fold the data in half with each pass.
        #pragma unroll
        for(int destCount = NT / 2; destCount >= 1; destCount /= 2) {
            if(tid < destCount) {
                // Read from the right half and store to the left half.
                x = op(x, storage.shared[destCount + tid]);
                storage.shared[tid] = x;
            }
            __syncthreads();
        }
        T total = storage.shared[0];
        __syncthreads();
        return total;
    }
};

#if __CUDA_ARCH__ >= 300

// Shuffle-based implementation for scalar numeric types on SM_30+ devices.
// Valid for T = {int, long long,
//                unsigned int, unsigned long long,
//                float, double}

template<int NT, typename T, typename Op>
SGPU_DEVICE T ReduceNumeric(int tid, T x, T* shared, Op op) {

    const int NumWarps = NT / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    int wid = tid / WARP_SIZE;

    // In the first phase, threads cooperatively find the reduction within
    // their warp.
    #pragma unroll
    for(int offset = 1; offset < WARP_SIZE; offset *= 2)
        x = op(x, shfl_up(x, offset));

    // The last thread in each warp stores the warp's reduction to shared
    // memory.
    if(lane == WARP_SIZE - 1) shared[wid] = x;
    __syncthreads();

    // Reduce the totals of each warp. The spine is NumWarps threads wide, and
    // NumWarps can be at most WARP_SIZE.
    if(tid < NumWarps) {
        x = shared[tid];
        #pragma unroll
        for(int offset = 1; offset < NumWarps; offset *= 2)
            x = op(x, shfl_up(x, offset));
        shared[tid] = x;
    }
    __syncthreads();

    T reduction = shared[NumWarps - 1];
    __syncthreads();

    return reduction;
}

// Specializations for scalar numeric types.
// Op inherits the default behaviour of the full template, i.e. sgpu::plus<T>.

template<int NT, typename Op>
struct CTAReduce<NT, int, Op> {
    typedef int T;
    enum { Size = NT, Capacity = WARP_SIZE };
    struct Storage { T shared[Capacity]; };

    SGPU_DEVICE static T Reduce(int tid, T x, Storage& storage, Op op = Op()) {

        return ReduceNumeric<NT, T, Op>(tid, x, storage.shared, op);
    }
};

template<int NT, typename Op>
struct CTAReduce<NT, sgpu::int64, Op> {
    typedef sgpu::int64 T;
    enum { Size = NT, Capacity = WARP_SIZE };
    struct Storage { T shared[Capacity]; };

    SGPU_DEVICE static T Reduce(int tid, T x, Storage& storage, Op op = Op()) {

        return ReduceNumeric<NT, T, Op>(tid, x, storage.shared, op);
    }
};

template<int NT, typename Op>
struct CTAReduce<NT, unsigned int, Op> {
    typedef unsigned int T;
    enum { Size = NT, Capacity = WARP_SIZE };
    struct Storage { T shared[Capacity]; };

    SGPU_DEVICE static T Reduce(int tid, T x, Storage& storage, Op op = Op()) {

        return ReduceNumeric<NT, T, Op>(tid, x, storage.shared, op);
    }
};

template<int NT, typename Op>
struct CTAReduce<NT, sgpu::uint64, Op> {
    typedef sgpu::uint64 T;
    enum { Size = NT, Capacity = WARP_SIZE };
    struct Storage { T shared[Capacity]; };

    SGPU_DEVICE static T Reduce(int tid, T x, Storage& storage, Op op = Op()) {

        return ReduceNumeric<NT, T, Op>(tid, x, storage.shared, op);
    }
};

template<int NT, typename Op>
struct CTAReduce<NT, float, Op> {
    typedef float T;
    enum { Size = NT, Capacity = WARP_SIZE };
    struct Storage { T shared[Capacity]; };

    SGPU_DEVICE static T Reduce(int tid, T x, Storage& storage, Op op = Op()) {

        return ReduceNumeric<NT, T, Op>(tid, x, storage.shared, op);
    }
};

template<int NT, typename Op>
struct CTAReduce<NT, double, Op> {
    typedef double T;
    enum { Size = NT, Capacity = WARP_SIZE };
    struct Storage { T shared[Capacity]; };

    SGPU_DEVICE static T Reduce(int tid, T x, Storage& storage, Op op = Op()) {

        return ReduceNumeric<NT, T, Op>(tid, x, storage.shared, op);
    }
};

#endif // __CUDA_ARCH__ >= 300

} // namespace sgpu
