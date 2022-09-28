// RUN: set +o pipefail
// RUN: dpct --format-range=none --out-root=%T/cuda_arch_lang --stop-on-parse-err %s -- -x c++ 2>&1 | grep "no CUDA code detected"
// RUN: set -o pipefail
// RUN: dpct --format-range=none --out-root=%T/cuda_arch_lang --stop-on-parse-err %s -- -x cuda
// RUN: FileCheck --input-file=%T/cuda_arch_lang/cuda_arch_lang.cpp.dp.cpp %s 

// CHECK: #ifdef DPCT_COMPATIBILITY_TEMP
#ifdef __CUDA_ARCH__
__device__ void foo1() { __fsqrt_rn(1.); }
#endif

// CHECK: #ifndef DPCT_COMPATIBILITY_TEMP
#ifndef __CUDA_ARCH__
#else
__device__ void foo2() { __popcll(2); }
#endif

// CHECK: #ifdef DPCT_COMPATIBILITY_TEMP
#ifdef __CUDA_ARCH__
#define DEVICE __device__
#else
#define DEVICE
#endif

DEVICE void foo3() {
// CHECK: #if defined(DPCT_COMPATIBILITY_TEMP)
#if defined(__CUDA_ARCH__)
  __syncthreads();
#endif
}
