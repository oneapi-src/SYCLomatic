// RUN: rm -rf %T/cuda_arch_lang && mkdir -p %T/cuda_arch_lang
// RUN: not dpct --format-range=none --out-root=%T/cuda_arch_lang --stop-on-parse-err --cuda-include-path="%cuda-path/include" %s -- -Wno-unused-command-line-argument -x c++ > %T/cuda_arch_lang/output 2>&1
// RUN: cat %T/cuda_arch_lang/output
// RUN: python -c "assert 'no CUDA code detected' in input()" < %T/cuda_arch_lang/output
// RUN: dpct --format-range=none --out-root=%T/cuda_arch_lang --cuda-include-path="%cuda-path/include" %s -- -x cuda
// RUN: FileCheck --input-file=%T/cuda_arch_lang/cuda_arch_lang.cpp.dp.cpp %s
// RUN: rm -rf %T/cuda_arch_lang

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
