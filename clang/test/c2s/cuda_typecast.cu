// RUN: c2s --format-range=none -out-root %T/cuda_typecast %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/cuda_typecast/cuda_typecast.dp.cpp

// CHECK:#include <CL/sycl.hpp>
// CHECK-NEXT:#include <c2s/c2s.hpp>
#include "cuda_fp16.h"

__device__ void foo() {
  long long a = 100000;
  double b = 2.5;
  // CHECK: double c = sycl::bit_cast<double>(a);
  double c = __longlong_as_double(a);
  // CHECK: long long d = sycl::bit_cast<long long>(b);
  long long d = __double_as_longlong(b);

  int e = 23;
  float f = 45.0;
  // CHECK: float g = sycl::bit_cast<float>(e);
  float g = __int_as_float(e);
  // CHECK: int h = sycl::bit_cast<int>(f);
  int h = __float_as_int(f);

  unsigned i = 23;
  float j = 45.0;
  // CHECK: float k = sycl::bit_cast<float>(i);
  float k = __uint_as_float(i);
  // CHECK: unsigned l = sycl::bit_cast<unsigned int>(j);
  unsigned l = __float_as_uint(j);
}

inline __device__ __half bar(uint32_t v) {
  uint32_t mask = 0x0000ffff;
  // CHECK: return sycl::bit_cast<sycl::half, unsigned short>(v ^ mask);
  return __ushort_as_half(v ^ mask);
}