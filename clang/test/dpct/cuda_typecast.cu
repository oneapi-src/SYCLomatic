// RUN: dpct --format-range=none -out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/cuda_typecast.dp.cpp

// CHECK:#include <CL/sycl.hpp>
// CHECK-NEXT:#include <dpct/dpct.hpp>
#include <cuda_runtime.h>

__device__ void foo() {
  long long a = 100000;
  double b = 2.5;
  // CHECK: double c = sycl::detail::bit_cast<double>(a);
  double c = __longlong_as_double(a);
  // CHECK: long long d = sycl::detail::bit_cast<long long>(b);
  long long d = __double_as_longlong(b);

  int e = 23;
  float f = 45.0;
  // CHECK: float g = sycl::detail::bit_cast<float>(e);
  float g = __int_as_float(e);
  // CHECK: int h = sycl::detail::bit_cast<int>(f);
  int h = __float_as_int(f);

  unsigned i = 23;
  float j = 45.0;
  // CHECK: float k = sycl::detail::bit_cast<float>(i);
  float k = __uint_as_float(i);
  // CHECK: unsigned l = sycl::detail::bit_cast<unsigned int>(j);
  unsigned l = __float_as_uint(j);
}
