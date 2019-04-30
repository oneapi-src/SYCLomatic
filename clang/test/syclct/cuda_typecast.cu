// RUN: syclct -out-root %T %s -- -x cuda --cuda-host-only --cuda-path=%cuda-path
// RUN: FileCheck %s --match-full-lines --input-file %T/cuda_typecast.sycl.cpp

// CHECK:#include <CL/sycl.hpp>
// CHECK-NEXT:#include <syclct/syclct.hpp>
#include <cuda_runtime.h>

__device__ void foo() {
  long long a = 100000;
  double b = 2.5;
  // CHECK: double c = syclct::bit_cast<long long, double>(a);
  double c = __longlong_as_double(a);
  // CHECK: long long d = syclct::bit_cast<double, long long>(b);
  long long d = __double_as_longlong(b);

  int e = 23;
  float f = 45.0;
  // CHECK: float g = syclct::bit_cast<int, float>(e);
  float g = __int_as_float(e);
  // CHECK: int h = syclct::bit_cast<float, int>(f);
  int h = __float_as_int(f);

  unsigned i = 23;
  float j = 45.0;
  // CHECK: float k = syclct::bit_cast<unsigned int, float>(i);
  float k = __uint_as_float(i);
  // CHECK: unsigned l = syclct::bit_cast<float, unsigned int>(j);
  unsigned l = __float_as_uint(j);
}
