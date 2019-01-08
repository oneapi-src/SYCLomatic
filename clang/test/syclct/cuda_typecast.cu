// RUN: syclct -out-root %T %s -- -x cuda --cuda-host-only --cuda-path=%cuda-path
// RUN: FileCheck %s --match-full-lines --input-file %T/cuda_typecast.sycl.cpp

// CHECK:#include <CL/sycl.hpp>
// CHECK-NEXT:#include <syclct/syclct.hpp>
#include <cuda_runtime.h>

__device__ void foo() {
  long long int a = 100000;
  double b = 2.5;
  // CHECK: double c = syclct::ll2d(a);
  double c = __longlong_as_double(a);
  // CHECK: long long int d = syclct::d2ll(b);
  long long int d = __double_as_longlong(b);
}
