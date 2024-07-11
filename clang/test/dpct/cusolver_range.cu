// UNSUPPORTED: cuda-8.0, cuda-9.2, cuda-10.0
// UNSUPPORTED: v8.0, v9.2, v10.0
// RUN: dpct --format-range=none -out-root %T/cusolver_range %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cusolver_range/cusolver_range.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/cusolver_range/cusolver_range.dp.cpp -o %T/cusolver_range/cusolver_range.dp.o %}

// CHECK: #include <sycl/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
// CHECK-NEXT: #include <dpct/lapack_utils.hpp>
#include "cusolverDn.h"

void test0() {
  // CHECK: oneapi::mkl::rangev range = oneapi::mkl::rangev::all;
  cusolverEigRange_t range = CUSOLVER_EIG_RANGE_ALL;
  // CHECK: range = oneapi::mkl::rangev::values;
  range = CUSOLVER_EIG_RANGE_V;
  // CHECK: range = oneapi::mkl::rangev::indices;
  range = CUSOLVER_EIG_RANGE_I;
}
