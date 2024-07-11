// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct -in-root %S -out-root %T/intrinsic/shr_add %S/shr_add.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/intrinsic/shr_add/shr_add.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/intrinsic/shr_add/shr_add.dp.cpp -o %T/intrinsic/shr_add/shr_add.dp.o %}

// CHECK:#include <sycl/sycl.hpp>
// CHECK:#include <dpct/dpct.hpp>
#include <cub/cub.cuh>
#include <limits>
#include <stdio.h>

// CHECK:void kernel1(int *res) {
// CHECK-NEXT:  *res = dpct::extend_shr_clamp<uint32_t>(1, 2, 3, sycl::plus<>());
// CHECK-NEXT:}
__global__ void kernel1(int *res) {
  *res = cub::SHR_ADD(1, 2, 3);
}
