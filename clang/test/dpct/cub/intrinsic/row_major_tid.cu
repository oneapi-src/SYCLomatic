// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct -in-root %S -out-root %T/intrinsic/row_major_tid %S/row_major_tid.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/intrinsic/row_major_tid/row_major_tid.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/intrinsic/row_major_tid/row_major_tid.dp.cpp -o %T/intrinsic/row_major_tid/row_major_tid.dp.o %}

// CHECK:#include <sycl/sycl.hpp>
// CHECK:#include <dpct/dpct.hpp>
#include <cub/cub.cuh>

__global__ void kernel(int *res) {
  // CHECK: *res = item_ct1.get_local_linear_id();
  *res = cub::RowMajorTid(blockDim.x, blockDim.y, blockDim.z);
}
