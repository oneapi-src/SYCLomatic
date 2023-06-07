// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -in-root %S -out-root %T/intrinsic/laneid %S/laneid.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/intrinsic/laneid/laneid.dp.cpp --match-full-lines %s

// CHECK:#include <sycl/sycl.hpp>
// CHECK:#include <dpct/dpct.hpp>
#include <cub/cub.cuh>
#include <limits>
#include <stdio.h>

// CHECK: void laneid(int *id, const sycl::nd_item<3> &item_ct1) {
// CHECK-NEXT:   id[item_ct1.get_local_id(2)] = item_ct1.get_sub_group().get_local_linear_id();
// CHECK-NEXT: }
__global__ void laneid(int *id) {
  id[threadIdx.x] = cub::LaneId();
}

int main() {
  int *id = nullptr;
  cudaMallocManaged(&id, sizeof(int) * 10);
  laneid<<<1, 10>>>(id);
  for (int i = 0; i < 10; ++i) {
    if (id[i] != i)
      return 1;
  }
  return 0;
}
