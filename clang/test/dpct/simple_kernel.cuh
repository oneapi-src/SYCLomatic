// CHECK:#include <CL/sycl.hpp>
// CHECK-NEXT:#include <dpct/dpct.hpp>
#include <stdio.h>

// CHECK:void simple_kernel(sycl::nd_item<3> item_{{[a-f0-9]+}},
// CHECK-NEXT: float *d_array) {
// CHECK-NEXT:  int index;
// CHECK-NEXT:  index = item_{{[a-f0-9]+}}.get_group(0) * item_{{[a-f0-9]+}}.get_local_range(0) + item_{{[a-f0-9]+}}.get_local_id(0);
// CHECK-NEXT:  if (index < 360) {
// CHECK-NEXT:    d_array[index] = 10.0;
// CHECK-NEXT:  }
// CHECK-NEXT:  return;
// CHECK-NEXT:}
// This file is included by cuda_kernel_include.cu
__global__ void simple_kernel(float *d_array) {
  int index;
  index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < 360) {
    d_array[index] = 10.0;
  }
  return;
}
