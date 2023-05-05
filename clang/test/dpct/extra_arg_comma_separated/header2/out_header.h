#include "cuda_runtime.h"

// CHECK: inline void helloFromGPU(int i, const cl::sycl::nd_item<3> &item_ct1) {
// CHECK-NEXT:     int a = item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2) + item_ct1.get_group(2) +
// CHECK-NEXT:     item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
// CHECK-NEXT: }
__global__ void helloFromGPU() {
  int a = blockIdx.x * blockDim.x + threadIdx.x + blockIdx.x +
          blockDim.x + threadIdx.x;
}
