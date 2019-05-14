#include "multi-files-kernel.cuh"

// CHECK: int global_id(cl::sycl::nd_item<3> item_{{[a-f0-9]+}}) {
__device__ int global_id() {
  unsigned x = 0;
  ATOMIC_UPDATE(x)
  return blockIdx.x * blockDim.x + threadIdx.x;
}
