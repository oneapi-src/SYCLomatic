#include "multi-files-kernel.cuh"

// CHECK: int global_id(sycl::nd_item<3> item_ct1) {
__device__ int global_id() {
  unsigned x = 0;
  ATOMIC_UPDATE(x)
  return blockIdx.x * blockDim.x + threadIdx.x;
}

void foo(){
  sgemm();
  randomGen();
}
