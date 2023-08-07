#include <cuda_runtime.h>
// CHECK: inline int test();
__host__ __device__ int test();
// CHECK: inline int test1() {
__host__ __device__ int test1() {
  return 5;
}

// CHECK: SYCL_EXTERNAL int test2();
__host__ __device__ int test2();
