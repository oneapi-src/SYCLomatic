#include <cuda_runtime.h>
// CHECK: int test();
__host__ __device__ int test();
// CHECK: inline int test1() {
__host__ __device__ int test1() {
  return 5;
}
