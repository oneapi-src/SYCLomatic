// RUN: echo

// CHECK: SYCL_EXTERNAL int test2() {
__host__ __device__ int test2() {
  return 5;
}
