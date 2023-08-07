// RUN: FileCheck %S/test.cu --match-full-lines --input-file %T/test_inline_optimization/test.dp.cpp

// CHECK: SYCL_EXTERNAL int test2() {
__host__ __device__ int test2() {
  return 5;
}