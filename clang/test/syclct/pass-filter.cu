// RUN: syclct -out-root %T %s -passes "IterationSpaceBuiltinRule" -- -x cuda --cuda-host-only --cuda-path=%cuda-path
// RUN: FileCheck %s --match-full-lines --input-file %T/pass-filter.sycl.cpp

// Test that only IterationSpaceBuiltinRule is being run
// CHECK:__global__ void test_00() {
__global__ void test_00() {
  // CHECK: size_t tix = item_{{[a-f0-9]+}}.get_local_id(0);
  // CHECK: size_t tiy = item_{{[a-f0-9]+}}.get_local_id(1);
  // CHECK: size_t tiz = item_{{[a-f0-9]+}}.get_local_id(2);
  size_t tix = threadIdx.x;
  size_t tiy = threadIdx.y;
  size_t tiz = threadIdx.z;
}
