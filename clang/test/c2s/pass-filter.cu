// RUN: c2s --format-range=none -out-root %T/pass-filter %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/pass-filter/pass-filter.dp.cpp

// Test that only IterationSpaceBuiltinRule is being run
// CHECK: void test_00(sycl::nd_item<3> item_ct1) {
__global__ void test_00() {
  // CHECK: size_t tix = item_ct1.get_local_id(2);
  // CHECK: size_t tiy = item_ct1.get_local_id(1);
  // CHECK: size_t tiz = item_ct1.get_local_id(0);
  size_t tix = threadIdx.x;
  size_t tiy = threadIdx.y;
  size_t tiz = threadIdx.z;
}

