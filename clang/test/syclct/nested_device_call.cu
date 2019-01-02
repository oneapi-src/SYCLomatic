// RUN: syclct -out-root %T %s -- -x cuda --cuda-host-only --cuda-path=%cuda-path
// RUN: FileCheck --input-file %T/nested_device_call.sycl.cpp --match-full-lines %s

#include <cstdio>

// CHECK: void test0(int a, cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
__device__ void test0(int a) {
  // CHECK: printf("Hello World %d\n", a);
  printf("Hello World %d\n", a);
  // CHECK: cl::sycl::sqrt(10.0);
  sqrt(10.0);
}

// CHECK: void test1(int a, cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
__device__ void test1(int a) {
  // CHECK: test0(a, [[ITEM]]);
  test0(a);
  // CHECK: test0(a + 1, [[ITEM]]);
  test0(a + 1);
}

// CHECK: void test2(int a, cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
__device__ void test2(int a) {
  // CHECK: test1(a, [[ITEM]]);
  test1(a);
  // CHECK: test1(a + 1, [[ITEM]]);
  test1(a + 1);
}

// CHECK: void test3(int a, cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
__device__ void test3(int a) {
  // CHECK: test2(a, [[ITEM]]);
  test2(a);
  // CHECK: test2(a + 1, [[ITEM]]);
  test2(a + 1);
}

// CHECK: void kernel(cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
__global__ void kernel() {
  // CHECK: test3(1, [[ITEM]]);
  test3(1);
  // CHECK: test3(2, [[ITEM]]);
  test3(2);
}

int main() { kernel<<<1, 1>>>(); }
