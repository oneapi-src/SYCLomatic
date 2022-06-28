// RUN: dpct --format-range=none -out-root %T/nested_device_call %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/nested_device_call/nested_device_call.dp.cpp --match-full-lines %s

#include <cstdio>

// CHECK: void test0_with_item(int a, sycl::nd_item<3> [[ITEM:item_ct1]]) {
__device__ void test0_with_item(int a) {
  int i = threadIdx.x;
}

// CHECK: void test0(int a) {
__device__ void test0(int a) {
  // CHECK: sqrt(10.0);
  sqrt(10.0);
}

// CHECK: void test1(int a) {
__device__ void test1(int a) {
  // CHECK: test0(a);
  test0(a);
  // CHECK: test0(a + 1);
  test0(a + 1);
}

// CHECK: void test1_with_item(int a, sycl::nd_item<3> [[ITEM:item_ct1]]) {
__device__ void test1_with_item(int a) {
  //CHECK: test0_with_item(a, [[ITEM]]);
  test0_with_item(a);
  test0(a);
}

// CHECK: void test2(int a) {
__device__ void test2(int a) {
  // CHECK: test1(a);
  test1(a);
  // CHECK: test1(a + 1);
  test1(a + 1);
}

// CHECK: void test3(int a) {
__device__ void test3(int a) {
  // CHECK: test2(a);
  test2(a);
  // CHECK: test2(a + 1);
  test2(a + 1);
}

// CHECK: void kernel() {
__global__ void kernel() {
  // CHECK: test3(1);
  test3(1);
  // CHECK: test3(2);
  test3(2);
}

// CHECK: void kernel_with_item(sycl::nd_item<3> [[ITEM:item_ct1]]) {
__global__ void kernel_with_item() {
  // CHECK: test1_with_item(1, [[ITEM]]);
  test1_with_item(1);
  // CHECK: test3(2);
  test3(2);
}

namespace n1 {
  __device__ void test4(int a);
}

namespace n2 {
using namespace n1;
// CHECK: void test4(int a, sycl::nd_item<3> [[ITEM:item_ct1]]) {
__device__ void test4(int a) {
  int i = threadIdx.x;
}
}

int main() {
  kernel<<<1, 1>>>();
  kernel_with_item<<<1, 1>>>();
}

