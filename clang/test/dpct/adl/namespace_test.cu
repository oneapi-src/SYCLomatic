// RUN: dpct --format-range=none -out-root %T/adl %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck %s --match-full-lines --input-file %T/adl/namespace_test.dp.cpp

#include <cuda.h>
#include <iostream>
#include <mma.h>

namespace test {
template <class T>
void norm2(T *p) {
}
} // namespace test
namespace hello {
// CHECK: void norm2(sycl::float3 test) {
__device__ void norm2(float3 test) {
}
} // namespace hello
// CHECK: void norm2(sycl::float3 test) {
__device__ void norm2(float3 test) {
}
namespace test {
void __global__ my_test() {
  float3 hello;

  // CHECK: hello::norm2(hello);
  hello::norm2(hello);

  hello::norm2(hello);
}
} // namespace test
int main() {
  // test::my_test();
  return 0;
}