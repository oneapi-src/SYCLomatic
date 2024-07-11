// RUN: dpct --format-range=none -out-root %T/adl %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck %s --match-full-lines --input-file %T/adl/global_test.dp.cpp

#include <cuda.h>
#include <iostream>

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

__device__ void norm2(int a, float3 test) {
}

__device__ void norm2(float3 test, int a) {
}

template <class T>
__device__ void test_norm2(T test) {
}

__global__ void global_test() {
  float3 f3;
  norm2(f3);
}

namespace ns {
__global__ void func();
};
__global__ void ns::func() {
   float3 f3;
  // CHECK: ::norm2(f3);
  norm2(f3);
}
namespace test {
void __global__ my_test() {
  float3 hello;

  // CHECK: ::norm2(hello);
  ::norm2(hello);
  // CHECK: ::norm2(hello);
  norm2(hello);
  // CHECK: ::norm2(3, hello);
  // CHECK: ::norm2(hello, 4);
  // CHECK: ::test_norm2(hello);
  norm2(3, hello);
  norm2(hello, 4);
  test_norm2(hello);
  // CHECK: hello::norm2(hello);
  hello::norm2(hello);
}
template <class T>
__global__ void test_temp(T a) {
  // CHECK:  test_norm2<T>(a);
  test_norm2<T>(a);
}

} // namespace test
