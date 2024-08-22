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
  // CHECK: norm2(f3);
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
  // Some CUDA struct (like float3) is defined in the CUDA global namespace and
  // the Argument-dependent lookup(ADL) will search the function in the global namespace. After migration,
  // the float3 will be in the sycl namespace and the ADL will search the function
  // implementation under the SYCL namepsace which cause the failure. Solution
  // is to force calling the global namespace function implementation.
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

void foo(float3);
namespace t {
void foo(int a);
template <class T>
void test(T a) {
  // CHECK: foo(a);
  foo(a);
}

void test1() {
  int a;
  float3 fa;
  test<int>(a);
  test<float3>(fa);
}
} // namespace t

namespace n {
__device__ void foo(int3 a);
void __global__ my_test() {
  int3 a;
  // CHECK: foo(a);
  foo(a);
}

} // namespace n
__device__ void n::foo(int3 a) {}

__device__ void foo(float3 b) {
}
void __global__ my_test() {
  float3 b;
  // CHECK: foo(b);
  foo(b);
}

typedef dim3 mydim;

__device__ void norm2(mydim test) {
}

namespace dim_test {
void __global__ my_test() {
  mydim hello;
  // CHECK: ::norm2(hello);
  norm2(hello);
}
} // namespace dim_test
