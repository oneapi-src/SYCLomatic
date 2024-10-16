// RUN: dpct --use-experimental-features=device_global -in-root %S -out-root %T/device_global %S/device_global.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/device_global/device_global.dp.cpp --match-full-lines %s

#include <cuda_runtime.h>
#include <iostream>
#include <vector>

class A{
public:
  int data = 0;
};

struct B{
  int data = 1;
};


// CHECK: static sycl::ext::oneapi::experimental::device_global<int> var_a;
// CHECK: /*
// CHECK: DPCT1127:{{[0-9]+}}: The constant compile-time initialization for device_global is supported when compiling with C++20. You may need to adjust the compile commands.
// CHECK: */
// CHECK: static sycl::ext::oneapi::experimental::device_global<int> var_b{0};
// CHECK: /*
// CHECK: DPCT1127:{{[0-9]+}}: The constant compile-time initialization for device_global is supported when compiling with C++20. You may need to adjust the compile commands.
// CHECK: */
// CHECK: static sycl::ext::oneapi::experimental::device_global<const float> var_c{2.f};
// CHECK: /*
// CHECK: DPCT1127:{{[0-9]+}}: The constant compile-time initialization for device_global is supported when compiling with C++20. You may need to adjust the compile commands.
// CHECK: */
// CHECK: static sycl::ext::oneapi::experimental::device_global<const float> var_d{1.f};
// CHECK: static sycl::ext::oneapi::experimental::device_global<A> var_e;
// CHECK: static sycl::ext::oneapi::experimental::device_global<const B> var_f;
// CHECK: static dpct::global_memory<int, 0> var_g;
__device__ int var_a;
__device__ int var_b = 0;
__constant__ float var_c = 2.f;
__constant__ float var_d = 1.f;
__device__ A var_e;
__constant__ B var_f;
__device__ int var_g;
// CHECK: static sycl::ext::oneapi::experimental::device_global<float[10]> arr_a;
// CHECK: /*
// CHECK: DPCT1127:{{[0-9]+}}: The constant compile-time initialization for device_global is supported when compiling with C++20. You may need to adjust the compile commands.
// CHECK: */
// CHECK: static sycl::ext::oneapi::experimental::device_global<float[10]> arr_b{1, 2, 3};
// CHECK: /*
// CHECK: DPCT1127:{{[0-9]+}}: The constant compile-time initialization for device_global is supported when compiling with C++20. You may need to adjust the compile commands.
// CHECK: */
// CHECK: static sycl::ext::oneapi::experimental::device_global<const int[10]> arr_c{3, 2, 1};
// CHECK: /*
// CHECK: DPCT1127:{{[0-9]+}}: The constant compile-time initialization for device_global is supported when compiling with C++20. You may need to adjust the compile commands.
// CHECK: */
// CHECK: static sycl::ext::oneapi::experimental::device_global<const int[10]> arr_d{2};
// CHECK: static sycl::ext::oneapi::experimental::device_global<const A[10]> arr_e;
// CHECK: static sycl::ext::oneapi::experimental::device_global<B[10]> arr_f;
__device__ float arr_a[10];
__device__ float arr_b[10] = {1, 2, 3};
__constant__ int arr_c[10] = {3, 2, 1};
__constant__ int arr_d[10] = {2};
__constant__ A arr_e[10];
__device__ B arr_f[10];


// CHECK: int device_func() {
// CHECK:   arr_a[0] = 1;
// CHECK:   return arr_a[0] + arr_b[0] + arr_c[0] + arr_d[0] + arr_e[0].data + arr_f[0].data;
// CHECK: }

// CHECK: void kernel(int *ptr) {
// CHECK:   var_a.get() = 0;
// CHECK:   var_e.get().data = 1;
// CHECK:   *ptr = var_a.get() + var_b.get() + var_c.get() + var_d.get() + var_e.get().data + var_f.get().data + device_func();
// CHECK: }
__device__ int device_func() {
  arr_a[0] = 1;
  return arr_a[0] + arr_b[0] + arr_c[0] + arr_d[0] + arr_e[0].data + arr_f[0].data;
}

__global__ void kernel(int *ptr) {
  var_a = 0;
  var_e.data = 1;
  *ptr = var_a + var_b + var_c + var_d + var_e.data + var_f.data + device_func();
}

// CHECK: void kernel2(int *ptr,
// CHECK:   int &var_g) {
// CHECK: *ptr = var_g;
// CHECK: }
__global__ void kernel2(int *ptr) {
  *ptr = var_g;
}

int main() {
    int *dev;
    cudaMallocManaged(&dev, sizeof(int));
    *dev = 0;
    kernel<<<1, 1>>>(dev);
    kernel2<<<1, 1>>>(dev);
    cudaDeviceSynchronize();
    std::cout << *dev << std::endl;
    size_t size;
    cudaGetSymbolSize(&size, var_g);
    return 0;
}
