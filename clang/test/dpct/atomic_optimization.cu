// RUN: dpct --format-range=none --optimize-migration -out-root %T/atomic_optimization %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/atomic_optimization/atomic_optimization.dp.cpp --match-full-lines %s
#include <cuda_runtime.h>

#include <iostream>
#include <memory>

// CHECK: inline void foo(unsigned int p){}
__device__ void foo(unsigned int p){}
// CHECK: dpct::global_memory<unsigned int, 0> a1(4 * 2);
// CHECK: __dpct_inline__ void kernel1(unsigned int &a1){
// CHECK:   /*
// CHECK:   DPCT1116:{{[0-9]+}}: The atomicInc was migrated to dpct::atomic_fetch_add(&a1, 2) / 2 for performance, and 2 is computed by (UINT_MAX + 1) / ('0x7fffffff' + 1). This migration requires the initial value of 'a1' to be scaled by multiplying 2, and any usage of value of 'a1' outside atomic function to be scaled by dividing 2.
// CHECK:   */
// CHECK:   dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(&a1, 2) / 2;
// CHECK:   int c = a1 / 2;
// CHECK:   foo(a1 / 2);
// CHECK: }
__device__ unsigned int a1 = 4;

__global__ void kernel1(){

  atomicInc(&a1, 0x7fffffff);

  int c = a1;

  foo(a1);

}
// CHECK: dpct::global_memory<unsigned int, 1> b1(sycl::range<1>(10), {0 * 2});
// CHECK: __dpct_inline__ void kernel2(unsigned int *b1){
// CHECK:   /*
// CHECK:   DPCT1116:{{[0-9]+}}: The atomicInc was migrated to dpct::atomic_fetch_add(b1, 2) / 2 for performance, and 2 is computed by (UINT_MAX + 1) / ('0x7fffffff' + 1). This migration requires the initial value of 'b1[index]' to be scaled by multiplying 2, and any usage of value of 'b1[index]' outside atomic function to be scaled by dividing 2.
// CHECK:   */
// CHECK:   dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(b1, 2) / 2;
// CHECK:   foo(b1[0] / 2);
// CHECK: }
__device__ unsigned int b1[10] = {0};

__global__ void kernel2(){

  atomicInc(b1, 0x7fffffff);

  foo(b1[0]);

}
// CHECK: dpct::global_memory<unsigned int, 0> a2(10);
// CHECK: __dpct_inline__ void kernel3(unsigned int &a2){
// CHECK:     dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(&a2, 1);
// CHECK: }
__device__ unsigned int a2 = 10;

__global__ void kernel3(){

    atomicInc(&a2, 0xffffffff);

}
// CHECK: dpct::global_memory<unsigned int, 1> b2(sycl::range<1>(10), {0 * 2});
// CHECK: __dpct_inline__ void kernel4(unsigned int *b2){
// CHECK:     /*
// CHECK:     DPCT1116:{{[0-9]+}}: The atomicInc was migrated to dpct::atomic_fetch_add(&b2[0], 2) / 2 for performance, and 2 is computed by (UINT_MAX + 1) / ('0x7fffffff' + 1). This migration requires the initial value of 'b2[index]' to be scaled by multiplying 2, and any usage of value of 'b2[index]' outside atomic function to be scaled by dividing 2.
// CHECK:     */
// CHECK:     dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(&b2[0], 2) / 2;
// CHECK:     dpct::atomic_fetch_max<sycl::access::address_space::generic_space>(&b2[1], b2[2]) / 2;
// CHECK: }
__device__ unsigned int b2[10] = {0};

__global__ void kernel4(){

    atomicInc(&b2[0], 0x7fffffff);

    atomicMax(&b2[1], b2[2]);

}
// CHECK: dpct::global_memory<unsigned int, 0> a3(10 * 2);
// CHECK: __dpct_inline__ void kernel5(unsigned int &a3){
// CHECK:     /*
// CHECK:     DPCT1116:{{[0-9]+}}: The atomicInc was migrated to dpct::atomic_fetch_add(&a3, 2) / 2 for performance, and 2 is computed by (UINT_MAX + 1) / ('0x7fffffff' + 1). This migration requires the initial value of 'a3' to be scaled by multiplying 2, and any usage of value of 'a3' outside atomic function to be scaled by dividing 2.
// CHECK:     */
// CHECK:     dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(&a3, 2) / 2;
// CHECK:     /*
// CHECK:     DPCT1116:{{[0-9]+}}: The atomicDec was migrated to dpct::atomic_fetch_sub(&a3, 2) / 2 for performance, and 2 is computed by (UINT_MAX + 1) / ('0x7fffffff' + 1). This migration requires the initial value of 'a3' to be scaled by multiplying 2, and any usage of value of 'a3' outside atomic function to be scaled by dividing 2.
// CHECK:     */
// CHECK:     dpct::atomic_fetch_sub<sycl::access::address_space::generic_space>(&a3, 2) / 2;
// CHECK:     foo(a3 / 2);
// CHECK: }
__device__ unsigned int a3 = 10;

__global__ void kernel5(){

    atomicInc(&a3, 0x7fffffff);

    atomicDec(&a3, 0x7fffffff);

    foo(a3);

}
// CHECK: dpct::global_memory<unsigned int, 1> b3(10);
// CHECK: void kernel6(unsigned int *b3){
// CHECK:     /*
// CHECK:     DPCT1116:{{[0-9]+}}: The atomicInc was migrated to dpct::atomic_fetch_add(&b3[0], 2) / 2 for performance, and 2 is computed by (UINT_MAX + 1) / ('0x7fffffff' + 1). This migration requires the initial value of 'b3[index]' to be scaled by multiplying 2, and any usage of value of 'b3[index]' outside atomic function to be scaled by dividing 2.
// CHECK:     */
// CHECK:     dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(&b3[0], 2) / 2;
// CHECK: }
__device__ unsigned int b3[10];

__global__ void kernel6(){

    atomicInc(&b3[0], 0x7fffffff);

}

int main(){
  unsigned int *d_addr;
// CHECK:   /*
// CHECK:   DPCT1117:{{[0-9]+}}: There is atomicInc/Dec operation on 'b3[index]' and value of 'b3[index]' was scaled by migration for performance, refer to DPCT1116. Using the value of 'b3[index]' through 'd_addr' should also be scaled by dividing or multiplying 2. You may need to adjust the code.
// CHECK:   */
  cudaGetSymbolAddress((void **)&d_addr, b3);
// CHECK:   /*
// CHECK:   DPCT1117:{{[0-9]+}}: There is atomicInc/Dec operation on 'a3' and value of 'a3' was scaled by migration for performance, refer to DPCT1116. Using the value of 'a3' through 'd_addr' should also be scaled by dividing or multiplying 2. You may need to adjust the code.
// CHECK:   */
  cudaGetSymbolAddress((void **)&d_addr, a3);
  cudaMemset(d_addr, 0, 10 * sizeof(int));

  kernel1<<<1, 1>>>();
  kernel2<<<1, 1>>>();
  kernel3<<<1, 1>>>();
  kernel4<<<1, 1>>>();
  kernel5<<<1, 1>>>();

  return 0;
}