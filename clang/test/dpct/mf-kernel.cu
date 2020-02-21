#include "mf-kernel.cuh"
// RUN: dpct --usm-level=none -in-root %S -out-root %T %s %S/mf-test.cu -extra-arg="-I %S" --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -std=c++14 -x cuda --cuda-host-only

  // CHECK: dpct::device_memory<volatile int, 0> g_mutex(0);
volatile __device__ int g_mutex=0;
// CHECK: SYCL_EXTERNAL void Reset_kernel_parameters(volatile int *g_mutex)
__global__ void Reset_kernel_parameters(void)
{
    g_mutex=0;
}

// CHECK: SYCL_EXTERNAL void test_foo(){
__device__ void test_foo(void){
}

// CHECK: static void local_foo_1() {}
__global__ static void local_foo_1() {}

// CHECK: static void local_foo_2();
// CHECK-NEXT: void local_foo_2();
// CHECK-NEXT: void local_foo_2() { }
__global__ static void local_foo_2();
__global__ void local_foo_2();
__global__ void local_foo_2() { }