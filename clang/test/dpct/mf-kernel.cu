#include "mf-kernel.cuh"
// RUN: dpct --usm-level=none -in-root %S -out-root %T %s %S/mf-test.cu -extra-arg="-I %S" --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -std=c++14 -x cuda --cuda-host-only

  // CHECK: dpct::device_memory<volatile int, 0> g_mutex(0);
volatile __device__ int g_mutex=0;
// CHECK: SYCL_EXTERNAL void Reset_kernel_parameters(dpct::accessor<volatile int, dpct::device, 0> g_mutex)
__global__ void Reset_kernel_parameters(void)
{
    g_mutex=0;
}

// CHECK: SYCL_EXTERNAL void test_foo(){
__device__ void test_foo(void){
}
