// RUN: syclct -in-root %S -out-root %T %s -- -std=c++14 -x cuda --cuda-host-only --cuda-path="%cuda-path"
// RUN: FileCheck %s --match-full-lines --input-file %T/mf-kernel.dp.cpp

#include "mf-kernel.cuh"

  // CHECK: dpct::device_memory<volatile int, 0> g_mutex(dpct::dpct_range<0>(), 0);
volatile __device__ int g_mutex=0;

__global__ void Reset_kernel_parameters(void)
{
    g_mutex=0;
}
