// UNSUPPORTED: -linux-
// RUN: cat %S/DemoCudaProj.vcxproj > %T/DemoCudaProj.vcxproj
// RUN: cat %S/d_kernel.cu > %T/d_kernel.cu
// RUN: cd %T
// RUN: dpct --format-range=none  --vcxprojfile=%T/DemoCudaProj.vcxproj  -in-root=%T -out-root=%T/out  d_kernel.cu --cuda-include-path="%cuda-path/include"
// RUN: dpct --format-range=none  -p %T  -in-root=%T -out-root=%T/out  d_kernel.cu --cuda-include-path="%cuda-path/include"
// RUN: dpct --format-range=none  --vcxprojfile=%T/DemoCudaProj.vcxproj  -in-root=%T -out-root=%T/out  d_kernel.cu --cuda-include-path="%cuda-path/include"
// RUN: FileCheck --match-full-lines --input-file %T/out/d_kernel.dp.cpp %T/d_kernel.cu

#include "cuda_runtime.h"
#include <stdio.h>

// CHECK: void addKernel(int *c, const int *a, const int *b, sycl::nd_item<3> item_ct1)
__global__ void addKernel(int *c, const int *a, const int *b)
{
    // CHECK: int i = item_ct1.get_local_id(2);
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}
