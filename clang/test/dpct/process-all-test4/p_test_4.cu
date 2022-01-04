// RUN: cat %S/readme_4.txt > %T/readme_4.txt
// RUN: cat %S/p_test_4.cu > %T/p_test_4.cu

// RUN: dpct --format-range=none  -in-root=%T -out-root=%T/dpct-output --process-all --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: dpct --format-range=none  -in-root=%T -out-root=%T/dpct-output --process-all --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only

// RUN: FileCheck --input-file %T/dpct-output/p_test_4.dp.cpp --match-full-lines %S/p_test_4.cu
// RUN: FileCheck --match-full-lines --input-file %T/dpct-output/readme_4.txt %T/dpct-output/readme_4.txt


#include "cuda_runtime.h"
#include <stdio.h>

// This test case is used to verify that
// the issue of loop copy out-root file

// CHECK: void addKernel(int *c, const int *a, const int *b, sycl::nd_item<3> item_ct1)
__global__ void addKernel(int *c, const int *a, const int *b)
{
    // CHECK: int i = item_ct1.get_local_id(2);
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}
