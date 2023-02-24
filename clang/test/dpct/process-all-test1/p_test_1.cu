// RUN: dpct --format-range=none  -in-root=%S -out-root=%T --process-all --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only

// RUN: FileCheck --input-file %T/p_test_1.dp.cpp --match-full-lines %S/p_test_1.cu
// RUN: FileCheck --match-full-lines --input-file %T/readme_1.txt %T/readme_1.txt
// RUN: FileCheck --match-full-lines --input-file %T/standalone_1.dp.cpp %S/standalone_1.cu

#include "cuda_runtime.h"
#include <stdio.h>

// This test case is used to verify that
// if process-all and in-root are passed and no other input specified, try to migrate or copy all files from in-root.

// CHECK: void addKernel(int *c, const int *a, const int *b,
// CHECK-NEXT: const sycl::nd_item<3> &item_ct1)
__global__ void addKernel(int *c, const int *a, const int *b)
{
    // CHECK: int i = item_ct1.get_local_id(2);
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}
