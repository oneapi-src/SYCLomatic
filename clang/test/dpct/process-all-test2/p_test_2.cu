// RUN: cat %S/readme_2_ref.txt  >%T/readme_2.txt

// RUN: dpct --format-range=none -output-file=output-file.txt -in-root=%S -out-root=%T %s --process-all --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only

// RUN: cat %S/readme_2.txt > %T/check_output-file.txt
// RUN: cat %T/output-file.txt >>%T/check_output-file.txt
// RUN: FileCheck --match-full-lines --input-file %T/check_output-file.txt %T/check_output-file.txt

// RUN: FileCheck --input-file %T/p_test_2.dp.cpp --match-full-lines %S/p_test_2.cu
// RUN: FileCheck --match-full-lines --input-file %S/readme_2_ref.txt %T/readme_2.txt

// This test case is used to verify that if process-all conflicts
// with other input files specified, ignore process-all, and isolated file will not be copied,
// and warning msg that “process-all” option was ignored, since input files were provided in command line will be emitted.
#include "cuda_runtime.h"
#include <stdio.h>

// CHECK: void addKernel(int *c, const int *a, const int *b, sycl::nd_item<3> item_ct1)
__global__ void addKernel(int *c, const int *a, const int *b)
{
    // CHECK: int i = item_ct1.get_local_id(2);
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}
