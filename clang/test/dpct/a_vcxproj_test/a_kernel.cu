// UNSUPPORTED: -linux-
// RUN: cat %S/DemoCudaProj.vcxproj > %T/DemoCudaProj.vcxproj
// RUN: cat %S/a_kernel.cu > %T/a_kernel.cu
// RUN: cat %S/readme_1.txt > %T/readme_1.txt
// RUN: cd %T
// RUN: dpct --format-range=none  --vcxprojfile=%T/DemoCudaProj.vcxproj  -in-root=%T -out-root=%T/out  %T/a_kernel.cu --cuda-include-path="%cuda-path/include"

// RUN: cat %S/check_compilation_ref.txt  >%T/check_compilation_db.txt
// RUN: cat %T/compile_commands.json >> %T/check_compilation_db.txt

// RUN: FileCheck --match-full-lines --input-file %T/check_compilation_db.txt %T/check_compilation_db.txt

// RUN: dpct --format-range=none  -p=%T  -in-root=%T -out-root=%T/2 --process-all --cuda-include-path="%cuda-path/include"

// RUN: FileCheck --input-file %T/2/a_kernel.dp.cpp --match-full-lines %S/a_kernel.cu
// RUN: FileCheck --match-full-lines --input-file %T/2/readme_1.txt %T/2/readme_1.txt

#include "cuda_runtime.h"
#include <stdio.h>

// CHECK: void addKernel(int *c, const int *a, const int *b, sycl::nd_item<3> item_ct1)
__global__ void addKernel(int *c, const int *a, const int *b)
{
    // CHECK: int i = item_ct1.get_local_id(2);
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}
