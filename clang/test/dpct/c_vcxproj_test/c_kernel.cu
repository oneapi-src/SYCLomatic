// UNSUPPORTED: system-linux
// RUN: cat %S/proj_c.vcxproj > %T/proj_c.vcxproj
// RUN: cd %T

// RUN: dpct --format-range=none  --vcxprojfile=%T/proj_c.vcxproj  -in-root=%S -out-root=%T  %s %S/CuTmp_1.cu --cuda-include-path="%cuda-path/include"

// RUN: cat %S/CuTmp_1.cu > %T/CuTmp_1.dp.cpp
// RUN: cat %S/check_compilation_ref.txt  >%T/check_compilation_db.txt
// RUN: cat %T/compile_commands.json >>%T/check_compilation_db.txt
// RUN: FileCheck --match-full-lines --input-file %T/check_compilation_db.txt %T/check_compilation_db.txt
// RUN: FileCheck %S/CuTmp_1.cu --match-full-lines --input-file %T/CuTmp_1.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/CuTmp_1.dp.cpp -o %T/CuTmp_1.dp.o %}

// RUN: dpct -output-file=output-file.txt --format-range=none -in-root=%S -out-root=%T/2 -p %S --process-all --cuda-include-path="%cuda-path/include"
// RUN: FileCheck --input-file %T/2/c_kernel.dp.cpp --match-full-lines %S/c_kernel.cu
// RUN: %if build_lit %{icpx -c -fsycl %T/2/c_kernel.dp.cpp -o %T/2/c_kernel.dp.o %}
// RUN: FileCheck --match-full-lines --input-file %T/2/readme_3.txt %T/2/readme_3.txt

#include "cuda_runtime.h"
#include <stdio.h>

// CHECK: void addKernel(int *c, const int *a, const int *b,
// CHECK-NEXT: const sycl::nd_item<3> &item_ct1)
__global__ void addKernel(int *c, const int *a, const int *b)
{
    // CHECK: int i = item_ct1.get_local_id(2);
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}
