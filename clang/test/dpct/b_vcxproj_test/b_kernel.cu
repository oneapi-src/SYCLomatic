// UNSUPPORTED: -linux-
// RUN: cat %S/SVMbenchmark.vcxproj > %T/SVMbenchmark.vcxproj
// RUN: cat %S/b_kernel.cu > %T/b_kernel.cu
// RUN: cat %S/header.cuh > %T/header.cuh
// RUN: cat %S/header.cuh > %T/header.cuh
// RUN: cat %S/header.cuh > %T/header.cuh
// RUN: dpct -output-file=b_kernel_outputfile_win.txt --format-range=none  --vcxprojfile=%T/SVMbenchmark.vcxproj  -in-root=%T -out-root=%T/out  %T/b_kernel.cu -extra-arg="-I %T" --cuda-include-path="%cuda-path/include"

// RUN: cat %S/check_compilation_ref.txt  >%T/check_compilation_db.txt
// RUN: cat %T/compile_commands.json >>%T/check_compilation_db.txt
// RUN: FileCheck --match-full-lines --input-file %T/check_compilation_db.txt %T/check_compilation_db.txt

// RUN: cat %S/b_kernel_outputfile_ref_window.txt > %T/check_b_kernel_outputfile_windows.txt
// RUN: cat %T/out/b_kernel_outputfile_win.txt >>%T/check_b_kernel_outputfile_windows.txt
// RUN: FileCheck --match-full-lines --input-file %T/check_b_kernel_outputfile_windows.txt %T/check_b_kernel_outputfile_windows.txt

// RUN: dpct --format-range=none -output-file=output-file.txt -in-root=%T -out-root=%T/2 %T/b_kernel.cu --process-all --cuda-include-path="%cuda-path/include"
// RUN: cat %S/readme_2_ref.txt > %T/2/readme_2.txt
// RUN: cat %S/readme_2.txt > %T/2/check_output-file.txt
// RUN: cat %T/2/output-file.txt >>%T/2/check_output-file.txt
// RUN: FileCheck --match-full-lines --input-file %T/2/check_output-file.txt %T/2/check_output-file.txt

// RUN: FileCheck --input-file %T/2/b_kernel.dp.cpp --match-full-lines %S/b_kernel.cu
// RUN: FileCheck --match-full-lines --input-file %S/readme_2_ref.txt %T/2/readme_2.txt

#include "header.cuh"
#include "cuda_runtime.h"
#include <stdio.h>

// CHECK: void addKernel(int *c, const int *a, const int *b, sycl::nd_item<3> item_ct1)
__global__ void addKernel(int *c, const int *a, const int *b)
{
    // CHECK: int i = item_ct1.get_local_id(2);
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

// To make dpct clang parser emit three lines below:
// In file included from \path\to\b_vcxproj_test\b_kernel.cu:line_number:
// \path\to\b_vcxproj_test\header.cuh:5:9: warning: DPCT1003:0: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
//        return cudaDeviceReset();
// This three lines are expected in check_b_kernel_outputfile_windows.txt.
void foo(){
    cudaError_t cudaStatus = resetDevice();
}
