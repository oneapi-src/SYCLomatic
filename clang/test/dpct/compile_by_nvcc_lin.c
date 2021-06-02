// UNSUPPORTED: -windows-
// RUN: cd %T
// RUN: rm -rf ./compile_by_nvcc
// RUN: mkdir %T/compile_by_nvcc
// RUN: cd %T/compile_by_nvcc
// RUN: cp %s %T/compile_by_nvcc
// RUN: echo "[" > %T/compile_by_nvcc/compile_commands.json
// RUN: echo "  {" >> %T/compile_by_nvcc/compile_commands.json
// RUN: echo "    \"command\": \"nvcc -c -m64 -o compile_by_nvcc_lin.o compile_by_nvcc_lin.c\"," >> %T/compile_by_nvcc/compile_commands.json
// RUN: echo "    \"directory\": \"%T/compile_by_nvcc\"," >> %T/compile_by_nvcc/compile_commands.json
// RUN: echo "    \"file\": \"%T/compile_by_nvcc/compile_by_nvcc_lin.c\"" >> %T/compile_by_nvcc/compile_commands.json
// RUN: echo "  }" >> %T/compile_by_nvcc/compile_commands.json
// RUN: echo "]" >> %T/compile_by_nvcc/compile_commands.json
// RUN: dpct --format-range=none --out-root %T/compile_by_nvcc -p=./ --cuda-include-path="%cuda-path/include"
// RUN: FileCheck %s --match-full-lines --input-file %T/compile_by_nvcc/compile_by_nvcc_lin.c.dp.cpp
// RUN: rm -rf %T/compile_by_nvcc


// CHECK: #include <CL/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
#include <cuda_runtime.h>

// CHECK: int a = 0;
// CHECK-NEXT: dpct::constant_memory<float, 1> const_angle(360);
// CHECK-NEXT: void simple_kernel(float *d_array, float *const_angle) {
// CHECK-NEXT:   d_array[0] = const_angle[0];
// CHECK-NEXT:   return;
// CHECK-NEXT: }
int a = 0;
__constant__ float const_angle[360];
__global__ void simple_kernel(float *d_array) {
  d_array[0] = const_angle[0];
  return;
}
