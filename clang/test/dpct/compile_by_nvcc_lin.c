// UNSUPPORTED: -windows-
// RUN: mkdir %T/compile_by_nvcc
// RUN: cd %T/compile_by_nvcc
// RUN: cp %s %T/compile_by_nvcc
// RUN: echo "[" > %T/compile_by_nvcc/compile_commands.json
// RUN: echo "  {" >> %T/compile_by_nvcc/compile_commands.json
// RUN: echo "    \"command\": \"nvcc -c -m64 -I/%cuda-path/include -o compile_by_nvcc_lin.o compile_by_nvcc_lin.c\"," >> %T/compile_by_nvcc/compile_commands.json
// RUN: echo "    \"directory\": \"%T/compile_by_nvcc\"," >> %T/compile_by_nvcc/compile_commands.json
// RUN: echo "    \"file\": \"%T/compile_by_nvcc/compile_by_nvcc_lin.c\"" >> %T/compile_by_nvcc/compile_commands.json
// RUN: echo "  }" >> %T/compile_by_nvcc/compile_commands.json
// RUN: echo "]" >> %T/compile_by_nvcc/compile_commands.json
// RUN: dpct --out-root %T/compile_by_nvcc -p=./ --cuda-include-path="%cuda-path/include"
// RUN: FileCheck %s --match-full-lines --input-file %T/compile_by_nvcc/compile_by_nvcc_lin.c.dp.cpp
// RUN: cd ..
// RUN: rm -r ./compile_by_nvcc


// CHECK: #include <CL/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
#include <cuda_runtime.h>

// CHECK: dpct::constant_memory<float, 1> const_angle(360);
// CHECK-NEXT: void simple_kernel(float *d_array, dpct::accessor<float, dpct::constant, 1> const_angle) {
// CHECK-NEXT:   d_array[0] = const_angle[0];
// CHECK-NEXT:   return;
// CHECK-NEXT: }
__constant__ float const_angle[360];
__global__ void simple_kernel(float *d_array) {
  d_array[0] = const_angle[0];
  return;
}