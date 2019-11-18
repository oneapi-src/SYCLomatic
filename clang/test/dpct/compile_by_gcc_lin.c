// UNSUPPORTED: -windows-
// RUN: mkdir %T/compile_by_gcc
// RUN: cd %T/compile_by_gcc
// RUN: cp %s %T/compile_by_gcc
// RUN: echo "[" > %T/compile_by_gcc/compile_commands.json
// RUN: echo "  {" >> %T/compile_by_gcc/compile_commands.json
// RUN: echo "    \"command\": \"gcc -c -m64 -I/%cuda-path/include -o compile_by_gcc_lin.o compile_by_gcc_lin.c\"," >> %T/compile_by_gcc/compile_commands.json
// RUN: echo "    \"directory\": \"%T/compile_by_gcc\"," >> %T/compile_by_gcc/compile_commands.json
// RUN: echo "    \"file\": \"%T/compile_by_gcc/compile_by_gcc_lin.c\"" >> %T/compile_by_gcc/compile_commands.json
// RUN: echo "  }" >> %T/compile_by_gcc/compile_commands.json
// RUN: echo "]" >> %T/compile_by_gcc/compile_commands.json
// RUN: dpct --out-root %T/compile_by_gcc -p=./ --cuda-include-path="%cuda-path/include"
// RUN: FileCheck %s --match-full-lines --input-file %T/compile_by_gcc/compile_by_gcc_lin.c.dp.cpp
// RUN: cd ..
// RUN: rm -r ./compile_by_gcc

// CHECK: #include <CL/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
#include <cuda_runtime.h>

// CHECK: __constant__ float const_angle[360];
// CHECK-NEXT: void simple_kernel(float *d_array) {
// CHECK-NEXT:   d_array[0] = const_angle[0];
// CHECK-NEXT:   return;
// CHECK-NEXT: }
__constant__ float const_angle[360];
__global__ void simple_kernel(float *d_array) {
  d_array[0] = const_angle[0];
  return;
}