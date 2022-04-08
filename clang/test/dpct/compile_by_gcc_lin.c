// UNSUPPORTED: -windows-
// RUN: mkdir %T/compile_by_gcc
// RUN: cd %T/compile_by_gcc
// RUN: cp %s %T/compile_by_gcc
// RUN: echo "[" > %T/compile_by_gcc/compile_commands.json
// RUN: echo "  {" >> %T/compile_by_gcc/compile_commands.json
// RUN: echo "    \"command\": \"gcc -c -m64 -o compile_by_gcc_lin.o compile_by_gcc_lin.c\"," >> %T/compile_by_gcc/compile_commands.json
// RUN: echo "    \"directory\": \"%T/compile_by_gcc\"," >> %T/compile_by_gcc/compile_commands.json
// RUN: echo "    \"file\": \"%T/compile_by_gcc/compile_by_gcc_lin.c\"" >> %T/compile_by_gcc/compile_commands.json
// RUN: echo "  }" >> %T/compile_by_gcc/compile_commands.json
// RUN: echo "]" >> %T/compile_by_gcc/compile_commands.json
// RUN: c2s --format-range=none --out-root %T/compile_by_gcc -p=./ --cuda-include-path="%cuda-path/include"
// RUN: FileCheck %s --match-full-lines --input-file %T/compile_by_gcc/compile_by_gcc_lin.c.dp.cpp
// RUN: rm -rf %T/compile_by_gcc

// CHECK: #include <CL/sycl.hpp>
// CHECK-NEXT: #include <c2s/c2s.hpp>
#include <cuda_runtime.h>

// CHECK: int a = 0;
// CHECK-NEXT: /*
// CHECK-NEXT: DPCT1056:{{[0-9]+}}: The use of variable const_angle in device code was not detected. If this variable is also used in device code, you need to rewrite the code.
// CHECK-NEXT: */
// CHECK-NEXT: float const_angle[360];
// CHECK-NEXT: void simple_kernel(float *d_array) {
// CHECK-NEXT:   d_array[0] = const_angle[0];
// CHECK-NEXT:   return;
// CHECK-NEXT: }
int a = 0;
__constant__ float const_angle[360];
__global__ void simple_kernel(float *d_array) {
  d_array[0] = const_angle[0];
  return;
}
