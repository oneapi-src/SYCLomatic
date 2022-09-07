// RUN: cd %T
// RUN: cat %S/compile_commands.json > %T/compile_commands.json
// RUN: cat %S/t2.c > %T/t2.c
// RUN: cat %S/withoutxcuda.c > %T/withoutxcuda.c
// RUN: cat %S/ref > %T/ref

// RUN: dpct --format-range=none -in-root=%T -out-root=%T/out -p %T withoutxcuda.c --format-range=none --cuda-include-path="%cuda-path/include" &> %T/notFound || echo "test"
// RUN: FileCheck %T/ref --match-full-lines --input-file %T/notFound

#include <cuda_runtime.h>
__constant__ float const_angle[360];
__global__ void simple_kernel(float *d_array) {
  d_array[0] = const_angle[0];
  return;
}
