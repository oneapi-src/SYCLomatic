// RUN: cd %S
// RUN: dpct --format-range=none -out-root=%T/db_not_correct case_in_db.c --format-range=none --cuda-include-path="%cuda-path/include" > %T/db_not_correct_output.txt 2>&1 || true
// RUN: grep "Compile command for this file not found in compile_commands.json." %T/db_not_correct_output.txt

#include <cuda_runtime.h>

#ifdef TEST
__constant__ float const_angle[360];
__global__ void simple_kernel(float *d_array) {
  d_array[0] = const_angle[0];
  return;
}
#else
__constant__ float const_angle[230];
#endif
