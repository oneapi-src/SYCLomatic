// RUN: cd %S
// RUN: dpct --format-range=none -out-root=%T/input_not_in_db case_not_in_db.c case_in_db.c --format-range=none --cuda-include-path="%cuda-path/include" > %T/input_not_in_db_output.txt 2>&1 || true
// RUN: grep "Compile command for this file not found in compile_commands.json." %T/input_not_in_db_output.txt

#include <cuda_runtime.h>

#ifdef TEST
__constant__ float const_angle[360];
__global__ void simple_kernel(float *d_array) {
  d_array[0] = const_angle[0];
  return;
}
#else
__constant__ float const_angle[240];
#endif
