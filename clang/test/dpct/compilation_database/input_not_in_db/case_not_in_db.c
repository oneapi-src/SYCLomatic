// RUN: cd %T
// RUN: cat %S/compile_commands.json > %T/compile_commands.json
// RUN: cat %S/case_in_db.c > %T/case_in_db.c
// RUN: cat %S/case_not_in_db.c > %T/case_not_in_db.c

// RUN: dpct --format-range=none -in-root=%T  -out-root=%T/out case_not_in_db.c case_in_db.c --format-range=none --cuda-include-path="%cuda-path/include"
// RUN: FileCheck %T/case_not_in_db.c --match-full-lines --input-file %T/out/case_not_in_db.c.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/out/case_not_in_db.c.dp.cpp -o %T/out/case_not_in_db.c.dp.o %}
// RUN: FileCheck %T/case_in_db.c --match-full-lines --input-file %T/out/case_in_db.c.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/out/case_in_db.c.dp.cpp -o %T/out/case_in_db.c.dp.o %}


// CHECK: #include <sycl/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
#include <cuda_runtime.h>

// CHECK: static dpct::constant_memory<float, 1> const_angle(240);
#ifdef TEST
__constant__ float const_angle[360];
__global__ void simple_kernel(float *d_array) {
  d_array[0] = const_angle[0];
  return;
}
#else
__constant__ float const_angle[240];
#endif
