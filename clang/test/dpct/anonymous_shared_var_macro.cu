// RUN: dpct -out-root %T/anonymous_shared_var_macro %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/anonymous_shared_var_macro/anonymous_shared_var_macro.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl -DNO_BUILD_TEST %T/anonymous_shared_var_macro/anonymous_shared_var_macro.dp.cpp -o %T/anonymous_shared_var_macro/anonymous_shared_var_macro.dp.o %}

#ifndef NO_BUILD_TEST
#include<cuda_runtime.h>

// CHECK: #define BSPLINE_DEFS                                                           \
// CHECK:   union type_ct1 {                                                             \
// CHECK:     float a2d[3][3];                                                           \
// CHECK:     float a1d[3];                                                              \
// CHECK:   };                                                                           \
// CHECK:   type_ct1 &bspline_coeffs = *(type_ct1 *)bspline_coeffs_ct1;                  \
// CHECK:   volatile union type_ct2 {                                                    \
// CHECK:     float a2d[3][7];                                                           \
// CHECK:     float a1d[7];                                                              \
// CHECK:   };                                                                           \
// CHECK:   type_ct2 *atoms = (type_ct2 *)atoms_ct1;                                     \
// CHECK:   if (item_ct1.get_local_id(2) < 7) {                                          \
// CHECK:     bspline_coeffs.a1d[item_ct1.get_local_id(2)] = 0;                          \
// CHECK:   };

#define BSPLINE_DEFS \
  __shared__ union { \
    float a2d[3][3]; \
    float a1d[3]; \
  } bspline_coeffs; \
  __shared__ volatile union { \
    float a2d[3][7]; \
    float a1d[7]; \
  } atoms[3]; \
  if ( threadIdx.x < 7 ) { \
    bspline_coeffs.a1d[threadIdx.x] = 0; \
  };

__global__ void kernel() {
  BSPLINE_DEFS
}

int main() {
  kernel<<<1, 1>>>();
  return 0;
}
#endif
