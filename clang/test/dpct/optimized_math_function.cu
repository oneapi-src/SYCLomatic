// RUN: dpct --format-range=none --optimize-migration --usm-level=none -out-root %T/optimized_math_function %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -x cuda --cuda-host-only -fno-delayed-template-parsing
// RUN: FileCheck --input-file %T/optimized_math_function/optimized_math_function.dp.cpp --match-full-lines %s
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <math_functions.h>

__device__ void sincos_1(float x, float* sptr, float* cptr) {
    // CHECK:  return [&](){ (*sptr = sycl::sin(x), *cptr = sycl::cos(x)); }();
    return sincosf(x, sptr, cptr);
}

__device__ void sincos_2(double x, double* sptr, double* cptr) {
    // CHECK:  [&](){ (*sptr = sycl::sin(x), *cptr = sycl::cos(x)); }();
    return sincos(x, sptr, cptr);
}

void sincos_3(float a, float *sptr, float *cptr) {
    // CHECK:  return sincos(a, sptr, cptr);
    return sincos(a, sptr, cptr);
}
