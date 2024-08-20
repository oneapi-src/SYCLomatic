// RUN: dpct --format-range=none --optimize-migration -out-root %T/optimized_math_function %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/optimized_math_function/optimized_math_function.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/optimized_math_function/optimized_math_function.dp.cpp -o %T/optimized_math_function/optimized_math_function.dp.o %}

#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ void sincos_1(float x, float* sptr, float* cptr) {
    // CHECK:  return [&](){ (*sptr = sycl::sin(x), *cptr = sycl::cos(x)); }();
    return sincosf(x, sptr, cptr);
}

__device__ void sincos_2(double x, double* sptr, double* cptr) {
    // CHECK:  return [&](){ (*sptr = sycl::sin(x), *cptr = sycl::cos(x)); }();
    return sincos(x, sptr, cptr);
}

void sincos_3(float a, float *sptr, float *cptr) {
    // CHECK:  return [&](){ *sptr = sycl::sincos(a, sycl::address_space_cast<sycl::access::address_space::generic_space, sycl::access::decorated::yes>(cptr)); }();
    return sincos(a, sptr, cptr);
}
