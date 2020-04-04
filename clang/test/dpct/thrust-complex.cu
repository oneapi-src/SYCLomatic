// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none -out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/thrust-complex.dp.cpp --match-full-lines %s
// CHECK: #include <CL/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
// CHECK-NEXT: #include <dpct/dpstd_utils.hpp>
// CHECK-NEXT: #include <dpstd/execution>
// CHECK-NEXT: #include <dpstd/algorithm>
// CHECK-NEXT: #include <complex>
#include <thrust/complex.h>

template<typename T>
// CHECK: std::complex<T> foo(std::complex<T> cp) {
thrust::complex<T> foo(thrust::complex<T> cp) {
// CHECK:   std::complex<T> c = std::complex<T>(0.0);
  thrust::complex<T> c = thrust::complex<T>(0.0);
  return c;
}

int main() {
// CHECK:   std::complex<float> cf = foo(std::complex<float>(1.0));
  thrust::complex<float> cf = foo(thrust::complex<float>(1.0));
// CHECK:   std::complex<double> cd = foo(std::complex<double>(1.0));
  thrust::complex<double> cd = foo(thrust::complex<double>(1.0));
// CHECK:   std::complex<float> log = sycl::log(cf);
  thrust::complex<float> log = thrust::log(cf);
// CHECK:   std::complex<double> exp = sycl::exp(cd);
  thrust::complex<double> exp = thrust::exp(cd);
  return 0;
}
