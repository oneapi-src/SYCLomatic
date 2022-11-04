// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none -out-root %T/thrust_math %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/thrust_math/thrust_math.dp.cpp --match-full-lines %s

// CHECK: #include <complex>
#include <thrust/complex.h>

int main() {
  // CHECK: std::log10(std::complex<float>(0.0));
  thrust::log10(thrust::complex<float>(0.0));
  // CHECK: std::sqrt(std::complex<float>(0.0));
  thrust::sqrt(thrust::complex<float>(0.0));
  // CHECK: std::pow(std::complex<float>(1.0), std::complex<float>(1.0));
  thrust::pow(thrust::complex<float>(1.0), thrust::complex<float>(1.0));
  // CHECK: std::pow(std::complex<float>(1.0), 1.0);
  thrust::pow(thrust::complex<float>(1.0), 1.0);
  // CHECK: std::pow(1.0, std::complex<float>(1.0));
  thrust::pow(1.0, thrust::complex<float>(1.0));
  // CHECK: std::sin(std::complex<float>(0.0));
  thrust::sin(thrust::complex<float>(0.0));
  // CHECK: std::cos(std::complex<float>(0.0));
  thrust::cos(thrust::complex<float>(0.0));
  // CHECK: std::tan(std::complex<float>(0.0));
  thrust::tan(thrust::complex<float>(0.0));
  // CHECK: std::asin(std::complex<float>(0.0));
  thrust::asin(thrust::complex<float>(0.0));
  // CHECK: std::acos(std::complex<float>(0.0));
  thrust::acos(thrust::complex<float>(0.0));
  // CHECK: std::atan(std::complex<float>(0.0));
  thrust::atan(thrust::complex<float>(0.0));
  // CHECK: std::sinh(std::complex<float>(0.0));
  thrust::sinh(thrust::complex<float>(0.0));
  // CHECK: std::cosh(std::complex<float>(0.0));
  thrust::cosh(thrust::complex<float>(0.0));
  // CHECK: std::tanh(std::complex<float>(0.0));
  thrust::tanh(thrust::complex<float>(0.0));
  // CHECK: std::asinh(std::complex<float>(0.0));
  thrust::asinh(thrust::complex<float>(0.0));
  // CHECK: std::acosh(std::complex<float>(0.0));
  thrust::acosh(thrust::complex<float>(0.0));
  // CHECK: std::atanh(std::complex<float>(0.0));
  thrust::atanh(thrust::complex<float>(0.0));
  // CHECK: std::abs(std::complex<float>(0.0));
  thrust::abs(thrust::complex<float>(0.0));
  // CHECK: std::polar(1.0, 1.0);
  thrust::polar(1.0, 1.0);
  // CHECK: std::exp(std::complex<float>(0.0));
  thrust::exp(thrust::complex<float>(0.0));
  // CHECK: std::log(std::complex<float>(0.0)); 
  thrust::log(thrust::complex<float>(0.0));

  return 0;
}