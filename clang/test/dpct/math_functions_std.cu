// RUN: dpct --format-range=none -out-root %T/math_functions_std %s --cuda-include-path="%cuda-path/include" --use-dpcpp-extensions=c_cxx_standard_library
// RUN: FileCheck --input-file %T/math_functions_std/math_functions_std.dp.cpp --match-full-lines %s

// CHECK: #include <cstdlib>
#include <cuda_runtime.h>

__device__ void d() {
  float f, f2;
  double d, d2;
  int i, i2;
  long l, l2;
  long long ll, ll2;

  //CHECK: f = std::abs(f2);
  //CHECK-NEXT: d = std::abs(d2);
  //CHECK-NEXT: i = std::abs(i2);
  //CHECK-NEXT: l = std::abs(l2);
  //CHECK-NEXT: ll = std::abs(ll2);
  //CHECK-NEXT: f = std::abs(f2);
  //CHECK-NEXT: d = std::abs(d2);
  //CHECK-NEXT: i = std::abs(i2);
  //CHECK-NEXT: l = std::abs(l2);
  //CHECK-NEXT: ll = std::abs(ll2);
  f = abs(f2);
  d = abs(d2);
  i = abs(i2);
  l = abs(l2);
  ll = abs(ll2);
  f = std::abs(f2);
  d = std::abs(d2);
  i = std::abs(i2);
  l = std::abs(l2);
  ll = std::abs(ll2);
}

void h() {
  float f, f2;
  double d, d2;
  int i, i2;
  long l, l2;
  long long ll, ll2;

  //CHECK: f = abs(f2);
  //CHECK-NEXT: d = abs(d2);
  //CHECK-NEXT: i = abs(i2);
  //CHECK-NEXT: l = abs(l2);
  //CHECK-NEXT: ll = abs(ll2);
  //CHECK-NEXT: f = std::abs(f2);
  //CHECK-NEXT: d = std::abs(d2);
  //CHECK-NEXT: i = std::abs(i2);
  //CHECK-NEXT: l = std::abs(l2);
  //CHECK-NEXT: ll = std::abs(ll2);
  f = abs(f2);
  d = abs(d2);
  i = abs(i2);
  l = abs(l2);
  ll = abs(ll2);
  f = std::abs(f2);
  d = std::abs(d2);
  i = std::abs(i2);
  l = std::abs(l2);
  ll = std::abs(ll2);
}

