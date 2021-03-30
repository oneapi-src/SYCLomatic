// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none -out-root %T/thrust-identity %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -fno-delayed-template-parsing -std=c++17
// RUN: FileCheck --input-file %T/thrust-identity/thrust-identity.dp.cpp --match-full-lines %s

#include <thrust/functional.h>

// CHECK: oneapi::dpl::identity identity;
thrust::identity<int> identity;
 
int main() {
  int i1 = identity(1);
// CHECK:     int i2 = oneapi::dpl::identity()(2);
  int i2 = thrust::identity<int>()(2);
}
