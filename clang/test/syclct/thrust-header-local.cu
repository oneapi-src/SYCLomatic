// UNSUPPORTED: cuda-8.0
// RUN: syclct -out-root %T %s  -- -x cuda --cuda-host-only --cuda-path=%cuda-path -I./thrust
// RUN: FileCheck --input-file %T/thrust-header-local.sycl.cpp --match-full-lines %s
// CHECK: #include <CL/sycl.hpp>
// CHECK-NEXT: #include <syclct/syclct.hpp>
// CHECK-NEXT: #include <syclct/syclct_thrust.hpp>
#include <thrust/copy.h>
int main() {
}
