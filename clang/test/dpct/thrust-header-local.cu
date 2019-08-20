// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct -out-root %T %s  -- -x cuda --cuda-host-only --cuda-path="%cuda-path" -I./thrust
// RUN: FileCheck --input-file %T/thrust-header-local.dp.cpp --match-full-lines %s
// CHECK: #include <CL/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
// CHECK-NEXT: #include <dpstd/algorithm>
// CHECK-NEXT: #include <dpstd/execution>
// CHECK-NEXT: #include <dpct/dpct_dpstd_utils.hpp>
#include <thrust/copy.h>
int main() {
}
