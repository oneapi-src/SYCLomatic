// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: c2s --format-range=none -out-root %T/thrust-header-local %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -I./thrust
// RUN: FileCheck --input-file %T/thrust-header-local/thrust-header-local.dp.cpp --match-full-lines %s
// CHECK: #include <oneapi/dpl/execution>
// CHECK-NEXT: #include <oneapi/dpl/algorithm>
// CHECK-NEXT: #include <CL/sycl.hpp>
// CHECK-NEXT: #include <c2s/c2s.hpp>
// CHECK-NEXT: #include <c2s/dpl_utils.hpp>
#include <thrust/copy.h>
int main() {
}

