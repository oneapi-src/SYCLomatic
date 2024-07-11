// RUN: dpct --enable-codepin --out-root %T/debug_test/inc_loc_and_name_conflict %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck %s --match-full-lines --input-file %T/debug_test/inc_loc_and_name_conflict_codepin_sycl/test.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/debug_test/inc_loc_and_name_conflict_codepin_sycl/test.dp.cpp -o %T/debug_test/inc_loc_and_name_conflict_codepin_sycl/test.dp.o %}

// CHECK: #define DEVICE ""
// CHECK-NEXT: #include <sycl/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
// CHECK-EMPTY: 
// CHECK-NEXT: #include <dpct/codepin/codepin.hpp>
// CHECK-NEXT: #include "codepin_autogen_util.hpp"
// CHECK-NEXT: #include <iostream>
// CHECK-EMPTY: 
// CHECK-NEXT: namespace user_ns {
// CHECK-NEXT: #include "test.h"
// CHECK-NEXT: }

#define DEVICE ""
#include <iostream>

namespace user_ns {
#include "test.h"
}

__global__ void k() {}

int main() {
  k<<<1, 1>>>();
  return 0;
}
