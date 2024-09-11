// RUN: dpct --out-root %T/uncanonical_path %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck --input-file %T/uncanonical_path/test.cpp.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/uncanonical_path/test.cpp.dp.cpp -o %T/uncanonical_path/test.o %}

// CHECK: #include <sycl/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
// CHECK-NEXT: #include "../uncanonical_path//test.h"
#include "../uncanonical_path//test.h"

int main() {
  return 0;
}
