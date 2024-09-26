// RUN: dpct --out-root %T/uppercase_filename %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck %s --match-full-lines --input-file %T/uppercase_filename/test.dp.cpp
// RUN: FileCheck %S/Header.cuh --match-full-lines --input-file %T/uppercase_filename/Header.dp.hpp
// RUN: %if build_lit %{icpx -c -fsycl %T/uppercase_filename/test.dp.cpp -o %T/uppercase_filename/test.dp.o %}

// CHECK: #include <sycl/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
// CHECK-NEXT: #include "Header.dp.hpp"
#include "Header.cuh"

int main() {
  f2.x = 123.f;
  f2.y = 456.f;
  return 0;
}
