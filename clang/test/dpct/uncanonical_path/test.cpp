// RUN: dpct --out-root %T/uncanonical_path %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck --input-file %T/uncanonical_path/test.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/uncanonical_path/test.cpp -o %T/uncanonical_path/test.o %}

// CHECK: #include "../uncanonical_path//test.h"
#include "../uncanonical_path//test.h"

int main() {
  return 0;
}
