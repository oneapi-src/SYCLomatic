// RUN: dpct --out-root %T/pure_c_file %s --cuda-include-path="%cuda-path/include" --extra-arg="-xc" || true
// RUN: FileCheck %s --match-full-lines --input-file %T/pure_c_file/pure_c_file.c
// RUN: %if build_lit %{gcc -c %T/pure_c_file/pure_c_file.c -o %T/pure_c_file/pure_c_file.o %}

// CHECK: // AAAAA
// CHECK-NEXT: #include "pure_c_file.h"
// CHECK-NEXT: // BBBBB
// AAAAA
#include "pure_c_file.h"
// BBBBB

int main() {
  double d = 0;
  sin(d);
  cos(d);
  return 0;
}
