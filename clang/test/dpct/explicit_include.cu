// RUN: dpct -out-root %T/implicit_include %s --cuda-include-path="%cuda-path/include" || true
// RUN: FileCheck --input-file %T/implicit_include/explicit_include.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/implicit_include/explicit_include.dp.cpp -o %T/implicit_include/explicit_include.dp.o %}

// CHECK: #include <cmath>
// CHECK-NOT: #include <cmath>
#include <cmath>

void test() {
  int a = -1;
  // CHECK: abs(a);
  abs(a);
}
