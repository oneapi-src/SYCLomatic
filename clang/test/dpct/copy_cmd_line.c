// RUN: dpct -out-root %T %s %s.cu --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/copy_cmd_line.c --match-full-lines %s

// CHECK: #include <stdio.h>
// CHECK-EMPTY:
// CHECK-NEXT: int main()
// CHECK-NEXT: {
// CHECK-NEXT:   printf("Hello world\n");
// CHECK-NEXT:   return 0;
// CHECK-NEXT: }
#include <stdio.h>

int main()
{
  printf("Hello world\n");
  return 0;
}
