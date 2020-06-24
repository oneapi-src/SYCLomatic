// RUN: dpct -out-root %T %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/copy_cmd_line.c.dp.cpp --match-full-lines %s

// CHECK: #include <stdio.h>
// CHECK-EMPTY:
// CHECK-NEXT: void k() {}
// CHECK-EMPTY:
// CHECK-NEXT: int main()
// CHECK-NEXT: {
// CHECK-NEXT:   printf("Hello world\n");
// CHECK-NEXT:   return 0;
// CHECK-NEXT: }
#include <stdio.h>

__global__ void k() {}

int main()
{
  printf("Hello world\n");
  return 0;
}
