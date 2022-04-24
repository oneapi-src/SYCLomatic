// RUN: dpct --format-range=none -out-root %T/OUT %s %S/test-dpct-header.cu --cuda-include-path="%cuda-path/include" --sycl-named-lambda -extra-arg="-I%S/inc" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --match-full-lines --input-file %T/OUT/test-dpct-header-dup.dp.cpp %s

// CHECK: #include "inc/header3.c.dp.cpp"
// CHECK-NEXT: #include "inc/header4.c"
#include "inc/header3.c"
#include "inc/header4.c"

// CHECK: void foo(int *) {
// CHECK-NEXT: }
__global__ void foo(int *) {
}
