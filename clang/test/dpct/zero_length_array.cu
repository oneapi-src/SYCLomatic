// RUN: dpct --format-range=none -out-root %T/zero_length_array %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck %s --match-full-lines --input-file %T/zero_length_array/zero_length_array.dp.cpp

#include <stdio.h>

// CHECK: void k() {
// CHECK-NEXT:   /*
// CHECK-NEXT:   DPCT1102:{{[0-9]+}}: Zero-length arrays are not permitted in SYCL device code.
// CHECK-NEXT:   */
// CHECK-NEXT:   int arr0[0];
// CHECK-NEXT:   int arr1[1];
// CHECK-NEXT: }
__global__ void k() {
  int arr0[0];
  int arr1[1];
}

// CHECK: int main() {
// CHECK-NEXT:   int arr0[0];
// CHECK-NEXT:   int arr1[1];
// CHECK-NEXT:   dpct::get_default_queue().parallel_for(
int main() {
  int arr0[0];
  int arr1[1];
  k<<<1,1>>>();
  return 0;
}
