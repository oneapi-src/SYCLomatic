// RUN: dpct --format-range=none -out-root %T/zero_length_array %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck %s --match-full-lines --input-file %T/zero_length_array/zero_length_array.dp.cpp

#include <stdio.h>

// CHECK: void k(int *arr2) {
// CHECK-NEXT:   /*
// CHECK-NEXT:   DPCT1102:{{[0-9]+}}: Zero-length arrays are not permitted in SYCL device code.
// CHECK-NEXT:   */
// CHECK-NEXT:   int arr0[0];
// CHECK-NEXT:   int arr1[1];
// CHECK-NEXT:   // The zero-sized shared array will be migrated to zero-sized local_accessor which is allowed in SYCL
// CHECK-NEXT: }
__global__ void k() {
  int arr0[0];
  int arr1[1];
  __shared__ int arr2[0];// The zero-sized shared array will be migrated to zero-sized local_accessor which is allowed in SYCL
}

// CHECK: int main() {
// CHECK-NEXT:   int arr0[0];
// CHECK-NEXT:   int arr1[1];
// CHECK-NEXT:   dpct::get_default_queue().submit(
int main() {
  int arr0[0];
  int arr1[1];
  k<<<1,1>>>();
  return 0;
}
