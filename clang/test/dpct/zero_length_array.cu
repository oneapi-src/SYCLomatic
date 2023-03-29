// RUN: dpct --format-range=none -out-root %T/zero_length_array %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck %s --match-full-lines --input-file %T/zero_length_array/zero_length_array.dp.cpp

#include <stdio.h>

struct S {
// CHECK: int abc;
// CHECK-NEXT: /*
// CHECK-NEXT: DPCT1102:{{[0-9]+}}: Zero-length arrays are not permitted in SYCL device code.
// CHECK-NEXT: */
// CHECK-NEXT: int arr[0];
  int abc;
  int arr[0];
};

__global__ void k(S* s) {
  int *arr0 = s->arr;
}

int main() {
  S* s;
  cudaMalloc(&s, sizeof(S) + sizeof(int) * 4);
  k<<<1,1>>>(s);
  return 0;
}
