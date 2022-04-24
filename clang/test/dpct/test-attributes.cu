// RUN: dpct --format-range=none -out-root %T/test-attributes %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/test-attributes/test-attributes.dp.cpp --match-full-lines %s
#include "cuda.h"

// CHECK:  void hd() {
// CHECK-NEXT:  float f;
// CHECK-NEXT:  sycl::isnan(f);
// CHECK-NEXT:}
__host__ __device__ void hd() {
  float f;
  isnan(f);
}

// CHECK: void h() {
// CHECK-NEXT:   float f;
// CHECK-NEXT:   isnan(f);
// CHECK-NEXT: }
void h() {
  float f;
  isnan(f);
}

// CHECK:  void h1() {
// CHECK-NEXT:   float f;
// CHECK-NEXT:   isnan(f);
// CHECK-NEXT: }
__host__ void h1() {
  float f;
  isnan(f);
}

// CHECK: void d() {
// CHECK-NEXT:  float f;
// CHECK-NEXT:  sycl::isnan(f);
// CHECK-NEXT:}
__device__ void d() {
  float f;
  isnan(f);
}

// CHECK: void g() {
// CHECK-NEXT:  float f;
// CHECK-NEXT:  sycl::isnan(f);
// CHECK-NEXT:}
__global__ void g() {
  float f;
  isnan(f);
}

