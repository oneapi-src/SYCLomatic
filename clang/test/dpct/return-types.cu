// RUN: c2s --format-range=none --usm-level=none -out-root %T/return-types %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/return-types/return-types.dp.cpp --match-full-lines %s

// CHECK: #include <CL/sycl.hpp>
// CHECK-NEXT: #include <c2s/c2s.hpp>
// CHECK-NEXT: #include <stdio.h>
// CHECK-EMPTY:
#include <stdio.h>

// CHECK: #define DEF_BAR sycl::queue * bar() { \
// CHECK-NEXT:   return 0; \
// CHECK-NEXT: }
#define DEF_BAR cudaStream_t bar() { \
  return 0; \
}
// CHECK: #define DEF_BAR2 sycl::event bar2() { \
// CHECK-NEXT:   return 0; \
// CHECK-NEXT: }
#define DEF_BAR2 cudaEvent_t bar2() { \
  return 0; \
}

DEF_BAR
DEF_BAR2

// CHECK: template <typename T>
// CHECK-NEXT: sycl::queue * bar() {
// CHECK-NEXT:   return 0;
// CHECK-NEXT: }
template <typename T>
cudaStream_t bar() {
  return 0;
}

// CHECK: template <typename T>
// CHECK-NEXT: sycl::event bar2() {
// CHECK-NEXT:   return 0;
// CHECK-NEXT: }
template <typename T>
cudaEvent_t bar2() {
  return 0;
}

// CHECK: sycl::queue * foo() {
cudaStream_t foo() {
  return 0;
}

// CHECK: sycl::event foo2() {
cudaEvent_t foo2() {
  return 0;
}

class S {
  // CHECK: sycl::queue * foo() {
  cudaStream_t foo() {
    return 0;
  }

  // CHECK: sycl::event foo2() {
  cudaEvent_t foo2() {
    return 0;
  }
};

class C {
  // CHECK: sycl::queue * foo() {
  cudaStream_t foo() {
    return 0;
  }

  // CHECK: sycl::event foo2() {
  cudaEvent_t foo2() {
    return 0;
  }
};

// CHECK: sycl::queue * *foo(int i) {
cudaStream_t *foo(int i) {
  return 0;
}

// CHECK: const sycl::queue * *foo(unsigned i) {
const cudaStream_t *foo(unsigned i) {
  return 0;
}

// CHECK: sycl::queue * **foo(char i) {
cudaStream_t **foo(char i) {
  return 0;
}

// CHECK: sycl::queue * &foo(short i) {
cudaStream_t &foo(short i) {
  cudaStream_t s;
  return s;
}

// CHECK: const sycl::queue * &foo(long i) {
const cudaStream_t &foo(long i) {
  cudaStream_t s;
  return s;
}

// CHECK: sycl::event *bar(int i) {
cudaEvent_t *bar(int i) {
  return 0;
}

// CHECK: const sycl::event *bar(unsigned i) {
const cudaEvent_t *bar(unsigned i) {
  return 0;
}

// CHECK: sycl::event **bar(char i) {
cudaEvent_t **bar(char i) {
  return 0;
}

// CHECK: sycl::event &bar(short i) {
cudaEvent_t &bar(short i) {
  cudaEvent_t e;
  return e;
}

// CHECK: const sycl::event &bar(long i) {
const cudaEvent_t &bar(long i) {
  cudaEvent_t e;
  return e;
}

