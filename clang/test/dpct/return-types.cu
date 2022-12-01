// RUN: dpct --format-range=none --usm-level=none -out-root %T/return-types %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/return-types/return-types.dp.cpp --match-full-lines %s

// CHECK: #include <sycl/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
// CHECK-NEXT: #include <stdio.h>
// CHECK-EMPTY:
#include <stdio.h>

// CHECK: #define DEF_BAR dpct::queue_ptr bar() { \
// CHECK-NEXT:   return &dpct::get_default_queue(); \
// CHECK-NEXT: }
#define DEF_BAR cudaStream_t bar() { \
  return 0; \
}
// CHECK: #define DEF_BAR2 dpct::event_ptr bar2() { \
// CHECK-NEXT:   return 0; \
// CHECK-NEXT: }
#define DEF_BAR2 cudaEvent_t bar2() { \
  return 0; \
}

DEF_BAR
DEF_BAR2

// CHECK: template <typename T>
// CHECK-NEXT: dpct::queue_ptr bar() {
// CHECK-NEXT:   return &dpct::get_default_queue();
// CHECK-NEXT: }
template <typename T>
cudaStream_t bar() {
  return 0;
}

// CHECK: template <typename T>
// CHECK-NEXT: dpct::event_ptr bar2() {
// CHECK-NEXT:   return 0;
// CHECK-NEXT: }
template <typename T>
cudaEvent_t bar2() {
  return 0;
}

// CHECK: dpct::queue_ptr foo() {
cudaStream_t foo() {
  // CHECK: return &dpct::get_default_queue();
  return 0;
}

// CHECK: dpct::event_ptr foo2() {
cudaEvent_t foo2() {
  return 0;
}

class S {
  // CHECK: dpct::queue_ptr foo() {
  cudaStream_t foo() {
    // CHECK: return &dpct::get_default_queue();
    return 0;
  }

  // CHECK: dpct::event_ptr foo2() {
  cudaEvent_t foo2() {
    return 0;
  }
};

class C {
  // CHECK: dpct::queue_ptr foo() {
  cudaStream_t foo() {
    // CHECK: return &dpct::get_default_queue();
    return 0;
  }

  // CHECK: dpct::event_ptr foo2() {
  cudaEvent_t foo2() {
    return 0;
  }
};

// CHECK: dpct::queue_ptr *foo(int i) {
cudaStream_t *foo(int i) {
  return 0;
}

// CHECK: const dpct::queue_ptr *foo(unsigned i) {
const cudaStream_t *foo(unsigned i) {
  return 0;
}

// CHECK: dpct::queue_ptr **foo(char i) {
cudaStream_t **foo(char i) {
  return 0;
}

// CHECK: dpct::queue_ptr &foo(short i) {
cudaStream_t &foo(short i) {
  cudaStream_t s;
  return s;
}

// CHECK: const dpct::queue_ptr &foo(long i) {
const cudaStream_t &foo(long i) {
  cudaStream_t s;
  return s;
}

// CHECK: dpct::event_ptr *bar(int i) {
cudaEvent_t *bar(int i) {
  return 0;
}

// CHECK: const dpct::event_ptr *bar(unsigned i) {
const cudaEvent_t *bar(unsigned i) {
  return 0;
}

// CHECK: dpct::event_ptr **bar(char i) {
cudaEvent_t **bar(char i) {
  return 0;
}

// CHECK: dpct::event_ptr &bar(short i) {
cudaEvent_t &bar(short i) {
  cudaEvent_t e;
  return e;
}

// CHECK: const dpct::event_ptr &bar(long i) {
const cudaEvent_t &bar(long i) {
  cudaEvent_t e;
  return e;
}

