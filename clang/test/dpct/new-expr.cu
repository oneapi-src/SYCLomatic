// RUN: c2s --format-range=none --usm-level=none -out-root %T/new-expr %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/new-expr/new-expr.dp.cpp --match-full-lines %s
#include <stdio.h>

// CHECK: #define NEW_STREAM new sycl::queue *
// CHECK-NEXT: #define NEW_EVENT new sycl::event
// CHECK-EMPTY:
// CHECK-NEXT: #define NEW(T) new T
#define NEW_STREAM new cudaStream_t
#define NEW_EVENT new cudaEvent_t

#define NEW(T) new T

void foo() {
  int n = 16;

  // CHECK: sycl::queue **stream = new sycl::queue *;
  // CHECK-NEXT: stream = new sycl::queue *();
  // CHECK-NEXT: stream = NEW_STREAM;
  // CHECK-NEXT: stream = NEW(sycl::queue *);
  // CHECK-NEXT: sycl::queue **streams = new sycl::queue *[n];
  cudaStream_t *stream = new cudaStream_t;
  stream = new cudaStream_t();
  stream = NEW_STREAM;
  stream = NEW(cudaStream_t);
  cudaStream_t *streams = new cudaStream_t[n];

  // CHECK: sycl::event *event = new sycl::event;
  // CHECK-NEXT: event = new sycl::event();
  // CHECK-NEXT: event = NEW_EVENT;
  // CHECK-NEXT: event = NEW(sycl::event);
  // CHECK-NEXT: sycl::event *events = new sycl::event[n];
  cudaEvent_t *event = new cudaEvent_t;
  event = new cudaEvent_t();
  event = NEW_EVENT;
  event = NEW(cudaEvent_t);
  cudaEvent_t *events = new cudaEvent_t[n];
}

