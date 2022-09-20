// RUN: dpct --format-range=none --usm-level=none -out-root %T/new-expr %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/new-expr/new-expr.dp.cpp --match-full-lines %s
#include <stdio.h>

// CHECK: #define NEW_STREAM new dpct::queue_ptr
// CHECK-NEXT: #define NEW_EVENT new dpct::event_ptr
// CHECK-EMPTY:
// CHECK-NEXT: #define NEW(T) new T
#define NEW_STREAM new cudaStream_t
#define NEW_EVENT new cudaEvent_t

#define NEW(T) new T

void foo() {
  int n = 16;

  // CHECK: dpct::queue_ptr *stream = new dpct::queue_ptr;
  // CHECK-NEXT: stream = new dpct::queue_ptr();
  // CHECK-NEXT: stream = NEW_STREAM;
  // CHECK-NEXT: stream = NEW(dpct::queue_ptr);
  // CHECK-NEXT: dpct::queue_ptr *streams = new dpct::queue_ptr[n];
  cudaStream_t *stream = new cudaStream_t;
  stream = new cudaStream_t();
  stream = NEW_STREAM;
  stream = NEW(cudaStream_t);
  cudaStream_t *streams = new cudaStream_t[n];

  // CHECK: dpct::event_ptr *event = new dpct::event_ptr;
  // CHECK-NEXT: event = new dpct::event_ptr();
  // CHECK-NEXT: event = NEW_EVENT;
  // CHECK-NEXT: event = NEW(dpct::event_ptr);
  // CHECK-NEXT: dpct::event_ptr *events = new dpct::event_ptr[n];
  cudaEvent_t *event = new cudaEvent_t;
  event = new cudaEvent_t();
  event = NEW_EVENT;
  event = NEW(cudaEvent_t);
  cudaEvent_t *events = new cudaEvent_t[n];
}

