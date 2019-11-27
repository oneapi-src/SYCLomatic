// RUN: dpct --usm-level=none -out-root %T %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/new-expr.dp.cpp --match-full-lines %s
#include <stdio.h>

// CHECK: #define NEW_STREAM new queue_p
// CHECK-NEXT: #define NEW_EVENT new cl::sycl::event
// CHECK-EMPTY:
// CHECK-NEXT: #define NEW(T) new T
#define NEW_STREAM new cudaStream_t
#define NEW_EVENT new cudaEvent_t

#define NEW(T) new T

void foo() {
  int n = 16;

  // CHECK: queue_p *stream = new queue_p;
  // CHECK-NEXT: stream = new queue_p();
  // CHECK-NEXT: stream = NEW_STREAM;
  // CHECK-NEXT: stream = NEW(queue_p);
  // CHECK-NEXT: queue_p *streams = new queue_p[n];
  cudaStream_t *stream = new cudaStream_t;
  stream = new cudaStream_t();
  stream = NEW_STREAM;
  stream = NEW(cudaStream_t);
  cudaStream_t *streams = new cudaStream_t[n];

  // CHECK: cl::sycl::event *event = new cl::sycl::event;
  // CHECK-NEXT: event = new cl::sycl::event();
  // CHECK-NEXT: event = NEW_EVENT;
  // CHECK-NEXT: event = NEW(cl::sycl::event);
  // CHECK-NEXT: cl::sycl::event *events = new cl::sycl::event[n];
  cudaEvent_t *event = new cudaEvent_t;
  event = new cudaEvent_t();
  event = NEW_EVENT;
  event = NEW(cudaEvent_t);
  cudaEvent_t *events = new cudaEvent_t[n];
}
