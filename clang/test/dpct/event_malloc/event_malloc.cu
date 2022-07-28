// RUN: dpct -out-root %T %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/event_malloc.dp.cpp --match-full-lines %s
// RUN: FileCheck --input-file %T/event_malloc.h --match-full-lines %S/event_malloc.h


#include "event_malloc.h"

// CHECK:C::~C(void) { free(kernelEvent); }
C::~C(void) { free(kernelEvent); }

void foo_1() {
  int n_streams = 4;

  // CHECK:  dpct::event_ptr *kernelEvent = (dpct::event_ptr *)malloc(n_streams * sizeof(dpct::event_ptr));
  // CHECK-NEXT:  free(kernelEvent);
  cudaEvent_t *kernelEvent = (cudaEvent_t *)malloc(n_streams * sizeof(cudaEvent_t));
  free(kernelEvent);
}

void foo_2() {
  int n_streams = 4;

  // CHECK:  dpct::event_ptr *kernelEvent;
  // CHECK-NEXT:  kernelEvent = (dpct::event_ptr *)malloc(n_streams * sizeof(dpct::event_ptr));
  // CHECK-NEXT:  free(kernelEvent);
  cudaEvent_t *kernelEvent;
  kernelEvent = (cudaEvent_t *)malloc(n_streams * sizeof(cudaEvent_t));
  free(kernelEvent);
}

void foo_3() {
  int n_streams = 4;

  // CHECK:  dpct::event_ptr *kernelEvent;
  // CHECK-NEXT:  int size = n_streams * sizeof(dpct::event_ptr);
  // CHECK-NEXT:  kernelEvent = (dpct::event_ptr *)malloc(size + sizeof(dpct::event_ptr));
  // CHECK-NEXT:  free(kernelEvent);
  cudaEvent_t *kernelEvent;
  int size = n_streams * sizeof(cudaEvent_t);
  kernelEvent = (cudaEvent_t *)malloc(size + sizeof(cudaEvent_t));
  free(kernelEvent);
}


cudaEvent_t *kernelEvent = NULL;

void foo_4() {
  int n_streams = 4;
  // CHECK:  dpct::event_ptr *kernelEvent = (dpct::event_ptr *)malloc(n_streams * sizeof(dpct::event_ptr));
  cudaEvent_t *kernelEvent = (cudaEvent_t *)malloc(n_streams * sizeof(cudaEvent_t));
}

inline void free(){}

void foo_5() {
  // CHECK:   free(kernelEvent);
   free(kernelEvent);
   free();
}
