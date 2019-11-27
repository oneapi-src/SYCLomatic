// UNSUPPORTED: -windows-
// RUN: dpct -out-root %T %s --cuda-include-path="%cuda-path/include" -- -std=c++14  -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/queue_ctn.dp.cpp


#include "cuda.h"

void bar();
#define SIZE 100

size_t size = 1234567 * sizeof(float);
float *h_A = (float *)malloc(size);
float *d_A = NULL;
__constant__ float constData[1234567 * 4];

void foo1() {
  // CHECK: cl::sycl::queue& q_ct0 = dpct::get_default_queue();
  // CHECK-NEXT: q_ct0.wait();
  // CHECK-NEXT: q_ct0.memcpy( d_A, h_A, sizeof(double)*SIZE*SIZE ).wait();
  // CHECK-NEXT: q_ct0.memcpy( d_A, h_A, sizeof(double)*SIZE*SIZE ).wait();
  // CHECK-NEXT: q_ct0.memcpy((char *)(constData.get_ptr()) + 1, h_A, size).wait();
  // CHECK-NEXT: q_ct0.memset(d_A, 23, size).wait();
  // CHECK-NEXT: q_ct0.memset(d_A, 23, size).wait();
  // CHECK-NEXT: bar();
  // CHECK-NEXT: cl::sycl::queue& q_ct1 = dpct::get_default_queue();
  // CHECK-NEXT: q_ct1.wait();
  // CHECK-NEXT: q_ct1.memset(d_A, 23, size).wait();
  // CHECK-NEXT: q_ct1.memset(d_A, 23, size).wait();
  cudaMemcpy( d_A, h_A, sizeof(double)*SIZE*SIZE, cudaMemcpyDeviceToHost );
  cudaMemcpy( d_A, h_A, sizeof(double)*SIZE*SIZE, cudaMemcpyDeviceToHost );
  cudaMemcpyToSymbol(constData, h_A, size, 1);
  cudaMemset(d_A, 23, size);
  cudaMemset(d_A, 23, size);
  bar();
  cudaMemset(d_A, 23, size);
  cudaMemset(d_A, 23, size);
}


void foo2() {
  // CHECK: cl::sycl::queue& q_ct2 = dpct::get_default_queue();
  // CHECK-NEXT: q_ct2.wait();
  // CHECK-NEXT: q_ct2.memcpy(h_A, (char *)(constData.get_ptr()) + 1, size).wait();
  // CHECK-NEXT: q_ct2.memcpy(h_A, (char *)(constData.get_ptr()) + 1, size).wait();
  // CHECK-NEXT: q_ct2.memcpy(h_A, (char *)(constData.get_ptr()) + 1, size).wait();
  // CHECK-NEXT: q_ct2.memset(d_A, 23, size).wait();
  // CHECK-NEXT: q_ct2.memset(d_A, 23, size).wait();
  // CHECK-NEXT: q_ct2.memset(d_A, 23, size).wait();
  // CHECK-NEXT: q_ct2.memset(d_A, 23, size).wait();
  cudaMemcpyFromSymbol(h_A, constData, size, 1);
  cudaMemcpyFromSymbol(h_A, constData, size, 1);
  cudaMemcpyFromSymbol(h_A, constData, size, 1);
  cudaMemset(d_A, 23, size);
  cudaMemset(d_A, 23, size);
  cudaMemset(d_A, 23, size);
  cudaMemset(d_A, 23, size);
}

void foo3() {
  // CHECK: cl::sycl::queue& q_ct3 = dpct::get_default_queue();
  // CHECK-NEXT: q_ct3.wait();
  // CHECK-NEXT: q_ct3.memcpy( d_A, h_A, sizeof(double)*SIZE*SIZE ).wait();
  // CHECK-NEXT: q_ct3.memset(d_A, 23, size).wait();
  cudaMemcpy( d_A, h_A, sizeof(double)*SIZE*SIZE, cudaMemcpyDeviceToHost );
  cudaMemset(d_A, 23, size);
}

void foo4() {
  // CHECK: dpct::get_default_queue_wait().memcpy( d_A, h_A, sizeof(double)*SIZE*SIZE ).wait();
  // CHECK-NEXT: bar();
  // CHECK-NEXT: dpct::get_default_queue_wait().memset(d_A, 23, size).wait();
  cudaMemcpy( d_A, h_A, sizeof(double)*SIZE*SIZE, cudaMemcpyDeviceToHost );
  bar();
  cudaMemset(d_A, 23, size);
}

void foo5() {
  // CHECK: cl::sycl::queue& q_ct4 = dpct::get_default_queue();
  // CHECK-NEXT: q_ct4.wait();
  // CHECK-NEXT: q_ct4.memcpy( d_A, h_A, sizeof(double)*SIZE*SIZE ).wait();
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: int Err = (q_ct4.memcpy( d_A, h_A, sizeof(double)*SIZE*SIZE ).wait(), 0);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: Err = (q_ct4.memcpy( d_A, h_A, sizeof(double)*SIZE*SIZE ).wait(), 0);
  cudaMemcpy( d_A, h_A, sizeof(double)*SIZE*SIZE, cudaMemcpyDeviceToHost );
  int Err = cudaMemcpy( d_A, h_A, sizeof(double)*SIZE*SIZE, cudaMemcpyDeviceToHost );
  Err = cudaMemcpy( d_A, h_A, sizeof(double)*SIZE*SIZE, cudaMemcpyDeviceToHost );
}

#define CUDA_CALL( call) call

// unsupported conditions
void foo6() {
  // CHECK: dpct::get_default_queue_wait().memcpy( d_A, h_A, sizeof(double)*SIZE*SIZE ).wait();
  // CHECK-NEXT: // call in macro
  // CHECK-NEXT: CUDA_CALL(dpct::get_default_queue_wait().memcpy( d_A, h_A, sizeof(double)*SIZE*SIZE ).wait());
  cudaMemcpy( d_A, h_A, sizeof(double)*SIZE*SIZE, cudaMemcpyDeviceToHost );
  // call in macro
  CUDA_CALL(cudaMemcpy( d_A, h_A, sizeof(double)*SIZE*SIZE, cudaMemcpyDeviceToHost ));
}

template <typename T>
void foo7() {
  T* d_A_unresolved, h_A_unresolved;
  // Types of d_A_unresolved and h_A_unresolved are unresolved
  // CHECK: dpct::get_default_queue_wait().memcpy( d_A_unresolved, h_A_unresolved, sizeof(T)*SIZE*SIZE ).wait();
  // CHECK-NEXT: dpct::get_default_queue_wait().memcpy( d_A_unresolved, h_A_unresolved, sizeof(T)*SIZE*SIZE ).wait();
  cudaMemcpy( d_A_unresolved, h_A_unresolved, sizeof(T)*SIZE*SIZE, cudaMemcpyDeviceToHost );
  cudaMemcpy( d_A_unresolved, h_A_unresolved, sizeof(T)*SIZE*SIZE, cudaMemcpyDeviceToHost );
}