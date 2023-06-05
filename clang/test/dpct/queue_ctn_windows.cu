// UNSUPPORTED: system-linux
// RUN: dpct --format-range=none -out-root %T/queue_ctn_windows %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/queue_ctn_windows/queue_ctn_windows.dp.cpp


#include "cuda.h"

void bar();
#define SIZE 100

size_t size = 1234567 * sizeof(float);
float *h_A = (float *)malloc(size);
float *d_A = NULL;
__constant__ float constData[123 * 4];
cudaStream_t s;

// CHECK: void bar1() {
// CHECK-NEXT: dpct::get_default_queue().memcpy(d_A, h_A, sizeof(double)).wait();
// CHECK-NEXT: }
void bar1() {
  cudaMemcpy(d_A, h_A, sizeof(double), cudaMemcpyDeviceToHost);
}
// CHECK: void bar2() {
// CHECK-NEXT: s = dpct::get_current_device().create_queue();
// CHECK-NEXT: }
void bar2() {
  cudaStreamCreate(&s);
}
// CHECK: void bar3() {
// CHECK-NEXT: s = dpct::get_current_device().create_queue();
// CHECK-NEXT: dpct::get_default_queue().memcpy(d_A, h_A, sizeof(double)).wait();
// CHECK-NEXT: }
void bar3() {
  cudaStreamCreate(&s);
  cudaMemcpy(d_A, h_A, sizeof(double), cudaMemcpyDeviceToHost);
}
// CHECK: void bar4() {
// CHECK-NEXT: dpct::device_ext &dev_ct1 = dpct::get_current_device();
// CHECK-NEXT: s = dev_ct1.create_queue();
// CHECK-NEXT: s = dev_ct1.create_queue();
// CHECK-NEXT: dpct::get_default_queue().memcpy(d_A, h_A, sizeof(double)).wait();
// CHECK-NEXT: }
void bar4() {
  cudaStreamCreate(&s);
  cudaStreamCreate(&s);
  cudaMemcpy(d_A, h_A, sizeof(double), cudaMemcpyDeviceToHost);
}
// CHECK: void bar5() {
// CHECK-NEXT: dpct::device_ext &dev_ct1 = dpct::get_current_device();
// CHECK-NEXT: sycl::queue &q_ct1 = dev_ct1.default_queue();
// CHECK-NEXT: s = dev_ct1.create_queue();
// CHECK-NEXT: q_ct1.memcpy(d_A, h_A, sizeof(double));
// CHECK-NEXT: q_ct1.memcpy(d_A, h_A, sizeof(double)).wait();
// CHECK-NEXT: }
void bar5() {
  cudaStreamCreate(&s);
  cudaMemcpy(d_A, h_A, sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(d_A, h_A, sizeof(double), cudaMemcpyDeviceToHost);
}
// CHECK: void bar6() {
// CHECK-NEXT: dpct::device_ext &dev_ct1 = dpct::get_current_device();
// CHECK-NEXT: sycl::queue &q_ct1 = dev_ct1.default_queue();
// CHECK-NEXT: s = dev_ct1.create_queue();
// CHECK-NEXT: s = dev_ct1.create_queue();
// CHECK-NEXT: q_ct1.memcpy(d_A, h_A, sizeof(double));
// CHECK-NEXT: q_ct1.memcpy(d_A, h_A, sizeof(double)).wait();
// CHECK-NEXT: }
void bar6() {
  cudaStreamCreate(&s);
  cudaStreamCreate(&s);
  cudaMemcpy(d_A, h_A, sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(d_A, h_A, sizeof(double), cudaMemcpyDeviceToHost);
}

void foo1() {
  // CHECK: dpct::device_ext &dev_ct1 = dpct::get_current_device();
  // CHECK-NEXT: sycl::queue &q_ct1 = dev_ct1.default_queue();
  // CHECK: q_ct1.memcpy( d_A, h_A, sizeof(double)*SIZE*SIZE );
  // CHECK-NEXT: q_ct1.memcpy( d_A, h_A, sizeof(double)*SIZE*SIZE ).wait();
  // CHECK-NEXT: q_ct1.memcpy((char *)(constData.get_ptr()) + 1, h_A, size).wait();
  // CHECK-NEXT: q_ct1.memset(d_A, 23, size).wait();
  // CHECK-NEXT: q_ct1.memset(d_A, 23, size).wait();
  // CHECK-NEXT: bar();
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
  // CHECK: dpct::device_ext &dev_ct1 = dpct::get_current_device();
  // CHECK-NEXT: sycl::queue &q_ct1 = dev_ct1.default_queue();
  // CHECK: q_ct1.memcpy(h_A, (char *)(constData.get_ptr()) + 1, size).wait();
  // CHECK-NEXT: q_ct1.memcpy(h_A, (char *)(constData.get_ptr()) + 1, size).wait();
  // CHECK-NEXT: q_ct1.memcpy(h_A, (char *)(constData.get_ptr()) + 1, size).wait();
  // CHECK-NEXT: q_ct1.memset(d_A, 23, size).wait();
  // CHECK-NEXT: q_ct1.memset(d_A, 23, size).wait();
  // CHECK-NEXT: q_ct1.memset(d_A, 23, size).wait();
  // CHECK-NEXT: q_ct1.memset(d_A, 23, size).wait();
  cudaMemcpyFromSymbol(h_A, constData, size, 1);
  cudaMemcpyFromSymbol(h_A, constData, size, 1);
  cudaMemcpyFromSymbol(h_A, constData, size, 1);
  cudaMemset(d_A, 23, size);
  cudaMemset(d_A, 23, size);
  cudaMemset(d_A, 23, size);
  cudaMemset(d_A, 23, size);
}

void foo3() {
  // CHECK: dpct::device_ext &dev_ct1 = dpct::get_current_device();
  // CHECK-NEXT: sycl::queue &q_ct1 = dev_ct1.default_queue();
  // CHECK: q_ct1.memcpy( d_A, h_A, sizeof(double)*SIZE*SIZE ).wait();
  // CHECK-NEXT: q_ct1.memset(d_A, 23, size).wait();
  cudaMemcpy( d_A, h_A, sizeof(double)*SIZE*SIZE, cudaMemcpyDeviceToHost );
  cudaMemset(d_A, 23, size);
}

void foo4() {
  // CHECK: dpct::device_ext &dev_ct1 = dpct::get_current_device();
  // CHECK-NEXT: sycl::queue &q_ct1 = dev_ct1.default_queue();
  // CHECK: q_ct1.memcpy( d_A, h_A, sizeof(double)*SIZE*SIZE ).wait();
  // CHECK-NEXT: bar();
  // CHECK-NEXT: q_ct1.memset(d_A, 23, size).wait();
  cudaMemcpy( d_A, h_A, sizeof(double)*SIZE*SIZE, cudaMemcpyDeviceToHost );
  bar();
  cudaMemset(d_A, 23, size);
}

void foo5() {
  // CHECK: dpct::device_ext &dev_ct1 = dpct::get_current_device();
  // CHECK-NEXT: sycl::queue &q_ct1 = dev_ct1.default_queue();
  // CHECK: q_ct1.memcpy( d_A, h_A, sizeof(double)*SIZE*SIZE );
  // CHECK-NEXT: int Err = DPCT_CHECK_ERROR(q_ct1.memcpy( d_A, h_A, sizeof(double)*SIZE*SIZE ));
  // CHECK-NEXT: Err = DPCT_CHECK_ERROR(q_ct1.memcpy( d_A, h_A, sizeof(double)*SIZE*SIZE ).wait());
  cudaMemcpy( d_A, h_A, sizeof(double)*SIZE*SIZE, cudaMemcpyDeviceToHost );
  int Err = cudaMemcpy( d_A, h_A, sizeof(double)*SIZE*SIZE, cudaMemcpyDeviceToHost );
  Err = cudaMemcpy( d_A, h_A, sizeof(double)*SIZE*SIZE, cudaMemcpyDeviceToHost );
}

#define CUDA_CALL( call) call

void foo6() {
  // CHECK: dpct::device_ext &dev_ct1 = dpct::get_current_device();
  // CHECK-NEXT: sycl::queue &q_ct1 = dev_ct1.default_queue();
  // CHECK: q_ct1.memcpy( d_A, h_A, sizeof(double)*SIZE*SIZE );
  // CHECK-NEXT: // call in macro
  // CHECK-NEXT: CUDA_CALL(q_ct1.memcpy( d_A, h_A, sizeof(double)*SIZE*SIZE ).wait());
  cudaMemcpy( d_A, h_A, sizeof(double)*SIZE*SIZE, cudaMemcpyDeviceToHost );
  // call in macro
  CUDA_CALL(cudaMemcpy( d_A, h_A, sizeof(double)*SIZE*SIZE, cudaMemcpyDeviceToHost ));
}

// On Windows, migration is not supported in normal case
//template <typename T>
//void foo7() {
//  T* d_A_unresolved, h_A_unresolved;
//  // Types of d_A_unresolved and h_A_unresolved are unresolved
//  cudaMemcpy( d_A_unresolved, h_A_unresolved, sizeof(T)*SIZE*SIZE, cudaMemcpyDeviceToHost );
//  cudaMemcpy( d_A_unresolved, h_A_unresolved, sizeof(T)*SIZE*SIZE, cudaMemcpyDeviceToHost );
//}

