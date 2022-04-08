// UNSUPPORTED: -windows-
// RUN: c2s --format-range=none -out-root %T/queue_ctn %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/queue_ctn/queue_ctn.dp.cpp


#include "cuda.h"

void bar();
#define SIZE 100

size_t size = 1234567 * sizeof(float);
float *h_A = (float *)malloc(size);
float *d_A = NULL;
__constant__ float constData[1234567 * 4];
cudaStream_t s;

// CHECK: void bar1() {
// CHECK-NEXT: c2s::get_default_queue().memcpy(d_A, h_A, sizeof(double)).wait();
// CHECK-NEXT: }
void bar1() {
  cudaMemcpy(d_A, h_A, sizeof(double), cudaMemcpyDeviceToHost);
}
// CHECK: void bar2() {
// CHECK-NEXT: s = c2s::get_current_device().create_queue();
// CHECK-NEXT: }
void bar2() {
  cudaStreamCreate(&s);
}
// CHECK: void bar3() {
// CHECK-NEXT: s = c2s::get_current_device().create_queue();
// CHECK-NEXT: c2s::get_default_queue().memcpy(d_A, h_A, sizeof(double)).wait();
// CHECK-NEXT: }
void bar3() {
  cudaStreamCreate(&s);
  cudaMemcpy(d_A, h_A, sizeof(double), cudaMemcpyDeviceToHost);
}
// CHECK: void bar4() {
// CHECK-NEXT: c2s::device_ext &dev_ct1 = c2s::get_current_device();
// CHECK-NEXT: s = dev_ct1.create_queue();
// CHECK-NEXT: s = dev_ct1.create_queue();
// CHECK-NEXT: c2s::get_default_queue().memcpy(d_A, h_A, sizeof(double)).wait();
// CHECK-NEXT: }
void bar4() {
  cudaStreamCreate(&s);
  cudaStreamCreate(&s);
  cudaMemcpy(d_A, h_A, sizeof(double), cudaMemcpyDeviceToHost);
}
// CHECK: void bar5() {
// CHECK-NEXT: c2s::device_ext &dev_ct1 = c2s::get_current_device();
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
// CHECK-NEXT: c2s::device_ext &dev_ct1 = c2s::get_current_device();
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
  // CHECK: c2s::device_ext &dev_ct1 = c2s::get_current_device();
  // CHECK-NEXT: sycl::queue &q_ct1 = dev_ct1.default_queue();
  // CHECK: q_ct1.memcpy( d_A, h_A, sizeof(double)*SIZE*SIZE );
  // CHECK-NEXT: q_ct1.memcpy( d_A, h_A, sizeof(double)*SIZE*SIZE );
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
  // CHECK: c2s::device_ext &dev_ct1 = c2s::get_current_device();
  // CHECK-NEXT: sycl::queue &q_ct1 = dev_ct1.default_queue();
  // CHECK: q_ct1.memcpy(h_A, (char *)(constData.get_ptr()) + 1, size);
  // CHECK-NEXT: q_ct1.memcpy(h_A, (char *)(constData.get_ptr()) + 1, size);
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
  // CHECK: c2s::device_ext &dev_ct1 = c2s::get_current_device();
  // CHECK-NEXT: sycl::queue &q_ct1 = dev_ct1.default_queue();
  // CHECK: q_ct1.memcpy( d_A, h_A, sizeof(double)*SIZE*SIZE ).wait();
  // CHECK-NEXT: q_ct1.memset(d_A, 23, size).wait();
  cudaMemcpy( d_A, h_A, sizeof(double)*SIZE*SIZE, cudaMemcpyDeviceToHost );
  cudaMemset(d_A, 23, size);
}

void foo4() {
  // CHECK: c2s::device_ext &dev_ct1 = c2s::get_current_device();
  // CHECK-NEXT: sycl::queue &q_ct1 = dev_ct1.default_queue();
  // CHECK: q_ct1.memcpy( d_A, h_A, sizeof(double)*SIZE*SIZE ).wait();
  // CHECK-NEXT: bar();
  // CHECK-NEXT: q_ct1.memset(d_A, 23, size).wait();
  cudaMemcpy( d_A, h_A, sizeof(double)*SIZE*SIZE, cudaMemcpyDeviceToHost );
  bar();
  cudaMemset(d_A, 23, size);
}

void foo5() {
  // CHECK: c2s::device_ext &dev_ct1 = c2s::get_current_device();
  // CHECK-NEXT: sycl::queue &q_ct1 = dev_ct1.default_queue();
  // CHECK: q_ct1.memcpy( d_A, h_A, sizeof(double)*SIZE*SIZE );
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: int Err = (q_ct1.memcpy( d_A, h_A, sizeof(double)*SIZE*SIZE ), 0);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: Err = (q_ct1.memcpy( d_A, h_A, sizeof(double)*SIZE*SIZE ).wait(), 0);
  cudaMemcpy( d_A, h_A, sizeof(double)*SIZE*SIZE, cudaMemcpyDeviceToHost );
  int Err = cudaMemcpy( d_A, h_A, sizeof(double)*SIZE*SIZE, cudaMemcpyDeviceToHost );
  Err = cudaMemcpy( d_A, h_A, sizeof(double)*SIZE*SIZE, cudaMemcpyDeviceToHost );
}

#define CUDA_CALL( call) call

void foo6() {
  // CHECK: c2s::device_ext &dev_ct1 = c2s::get_current_device();
  // CHECK-NEXT: sycl::queue &q_ct1 = dev_ct1.default_queue();
  // CHECK: q_ct1.memcpy( d_A, h_A, sizeof(double)*SIZE*SIZE );
  // CHECK-NEXT: // call in macro
  // CHECK-NEXT: CUDA_CALL(q_ct1.memcpy( d_A, h_A, sizeof(double)*SIZE*SIZE ).wait());
  cudaMemcpy( d_A, h_A, sizeof(double)*SIZE*SIZE, cudaMemcpyDeviceToHost );
  // call in macro
  CUDA_CALL(cudaMemcpy( d_A, h_A, sizeof(double)*SIZE*SIZE, cudaMemcpyDeviceToHost ));
}

template <typename T>
void foo7() {
  // CHECK: c2s::device_ext &dev_ct1 = c2s::get_current_device();
  // CHECK-NEXT: sycl::queue &q_ct1 = dev_ct1.default_queue();
  T* d_A_unresolved, h_A_unresolved;
  // CHECK: q_ct1.memcpy( d_A_unresolved, h_A_unresolved, sizeof(T)*SIZE*SIZE );
  // CHECK-NEXT: q_ct1.memcpy( d_A_unresolved, h_A_unresolved, sizeof(T)*SIZE*SIZE ).wait();
  cudaMemcpy( d_A_unresolved, h_A_unresolved, sizeof(T)*SIZE*SIZE, cudaMemcpyDeviceToHost );
  cudaMemcpy( d_A_unresolved, h_A_unresolved, sizeof(T)*SIZE*SIZE, cudaMemcpyDeviceToHost );
}


template <typename T>
int writeNStage2DDWT() {
    // CHECK: c2s::device_ext &dev_ct1 = c2s::get_current_device();
    // CHECK-NEXT: sycl::queue &q_ct1 = dev_ct1.default_queue();
    T *src;
    // CHECK: src = (T *)sycl::malloc_host(10, q_ct1);
    cudaMallocHost((void **)&src, 10);

    // CHECK: q_ct1.memcpy(src, src, 10);
    // CHECK-NEXT: q_ct1.memcpy(src, src, 10);
    // CHECK-NEXT: q_ct1.memcpy(src, src, 10).wait();
    cudaMemcpy(src, src, 10, cudaMemcpyHostToDevice);
    cudaMemcpy(src, src, 10, cudaMemcpyHostToDevice);
    cudaMemcpy(src, src, 10, cudaMemcpyHostToDevice);

    return 0;
}
template int writeNStage2DDWT<float>();
template int writeNStage2DDWT<int>();

