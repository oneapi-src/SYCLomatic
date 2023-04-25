// UNSUPPORTED: system-windows
// RUN: dpct --format-range=none -out-root %T/queue_ctn %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/queue_ctn/queue_ctn.dp.cpp


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

__global__ void kernel(float *a, float *b, float *c){
  int i = threadIdx.x;
  c[i] = a[i] + b[i];
}

void bar7(){
// CHECK: float *A, *B, *C;
// CHECK: A = sycl::malloc_device<float>(100, q_ct1);
// CHECK: B = sycl::malloc_device<float>(100, q_ct1);
// CHECK: C = sycl::malloc_device<float>(100, q_ct1);
// CHECK: q_ct1.memcpy(A, h_A, 100 * sizeof(float));
// CHECK: q_ct1.memcpy(B, h_A, 100 * sizeof(float));
  float *A, *B, *C;
  cudaMalloc(&A, 100 * sizeof(float));
  cudaMalloc(&B, 100 * sizeof(float));
  cudaMalloc(&C, 100 * sizeof(float));
  cudaMemcpy(A, h_A, 100 * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(B, h_A, 100 * sizeof(float), cudaMemcpyDeviceToHost);
  kernel<<<1, 100>>>(A, B, C);
// CHECK: q_ct1.memcpy(h_A, C, 100 * sizeof(float)).wait();
  cudaMemcpy(h_A, C, 100 * sizeof(float), cudaMemcpyHostToDevice);
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
  // CHECK: dpct::device_ext &dev_ct1 = dpct::get_current_device();
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
  // CHECK: dpct::device_ext &dev_ct1 = dpct::get_current_device();
  // CHECK-NEXT: sycl::queue &q_ct1 = dev_ct1.default_queue();
  T* d_A_unresolved, h_A_unresolved;
  // CHECK: q_ct1.memcpy( d_A_unresolved, h_A_unresolved, sizeof(T)*SIZE*SIZE );
  // CHECK-NEXT: q_ct1.memcpy( d_A_unresolved, h_A_unresolved, sizeof(T)*SIZE*SIZE ).wait();
  cudaMemcpy( d_A_unresolved, h_A_unresolved, sizeof(T)*SIZE*SIZE, cudaMemcpyDeviceToHost );
  cudaMemcpy( d_A_unresolved, h_A_unresolved, sizeof(T)*SIZE*SIZE, cudaMemcpyDeviceToHost );
}


template <typename T>
int writeNStage2DDWT() {
    // CHECK: dpct::device_ext &dev_ct1 = dpct::get_current_device();
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

