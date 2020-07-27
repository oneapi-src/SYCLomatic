#include "mf-kernel.cuh"
// RUN: dpct --usm-level=none -in-root %S -out-root %T %s %S/mf-test.cu -extra-arg="-I %S" --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -std=c++14 -x cuda --cuda-host-only

  // CHECK: dpct::device_memory<volatile int, 0> g_mutex(0);
volatile __device__ int g_mutex=0;
// CHECK: SYCL_EXTERNAL void Reset_kernel_parameters(volatile int *g_mutex)
__global__ void Reset_kernel_parameters(void)
{
    g_mutex=0;
}

// CHECK: SYCL_EXTERNAL void test_foo(){
__device__ void test_foo(void){
}

// CHECK: static void local_foo_1() {}
__global__ static void local_foo_1() {}

// CHECK: static void local_foo_2();
// CHECK-NEXT: void local_foo_2();
// CHECK-NEXT: void local_foo_2() { }
__global__ static void local_foo_2();
__global__ void local_foo_2();
__global__ void local_foo_2() { }



// CHECK: dpct::constant_memory<float, 0> A1_ct;
// CHECK-NEXT: dpct::constant_memory<float, 0> A2;
__constant__ float A1, A2;

// CHECK: dpct::constant_memory<float, 0> A4_ct;
// CHECK-NEXT: dpct::constant_memory<float, 0> A5_ct;
__constant__ float A4, A5;

// CHECK: dpct::constant_memory<float, 1> A_ct(sycl::range<1>(3 * 3), {0.0625f, 0.125f,  0.0625f, 0.1250f, 0.250f,
// CHECK-NEXT:                                0.1250f, 0.0625f, 0.125f,  0.0625f});
// CHECK-NEXT: dpct::constant_memory<float, 0> A3;
__constant__ float A[3 * 3] = {0.0625f, 0.125f,  0.0625f, 0.1250f, 0.250f,
                               0.1250f, 0.0625f, 0.125f,  0.0625f}, A3;

// CHECK: void constAdd(float *C, sycl::nd_item<3> item_ct1, float *A) {
// CHECK-NEXT:  int i = item_ct1.get_group(2);
// CHECK-NEXT:  int j = item_ct1.get_local_id(2);
// CHECK-NEXT:  int k = 3 * i + j;
// CHECK-NEXT:  if (i < 3 && j < 3) {
// CHECK-NEXT:    C[k] = A[k] + 1.0;
// CHECK-NEXT:  }
// CHECK-NEXT:}
__global__ void constAdd(float *C) {
  int i = blockIdx.x;
  int j = threadIdx.x;
  int k = 3 * i + j;
  if (i < 3 && j < 3) {
    C[k] = A[k] + 1.0;
  }
}

// CHECK: void call_constAdd(float *h_C, int size) {
// CHECK-NEXT:  float *d_C = NULL;
// CHECK-NEXT:  {
// CHECK-NEXT:    dpct::buffer_t d_C_buf_ct0 = dpct::get_buffer(d_C);
// CHECK-NEXT:    dpct::get_default_queue().submit(
// CHECK-NEXT:      [&](sycl::handler &cgh) {
// CHECK-NEXT:        auto A_ct_acc_ct1 = A_ct.get_access(cgh);
// CHECK-NEXT:        auto d_C_acc_ct0 = d_C_buf_ct0.get_access<sycl::access::mode::read_write>(cgh);
// CHECK-EMPTY:
// CHECK-NEXT:        cgh.parallel_for<dpct_kernel_name<class constAdd_{{[a-f0-9]+}}>>(
// CHECK-NEXT:          sycl::nd_range<3>(sycl::range<3>(1, 1, 3) * sycl::range<3>(1, 1, 3), sycl::range<3>(1, 1, 3)), 
// CHECK-NEXT:          [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:            constAdd((float *)(&d_C_acc_ct0[0]), item_ct1, A_ct_acc_ct1.get_pointer());
// CHECK-NEXT:          });
// CHECK-NEXT:      });
// CHECK-NEXT:  }
// CHECK-NEXT:}
void call_constAdd(float *h_C, int size) {
  float *d_C = NULL;
  constAdd<<<3, 3>>>(d_C);
}
