// RUN: dpct --format-range=none -out-root %T %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/devicemem_usm.dp.cpp

#include <cuda_runtime.h>

#include <cassert>

#define NUM_ELEMENTS (/* Threads per block */ 16)

class TestStruct {
public:
  __device__ void test() {}
};

// CHECK: dpct::device_memory<TestStruct, 0> t1;
__device__ TestStruct t1;

// CHECK: void member_acc(TestStruct *t1) {
// CHECK-NEXT:  t1->test();
// CHECK-NEXT:}
__global__ void member_acc() {
  t1.test();
}

// CHECK: dpct::device_memory<float, 1> in(NUM_ELEMENTS);
__device__ float in[NUM_ELEMENTS];
// CHECK: dpct::device_memory<int, 1> init(cl::sycl::range<1>(4), {1, 2, 3, 4});
__device__ int init[4] = {1, 2, 3, 4};

// CHECK: void kernel1(float *out, cl::sycl::nd_item<3> [[ITEM:item_ct1]], float *in) {
// CHECK:   out[{{.*}}[[ITEM]].get_local_id(2)] = in[{{.*}}[[ITEM]].get_local_id(2)];
// CHECK: }
__global__ void kernel1(float *out) {
  out[threadIdx.x] = in[threadIdx.x];
}

// CHECK: dpct::device_memory<int, 0> al;
__device__ int al;
// CHECK: dpct::device_memory<int, 0> ainit(NUM_ELEMENTS);
__device__ int ainit = NUM_ELEMENTS;

const int num_elements = 16;
// CHECK: dpct::device_memory<float, 1> fx(2);
// CHECK: dpct::device_memory<float, 2> fy(num_elements, 4 * num_elements);
__device__ float fx[2], fy[num_elements][4 * num_elements];

// CHECK: void kernel2(float *out, cl::sycl::nd_item<3> [[ITEM:item_ct1]], int *al, float *fx, dpct::accessor<float, dpct::device, 2> fy, float *tmp) {
// CHECK:   out[{{.*}}[[ITEM]].get_local_id(2)] += *al;
// CHECK:   fx[{{.*}}[[ITEM]].get_local_id(2)] = fy[{{.*}}[[ITEM]].get_local_id(2)][{{.*}}[[ITEM]].get_local_id(2)];
// CHECK: }
__global__ void kernel2(float *out) {
  const int size = 64;
  __device__ float tmp[size];
  out[threadIdx.x] += al;
  fx[threadIdx.x] = fy[threadIdx.x][threadIdx.x];
}

int main() {
  float h_in[NUM_ELEMENTS] = {0};
  float h_out[NUM_ELEMENTS] = {0};

  for (int i = 0; i < NUM_ELEMENTS; ++i) {
    h_in[i] = i;
    h_out[i] = -i;
  }

  const size_t array_size = sizeof(float) * NUM_ELEMENTS;
  // CTST-50
  cudaMemcpyToSymbol(in, h_in, array_size);

  const int h_a = 3;
  // CTST-50
  cudaMemcpyToSymbol(al, &h_a, sizeof(int));

  float *d_out = NULL;
  cudaMalloc((void **)&d_out, array_size);

  const int threads_per_block = NUM_ELEMENTS;
  // CHECK:   dpct::get_default_queue_wait().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto t1_ptr_ct1 = t1.get_ptr();
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class member_acc_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, threads_per_block), cl::sycl::range<3>(1, 1, threads_per_block)),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           member_acc(t1_ptr_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  member_acc<<<1, threads_per_block>>>();
  // CHECK:   dpct::get_default_queue_wait().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto in_ptr_ct1 = in.get_ptr();
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class kernel1_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, threads_per_block), cl::sycl::range<3>(1, 1, threads_per_block)),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           kernel1(d_out, item_ct1, in_ptr_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  kernel1<<<1, threads_per_block>>>(d_out);

  // CHECK:   dpct::get_default_queue_wait().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       dpct::device_memory<float, 1> tmp(64/*size*/);
  // CHECK-NEXT:       auto tmp_ptr_ct1 = tmp.get_ptr();
  // CHECK-NEXT:       auto al_ptr_ct1 = al.get_ptr();
  // CHECK-NEXT:       auto fx_ptr_ct1 = fx.get_ptr();
  // CHECK-NEXT:       auto fy_acc_ct1 = fy.get_access(cgh);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class kernel2_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, threads_per_block), cl::sycl::range<3>(1, 1, threads_per_block)),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           kernel2(d_out, item_ct1, al_ptr_ct1, fx_ptr_ct1, fy_acc_ct1, tmp_ptr_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  kernel2<<<1, threads_per_block>>>(d_out);

  cudaMemcpy(h_out, d_out, array_size, cudaMemcpyDeviceToHost);

  for (int i = 0; i < NUM_ELEMENTS; ++i) {
    assert(h_out[i] == i + h_a && "Value mis-calculated!");
  }

  return 0;
}
