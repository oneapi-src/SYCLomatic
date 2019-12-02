// RUN: dpct -out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/cuda_const_usm.dp.cpp

#include <stdio.h>

#define NUM_ELEMENTS 16
const unsigned num_elements = 16;
// CHECK: dpct::constant_memory<float, 1> const_angle(360);
// CHECK: dpct::constant_memory<float, 2> const_float(NUM_ELEMENTS, num_elements * 2);
__constant__ float const_angle[360], const_float[NUM_ELEMENTS][num_elements * 2];
// CHECK: dpct::constant_memory<cl::sycl::double2, 0> vec_d;
__constant__ double2 vec_d;

// CHECK:void simple_kernel(float *d_array, cl::sycl::nd_item<3> [[ITEM:item_ct1]], dpct::accessor<float, dpct::constant, 1> const_angle) {
// CHECK-NEXT:  int index;
// CHECK-NEXT:  index = [[ITEM]].get_group(2) * [[ITEM]].get_local_range().get(2) + [[ITEM]].get_local_id(2);
// CHECK-NEXT:  if (index < 360) {
// CHECK-NEXT:    d_array[index] = const_angle[index];
// CHECK-NEXT:  }
// CHECK-NEXT:  return;
// CHECK-NEXT:}
__global__ void simple_kernel(float *d_array) {
  int index;
  index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < 360) {
    d_array[index] = const_angle[index];
  }
  return;
}

// CHECK: dpct::constant_memory<float, 0> const_one;
__constant__ float const_one;

// CHECK:void simple_kernel_one(float *d_array, cl::sycl::nd_item<3> [[ITEM:item_ct1]], dpct::accessor<float, dpct::constant, 2> const_float, dpct::accessor<float, dpct::constant, 0> const_one) {
// CHECK-NEXT:  int index;
// CHECK-NEXT:  index = [[ITEM]].get_group(2) * [[ITEM]].get_local_range().get(2) + [[ITEM]].get_local_id(2);
// CHECK-NEXT:  if (index < 33) {
// CHECK-NEXT:    d_array[index] = (float)const_one + const_float[index][index];
// CHECK-NEXT:  }
// CHECK-NEXT:  return;
// CHECK-NEXT:}
__global__ void simple_kernel_one(float *d_array) {
  int index;
  index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < 33) {
    d_array[index] = const_one + const_float[index][index];
  }
  return;
}

int main(int argc, char **argv) {
  int size = 3200;
  float *d_array;
  float h_array[360];

  // CHECK: d_array = (float *)cl::sycl::malloc_device(sizeof(float) * size, dpct::get_current_device(), dpct::get_default_context());
  cudaMalloc((void **)&d_array, sizeof(float) * size);

  // CHECK: dpct::get_default_queue_wait().memset(d_array, 0, sizeof(float) * size).wait();
  cudaMemset(d_array, 0, sizeof(float) * size);

  for (int loop = 0; loop < 360; loop++)
    h_array[loop] = acos(-1.0f) * loop / 180.0f;

  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:   (dpct::get_default_queue_wait().memcpy(const_angle.get_ptr(), &h_array[0], sizeof(float) * 360).wait(), 0);
  cudaMemcpyToSymbol(&const_angle[0], &h_array[0], sizeof(float) * 360);

  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:   (dpct::get_default_queue_wait().memcpy(const_angle.get_ptr() + sizeof(float) * (3), &h_array[0], sizeof(float) * 357).wait(), 0);
  cudaMemcpyToSymbol(&const_angle[3], &h_array[0], sizeof(float) * 357);

  // CHECK:  /*
  // CHECK-NEXT:  DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:  */
  // CHECK-NEXT:  (dpct::get_default_queue_wait().memcpy(&h_array[0], const_angle.get_ptr() + sizeof(float) * (3), sizeof(float) * 357).wait(), 0);
  cudaMemcpyFromSymbol(&h_array[0], &const_angle[3], sizeof(float) * 357);

  #define NUM 3
  // CHECK:/*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: (dpct::get_default_queue_wait().memcpy(const_angle.get_ptr() + sizeof(float) * (3+NUM), &h_array[0], sizeof(float) * 354).wait(), 0);
  cudaMemcpyToSymbol(&const_angle[3+NUM], &h_array[0], sizeof(float) * 354);

  // CHECK:  /*
  // CHECK-NEXT:  DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:  */
  // CHECK-NEXT:  (dpct::get_default_queue_wait().memcpy(&h_array[0], const_angle.get_ptr() + sizeof(float) * (3+NUM), sizeof(float) * 354).wait(), 0);
  cudaMemcpyFromSymbol(&h_array[0], &const_angle[3+NUM], sizeof(float) * 354);
  // CHECK:   dpct::get_default_queue_wait().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto const_angle_acc_ct1 = const_angle.get_access(cgh);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class simple_kernel_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(1, 1, size / 64) * cl::sycl::range<3>(1, 1, 64), cl::sycl::range<3>(1, 1, 64)),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           simple_kernel(d_array, item_ct1, const_angle_acc_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  simple_kernel<<<size / 64, 64>>>(d_array);

  float hangle_h[360];
  // CHECK:  dpct::get_default_queue_wait().memcpy(hangle_h, d_array, 360 * sizeof(float)).wait();
  cudaMemcpy(hangle_h, d_array, 360 * sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i < 360; i++) {
    if (fabs(h_array[i] - hangle_h[i]) > 1e-5) {
      exit(-1);
    }
  }

  h_array[0] = 10.0f; // Just to test
  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT:  (dpct::get_default_queue_wait().memcpy(const_one.get_ptr(), &h_array[0], sizeof(float) * 1).wait(), 0);
  cudaMemcpyToSymbol(&const_one, &h_array[0], sizeof(float) * 1);

  // CHECK:   dpct::get_default_queue_wait().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto const_float_acc_ct1 = const_float.get_access(cgh);
  // CHECK-NEXT:       auto const_one_acc_ct1 = const_one.get_access(cgh);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class simple_kernel_one_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(1, 1, size / 64) * cl::sycl::range<3>(1, 1, 64), cl::sycl::range<3>(1, 1, 64)),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           simple_kernel_one(d_array, item_ct1, const_float_acc_ct1, const_one_acc_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  simple_kernel_one<<<size / 64, 64>>>(d_array);

  // CHECK:  dpct::get_default_queue_wait().memcpy(hangle_h, d_array, 360 * sizeof(float)).wait();
  cudaMemcpy(hangle_h, d_array, 360 * sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 1; i < 360; i++) {
    if (fabs(h_array[i] + 30.0f - hangle_h[i]) > 1e-5) {
      exit(-1);
    }
  }

  cudaFree(d_array);

  printf("Test Passed!\n");
  return 0;
}
