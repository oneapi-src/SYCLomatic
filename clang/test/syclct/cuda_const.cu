// RUN: syclct -out-root %T %s -- -x cuda --cuda-host-only --cuda-path=%cuda-path
// RUN: FileCheck %s --match-full-lines --input-file %T/cuda_const.sycl.cpp

#include <stdio.h>

// CHECK: syclct::ConstMem  const_angle(360* sizeof(float));
__constant__ float const_angle[360];

// CHECK:void simple_kernel(cl::sycl::nd_item<3> item_{{[a-f0-9]+}},
// CHECK-NEXT: cl::sycl::accessor<float, 1, cl::sycl::access::mode::read, cl::sycl::access::target::constant_buffer>  const_acc_{{[a-f0-9]+}}_{{[a-f0-9]+}},
// CHECK-NEXT: float *d_array) {
// CHECK-NEXT:  int index;
// CHECK-NEXT:  index = item_{{[a-f0-9]+}}.get_group(0) * item_{{[a-f0-9]+}}.get_local_range().get(0) + item_{{[a-f0-9]+}}.get_local_id(0);
// CHECK-NEXT:  if (index < 360) {
// CHECK-NEXT:    d_array[index] = const_acc_{{[a-f0-9]+}}_{{[a-f0-9]+}}[index];
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

// CHECK: syclct::ConstMem  const_one(1* sizeof(float));
// CHECK-NEXT: // const_one;
__constant__ float const_one;

// CHECK:void simple_kernel_one(cl::sycl::nd_item<3> item_{{[a-f0-9]+}},
// CHECK-NEXT: cl::sycl::accessor<float, 1, cl::sycl::access::mode::read, cl::sycl::access::target::constant_buffer>  const_acc_{{[a-f0-9]+}}_{{[a-f0-9]+}},
// CHECK-NEXT: cl::sycl::accessor<float, 1, cl::sycl::access::mode::read, cl::sycl::access::target::constant_buffer>  const_acc_{{[a-f0-9]+}}_{{[a-f0-9]+}},
// CHECK-NEXT: float *d_array) {
// CHECK-NEXT:  int index;
// CHECK-NEXT:  index = item_{{[a-f0-9]+}}.get_group(0) * item_{{[a-f0-9]+}}.get_local_range().get(0) + item_{{[a-f0-9]+}}.get_local_id(0);
// CHECK-NEXT:  if (index < 360) {
// CHECK-NEXT:    d_array[index] = const_acc_{{[a-f0-9]+}}_{{[a-f0-9]+}}[0] + const_acc_{{[a-f0-9]+}}_{{[a-f0-9]+}}[index] + const_acc_{{[a-f0-9]+}}_{{[a-f0-9]+}}[0] + const_acc_{{[a-f0-9]+}}_{{[a-f0-9]+}}[0];
// CHECK-NEXT:  }
// CHECK-NEXT:  return;
// CHECK-NEXT:}
__global__ void simple_kernel_one(float *d_array) {
  int index;
  index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < 360) {
    d_array[index] = const_one + const_angle[index] + const_one + const_one;
  }
  return;
}

int main(int argc, char **argv) {
  int size = 3200;
  float *d_array;
  float h_array[360];

  // CHECK: syclct::sycl_malloc((void **)&d_array, sizeof(float) * size);
  cudaMalloc((void **)&d_array, sizeof(float) * size);

  // CHECK: syclct::sycl_memset((void*)(d_array), (int)(0), (size_t)(sizeof(float) * size));
  cudaMemset(d_array, 0, sizeof(float) * size);

  for (int loop = 0; loop < 360; loop++)
    h_array[loop] = acos(-1.0f) * loop / 180.0f;

  // CHECK:/*
  // CHECK-NEXT:SYCLCT1003: Translated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT:*/
  // CHECK-NEXT:   (syclct::sycl_memcpy_to_symbol(const_angle.get_ptr(), (void*)(&h_array[0]), sizeof(float) * 360), 0);
  cudaMemcpyToSymbol(&const_angle[0], &h_array[0], sizeof(float) * 360);

  // CHECK:    {
  // CHECK-NEXT:    std::pair<syclct::buffer_t, size_t> d_array_buf = syclct::get_buffer_and_offset(d_array);
  // CHECK-NEXT:    size_t d_array_offset = d_array_buf.second;
  // CHECK-NEXT:    syclct::get_default_queue().submit(
  // CHECK-NEXT:      [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:        auto d_array_acc = d_array_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:        auto buffer_and_offset_const_acc_{{[a-f0-9]+}}_{{[a-f0-9]+}} = syclct::get_buffer_and_offset(const_angle.get_ptr());
  // CHECK-NEXT:				auto buffer_const_acc_{{[a-f0-9]+}}_{{[a-f0-9]+}} = buffer_and_offset_const_acc_{{[a-f0-9]+}}_{{[a-f0-9]+}}.first.reinterpret<float>(cl::sycl::range<1>(360));
  // CHECK-NEXT:				auto const_acc_{{[a-f0-9]+}}_{{[a-f0-9]+}}= buffer_const_acc_{{[a-f0-9]+}}_{{[a-f0-9]+}}.get_access<cl::sycl::access::mode::read,  cl::sycl::access::target::constant_buffer>(cgh);
  // CHECK-NEXT:        cgh.parallel_for<SyclKernelName<class simple_kernel_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:          cl::sycl::nd_range<3>((cl::sycl::range<3>(size / 64, 1, 1) * cl::sycl::range<3>(64, 1, 1)), cl::sycl::range<3>(64, 1, 1)),
  // CHECK-NEXT:          [=](cl::sycl::nd_item<3> it) {
  // CHECK-NEXT:            float *d_array = (float*)(&d_array_acc[0] + d_array_offset);
  // CHECK-NEXT:            simple_kernel(it, const_acc_{{[a-f0-9]+}}_{{[a-f0-9]+}}, d_array);
  // CHECK-NEXT:          });
  // CHECK-NEXT:      });
  // CHECK-NEXT:  };
  simple_kernel<<<size / 64, 64>>>(d_array);

  float hangle_h[360];
  // CHECK:  syclct::sycl_memcpy((void*)(hangle_h), (void*)(d_array), 360 * sizeof(float), syclct::device_to_host);
  cudaMemcpy(hangle_h, d_array, 360 * sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i < 360; i++) {
    if (fabs(h_array[i] - hangle_h[i]) > 1e-5) {
      exit(-1);
    }
  }

  h_array[0] = 10.0f; // Just to test
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003: Translated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT:  (syclct::sycl_memcpy_to_symbol(const_one.get_ptr(), (void*)(&h_array[0]), sizeof(float) * 1), 0);
  cudaMemcpyToSymbol(&const_one, &h_array[0], sizeof(float) * 1);

  // CHECK:  {
  // CHECK-NEXT:    std::pair<syclct::buffer_t, size_t> d_array_buf = syclct::get_buffer_and_offset(d_array);
  // CHECK-NEXT:    size_t d_array_offset = d_array_buf.second;
  // CHECK-NEXT:    syclct::get_default_queue().submit(
  // CHECK-NEXT:      [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:        auto d_array_acc = d_array_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:        auto buffer_and_offset_const_acc_{{[a-f0-9]+}}_{{[a-f0-9]+}} = syclct::get_buffer_and_offset(const_angle.get_ptr());
  // CHECK-NEXT:				auto buffer_const_acc_{{[a-f0-9]+}}_{{[a-f0-9]+}} = buffer_and_offset_const_acc_{{[a-f0-9]+}}_{{[a-f0-9]+}}.first.reinterpret<float>(cl::sycl::range<1>(360));
  // CHECK-NEXT:				auto const_acc_{{[a-f0-9]+}}_{{[a-f0-9]+}}= buffer_const_acc_{{[a-f0-9]+}}_{{[a-f0-9]+}}.get_access<cl::sycl::access::mode::read,  cl::sycl::access::target::constant_buffer>(cgh);
  // CHECK-NEXT:        auto buffer_and_offset_const_acc_{{[a-f0-9]+}}_{{[a-f0-9]+}} = syclct::get_buffer_and_offset(const_one.get_ptr());
  // CHECK-NEXT:				auto buffer_const_acc_{{[a-f0-9]+}}_{{[a-f0-9]+}} = buffer_and_offset_const_acc_{{[a-f0-9]+}}_{{[a-f0-9]+}}.first.reinterpret<float>(cl::sycl::range<1>(1));
  // CHECK-NEXT:				auto const_acc_{{[a-f0-9]+}}_{{[a-f0-9]+}}= buffer_const_acc_{{[a-f0-9]+}}_{{[a-f0-9]+}}.get_access<cl::sycl::access::mode::read,  cl::sycl::access::target::constant_buffer>(cgh);
  // CHECK-NEXT:        cgh.parallel_for<SyclKernelName<class simple_kernel_one_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:          cl::sycl::nd_range<3>((cl::sycl::range<3>(size / 64, 1, 1) * cl::sycl::range<3>(64, 1, 1)), cl::sycl::range<3>(64, 1, 1)),
  // CHECK-NEXT:          [=](cl::sycl::nd_item<3> it) {
  // CHECK-NEXT:            float *d_array = (float*)(&d_array_acc[0] + d_array_offset);
  // CHECK-NEXT:            simple_kernel_one(it, const_acc_{{[a-f0-9]+}}_{{[a-f0-9]+}}, const_acc_{{[a-f0-9]+}}_{{[a-f0-9]+}}, d_array);
  // CHECK-NEXT:          });
  // CHECK-NEXT:      });
  // CHECK-NEXT:  };
  simple_kernel_one<<<size / 64, 64>>>(d_array);

  hangle_h[360];
  // CHECK:  syclct::sycl_memcpy((void*)(hangle_h), (void*)(d_array), 360 * sizeof(float), syclct::device_to_host);
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
