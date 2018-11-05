// RUN: syclct -out-root %T %s -- -x cuda --cuda-host-only --cuda-path=%cuda-path
// RUN: FileCheck %s --match-full-lines --input-file %T/cuda_const.sycl.cpp

#include <stdio.h>

// CHECK: float const_angle[360];
__constant__ float const_angle[360];

// CHECK: void simple_kernel(cl::sycl::nd_item<3> item_{{[a-f0-9]+}}, cl::sycl::accessor<float, 1, cl::sycl::access::mode::read, cl::sycl::access::target::constant_buffer>  const_acc, float *d_array) {
// CHECK-NEXT:    int index;
// CHECK-NEXT:    index = item_{{[a-f0-9]+}}.get_group(0) * item_{{[a-f0-9]+}}.get_local_range().get(0) + item_{{[a-f0-9]+}}.get_local_id(0);
// CHECK-NEXT:    if (index < 360) {
// CHECK-NEXT:      d_array[index] = const_acc[index];
// CHECK-NEXT:    }
// CHECK-NEXT:    return;
// CHECK-NEXT:  }
__global__ void simple_kernel(float *d_array) {
  int index;
  index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < 360) {
    d_array[index] = const_angle[index];
  }
  return;
}

// CHECK: float const_one;
__constant__ float const_one;

// CHECK: void simple_kernel_one(cl::sycl::nd_item<3> item_{{[a-f0-9]+}}, cl::sycl::accessor<float, 1, cl::sycl::access::mode::read, cl::sycl::access::target::constant_buffer>  const_acc, float *d_array) {
// CHECK-NEXT:    int index;
// CHECK-NEXT:    index = item_{{[a-f0-9]+}}.get_group(0) * item_{{[a-f0-9]+}}.get_local_range().get(0) + item_{{[a-f0-9]+}}.get_local_id(0);
// CHECK-NEXT:    if (index < 360) {
// CHECK-NEXT:      d_array[index] = const_acc[0];
// CHECK-NEXT:    }
// CHECK-NEXT:    return;
// CHECK-NEXT:  }
__global__ void simple_kernel_one(float *d_array) {
  int index;
  index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < 360) {
    d_array[index] = const_one;
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

  // Need to do translation, will be fixed in CTST-50
  cudaMemcpyToSymbol(&const_angle[0], &h_array[0], sizeof(float) * 360);

  // CHECK:   {
  // CHECK-NEXT:   std::pair<syclct::buffer_t, size_t> d_array_buf = syclct::get_buffer_and_offset(d_array);
  // CHECK-NEXT:   size_t d_array_offset = d_array_buf.second;
  // CHECK-NEXT:   syclct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto d_array_acc = d_array_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       cl::sycl::buffer<cl::sycl::cl_float, 1> const_buf(&const_angle[0], cl::sycl::range<1>(360));
  // CHECK-NEXT:       auto  const_acc_{{[a-f0-9]+}} = const_buf.get_access<cl::sycl::access::mode::read, cl::sycl::access::target::constant_buffer>(cgh);
  // CHECK-NEXT:       cgh.parallel_for<SyclKernelName<class simple_kernel_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<1>((cl::sycl::range<1>(size / 64) * cl::sycl::range<1>(64)), cl::sycl::range<1>(64)),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<1> it) {
  // CHECK-NEXT:           float *d_array = (float*)(&d_array_acc[0] + d_array_offset);
  // CHECK-NEXT:           simple_kernel(it, const_acc_{{[a-f0-9]+}}, d_array);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK-NEXT: };
  simple_kernel<<<size / 64, 64>>>(d_array);

  // Need to do translation, will be fixed in CTST-50
  cudaMemcpyToSymbol(&const_one, &h_array[0], sizeof(float) * 1);

  // CHECK:  {
  // CHECK-NEXT:    std::pair<syclct::buffer_t, size_t> d_array_buf = syclct::get_buffer_and_offset(d_array);
  // CHECK-NEXT:    size_t d_array_offset = d_array_buf.second;
  // CHECK-NEXT:    syclct::get_default_queue().submit(
  // CHECK-NEXT:      [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:        auto d_array_acc = d_array_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:        cl::sycl::buffer<cl::sycl::cl_float, 1> const_buf(&const_one, cl::sycl::range<1>(1));
  // CHECK-NEXT:        auto  const_acc_{{[a-f0-9]+}} = const_buf.get_access<cl::sycl::access::mode::read, cl::sycl::access::target::constant_buffer>(cgh);
  // CHECK-NEXT:        cgh.parallel_for<SyclKernelName<class simple_kernel_one_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:          cl::sycl::nd_range<1>((cl::sycl::range<1>(size / 64) * cl::sycl::range<1>(64)), cl::sycl::range<1>(64)),
  // CHECK-NEXT:          [=](cl::sycl::nd_item<1> it) {
  // CHECK-NEXT:            float *d_array = (float*)(&d_array_acc[0] + d_array_offset);
  // CHECK-NEXT:            simple_kernel_one(it, const_acc_{{[a-f0-9]+}}, d_array);
  // CHECK-NEXT:          });
  // CHECK-NEXT:      });
  // CHECK-NEXT:  };
  simple_kernel_one<<<size / 64, 64>>>(d_array);

  cudaFree(d_array);

  return 0;
}
