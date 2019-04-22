// RUN: syclct -in-root %S -out-root %T %s %S/multi-files-device.cuh -- -x cuda --cuda-host-only --cuda-path=%cuda-path
// RUN: FileCheck %s --match-full-lines --input-file %T/multi-files-main.sycl.cpp
// RUN: FileCheck %S/multi-files-device.cuh --match-full-lines --input-file %T/multi-files-device.sycl.hpp
// RUN: FileCheck %S/multi-files-kernel.cuh --match-full-lines --input-file %T/multi-files-kernel.sycl.hpp

#include "multi-files-kernel.cuh"

int main() {
  int *i_array;  
  cudaMalloc((void **)&i_array, sizeof(int) * 360);

  // CHECK:  {
  // CHECK-NEXT:    std::pair<syclct::buffer_t, size_t> i_array_buf = syclct::get_buffer_and_offset(i_array);
  // CHECK-NEXT:    size_t i_array_offset = i_array_buf.second;
  // CHECK-NEXT:    syclct::get_default_queue().submit(
  // CHECK-NEXT:      [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:        auto i_array_acc = i_array_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:        cgh.parallel_for<syclct_kernel_name<class simple_kernel_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:          cl::sycl::nd_range<3>((cl::sycl::range<3>(16, 1, 1) * cl::sycl::range<3>(16, 1, 1)), cl::sycl::range<3>(16, 1, 1)),
  // CHECK-NEXT:          [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK-NEXT:            int *i_array = (int*)(&i_array_acc[0] + i_array_offset);
  // CHECK-NEXT:            simple_kernel(i_array, [[ITEM]]);
  // CHECK-NEXT:          });
  // CHECK-NEXT:      });
  // CHECK-NEXT:  }
  simple_kernel<<<16, 16>>>(i_array);

  cudaFree(i_array);
}
