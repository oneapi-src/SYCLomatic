// RUN: dpct --usm-level=none -out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --match-full-lines --input-file %T/decay.dp.cpp %s
#include <cuda_runtime.h>

template<typename T>
__global__ void foo_kernel(T** R)
{
}


//CHECK: void foo(float* R[])
//CHECK-NEXT: {
//CHECK-NEXT:   {
//CHECK-NEXT:     std::pair<dpct::buffer_t, size_t> R_buf_ct0 = dpct::get_buffer_and_offset(R);
//CHECK-NEXT:     size_t R_offset_ct0 = R_buf_ct0.second;
//CHECK-NEXT:     dpct::get_default_queue().submit(
//CHECK-NEXT:       [&](cl::sycl::handler &cgh) {
//CHECK-NEXT:         auto R_acc_ct0 = R_buf_ct0.first.get_access<cl::sycl::access::mode::read_write>(cgh);
//CHECK-NEXT:         auto dpct_global_range = cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1);
//CHECK-NEXT:         auto dpct_local_range = cl::sycl::range<3>(1, 1, 1);
//CHECK-NEXT:         cgh.parallel_for<dpct_kernel_name<class foo_kernel_{{[a-f0-9]+}}, float>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), cl::sycl::range<3>(dpct_local_range.get(2), dpct_local_range.get(1), dpct_local_range.get(0))),
//CHECK-NEXT:           [=](cl::sycl::nd_item<3> item_ct1) {
//CHECK-NEXT:             float * *R_ct0 = (float * *)(&R_acc_ct0[0] + R_offset_ct0);
//CHECK-NEXT:             foo_kernel<float>(R_ct0);
//CHECK-NEXT:           });
//CHECK-NEXT:       });
//CHECK-NEXT:   }
void foo(float* R[])
{
  foo_kernel<float><<<1, 1>>>(R);
}
