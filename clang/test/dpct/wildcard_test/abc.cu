// RUN: dpct --usm-level=none -out-root=%T/abc -in-root=%S %S/*.cu --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/abc/abc.dp.cpp --match-full-lines %S/abc.cu
// RUN: FileCheck --input-file %T/abc/abd.dp.cpp --match-full-lines %S/abd.cu

// CHECK: void testKernelPtr(const int *L, const int *M, int N, cl::sycl::nd_item<3> [[ITEMNAME:item_ct1]]) {
__global__ void testKernelPtr(const int *L, const int *M, int N) {
  // CHECK: int gtid = [[ITEMNAME]].get_group(2) * [[ITEMNAME]].get_local_range().get(2) + [[ITEMNAME]].get_local_id(2);
  int gtid = blockIdx.x * blockDim.x + threadIdx.x;
}

int main() {
  dim3 griddim = 2;
  dim3 threaddim = 32;
  void *karg1 = 0;
  const int *karg2 = 0;
  int karg3 = 80;
  // CHECK: {
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> karg2_buf_ct0 = dpct::get_buffer_and_offset((const int *)karg2);
  // CHECK-NEXT:   size_t karg2_offset_ct0 = karg2_buf_ct0.second;
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> karg2_buf_ct1 = dpct::get_buffer_and_offset(karg2);
  // CHECK-NEXT:   size_t karg2_offset_ct1 = karg2_buf_ct1.second;
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto karg2_acc_ct0 = karg2_buf_ct0.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto karg2_acc_ct1 = karg2_buf_ct1.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto dpct_global_range = griddim * threaddim;
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class testKernelPtr_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), cl::sycl::range<3>(threaddim.get(2), threaddim.get(1), threaddim.get(0))),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           int const *karg2_ct0 = (int const *)(&karg2_acc_ct0[0] + karg2_offset_ct0);
  // CHECK-NEXT:           int const *karg2_ct1 = (int const *)(&karg2_acc_ct1[0] + karg2_offset_ct1);
  // CHECK-NEXT:           testKernelPtr(karg2_ct0, karg2_ct1, karg3, item_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK-NEXT: }
  testKernelPtr<<<griddim, threaddim>>>((const int *)karg2, karg2, karg3);

}
