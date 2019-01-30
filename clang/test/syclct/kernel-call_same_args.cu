// RUN: syclct -out-root %T %s -- -x cuda --cuda-host-only --cuda-path=%cuda-path
// RUN: FileCheck --input-file %T/kernel-call_same_args.sycl.cpp --match-full-lines %s

// CHECK: void testKernelPtr(const int *L, const int *M, int N, cl::sycl::nd_item<3> [[ITEMNAME:item_[a-f0-9]+]]) {
__global__ void testKernelPtr(const int *L, const int *M, int N) {
  // CHECK: int gtid = [[ITEMNAME]].get_group(0) * [[ITEMNAME]].get_local_range().get(0) + [[ITEMNAME]].get_local_id(0);
  int gtid = blockIdx.x * blockDim.x + threadIdx.x;
}

int main() {
  dim3 griddim = 2;
  dim3 threaddim = 32;
  void *karg1 = 0;
  const int *karg2 = 0;
  int karg3 = 80;
  // CHECK:  {
  // CHECK-NEXT:    std::pair<syclct::buffer_t, size_t> karg2_buf = syclct::get_buffer_and_offset(karg2);
  // CHECK-NEXT:    size_t karg2_offset = karg2_buf.second;
  // CHECK-NEXT:    syclct::get_default_queue().submit(
  // CHECK-NEXT:      [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:        auto karg2_acc = karg2_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:        cgh.parallel_for<syclct_kernel_name<class testKernelPtr_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:          cl::sycl::nd_range<3>((griddim * threaddim), threaddim),
  // CHECK-NEXT:          [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK-NEXT:            const int *karg2 = (const int*)(&karg2_acc[0] + karg2_offset);
  // CHECK-NEXT:            testKernelPtr((const int *)karg2, karg2, karg3, [[ITEM]]);
  // CHECK-NEXT:          });
  // CHECK-NEXT:      });
  // CHECK-NEXT:  };
  testKernelPtr<<<griddim, threaddim>>>((const int *)karg2, karg2, karg3);

}
