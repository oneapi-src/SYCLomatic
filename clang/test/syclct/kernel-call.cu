// RUN: syclct -out-root %T %s -- -x cuda --cuda-host-only --cuda-path=%cuda-path
// RUN: FileCheck --input-file %T/kernel-call.sycl.cpp --match-full-lines %s

// CHECK: void testKernelPtr(const int *L, const int *M, int N, cl::sycl::nd_item<3> [[ITEMNAME:item_[a-f0-9]+]]) {
__global__ void testKernelPtr(const int *L, const int *M, int N) {
  // CHECK: int gtid = [[ITEMNAME]].get_group(0) * [[ITEMNAME]].get_local_range().get(0) + [[ITEMNAME]].get_local_id(0);
  int gtid = blockIdx.x * blockDim.x + threadIdx.x;
}

// CHECK: void testKernel(int L, int M, int N, cl::sycl::nd_item<3> [[ITEMNAME:item_[a-f0-9]+]]) {
__global__ void testKernel(int L, int M, int N) {
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
  // CHECK-NEXT:    std::pair<syclct::buffer_t, size_t> karg1_buf = syclct::get_buffer_and_offset(karg1);
  // CHECK-NEXT:    size_t karg1_offset = karg1_buf.second;
  // CHECK-NEXT:    std::pair<syclct::buffer_t, size_t> karg2_buf = syclct::get_buffer_and_offset(karg2);
  // CHECK-NEXT:    size_t karg2_offset = karg2_buf.second;
  // CHECK-NEXT:    syclct::get_default_queue().submit(
  // CHECK-NEXT:      [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:        auto karg1_acc = karg1_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:        auto karg2_acc = karg2_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:        cgh.parallel_for<syclct_kernel_name<class testKernelPtr_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:          cl::sycl::nd_range<3>((griddim * threaddim), threaddim),
  // CHECK-NEXT:          [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK-NEXT:            void *karg1 = (void*)(&karg1_acc[0] + karg1_offset);
  // CHECK-NEXT:            const int *karg2 = (const int*)(&karg2_acc[0] + karg2_offset);
  // CHECK-NEXT:            testKernelPtr((const int *)karg1, karg2, karg3, [[ITEM]]);
  // CHECK-NEXT:          });
  // CHECK-NEXT:      });
  // CHECK-NEXT:  };
  testKernelPtr<<<griddim, threaddim>>>((const int *)karg1, karg2, karg3);

  // CHECK:  {
  // CHECK-NEXT:    syclct::get_default_queue().submit(
  // CHECK-NEXT:      [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:        cgh.parallel_for<syclct_kernel_name<class testKernel_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:          cl::sycl::nd_range<3>((cl::sycl::range<3>(10, 1, 1) * cl::sycl::range<3>(intvar, 1, 1)), cl::sycl::range<3>(intvar, 1, 1)),
  // CHECK-NEXT:          [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK-NEXT:            testKernel(karg1int, karg2int, karg3int, [[ITEM]]);
  // CHECK-NEXT:          });
  // CHECK-NEXT:      });
  // CHECK-NEXT:  };
  int karg1int = 1;
  int karg2int = 2;
  int karg3int = 3;
  int intvar = 20;
  testKernel<<<10, intvar>>>(karg1int, karg2int, karg3int);

  // CHECK:  {
  // CHECK-NEXT:    syclct::get_default_queue().submit(
  // CHECK-NEXT:      [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:        cgh.parallel_for<syclct_kernel_name<class testKernel_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:          cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 2, 1)), cl::sycl::range<3>(1, 2, 1)),
  // CHECK-NEXT:          [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK-NEXT:            testKernel(karg1int, karg2int, karg3int, [[ITEM]]);
  // CHECK-NEXT:          });
  // CHECK-NEXT:      });
  // CHECK-NEXT:  };
  testKernel<<<dim3(1), dim3(1, 2)>>>(karg1int, karg2int, karg3int);

  // CHECK:  {
  // CHECK-NEXT:    syclct::get_default_queue().submit(
  // CHECK-NEXT:      [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:        cgh.parallel_for<syclct_kernel_name<class testKernel_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:          cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 2, 1) * cl::sycl::range<3>(1, 2, 3)), cl::sycl::range<3>(1, 2, 3)),
  // CHECK-NEXT:          [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK-NEXT:            testKernel(karg1int, karg2int, karg3int, [[ITEM]]);
  // CHECK-NEXT:          });
  // CHECK-NEXT:      });
  // CHECK-NEXT:  };
  testKernel<<<dim3(1, 2), dim3(1, 2, 3)>>>(karg1int, karg2int, karg3int);

  // CHECK:  {
  // CHECK-NEXT:    syclct::get_default_queue().submit(
  // CHECK-NEXT:      [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:      cgh.parallel_for<syclct_kernel_name<class testKernel_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:        cl::sycl::nd_range<3>((cl::sycl::range<3>(griddim[0], 1, 1) * cl::sycl::range<3>(griddim[1] + 2, 1, 1)), cl::sycl::range<3>(griddim[1] + 2, 1, 1)),
  // CHECK-NEXT:        [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK-NEXT:        testKernel(karg1int, karg2int, karg3int, [[ITEM]]);
  // CHECK-NEXT:      });
  // CHECK-NEXT:    });
  // CHECK-NEXT:  };
  testKernel <<<griddim.x, griddim.y + 2 >>>(karg1int, karg2int, karg3int);
}
