// RUN: dpct -out-root %T %s -- -x cuda --cuda-host-only --cuda-path="%cuda-path"
// RUN: FileCheck --input-file %T/kernel-call.dp.cpp --match-full-lines %s

// CHECK: void testKernel(int L, int M, int N, cl::sycl::nd_item<3> [[ITEMNAME:item_ct1]]);
__global__ void testKernel(int L, int M, int N);

// CHECK: void testKernelPtr(const int *L, const int *M, int N, cl::sycl::nd_item<3> [[ITEMNAME:item_ct1]]) {
__global__ void testKernelPtr(const int *L, const int *M, int N) {
  // CHECK: int gtid = [[ITEMNAME]].get_group(0) * [[ITEMNAME]].get_local_range().get(0) + [[ITEMNAME]].get_local_id(0);
  int gtid = blockIdx.x * blockDim.x + threadIdx.x;
}

// CHECK: void testKernel(int L, int M, int N, cl::sycl::nd_item<3> [[ITEMNAME:item_ct1]]) {
__global__ void testKernel(int L, int M, int N) {
  // CHECK: int gtid = [[ITEMNAME]].get_group(0) * [[ITEMNAME]].get_local_range().get(0) + [[ITEMNAME]].get_local_id(0);
  int gtid = blockIdx.x * blockDim.x + threadIdx.x;
}

// CHECK: void helloFromGPU(int i, cl::sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:     int a = item_ct1.get_group(0) * item_ct1.get_local_range().get(0) + item_ct1.get_local_id(0) + item_ct1.get_group(0) +
// CHECK-NEXT:     item_ct1.get_local_range().get(0) + item_ct1.get_local_id(0);
// CHECK-NEXT: }
__global__ void helloFromGPU(int i) {
  int a = blockIdx.x * blockDim.x + threadIdx.x + blockIdx.x +
          blockDim.x + threadIdx.x;
}

// CHECK: void helloFromGPU(cl::sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:     int a = item_ct1.get_group(0) * item_ct1.get_local_range().get(0) + item_ct1.get_local_id(0) + item_ct1.get_group(0) +
// CHECK-NEXT:     item_ct1.get_local_range().get(0) + item_ct1.get_local_id(0);
// CHECK-NEXT: }
__global__ void helloFromGPU(void) {
  int a = blockIdx.x * blockDim.x + threadIdx.x + blockIdx.x +
          blockDim.x + threadIdx.x;
}

// CHECK: void helloFromGPU2(cl::sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:     int a = item_ct1.get_group(0) * item_ct1.get_local_range().get(0) + item_ct1.get_local_id(0) + item_ct1.get_group(0) +
// CHECK-NEXT:     item_ct1.get_local_range().get(0) + item_ct1.get_local_id(0);
// CHECK-NEXT: }
__global__ void helloFromGPU2() {
  int a = blockIdx.x * blockDim.x + threadIdx.x + blockIdx.x +
          blockDim.x + threadIdx.x;
}

void testReference(const int &i) {
  dim3 griddim = 2;
  dim3 threaddim = 32;
  // CHECK: {
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class helloFromGPU_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>((griddim * threaddim), threaddim),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           helloFromGPU(i, item_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK-NEXT: }
  helloFromGPU<<<griddim, threaddim>>>(i);

}

struct TestThis {
  struct TestMember {
    int arg1, arg2;
  } args;
  int arg3;
  dim3 griddim, threaddim;
  void test() {
    /// Kernel function is called in method declaration, and fields are used as arguments.
    /// Check the miggration of implicit "this" pointer.
    // CHECK: {
    // CHECK-NEXT:   dpct::get_default_queue().submit(
    // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
    // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class testKernel_{{[a-f0-9]+}}>>(
    // CHECK-NEXT:         cl::sycl::nd_range<3>((griddim * threaddim), threaddim),
    // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
    // CHECK-NEXT:           testKernel(args.arg1, args.arg2, arg3, item_ct1);
    // CHECK-NEXT:         });
    // CHECK-NEXT:     });
    // CHECK-NEXT: }
    testKernel<<<griddim, threaddim>>>(args.arg1, args.arg2, arg3);
  }
};

int main() {
  dim3 griddim = 2;
  dim3 threaddim = 32;
  void *karg1 = 0;
  const int *karg2 = 0;
  int karg3 = 80;
  // CHECK: {
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> arg_ct0_buf = dpct::get_buffer_and_offset((const int *)karg1);
  // CHECK-NEXT:   size_t arg_ct0_offset = arg_ct0_buf.second;
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> arg_ct1_buf = dpct::get_buffer_and_offset(karg2);
  // CHECK-NEXT:   size_t arg_ct1_offset = arg_ct1_buf.second;
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto arg_ct0_acc = arg_ct0_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto arg_ct1_acc = arg_ct1_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class testKernelPtr_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>((griddim * threaddim), threaddim),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           const int *arg_ct0 = (const int *)(&arg_ct0_acc[0] + arg_ct0_offset);
  // CHECK-NEXT:           const int *arg_ct1 = (const int *)(&arg_ct1_acc[0] + arg_ct1_offset);
  // CHECK-NEXT:           testKernelPtr(arg_ct0, arg_ct1, karg3, item_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK-NEXT: }
  testKernelPtr<<<griddim, threaddim>>>((const int *)karg1, karg2, karg3);

  int karg1int = 1;
  int karg2int = 2;
  int karg3int = 3;
  int intvar = 20;
  // CHECK: {
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class testKernel_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>((cl::sycl::range<3>(10, 1, 1) * cl::sycl::range<3>(intvar, 1, 1)), cl::sycl::range<3>(intvar, 1, 1)),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           testKernel(karg1int, karg2int, karg3int, item_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK-NEXT: }
  testKernel<<<10, intvar>>>(karg1int, karg2int, karg3int);

  struct KernelPointer {
    const int *arg1, *arg2;
  } args;
  // CHECK: {
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> arg_ct0_buf = dpct::get_buffer_and_offset(args.arg1);
  // CHECK-NEXT:   size_t arg_ct0_offset = arg_ct0_buf.second;
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> arg_ct1_buf = dpct::get_buffer_and_offset(args.arg2);
  // CHECK-NEXT:   size_t arg_ct1_offset = arg_ct1_buf.second;
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto arg_ct0_acc = arg_ct0_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto arg_ct1_acc = arg_ct1_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class testKernelPtr_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 2, 1)), cl::sycl::range<3>(1, 2, 1)),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           const int *arg_ct0 = (const int *)(&arg_ct0_acc[0] + arg_ct0_offset);
  // CHECK-NEXT:           const int *arg_ct1 = (const int *)(&arg_ct1_acc[0] + arg_ct1_offset);
  // CHECK-NEXT:           testKernelPtr(arg_ct0, arg_ct1, karg3int, item_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK-NEXT: }
  testKernelPtr<<<dim3(1), dim3(1, 2)>>>(args.arg1, args.arg2, karg3int);

  // CHECK: {
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class testKernel_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 2, 1) * cl::sycl::range<3>(1, 2, 3)), cl::sycl::range<3>(1, 2, 3)),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           testKernel(karg1int, karg2int, karg3int, item_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK-NEXT: }
  testKernel<<<dim3(1, 2), dim3(1, 2, 3)>>>(karg1int, karg2int, karg3int);

  // CHECK: {
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class testKernel_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>((cl::sycl::range<3>(griddim[0], 1, 1) * cl::sycl::range<3>(griddim[1] + 2, 1, 1)), cl::sycl::range<3>(griddim[1] + 2, 1, 1)),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           testKernel(karg1int, karg2int, karg3int, item_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK-NEXT: }
  testKernel <<<griddim.x, griddim.y + 2 >>>(karg1int, karg2int, karg3int);

  // CHECK: {
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class helloFromGPU_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>((cl::sycl::range<3>(2, 1, 1) * cl::sycl::range<3>(4, 1, 1)), cl::sycl::range<3>(4, 1, 1)),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           helloFromGPU(23, item_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK-NEXT: }
  helloFromGPU <<<2, 4>>>(23);

  // CHECK: {
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class helloFromGPU_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>((cl::sycl::range<3>(2, 1, 1) * cl::sycl::range<3>(4, 1, 1)), cl::sycl::range<3>(4, 1, 1)),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           helloFromGPU(item_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK-NEXT: }
  helloFromGPU <<<2, 4>>>();

  // CHECK: {
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class helloFromGPU2_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>((cl::sycl::range<3>(2, 1, 1) * cl::sycl::range<3>(3, 1, 1)), cl::sycl::range<3>(3, 1, 1)),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           helloFromGPU2(item_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK-NEXT: }
  helloFromGPU2 <<<2, 3>>>();
}
