// UNSUPPORTED: -linux-
// RUN: dpct -out-root=%T -in-root=%S %S/test_PATH_in_Windows.cu  -- -x cuda --cuda-host-only --cuda-path="%cuda-path"
// RUN: FileCheck --input-file %T/test_path_in_windows.dp.cpp --match-full-lines %S/test_path_in_windows.cu


// CHECK: void testKernelPtr(const int *L, const int *M, int N, cl::sycl::nd_item<3> item_ct1) {
__global__ void testKernelPtr(const int *L, const int *M, int N) {
  // CHECK: int gtid = item_ct1.get_group(0) * item_ct1.get_local_range().get(0) + item_ct1.get_local_id(0);
  int gtid = blockIdx.x * blockDim.x + threadIdx.x;
}

int main() {
  dim3 griddim = 2;
  dim3 threaddim = 32;
  void *karg1 = 0;
  const int *karg2 = 0;
  int karg3 = 80;
  // CHECK:    {
  // CHECK-NEXT:      std::pair<dpct::buffer_t, size_t> arg_ct0_buf = dpct::get_buffer_and_offset((const int *)karg2);
  // CHECK-NEXT:      size_t arg_ct0_offset = arg_ct0_buf.second;
  // CHECK-NEXT:      std::pair<dpct::buffer_t, size_t> arg_ct1_buf = dpct::get_buffer_and_offset(karg2);
  // CHECK-NEXT:      size_t arg_ct1_offset = arg_ct1_buf.second;
  // CHECK-NEXT:      dpct::get_default_queue().submit(
  // CHECK-NEXT:        [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:          auto arg_ct0_acc = arg_ct0_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:          auto arg_ct1_acc = arg_ct1_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:          cgh.parallel_for<dpct_kernel_name<class testKernelPtr_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:            cl::sycl::nd_range<3>((griddim * threaddim), threaddim),
  // CHECK-NEXT:            [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:              const int *arg_ct0 = (const int *)(&arg_ct0_acc[0] + arg_ct0_offset);
  // CHECK-NEXT:              const int *arg_ct1 = (const int *)(&arg_ct1_acc[0] + arg_ct1_offset);
  // CHECK-NEXT:              testKernelPtr(arg_ct0, arg_ct1, karg3, item_ct1);
  // CHECK-NEXT:            });
  // CHECK-NEXT:        });
  // CHECK-NEXT:    }
  testKernelPtr<<<griddim, threaddim>>>((const int *)karg2, karg2, karg3);

}
