// UNSUPPORTED: -linux-
// RUN: c2s --format-range=none --usm-level=none -out-root=%T/test_path_in_windows -in-root=%S %S/test_PATH_in_Windows.cu --cuda-include-path="%cuda-path/include" --sycl-named-lambda --comments -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/test_path_in_windows/test_path_in_windows.dp.cpp --match-full-lines %S/test_path_in_windows.cu


// CHECK: void testKernelPtr(const int *L, const int *M, int N, sycl::nd_item<3> item_ct1) {
__global__ void testKernelPtr(const int *L, const int *M, int N) {
  // CHECK: int gtid = item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
  int gtid = blockIdx.x * blockDim.x + threadIdx.x;
}

int main() {
  dim3 griddim = 2;
  dim3 threaddim = 32;
  void *karg1 = 0;
  const int *karg2 = 0;
  int karg3 = 80;
  // CHECK: c2s::get_default_queue().submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     // accessors to device memory
  // CHECK-NEXT:     c2s::access_wrapper<const int *> karg2_acc_ct0((const int *)karg2, cgh);
  // CHECK-NEXT:     c2s::access_wrapper<const int *> karg2_acc_ct1(karg2, cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for<c2s_kernel_name<class testKernelPtr_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:       sycl::nd_range<3>(griddim * threaddim, threaddim),
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         testKernelPtr(karg2_acc_ct0.get_raw_pointer(), karg2_acc_ct1.get_raw_pointer(), karg3, item_ct1);
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  testKernelPtr<<<griddim, threaddim>>>((const int *)karg2, karg2, karg3);

}
