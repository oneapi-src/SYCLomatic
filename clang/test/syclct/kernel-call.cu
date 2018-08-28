// RUN: syclct -out-root %T %s -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/kernel-call.sycl.cpp --match-full-lines %s
__global__ void testKernel(int L, int M, int N) {
  int gtid = blockIdx.x * blockDim.x + threadIdx.x;
}

int main() {
  dim3 griddim = 2;
  dim3 threaddim = 32;
  int karg1 = 60;
  int karg2 = 70;
  int karg3 = 80;
  // CHECK: syclct::get_device_manager().current_device().default_queue().submit(
  // CHECK-NEXT: [=](cl::sycl::handler &cgh) {
  // CHECK-NEXT:   cgh.parallel_for<class testKernel>(
  // CHECK-NEXT:     cl::sycl::nd_range<3>(griddim, threaddim),
  // CHECK-NEXT:     [=](cl::sycl::nd_item<3> it) {
  // CHECK-NEXT:       testKernel(it, karg1, karg2, karg3);
  // CHECK-NEXT:     });
  // CHECK-NEXT:   });
  testKernel<<<griddim, threaddim>>>(karg1, karg2, karg3);
}
