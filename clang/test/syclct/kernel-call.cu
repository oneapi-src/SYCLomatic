// RUN: syclct -out-root %T %s -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/kernel-call.sycl.cpp --match-full-lines %s
__global__ void testKernel(const int *L, const int *M, int N) {
  int gtid = blockIdx.x * blockDim.x + threadIdx.x;
}

int main() {
  dim3 griddim = 2;
  dim3 threaddim = 32;
  void *karg1 = 0;
  const int *karg2 = 0;
  int karg3 = 80;
// CHECK:  {
// CHECK-NEXT:    std::pair<cl::sycl::buffer<char, 1 >*, size_t> karg1_buf = cu2sycl::get_buffer_and_offset(karg1);
// CHECK-NEXT:    size_t karg1_offset = karg1_buf.second;
// CHECK-NEXT:    std::pair<cl::sycl::buffer<char, 1 >*, size_t> karg2_buf = cu2sycl::get_buffer_and_offset(karg2);
// CHECK-NEXT:    size_t karg2_offset = karg2_buf.second;
// CHECK-NEXT:    cu2sycl::get_device_manager().current_device().default_queue().submit(
// CHECK-NEXT:      [=](cl::sycl::handler &cgh) {
// CHECK-NEXT:        auto karg1_acc = karg1_buf.first->get_access<cl::sycl::access::mode::read_write>(cgh);
// CHECK-NEXT:        auto karg2_acc = karg2_buf.first->get_access<cl::sycl::access::mode::read_write>(cgh);
// CHECK-NEXT:        cgh.parallel_for<class testKernel>(
// CHECK-NEXT:          cl::sycl::nd_range<3>(griddim, threaddim),
// CHECK-NEXT:          [=](cl::sycl::nd_item<3> it) {
// CHECK-NEXT:            void *karg1 = (void*)(&karg1_acc[0] + karg1_offset);
// CHECK-NEXT:            const int *karg2 = (const int*)(&karg2_acc[0] + karg2_offset);
// CHECK-NEXT:            testKernel(it, (const int *)karg1, karg2, karg3);
// CHECK-NEXT:          });
// CHECK-NEXT:      })
// CHECK-NEXT:  };
  testKernel<<<griddim, threaddim>>>((const int *)karg1, karg2, karg3);
}
