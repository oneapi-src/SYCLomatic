// RUN: dpct --rule-file=%S/../../../tools/dpct/DpctOptRules/pytorch_api.yaml --format-range=none -out-root %T/pytoch_api_test2 %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/pytoch_api_test2/pytoch_api_test2.dp.cpp --match-full-lines %s

#include <iostream>
#include <vector>

#define AT_CUDA_CHECK(stmt)  (stmt)

namespace c10 {
using DeviceIndex = int8_t;
namespace cuda {
class CUDAStream {
public:
  CUDAStream() {}
  cudaStream_t stream() { return 0; }
  operator cudaStream_t() const {
    return stream();
  }
  cudaStream_t stream() const;
};
CUDAStream getCurrentCUDAStream(DeviceIndex device_index = -1) {
  return CUDAStream();
}
} // namespace cuda
} // namespace c10

namespace at {
using namespace c10;
} // namespace at

__global__ void kernel() {}

int main() {
  dim3 gridSize(2, 2, 1);
  dim3 blockSize(8, 8, 1);
  void *args[] = {nullptr}; 

  //      CHECK:([&](){
  // CHECK-NEXT:  ((sycl::queue*)(c10::xpu::getCurrentXPUStream()))->parallel_for(
  // CHECK-NEXT:    sycl::nd_range<3>(gridSize * blockSize, blockSize),
  // CHECK-NEXT:    [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:      kernel();
  // CHECK-NEXT:    });
  // CHECK-NEXT:  return 0;
  // CHECK-NEXT:}());
  AT_CUDA_CHECK(cudaLaunchKernel((const void *)kernel, gridSize, blockSize, args, 0, at::cuda::getCurrentCUDAStream()));
  at::DeviceIndex d = 1;
  // CHECK: c10::xpu::getCurrentXPUStream(d);
  at::cuda::getCurrentCUDAStream(d);
  // CHECK: dpct::queue_ptr s = &static_cast<sycl::queue &>(c10::xpu::getCurrentXPUStream(). queue());
  cudaStream_t s = at::cuda::getCurrentCUDAStream().stream();
  return 0;
}
