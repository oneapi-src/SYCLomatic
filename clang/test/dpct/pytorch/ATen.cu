// RUN: rm -rf %T/pytorch/ATen
// RUN: mkdir -p %T/pytorch/ATen/src
// RUN: cp %S/ATen.cu %T/pytorch/ATen/src/
// RUN: cp -r %S/pytorch_inc %T/pytorch/ATen/
// RUN: cd %T/pytorch/ATen
// RUN: mkdir dpct_out
// RUN: dpct --out-root dpct_out %T/pytorch/ATen/src/ATen.cu --extra-arg="-I%T/pytorch/ATen/pytorch_inc" --cuda-include-path="%cuda-path/include" --rule-file=%S/../../../tools/dpct/extensions/pytorch_api_rules/pytorch_api.yaml --analysis-scope-path %T/pytorch/ATen/pytorch_inc --analysis-scope-path %T/pytorch/ATen/src --in-root %T/pytorch/ATen/src
// RUN: FileCheck --input-file %T/pytorch/ATen/dpct_out/ATen.dp.cpp --match-full-lines %T/pytorch/ATen/src/ATen.cu

// CHECK: #include <c10/xpu/XPUStream.h>
#include <iostream>
// CHECK: #include <ATen/xpu/XPUContext.h>
#include <ATen/cuda/CUDAContext.h>
// CHECK: #include <ATen/core/Tensor.h>
#include <ATen/core/Tensor.h>

// CHECK: #include <ATen/Tensor.h>
// CHECK-NEXT: #include <c10/util/Half.h>
#include <ATen/cuda/CUDATensorMethods.cuh>

#define AT_CUDA_CHECK(stmt)  (stmt)

// CHECK: #define BE_AT_CHECK
#define BE_AT_CHECK AT_CUDA_CHECK


__global__ void kernel() {}

void test_CUDAStream_as_arg() {
  dim3 gridSize(2, 2, 1);
  dim3 blockSize(8, 8, 1);
  void *args[] = {nullptr}; 

  // CHECK: ([&]() {
  // CHECK-NEXT:   ((sycl::queue *)(c10::xpu::getCurrentXPUStream()))
  // CHECK-NEXT:       ->parallel_for(sycl::nd_range<3>(gridSize * blockSize, blockSize),
  // CHECK-NEXT:                      [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:                        kernel();
  // CHECK-NEXT:                      });
  // CHECK-NEXT:   return 0;
  // CHECK-NEXT: }());
  AT_CUDA_CHECK(cudaLaunchKernel((const void *)kernel, gridSize, blockSize, args, 0, at::cuda::getCurrentCUDAStream()));
}

int main() {
  // CHECK: dpct::queue_ptr st =
  // CHECK-NEXT: &static_cast<sycl::queue &>(c10::xpu::getCurrentXPUStream());
  cudaStream_t st = 0;

  // stream APIs
  at::DeviceIndex devInd = 1;

  // CHECK: auto currentStream = c10::xpu::getCurrentXPUStream();
  auto currentStream = at::cuda::getCurrentCUDAStream();
  // CHECK: auto deviceStream = c10::xpu::getCurrentXPUStream(devInd);
  auto deviceStream = at::cuda::getCurrentCUDAStream(devInd);

  // CHECK: dpct::queue_ptr curr_cuda_st =
  // CHECK-NEXT:    &static_cast<sycl::queue &>(c10::xpu::getCurrentXPUStream().queue());
  cudaStream_t curr_cuda_st = at::cuda::getCurrentCUDAStream().stream();
  // CHECK: dpct::queue_ptr dev_cuda_st = &static_cast<sycl::queue &>(
  // CHECK-NEXT:    c10::xpu::getCurrentXPUStream(devInd).queue());
  cudaStream_t dev_cuda_st = at::cuda::getCurrentCUDAStream(devInd).stream();

  test_CUDAStream_as_arg();

  return 0;
}
