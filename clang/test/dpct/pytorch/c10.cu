// RUN: rm -rf %T/pytorch/c10
// RUN: mkdir -p %T/pytorch/c10/src
// RUN: cp %S/c10.cu %T/pytorch/c10/src/
// RUN: cp -r %S/pytorch_inc %T/pytorch/c10/
// RUN: cd %T/pytorch/c10
// RUN: mkdir dpct_out
// RUN: dpct -out-root dpct_out %T/pytorch/c10/src/c10.cu --extra-arg="-I%T/pytorch/c10/pytorch_inc" --cuda-include-path="%cuda-path/include" --rule-file=%S/../../../tools/dpct/DpctOptRules/pytorch_api.yaml  -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/pytorch/c10/dpct_out/c10.dp.cpp --match-full-lines %T/pytorch/c10/src/c10.cu

#include <iostream>
// CHECK: #include <c10/core/DeviceGuard.h>
#include <c10/cuda/CUDAGuard.h>
// CHECK: #include <c10/xpu/XPUStream.h>
#include <c10/cuda/CUDAStream.h>
// CHECK: #include <c10/xpu/XPUFunctions.h>
#include <c10/cuda/CUDAFunctions.h>
// CHECK: #include <c10/xpu/XPUMacros.h>
#include <c10/cuda/CUDAMacros.h>

// CHECK: #define BE_BUILD_SHARED_LIBS C10_XPU_BUILD_SHARED_LIBS
// CHECK-NEXT: #define BE_EXPORT C10_XPU_EXPORT
// CHECK-NEXT: #define BE_IMPORT C10_XPU_IMPORT
// CHECK-NEXT: #define BE_API C10_XPU_API
// CHECK-NEXT: #define BE_BUILD_MAIN_LIB C10_XPU_BUILD_MAIN_LIB
#define BE_BUILD_SHARED_LIBS C10_CUDA_BUILD_SHARED_LIBS
#define BE_EXPORT C10_CUDA_EXPORT
#define BE_IMPORT C10_CUDA_IMPORT
#define BE_API C10_CUDA_API
#define BE_BUILD_MAIN_LIB C10_CUDA_BUILD_MAIN_LIB

int main() {
  // device APIs
  // CHECK: c10::DeviceIndex num_devices = c10::xpu::device_count();
  c10::DeviceIndex num_devices = c10::cuda::device_count();

  // CHECK: c10::DeviceIndex num_devices_ensured =
  // CHECK-NEXT:     c10::xpu::device_count_ensure_non_zero();
  c10::DeviceIndex num_devices_ensured = c10::cuda::device_count_ensure_non_zero();

  // CHECK: c10::DeviceIndex current_device = c10::xpu::current_device();
  c10::DeviceIndex current_device = c10::cuda::current_device();

  c10::DeviceIndex new_device = 1;
  // CHECK: c10::xpu::set_device(new_device);
  c10::cuda::set_device(new_device);

  // CHECK: c10::DeviceIndex exchanged_device = c10::xpu::exchange_device(0);
  c10::DeviceIndex exchanged_device = c10::cuda::ExchangeDevice(0);

  // CHECK: c10::DeviceIndex maybe_exchanged_device = c10::xpu::maybe_exchange_device(1);
  c10::DeviceIndex maybe_exchanged_device = c10::cuda::MaybeExchangeDevice(1);

  std::optional<c10::Device> device;
  try {
    // CHECK: c10::OptionalDeviceGuard device_guard(device);
    c10::cuda::OptionalCUDAGuard device_guard(device);
  } catch (const std::exception &e) {
    std::cerr << "Runtime error: " << e.what() << std::endl;
    return -1;
  }

  // stream APIs
  // CHECK: auto currentStream = c10::xpu::getCurrentXPUStream();
  auto currentStream = c10::cuda::getCurrentCUDAStream();

  // CHECK: dpct::queue_ptr curr_cuda_st =
  // CHECK-NEXT: &static_cast<sycl::queue &>(currentStream.queue());
  cudaStream_t curr_cuda_st = currentStream.stream();
  // CHECK: curr_cuda_st =
  // CHECK-NEXT: &static_cast<sycl::queue &>(c10::xpu::getCurrentXPUStream().queue());
  curr_cuda_st = c10::cuda::getCurrentCUDAStream().stream();

  // CHECK: auto deviceStream = c10::xpu::getCurrentXPUStream(0);
  auto deviceStream = c10::cuda::getCurrentCUDAStream(0);

  return 0;
}
