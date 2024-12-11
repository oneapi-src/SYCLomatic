// RUN: rm -rf %T/pytorch/c10
// RUN: mkdir -p %T/pytorch/c10/src
// RUN: cp %S/c10.cu %T/pytorch/c10/src/
// RUN: cp %S/user_defined_rule_pytorch.yaml %T/pytorch/c10/
// RUN: cp -r %S/pytorch_cuda_inc %T/pytorch/c10/
// RUN: cd %T/pytorch/c10
// RUN: mkdir dpct_out
// RUN: dpct -out-root dpct_out %T/pytorch/c10/src/c10.cu --extra-arg="-I%T/pytorch/c10/pytorch_cuda_inc" --cuda-include-path="%cuda-path/include" --rule-file=%T/pytorch/c10/user_defined_rule_pytorch.yaml  -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/pytorch/c10/dpct_out/c10.dp.cpp --match-full-lines %T/pytorch/c10/src/c10.cu
// RUN: %if build_lit %{icpx -c -fsycl -DNO_BUILD_TEST  %T/pytorch/c10/dpct_out/c10.dp.cpp -o %T/pytorch/c10/dpct_out/c10.dp.o %}

#ifndef NO_BUILD_TEST
#include <iostream>
// CHECK: #include <c10/xpu/XPUStream.h>
#include <c10/cuda/CUDAStream.h>
// CHECK: #include <c10/core/DeviceGuard.h>
#include <c10/cuda/CUDAGuard.h>

int main() {
  std::optional<c10::Device> device;

  try {
    // CHECK: c10::OptionalDeviceGuard device_guard(device);
    c10::cuda::OptionalCUDAGuard device_guard(device);
  } catch (const std::exception &e) {
    std::cerr << "Runtime error: " << e.what() << std::endl;
    return -1;
  }

  // CHECK: auto currentStream = c10::xpu::getCurrentXPUStream();
  auto currentStream = c10::cuda::getCurrentCUDAStream();

  // CHECK: dpct::queue_ptr curr_cuda_st = &(currentStream.queue());
  // CHECK-NEXT: curr_cuda_st = &(c10::xpu::getCurrentXPUStream().queue());
  cudaStream_t curr_cuda_st = currentStream.stream();
  curr_cuda_st = c10::cuda::getCurrentCUDAStream().stream();

  // CHECK: auto deviceStream = c10::xpu::getCurrentXPUStream(0);
  auto deviceStream = c10::cuda::getCurrentCUDAStream(0);

  return 0;
}

#endif
