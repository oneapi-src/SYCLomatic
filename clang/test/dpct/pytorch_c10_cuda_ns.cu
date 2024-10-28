// RUN: rm -rf %T/src
// RUN: mkdir %T/src
// RUN: cp %S/pytorch_c10_cuda_ns.cu %T/src/
// RUN: cp %S/user_defined_rule_pytorch.yaml %T/
// RUN: cp -r %S/pytorch_cuda_inc %T/
// RUN: cd %T
// RUN: rm -rf %T/pytorch_c10_cuda_ns_output
// RUN: mkdir %T/pytorch_c10_cuda_ns_output
// RUN: dpct -out-root %T/pytorch_c10_cuda_ns_output src/pytorch_c10_cuda_ns.cu --extra-arg="-I./pytorch_cuda_inc" --cuda-include-path="%cuda-path/include" --rule-file=user_defined_rule_pytorch.yaml  -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/pytorch_c10_cuda_ns_output/pytorch_c10_cuda_ns.dp.cpp --match-full-lines %T/src/pytorch_c10_cuda_ns.cu
// RUN: %if build_lit %{icpx -c -fsycl -DNO_BUILD_TEST  %T/pytorch_c10_cuda_ns_output/pytorch_c10_cuda_ns.dp.cpp -o %T/pytorch_c10_cuda_ns_output/pytorch_c10_cuda_ns.dp.o %}

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

  // CHECK: std::cout << "Current Stream (Default Device): " << currentStream.queue() << std::endl;
  std::cout << "Current Stream (Default Device): " << currentStream.stream() << std::endl;

  // CHECK: auto deviceStream = c10::xpu::getCurrentXPUStream(0);
  auto deviceStream = c10::cuda::getCurrentCUDAStream(0);

  return 0;
}

#endif
