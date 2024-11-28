// RUN: rm -rf %T/pytorch_ATen_cuda_ns
// RUN: mkdir %T/pytorch_ATen_cuda_ns
// RUN: cp %S/pytorch_ATen_cuda_ns.cu %T/pytorch_ATen_cuda_ns/
// RUN: cp %S/user_defined_rule_pytorch.yaml %T/
// RUN: cp -r %S/pytorch_cuda_inc %T/
// RUN: cd %T
// RUN: rm -rf %T/pytorch_ATen_cuda_ns_output
// RUN: mkdir %T/pytorch_ATen_cuda_ns_output
// RUN: dpct -out-root %T/pytorch_ATen_cuda_ns_output %T/pytorch_ATen_cuda_ns/pytorch_ATen_cuda_ns.cu --extra-arg="-I./pytorch_cuda_inc" --cuda-include-path="%cuda-path/include" --rule-file=%T/user_defined_rule_pytorch.yaml  -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/pytorch_ATen_cuda_ns_output/pytorch_ATen_cuda_ns.dp.cpp --match-full-lines %T/pytorch_ATen_cuda_ns/pytorch_ATen_cuda_ns.cu
// RUN: %if build_lit %{icpx -c -fsycl -DNO_BUILD_TEST  %T/pytorch_ATen_cuda_ns_output/pytorch_ATen_cuda_ns.dp.cpp -o %T/pytorch_ATen_cuda_ns_output/pytorch_ATen_cuda_ns.dp.o %}

#ifndef NO_BUILD_TEST
#include <iostream>
// CHECK: #include <ATen/xpu/XPUContext.h>
#include <ATen/cuda/CUDAContext.h>
// CHECK: #include <ATen/core/Tensor.h>
#include <ATen/core/Tensor.h>

int main() {

  return 0;
}
#endif
