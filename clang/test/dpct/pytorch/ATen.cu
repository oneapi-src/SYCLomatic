// RUN: rm -rf %T/pytorch/ATen
// RUN: mkdir -p %T/pytorch/ATen/src
// RUN: cp %S/ATen.cu %T/pytorch/ATen/src/
// RUN: cp %S/user_defined_rule_pytorch.yaml %T/pytorch/ATen/
// RUN: cp -r %S/pytorch_cuda_inc %T/pytorch/ATen/
// RUN: cd %T/pytorch/ATen
// RUN: mkdir dpct_out
// RUN: dpct -out-root dpct_out src/ATen.cu --extra-arg="-I./pytorch_cuda_inc" --cuda-include-path="%cuda-path/include" --rule-file=user_defined_rule_pytorch.yaml  -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file dpct_out/ATen.dp.cpp --match-full-lines src/ATen.cu
// RUN: %if build_lit %{icpx -c -fsycl -DNO_BUILD_TEST  dpct_out/ATen.dp.cpp -o dpct_out/ATen.dp.o %}

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
