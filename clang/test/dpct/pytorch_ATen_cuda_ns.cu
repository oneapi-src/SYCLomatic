// RUN: rm -rf %T/src
// RUN: mkdir %T/src
// RUN: cat %s > %T/src/pytorch_ATen_cuda_ns.cu
// RUN: cat %S/user_defined_rule_pytorch.yaml > %T/user_defined_rule_pytorch.yaml
// RUN: cp -r %S/pytorch_cuda_inc %T/
// RUN: cd %T
// RUN: rm -rf %T/pytorch_ATen_cuda_ns_output
// RUN: mkdir %T/pytorch_ATen_cuda_ns_output
// RUN: dpct -out-root %T/pytorch_ATen_cuda_ns_output src/pytorch_ATen_cuda_ns.cu --extra-arg="-I./pytorch_cuda_inc" --cuda-include-path="%cuda-path/include" --rule-file=user_defined_rule_pytorch.yaml  -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/pytorch_ATen_cuda_ns_output/pytorch_ATen_cuda_ns.dp.cpp --match-full-lines pytorch_ATen_cuda_ns.cu
// RUN: %if build_lit %{icpx -c -fsycl -DNO_BUILD_TEST  %T/pytorch_ATen_cuda_ns_output/pytorch_ATen_cuda_ns.dp.cpp -o %T/pytorch_ATen_cuda_ns_output/pytorch_ATen_cuda_ns.dp.o %}

#ifndef NO_BUILD_TEST
#include <iostream>
// CHECK: #include <ATen/xpu/XPUContext.h>
#include <ATen/cuda/CUDAContext.h>
// CHECK: #include <ATen/core/Tensor.h>
#include <ATen/core/Tensor.h>
