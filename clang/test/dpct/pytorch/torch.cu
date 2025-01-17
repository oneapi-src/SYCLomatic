// RUN: rm -rf %T/pytorch/torch
// RUN: mkdir -p %T/pytorch/torch/src
// RUN: cp %S/torch.cu %T/pytorch/torch/src/
// RUN: cp -r %S/pytorch_inc %T/pytorch/torch/
// RUN: cd %T/pytorch/torch
// RUN: mkdir dpct_out
// RUN: dpct --out-root dpct_out %T/pytorch/torch/src/torch.cu --extra-arg="-I%T/pytorch/torch/pytorch_inc" --cuda-include-path="%cuda-path/include" --rule-file=%S/../../../tools/dpct/extensions/pytorch_api_rules/pytorch_api.yaml --analysis-scope-path %T/pytorch/torch/pytorch_inc --analysis-scope-path %T/pytorch/torch/src --in-root %T/pytorch/torch/src
// RUN: FileCheck --input-file %T/pytorch/torch/dpct_out/torch.dp.cpp --match-full-lines %T/pytorch/torch/src/torch.cu

// CHECK: #include <c10/xpu/XPUStream.h>
#include <cuda.h>
#include <iostream>
#include <stdexcept>
#include <torch/torch.h>

#define MY_CHECK(condition, message)                              \
  do {                                                            \
    if (!(condition)) {                                           \
      throw std::runtime_error("Error: " + std::string(message)); \
    }                                                             \
  } while (0)

// void foo(torch::Tensor x) {
void foo(torch::Tensor x) {
  // CHECK: MY_CHECK(x.is_xpu(), "x must reside on device");
  MY_CHECK(x.is_cuda(), "x must reside on device");
}
