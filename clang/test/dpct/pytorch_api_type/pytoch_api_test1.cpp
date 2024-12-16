// RUN: cat %s > %T/pytoch_api_test1.cpp
// RUN: cd %T && mkdir -p ../ATen/cuda
// RUN: cat %S/CUDAContext.h >  %T/../ATen/cuda/CUDAContext.h
// RUN: dpct --rule-file=%S/../../../tools/dpct/DpctOptRules/pytorch_api.yaml --extra-arg="-I ../" --format-range=none -in-root %T -out-root %T/out pytoch_api_test1.cpp --cuda-include-path="%cuda-path/include" 
// RUN: FileCheck --input-file %T/out/pytoch_api_test1.cpp.dp.cpp --match-full-lines %s

#include <cuda.h>
#include <iostream>
#include <stdexcept>

// CHECK: #include "c10/xpu/XPUStream.h"
#include "ATen/cuda/CUDAContext.h"

class TensorStub {
public:
  bool is_cuda() const {
    return true;
  }
};

#define MY_CHECK(condition, message)                              \
  do {                                                            \
    if (!(condition)) {                                           \
      throw std::runtime_error("Error: " + std::string(message)); \
    }                                                             \
  } while (0)

int main() {
  TensorStub x;
  // CHECK: MY_CHECK(x.is_xpu(), "x must reside on device");
  MY_CHECK(x.is_cuda(), "x must reside on device");

  return 0;
}
