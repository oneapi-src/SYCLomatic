// RUN: c2s --format-range=none --usm-level=none -in-root %S -out-root %T/circular_inclusion %S/kernel.cu %s -extra-arg="-I %S" --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -std=c++14 -x cuda --cuda-host-only
#include <cuda_runtime.h>
#include "kernel.cuh"
// c2s need to know whether the kernel.cu is included by test.cu,
// but kernel.cuh and dirty.cuh are included each other. In this case,
// c2s will hang in indefinite loop.
int main(){
  kernel<<<1,1>>>();
  return 0;
}
