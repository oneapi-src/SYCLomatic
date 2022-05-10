// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=file -out-root %T/out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/out/test.dp.cpp --match-full-lines %s
// RUN: FileCheck --input-file %T/out/MainSourceFiles.yaml --match-full-lines %S/check_yaml.txt
// RUN: FileCheck --input-file %T/out/include/dpct/dpct.hpp --match-full-lines %S/check_dpct.txt

#include "header.h"

// CHECK: void k(dpct::device_info *CDP) {
__global__ void k() {
  d();
}

int main() {
  float* a;
  cudaMalloc(&a, 4);
  k<<<1,1>>>();
  return 0;
}
