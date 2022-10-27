// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/DplExtrasVector/api_test1_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++17 -fsized-deallocation
// RUN: grep "IsCalled" %T/DplExtrasVector/api_test1_out/MainSourceFiles.yaml | wc -l > %T/DplExtrasVector/api_test1_out/count.txt
// RUN: FileCheck --input-file %T/DplExtrasVector/api_test1_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/DplExtrasVector/api_test1_out

// CHECK: 38
// TEST_FEATURE: DplExtrasVector_device_vector

#include <thrust/device_vector.h>

int main() {
  float* x;
  thrust::device_vector<float> d_x(x, x + 4);
  return 0;
}
