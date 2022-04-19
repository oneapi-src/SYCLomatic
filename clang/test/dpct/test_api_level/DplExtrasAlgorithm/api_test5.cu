// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: c2s --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/DplExtrasAlgorithm/api_test5_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++17
// RUN: grep "IsCalled" %T/DplExtrasAlgorithm/api_test5_out/MainSourceFiles.yaml | wc -l > %T/DplExtrasAlgorithm/api_test5_out/count.txt
// RUN: FileCheck --input-file %T/DplExtrasAlgorithm/api_test5_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/DplExtrasAlgorithm/api_test5_out

// CHECK: 4
// TEST_FEATURE: DplExtrasAlgorithm_iota

#include <thrust/sequence.h>

int main() {
  int *a;
  thrust::sequence(a, a + 10);
  return 0;
}
