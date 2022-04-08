// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: c2s --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/DplExtrasAlgorithm/api_test4_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++17
// RUN: grep "IsCalled" %T/DplExtrasAlgorithm/api_test4_out/MainSourceFiles.yaml | wc -l > %T/DplExtrasAlgorithm/api_test4_out/count.txt
// RUN: FileCheck --input-file %T/DplExtrasAlgorithm/api_test4_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/DplExtrasAlgorithm/api_test4_out

// CHECK: 5
// TEST_FEATURE: DplExtrasAlgorithm_copy_if

#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>

int main() {
  int A[10];
  auto range = thrust::make_counting_iterator(0);
  thrust::copy_if(A, A + 10, range, A,[=] __device__(int idx) { return true; });
  return 0;
}
