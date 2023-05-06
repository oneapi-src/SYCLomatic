// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/DplExtrasAlgorithm/api_test26_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++17
// RUN: grep "IsCalled" %T/DplExtrasAlgorithm/api_test26_out/MainSourceFiles.yaml | wc -l > %T/DplExtrasAlgorithm/api_test26_out/count.txt
// RUN: FileCheck --input-file %T/DplExtrasAlgorithm/api_test26_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/DplExtrasAlgorithm/api_test26_out

// CHECK: 36
// TEST_FEATURE: DplExtrasAlgorithm_equal_range

#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>


void test_1() {
  int data[] = {0, 2, 5, 7, 8};
  const int N = 5;
  thrust::equal_range(thrust::host, data, data + N, 0);   // returns [input.begin(), input.begin() + 1)
}
