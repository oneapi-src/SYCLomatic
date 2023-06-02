// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/DplExtrasAlgorithm/api_test27_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++17
// RUN: grep "IsCalled" %T/DplExtrasAlgorithm/api_test27_out/MainSourceFiles.yaml | wc -l > %T/DplExtrasAlgorithm/api_test27_out/count.txt
// RUN: FileCheck --input-file %T/DplExtrasAlgorithm/api_test27_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/DplExtrasAlgorithm/api_test27_out

// CHECK: 37
// TEST_FEATURE: DplExtrasAlgorithm_set_symmetric_difference

#include <thrust/device_malloc.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/set_operations.h>


void test_1() {
  int A_keys[7] = {0, 1, 2, 2, 4, 6, 7}; // 0 2 4 6 7
  int A_vals[7] = {0, 0, 0, 0, 0, 0, 0};
  int B_keys[5] = {1, 1, 2, 5, 8}; // 1 5 8
  int B_vals[5] = {1, 1, 1, 1, 1};
  int keys_result[8];
  int vals_result[8];
  thrust::pair<int *, int *> end = thrust::set_symmetric_difference_by_key(
      thrust::host, A_keys, A_keys + 7, B_keys, B_keys + 5, A_vals, B_vals,
      keys_result, vals_result);
}
