// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/DplExtrasAlgorithm/api_test7_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++17 -fsized-deallocation
// RUN: grep "IsCalled" %T/DplExtrasAlgorithm/api_test7_out/MainSourceFiles.yaml | wc -l > %T/DplExtrasAlgorithm/api_test7_out/count.txt
// RUN: FileCheck --input-file %T/DplExtrasAlgorithm/api_test7_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/DplExtrasAlgorithm/api_test7_out

// CHECK: 39
// TEST_FEATURE: DplExtrasAlgorithm_gather

#include <thrust/gather.h>
#include <thrust/device_vector.h>

int main() {
  int values[10] = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
  thrust::device_vector<int> d_values(values, values + 10);
  int map[10] = {0, 2, 4, 6, 8, 1, 3, 5, 7, 9};
  thrust::device_vector<int> d_map(map, map + 10);
  thrust::device_vector<int> d_output(10);
  thrust::gather(d_map.begin(), d_map.end(), d_values.begin(), d_output.begin());
  return 0;
}
