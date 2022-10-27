// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/DplExtrasAlgorithm/api_test8_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++17 -fsized-deallocation
// RUN: grep "IsCalled" %T/DplExtrasAlgorithm/api_test8_out/MainSourceFiles.yaml | wc -l > %T/DplExtrasAlgorithm/api_test8_out/count.txt
// RUN: FileCheck --input-file %T/DplExtrasAlgorithm/api_test8_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/DplExtrasAlgorithm/api_test8_out

// CHECK: 39
// TEST_FEATURE: DplExtrasAlgorithm_scatter

#include <thrust/scatter.h>
#include <thrust/device_vector.h>

int main() {
  int values[10] = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
  thrust::device_vector<int> d_values(values, values + 10);
  int map[10] = {0, 5, 1, 6, 2, 7, 3, 8, 4, 9};
  thrust::device_vector<int> d_map(map, map + 10);
  thrust::device_vector<int> d_output(10);
  thrust::scatter(d_values.begin(), d_values.end(), d_map.begin(), d_output.begin());
  return 0;
}
