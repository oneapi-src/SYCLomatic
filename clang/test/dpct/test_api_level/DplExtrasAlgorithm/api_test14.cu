// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/DplExtrasAlgorithm/api_test14 %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++17
// RUN: grep "IsCalled" %T/DplExtrasAlgorithm/api_test14/MainSourceFiles.yaml | wc -l > %T/DplExtrasAlgorithm/api_test14/count.txt
// RUN: FileCheck --input-file %T/DplExtrasAlgorithm/api_test14/count.txt --match-full-lines %s
// RUN: rm -rf %T/DplExtrasAlgorithm/api_test14

// CHECK: 23
// TEST_FEATURE: DplExtrasAlgorithm_sort_pairs

#include <cub/cub.cuh>

int main() {
   void *temp_storage;
  size_t temp_storage_size;
  int n, *d_keys_in, *d_keys_out, *d_values_in, *d_values_out;
  cub::DeviceRadixSort::SortPairs(temp_storage, temp_storage_size, d_keys_in, d_keys_out, d_values_in, d_values_out, n);
  return 0;
}
