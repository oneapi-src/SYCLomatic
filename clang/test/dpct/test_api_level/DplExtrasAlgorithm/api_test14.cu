// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.2, cuda-11.4
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0, v11.2, v11.4
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
