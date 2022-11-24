// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/DplExtrasAlgorithm/api_test13 %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++17
// RUN: grep "IsCalled" %T/DplExtrasAlgorithm/api_test13/MainSourceFiles.yaml | wc -l > %T/DplExtrasAlgorithm/api_test13/count.txt
// RUN: FileCheck --input-file %T/DplExtrasAlgorithm/api_test13/count.txt --match-full-lines %s
// RUN: rm -rf %T/DplExtrasAlgorithm/api_test13

// CHECK: 21
// TEST_FEATURE: DplExtrasAlgorithm_sort_keys

#include <cub/cub.cuh>

int main() {
  void *temp_storage;
  size_t temp_storage_size;
  int n, *d_keys_in, *d_keys_out;
  cub::DeviceRadixSort::SortKeys(temp_storage, temp_storage_size, d_keys_in, d_keys_out, n);
  return 0;
}
