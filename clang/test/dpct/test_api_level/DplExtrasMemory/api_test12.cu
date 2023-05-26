// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/DplExtrasMemory/api_test12_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++17 -fsized-deallocation
// RUN: grep "IsCalled" %T/DplExtrasMemory/api_test12_out/MainSourceFiles.yaml | wc -l > %T/DplExtrasMemory/api_test12_out/count.txt
// RUN: FileCheck --input-file %T/DplExtrasMemory/api_test12_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/DplExtrasMemory/api_test12_out

// CHECK: 42
// TEST_FEATURE: DplExtrasMemory_get_raw_reference

#include <thrust/device_vector.h>
#include <thrust/memory.h>

int main() {
  thrust::device_vector<int> d_vec;
  int &ref1 = thrust::raw_reference_cast(d_vec[0]);
  return 0;
}
