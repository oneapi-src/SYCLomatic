// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/DplExtrasIterators/api_test4_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++17
// RUN: grep "IsCalled" %T/DplExtrasIterators/api_test4_out/MainSourceFiles.yaml | wc -l > %T/DplExtrasIterators/api_test4_out/count.txt
// RUN: FileCheck --input-file %T/DplExtrasIterators/api_test4_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/DplExtrasIterators/api_test4_out

// CHECK: 2
// TEST_FEATURE: DplExtrasIterators_key_value_pair

#include <cub/cub.cuh>

int main() {
  cub::KeyValuePair<int, int> Pair;
  return 0;
}
