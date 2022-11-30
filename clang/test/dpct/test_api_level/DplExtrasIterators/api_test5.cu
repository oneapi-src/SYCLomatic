// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/DplExtrasIterators/api_test5_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++17
// RUN: grep "IsCalled" %T/DplExtrasIterators/api_test5_out/MainSourceFiles.yaml | wc -l > %T/DplExtrasIterators/api_test5_out/count.txt
// RUN: FileCheck --input-file %T/DplExtrasIterators/api_test5_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/DplExtrasIterators/api_test5_out

// CHECK: 4
// TEST_FEATURE: DplExtrasIterators_arg_index_input_iterator

#include <cub/cub.cuh>

int main() {
  cub::ArgIndexInputIterator<int *> Iter(nullptr);
  return 0;
}
