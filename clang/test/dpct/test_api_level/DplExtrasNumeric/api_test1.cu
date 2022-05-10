// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/DplExtrasNumeric/api_test1_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++17
// RUN: grep "IsCalled" %T/DplExtrasNumeric/api_test1_out/MainSourceFiles.yaml | wc -l > %T/DplExtrasNumeric/api_test1_out/count.txt
// RUN: FileCheck --input-file %T/DplExtrasNumeric/api_test1_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/DplExtrasNumeric/api_test1_out

// CHECK: 3
// TEST_FEATURE: DplExtrasNumeric_inner_product

#include <thrust/inner_product.h>
int main() {
  int* a;
  thrust::inner_product(thrust::seq, a, a + 10, a, 0.0f);
  return 0;
}
