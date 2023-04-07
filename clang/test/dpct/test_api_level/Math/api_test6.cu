// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Math/api_test6_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Math/api_test6_out/MainSourceFiles.yaml | wc -l > %T/Math/api_test6_out/count.txt
// RUN: FileCheck --input-file %T/Math/api_test6_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Math/api_test6_out

// CHECK: 11
// TEST_FEATURE: Math_abs
// TEST_FEATURE: Math_abs_diff
// TEST_FEATURE: Math_add_sat
// TEST_FEATURE: Math_rhadd
// TEST_FEATURE: Math_hadd
// TEST_FEATURE: Math_max
// TEST_FEATURE: Math_min
// TEST_FEATURE: Math_sub_sat
// TEST_FEATURE: Math_vectorized_binary
// TEST_FEATURE: Math_vectorized_unary
// TEST_FEATURE: Math_vectorized_sum_abs_diff

__device__ void
foo() {
  unsigned u, u2;
  u = __vabsdiffs2(u, u2);
  u = __vabsss2(u);
  u = __vsads2(u, u2);
}

int main() {
  return 0;
}
