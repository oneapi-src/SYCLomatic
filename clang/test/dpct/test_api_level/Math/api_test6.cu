// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Math/api_test6_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Math/api_test6_out/MainSourceFiles.yaml | wc -l > %T/Math/api_test6_out/count.txt
// RUN: FileCheck --input-file %T/Math/api_test6_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Math/api_test6_out

// CHECK: 12
// TEST_FEATURE: Math_vectorized_abs
// TEST_FEATURE: Math_vectorized_abs_diff
// TEST_FEATURE: Math_vectorized_abs_sat
// TEST_FEATURE: Math_vectorized_binary
// TEST_FEATURE: Math_vectorized_add_sat
// TEST_FEATURE: Math_vectorized_avg_sat
// TEST_FEATURE: Math_vectorized_hadd
// TEST_FEATURE: Math_vectorized_unary
// TEST_FEATURE: Math_vectorized_neg_sat
// TEST_FEATURE: Math_vectorized_sum_abs_diff
// TEST_FEATURE: Math_vectorized_set_compare
// TEST_FEATURE: Math_vectorized_sub_sat

__device__ void
foo() {
  unsigned u, u2;
  u = __vabs2(u);
  u = __vabsdiffs2(u, u2);
  u = __vabsss2(u);
  u = __vcmpgtu2(u, u2);
  u = __vaddss2(u, u2);
  u = __vavgs2(u, u2);
  u = __vhaddu2(u, u2);
  u = __vneg2(u);
  u = __vnegss2(u);
  u = __vsads2(u, u2);
  u = __vseteq2(u, u2);
  u = __vsubss2(u, u2);
}

int main() {
  return 0;
}
