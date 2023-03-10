// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Math/api_test6_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Math/api_test6_out/MainSourceFiles.yaml | wc -l > %T/Math/api_test6_out/count.txt
// RUN: FileCheck --input-file %T/Math/api_test6_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Math/api_test6_out

// CHECK: 3
// TEST_FEATURE: Math_vectorized_isgreater_T
// TEST_FEATURE: Math_vectorized_isgreater_unsigned
// TEST_FEATURE: Math_vectorized_isgreater_uchar4

__device__ void foo() {
  unsigned u, u2;
  u = __vcmpgtu2(u, u2);
  u = __vcmpgtu4(u, u2);
}

int main() {
  return 0;
}
