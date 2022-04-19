// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Util/api_test6_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Util/api_test6_out/MainSourceFiles.yaml | wc -l > %T/Util/api_test6_out/count.txt
// RUN: FileCheck --input-file %T/Util/api_test6_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Util/api_test6_out

// CHECK: 3
// TEST_FEATURE: Util_vectorized_isgreater_T
// TEST_FEATURE: Util_vectorized_isgreater_unsigned

__device__ void foo() {
  unsigned u, u2;
  u = __vcmpgtu2(u, u2);
}

int main() {
  return 0;
}
