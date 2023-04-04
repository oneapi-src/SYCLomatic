// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Math/api_test4_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Math/api_test4_out/MainSourceFiles.yaml | wc -l > %T/Math/api_test4_out/count.txt
// RUN: FileCheck --input-file %T/Math/api_test4_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Math/api_test4_out

// CHECK: 2
// TEST_FEATURE: Math_vectorized_max

__device__ void foo() {
  unsigned u, u2;
  u = __vmaxs4(u, u2);
}

int main() {
  return 0;
}
