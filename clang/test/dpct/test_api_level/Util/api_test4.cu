// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Util/api_test4_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Util/api_test4_out/MainSourceFiles.yaml | wc -l > %T/Util/api_test4_out/count.txt
// RUN: FileCheck --input-file %T/Util/api_test4_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Util/api_test4_out

// CHECK: 2
// TEST_FEATURE: Util_vectorized_max

__device__ void foo() {
  unsigned u, u2;
  u = __vmaxs4(u, u2);
}

int main() {
  return 0;
}
