// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Math/api_test7_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Math/api_test7_out/MainSourceFiles.yaml | wc -l > %T/Math/api_test7_out/count.txt
// RUN: FileCheck --input-file %T/Math/api_test7_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Math/api_test7_out

// CHECK: 2
// TEST_FEATURE: Math_reverse_bits

__device__ void foo() {
  unsigned u;
  u = __brev(u);
}

int main() {
  return 0;
}
