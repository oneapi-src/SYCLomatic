// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Math/api_test3_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Math/api_test3_out/MainSourceFiles.yaml | wc -l > %T/Math/api_test3_out/count.txt
// RUN: FileCheck --input-file %T/Math/api_test3_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Math/api_test3_out

// CHECK: 1
// TEST_FEATURE: Math_fast_length

__device__ void foo() {
  double d;
  d = norm3d(d, d, d);
}

int main() {
  return 0;
}
