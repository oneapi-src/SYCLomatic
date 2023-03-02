// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Math/api_test1_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Math/api_test1_out/MainSourceFiles.yaml | wc -l > %T/Math/api_test1_out/count.txt
// RUN: FileCheck --input-file %T/Math/api_test1_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Math/api_test1_out

// CHECK: 2
// TEST_FEATURE: Math_cast_double_to_int

__device__ void foo() {
  double d;
  int i = __double2hiint(d);
}

int main() {
  return 0;
}
