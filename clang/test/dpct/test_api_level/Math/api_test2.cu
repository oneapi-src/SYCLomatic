// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Math/api_test2_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Math/api_test2_out/MainSourceFiles.yaml | wc -l > %T/Math/api_test2_out/count.txt
// RUN: FileCheck --input-file %T/Math/api_test2_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Math/api_test2_out

// CHECK: 2
// TEST_FEATURE: Math_cast_ints_to_double

__device__ void foo() {
  int i;
  double d = __hiloint2double(i, i);
}

int main() {
  return 0;
}
