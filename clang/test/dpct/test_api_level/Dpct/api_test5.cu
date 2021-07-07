// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Dpct/api_test5_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Dpct/api_test5_out/MainSourceFiles.yaml | wc -l > %T/Dpct/api_test5_out/count.txt
// RUN: FileCheck --input-file %T/Dpct/api_test5_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Dpct/api_test5_out

// CHECK: 2
// TEST_FEATURE: Dpct_dpct_pi

__device__ void foo() {
  double d;
  sincospi(d, &d, &d);
}

int main() {
  return 0;
}
