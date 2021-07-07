// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Kernel/api_test3_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Kernel/api_test3_out/MainSourceFiles.yaml | wc -l > %T/Kernel/api_test3_out/count.txt
// RUN: FileCheck --input-file %T/Kernel/api_test3_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Kernel/api_test3_out

// CHECK: 2
// TEST_FEATURE: Kernel_kernel_functor

int main() {
  CUfunction F;
  return 0;
}
