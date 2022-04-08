// RUN: c2s --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Kernel/api_test1_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Kernel/api_test1_out/MainSourceFiles.yaml | wc -l > %T/Kernel/api_test1_out/count.txt
// RUN: FileCheck --input-file %T/Kernel/api_test1_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Kernel/api_test1_out

// CHECK: 2
// TEST_FEATURE: Kernel_kernel_function_info

int main() {
  cudaFuncAttributes attrs;
  return 0;
}
