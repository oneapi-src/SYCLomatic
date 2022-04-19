// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Memory/api_test6_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Memory/api_test6_out/MainSourceFiles.yaml | wc -l > %T/Memory/api_test6_out/count.txt
// RUN: FileCheck --input-file %T/Memory/api_test6_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Memory/api_test6_out

// CHECK: 24
// TEST_FEATURE: Memory_dpct_free

int main() {
  float* a;
  cudaFree(a);
  return 0;
}
