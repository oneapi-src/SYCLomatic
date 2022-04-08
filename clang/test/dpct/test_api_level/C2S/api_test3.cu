// RUN: c2s --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/C2S/api_test3_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/C2S/api_test3_out/MainSourceFiles.yaml | wc -l > %T/C2S/api_test3_out/count.txt
// RUN: FileCheck --input-file %T/C2S/api_test3_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/C2S/api_test3_out

// CHECK: 2

// TEST_FEATURE: C2S_c2s_compatibility_temp

#define AAA __CUDA_ARCH__

int main() {
  return 0;
}
