// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Image/api_test5_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Image/api_test5_out/MainSourceFiles.yaml | wc -l > %T/Image/api_test5_out/count.txt
// RUN: FileCheck --input-file %T/Image/api_test5_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Image/api_test5_out

// CHECK: 4
// TEST_FEATURE: Image_sampling_info

int main() {
  cudaTextureDesc a;
  return 0;
}
