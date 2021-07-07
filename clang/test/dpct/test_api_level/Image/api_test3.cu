// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Image/api_test3_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Image/api_test3_out/MainSourceFiles.yaml | wc -l > %T/Image/api_test3_out/count.txt
// RUN: FileCheck --input-file %T/Image/api_test3_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Image/api_test3_out

// CHECK: 7
// TEST_FEATURE: Image_image_channel

int main() {
  cudaChannelFormatDesc a;
  return 0;
}
