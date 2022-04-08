// RUN: c2s --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Image/api_test9_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Image/api_test9_out/MainSourceFiles.yaml | wc -l > %T/Image/api_test9_out/count.txt
// RUN: FileCheck --input-file %T/Image/api_test9_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Image/api_test9_out

// CHECK: 7
// TEST_FEATURE: Image_image_channel
// TEST_FEATURE: Image_image_channel_create

int main() {
  cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
  return 0;
}
