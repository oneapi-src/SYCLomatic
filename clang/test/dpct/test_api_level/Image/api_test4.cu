// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Image/api_test4_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Image/api_test4_out/MainSourceFiles.yaml | wc -l > %T/Image/api_test4_out/count.txt
// RUN: FileCheck --input-file %T/Image/api_test4_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Image/api_test4_out

// CHECK: 2
// TEST_FEATURE: Image_image_channel_data_type

int main() {
  cudaChannelFormatKind a;
  return 0;
}
