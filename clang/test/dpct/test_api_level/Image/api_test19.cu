// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Image/api_test19_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Image/api_test19_out/MainSourceFiles.yaml | wc -l > %T/Image/api_test19_out/count.txt
// RUN: FileCheck --input-file %T/Image/api_test19_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Image/api_test19_out

// CHECK: 16
// TEST_FEATURE: Image_image_matrix_get_channel

int main() {
  cudaChannelFormatDesc desc21;
  cudaArray_t a42;
  cudaGetChannelDesc(&desc21, a42);
  return 0;
}
