// RUN: c2s --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Image/api_test14_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Image/api_test14_out/MainSourceFiles.yaml | wc -l > %T/Image/api_test14_out/count.txt
// RUN: FileCheck --input-file %T/Image/api_test14_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Image/api_test14_out

// CHECK: 61
// TEST_FEATURE: Image_image_wrapper_attach
// TEST_FEATURE: Image_image_wrapper_base_set_data

int main() {
  texture<uint2, cudaTextureType1DLayered> tex21;
  uint2 *d_data21;
  cudaChannelFormatDesc desc21;
  cudaBindTexture(0, tex21, d_data21, desc21, 32);
  return 0;
}
