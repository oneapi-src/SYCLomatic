// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Image/api_test36_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Image/api_test36_out/MainSourceFiles.yaml | wc -l > %T/Image/api_test36_out/count.txt
// RUN: FileCheck --input-file %T/Image/api_test36_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Image/api_test36_out

// CHECK: 29
// TEST_FEATURE: Image_sampling_info_get_filtering_mode
// TEST_FEATURE: Image_sampling_info_get_addressing_mode

int main() {
  cudaTextureDesc texDesc42;
  cudaTextureFilterMode a = texDesc42.filterMode;
  cudaTextureAddressMode b = texDesc42.addressMode[0];
  return 0;
}
