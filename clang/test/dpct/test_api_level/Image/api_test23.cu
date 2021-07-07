// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Image/api_test23_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Image/api_test23_out/MainSourceFiles.yaml | wc -l > %T/Image/api_test23_out/count.txt
// RUN: FileCheck --input-file %T/Image/api_test23_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Image/api_test23_out

// CHECK: 37
// TEST_FEATURE: Image_sampling_info_set_filtering_mode
// TEST_FEATURE: Image_sampling_info_set_addressing_mode
// TEST_FEATURE: Image_sampling_info_set_addressing_mode_filtering_mode_coordinate_normalization_mode

// Below feature is triggered because all features named "set" of sampling_info are added
// when the migrated field name is empty.
// TEST_FEATURE: Image_sampling_info_set_coordinate_normalization_mode_enum

int main() {
  cudaTextureDesc texDesc42;
  texDesc42.addressMode[0] = cudaAddressModeClamp;
  texDesc42.addressMode[1] = cudaAddressModeClamp;
  texDesc42.addressMode[2] = cudaAddressModeClamp;
  texDesc42.filterMode = cudaFilterModePoint;
  texDesc42.normalizedCoords = 1;
  return 0;
}
