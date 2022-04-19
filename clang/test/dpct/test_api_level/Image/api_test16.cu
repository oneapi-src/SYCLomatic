// RUN: c2s --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Image/api_test16_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Image/api_test16_out/MainSourceFiles.yaml | wc -l > %T/Image/api_test16_out/count.txt
// RUN: FileCheck --input-file %T/Image/api_test16_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Image/api_test16_out

// CHECK: 73
// TEST_FEATURE: Image_image_wrapper_base_set_addressing_mode
// TEST_FEATURE: Image_image_wrapper_base_set_channel
// TEST_FEATURE: Image_image_wrapper_base_set_channel_data_type
// TEST_FEATURE: Image_image_wrapper_base_set_channel_size
// TEST_FEATURE: Image_image_wrapper_base_set_coordinate_normalization_mode_enum
// TEST_FEATURE: Image_image_wrapper_base_set_filtering_mode
// TEST_FEATURE: Image_image_wrapper_base_set_coordinate_normalization_mode
// TEST_FEATURE: Image_image_wrapper_base_set_addressing_mode_filtering_mode_coordinate_normalization_mode
// TEST_FEATURE: Image_image_wrapper_base_set_addressing_mode_filtering_mode_is_normalized

static texture<float4, 2> tex42;

int main() {
  tex42.addressMode[0] = cudaAddressModeClamp;
  cudaChannelFormatDesc desc;
  tex42.channelDesc = desc;
  tex42.normalized = 1;
  tex42.channelDesc.f = cudaChannelFormatKindSigned;
  tex42.channelDesc.z = 1;
  tex42.filterMode = cudaFilterModePoint;
  bool b;
  tex42.normalized = b;

  {
    texture<unsigned int, 1, cudaReadModeElementType> tex_tmp;
    tex_tmp.normalized = false;
    tex_tmp.addressMode[0] = cudaAddressModeClamp;
    tex_tmp.filterMode = cudaFilterModePoint;    
  }

  return 0;
}
