// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Image/api_test33_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Image/api_test33_out/MainSourceFiles.yaml | wc -l > %T/Image/api_test33_out/count.txt
// RUN: FileCheck --input-file %T/Image/api_test33_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Image/api_test33_out

// CHECK: 28
// TEST_FEATURE: Image_image_data_set_data
// TEST_FEATURE: Image_image_data_set_data_type
// TEST_FEATURE: Image_image_data_set_channel_num
// TEST_FEATURE: Image_image_data_set_channel_type
// TEST_FEATURE: Image_image_data_set_data_ptr
// TEST_FEATURE: Image_image_data_set_pitch
// TEST_FEATURE: Image_image_data_set_x
// TEST_FEATURE: Image_image_data_set_y
// TEST_FEATURE: Image_image_data_set_channel

int main() {
  float* a;
  CUDA_RESOURCE_DESC res42;
  res42.resType = CU_RESOURCE_TYPE_ARRAY;
  res42.res.pitch2D.numChannels = 4;
  res42.res.pitch2D.format = CU_AD_FORMAT_FLOAT;
  res42.res.pitch2D.devPtr = (CUdeviceptr)a;
  res42.res.pitch2D.pitchInBytes = 4;
  res42.res.pitch2D.width = 4;
  res42.res.pitch2D.height = 4;

  cudaChannelFormatDesc desc;
  cudaResourceDesc res;
  res.res.pitch2D.desc = desc;
}
