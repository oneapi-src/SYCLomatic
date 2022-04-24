// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Image/api_test15_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Image/api_test15_out/MainSourceFiles.yaml | wc -l > %T/Image/api_test15_out/count.txt
// RUN: FileCheck --input-file %T/Image/api_test15_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Image/api_test15_out

// CHECK: 69
// TEST_FEATURE: Image_create_image_wrapper
// TEST_FEATURE: Image_image_wrapper_base_set_sampling_info

int main() {
  cudaResourceDesc res42;
  cudaTextureDesc texDesc42;
  cudaTextureObject_t tex;
  cudaCreateTextureObject(&tex, &res42, &texDesc42, NULL);
  return 0;
}
