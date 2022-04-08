// RUN: c2s --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Image/api_test24_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Image/api_test24_out/MainSourceFiles.yaml | wc -l > %T/Image/api_test24_out/count.txt
// RUN: FileCheck --input-file %T/Image/api_test24_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Image/api_test24_out

// CHECK: 27
// TEST_FEATURE: Image_sampling_info_set_coordinate_normalization_mode

int main() {
  cudaTextureDesc tex_tmp;
  int normalized;
  tex_tmp.normalizedCoords = normalized;
  return 0;
}
