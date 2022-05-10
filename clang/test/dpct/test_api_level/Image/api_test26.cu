// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Image/api_test26_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Image/api_test26_out/MainSourceFiles.yaml | wc -l > %T/Image/api_test26_out/count.txt
// RUN: FileCheck --input-file %T/Image/api_test26_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Image/api_test26_out

// CHECK: 69
// TEST_FEATURE: Image_sampling_info_set_addressing_mode_filtering_mode_is_normalized

int main() {
  texture<unsigned int, 1, cudaReadModeElementType> tex_tmp;
  if (true) {
    tex_tmp.filterMode = cudaFilterModePoint;
    tex_tmp.addressMode[1] = cudaAddressModeWrap;
    tex_tmp.addressMode[2] = cudaAddressModeWrap;
    tex_tmp.normalized = 1;
  }
  return 0;
}
