// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Image/api_test25_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Image/api_test25_out/MainSourceFiles.yaml | wc -l > %T/Image/api_test25_out/count.txt
// RUN: FileCheck --input-file %T/Image/api_test25_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Image/api_test25_out

// CHECK: 59
// TEST_FEATURE: Image_sampling_info_is_coordinate_normalized

int main() {
  texture<unsigned int, 1, cudaReadModeElementType> tex_tmp;
  int normalized = tex_tmp.normalized;
  return 0;
}
