// RUN: c2s --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Image/api_test1_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Image/api_test1_out/MainSourceFiles.yaml | wc -l > %T/Image/api_test1_out/count.txt
// RUN: FileCheck --input-file %T/Image/api_test1_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Image/api_test1_out

// CHECK: 15
// TEST_FEATURE: Image_image_matrix_p_alias

int main() {
  cudaArray_t a;
  return 0;
}
