// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Image/api_test8_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Image/api_test8_out/MainSourceFiles.yaml | wc -l > %T/Image/api_test8_out/count.txt
// RUN: FileCheck --input-file %T/Image/api_test8_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Image/api_test8_out

// CHECK: 29
// TEST_FEATURE: Image_image_wrapper_base_p_alias
// TEST_FEATURE: Image_get_image_from_module

int main() {
  cudaTextureObject_t a;
  CUmodule M;
  CUtexref tex;
  cuModuleGetTexRef(&tex, M, "tex");
  return 0;
}
