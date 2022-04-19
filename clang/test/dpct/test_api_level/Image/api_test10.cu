// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Image/api_test10_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Image/api_test10_out/MainSourceFiles.yaml | wc -l > %T/Image/api_test10_out/count.txt
// RUN: FileCheck --input-file %T/Image/api_test10_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Image/api_test10_out

// CHECK: 57
// TEST_FEATURE: Image_image_wrapper

int main() {
  texture<float4, 2> tex42;
  cudaUnbindTexture(tex42);
  return 0;
}
