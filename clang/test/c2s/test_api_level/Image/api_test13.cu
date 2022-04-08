// RUN: c2s --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Image/api_test13_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Image/api_test13_out/MainSourceFiles.yaml | wc -l > %T/Image/api_test13_out/count.txt
// RUN: FileCheck --input-file %T/Image/api_test13_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Image/api_test13_out

// CHECK: 28
// TEST_FEATURE: Image_image_accessor_ext

__global__ void foo(cudaTextureObject_t tex21) {
  uint2 u21;
  tex1D(&u21, tex21, 0.5f);
}

int main() {
  return 0;
}
