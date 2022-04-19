// RUN: c2s --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Image/api_test28_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Image/api_test28_out/MainSourceFiles.yaml | wc -l > %T/Image/api_test28_out/count.txt
// RUN: FileCheck --input-file %T/Image/api_test28_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Image/api_test28_out

// CHECK: 21
// TEST_FEATURE: Image_image_data_get_data_type

void foo(cudaResourceType res21) {}

int main() {
  cudaResourceDesc res21;
  foo(res21.resType);
  return 0;
}
