// RUN: c2s --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Image/api_test21_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Image/api_test21_out/MainSourceFiles.yaml | wc -l > %T/Image/api_test21_out/count.txt
// RUN: FileCheck --input-file %T/Image/api_test21_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Image/api_test21_out

// CHECK: 55
// TEST_FEATURE: Image_image_matrix_to_pitched_data

int main() {
  cudaArray_t a1;
  cudaArray* a2;
  size_t width, height, woffset, hoffset;
  cudaMemcpy2DArrayToArray(a1, woffset, hoffset, a2, woffset, hoffset, width, height, cudaMemcpyDeviceToHost);
  return 0;
}
