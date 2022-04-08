// RUN: c2s --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Memory/api_test40_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Memory/api_test40_out/MainSourceFiles.yaml | wc -l > %T/Memory/api_test40_out/count.txt
// RUN: FileCheck --input-file %T/Memory/api_test40_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Memory/api_test40_out

// CHECK: 35
// TEST_FEATURE: Memory_c2s_memset
// TEST_FEATURE: Memory_c2s_memset_2d
// TEST_FEATURE: Memory_c2s_memset_3d

int main() {
  cudaExtent e = make_cudaExtent(1, 1, 1);
  cudaPitchedPtr p_A;
  cudaMemset3D(p_A, 0xf, e);
  return 0;
}
