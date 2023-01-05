// RUN: dpct --format-range=none   --use-custom-helper=api -out-root %T/Memory/api_test41_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Memory/api_test41_out/MainSourceFiles.yaml | wc -l > %T/Memory/api_test41_out/count.txt
// RUN: FileCheck --input-file %T/Memory/api_test41_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Memory/api_test41_out

// CHECK: 29
// TEST_FEATURE: Memory_dpct_memset
// TEST_FEATURE: Memory_dpct_memset_2d
// TEST_FEATURE: Memory_dpct_memset_3d

int main() {
  cudaExtent e = make_cudaExtent(1, 1, 1);
  cudaPitchedPtr p_A;
  cudaMemset3D(p_A, 0xf, e);
  return 0;
}
