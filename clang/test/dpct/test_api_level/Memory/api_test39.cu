// RUN: dpct --format-range=none   --use-custom-helper=api -out-root %T/Memory/api_test39_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Memory/api_test39_out/MainSourceFiles.yaml | wc -l > %T/Memory/api_test39_out/count.txt
// RUN: FileCheck --input-file %T/Memory/api_test39_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Memory/api_test39_out

// CHECK: 36
// TEST_FEATURE: Memory_async_dpct_memcpy
// TEST_FEATURE: Memory_async_dpct_memcpy_2d
// TEST_FEATURE: Memory_async_dpct_memcpy_3d

int main() {
  cudaMemcpy3DParms parms;
  cudaMemcpy3DAsync(&parms);
  return 0;
}
