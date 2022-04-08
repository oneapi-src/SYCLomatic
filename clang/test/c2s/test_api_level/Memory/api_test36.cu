// RUN: c2s --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Memory/api_test36_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Memory/api_test36_out/MainSourceFiles.yaml | wc -l > %T/Memory/api_test36_out/count.txt
// RUN: FileCheck --input-file %T/Memory/api_test36_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Memory/api_test36_out

// CHECK: 40
// TEST_FEATURE: Memory_c2s_memcpy
// TEST_FEATURE: Memory_c2s_memcpy_2d
// TEST_FEATURE: Memory_c2s_memcpy_3d

int main() {
  cudaMemcpy3DParms parms;
  cudaMemcpy3D(&parms);
  return 0;
}
