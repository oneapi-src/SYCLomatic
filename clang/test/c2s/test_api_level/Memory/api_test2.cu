// RUN: c2s --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Memory/api_test2_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Memory/api_test2_out/MainSourceFiles.yaml | wc -l > %T/Memory/api_test2_out/count.txt
// RUN: FileCheck --input-file %T/Memory/api_test2_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Memory/api_test2_out

// CHECK: 30
// TEST_FEATURE: Memory_c2s_malloc_3d
// TEST_FEATURE: Memory_c2s_malloc_2d

int main() {
  cudaExtent extent = make_cudaExtent(1, 1, 1);
  cudaPitchedPtr p3;
  cudaMalloc3D(&p3, extent);
  return 0;
}
