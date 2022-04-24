// RUN: dpct --format-range=none   --use-custom-helper=api -out-root %T/Memory/api_test3_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Memory/api_test3_out/MainSourceFiles.yaml | wc -l > %T/Memory/api_test3_out/count.txt
// RUN: FileCheck --input-file %T/Memory/api_test3_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Memory/api_test3_out

// CHECK: 24
// TEST_FEATURE: Memory_dpct_malloc_3d
// TEST_FEATURE: Memory_dpct_malloc_2d

int main() {
  cudaExtent extent = make_cudaExtent(1, 1, 1);
  cudaPitchedPtr p3;
  cudaMalloc3D(&p3, extent);
  return 0;
}
