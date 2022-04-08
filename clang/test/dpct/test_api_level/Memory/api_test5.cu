// RUN: c2s --format-range=none  --use-custom-helper=api -out-root %T/Memory/api_test5_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Memory/api_test5_out/MainSourceFiles.yaml | wc -l > %T/Memory/api_test5_out/count.txt
// RUN: FileCheck --input-file %T/Memory/api_test5_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Memory/api_test5_out

// CHECK: 16
// TEST_FEATURE: Memory_c2s_malloc

int main() {
  float* a;
  cudaMalloc(&a, 4);
  return 0;
}
