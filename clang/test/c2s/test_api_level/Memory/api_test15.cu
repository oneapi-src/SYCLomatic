// RUN: c2s --format-range=none  --use-custom-helper=api -out-root %T/Memory/api_test15_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Memory/api_test15_out/MainSourceFiles.yaml | wc -l > %T/Memory/api_test15_out/count.txt
// RUN: FileCheck --input-file %T/Memory/api_test15_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Memory/api_test15_out

// CHECK: 42
// TEST_FEATURE: Memory_global_memory_alias

__device__ float A[1024];

int main() {
  return 0;
}
