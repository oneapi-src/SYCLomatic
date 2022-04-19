// RUN: c2s --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Memory/api_test27_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Memory/api_test27_out/MainSourceFiles.yaml | wc -l > %T/Memory/api_test27_out/count.txt
// RUN: FileCheck --input-file %T/Memory/api_test27_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Memory/api_test27_out

// CHECK: 51
// TEST_FEATURE: Memory_device_memory_get_ptr

__constant__ float constData[4];

int main() {
  float* host;
  cudaMemcpyToSymbolAsync(constData, host, 1, 3, cudaMemcpyHostToDevice, 0);
  return 0;
}
