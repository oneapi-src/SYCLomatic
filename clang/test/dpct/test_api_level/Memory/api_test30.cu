// RUN: dpct --format-range=none  --use-custom-helper=api -out-root %T/Memory/api_test30_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Memory/api_test30_out/MainSourceFiles.yaml | wc -l > %T/Memory/api_test30_out/count.txt
// RUN: FileCheck --input-file %T/Memory/api_test30_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Memory/api_test30_out

// CHECK: 47
// TEST_FEATURE: Memory_device_memory_get_ptr_q

__constant__ float constData[4];

int main() {
  cudaStream_t stream;
  float* host;
  cudaMemcpyToSymbolAsync(constData, host, 1, 3, cudaMemcpyHostToDevice, stream);
  return 0;
}
