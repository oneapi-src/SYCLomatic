// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Memory/api_test20_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Memory/api_test20_out/MainSourceFiles.yaml | wc -l > %T/Memory/api_test20_out/count.txt
// RUN: FileCheck --input-file %T/Memory/api_test20_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Memory/api_test20_out

// CHECK: 52
// TEST_FEATURE: Memory_device_memory_get_size

static __device__ float d_A[1234567];

int main() {
  size_t size2;
  cudaGetSymbolSize(&size2, d_A);
  return 0;
}
