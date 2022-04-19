// RUN: c2s --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Memory/api_test35_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Memory/api_test35_out/MainSourceFiles.yaml | wc -l > %T/Memory/api_test35_out/count.txt
// RUN: FileCheck --input-file %T/Memory/api_test35_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Memory/api_test35_out

// CHECK: 30
// TEST_FEATURE: Memory_c2s_memcpy

int main() {
  float constData[1234567 * 4];
  float* h_A;
  int size;
  cudaMemcpyToSymbol(constData, h_A, size, 1, cudaMemcpyHostToDevice);
  return 0;
}
