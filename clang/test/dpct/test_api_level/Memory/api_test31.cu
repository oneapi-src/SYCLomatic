// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Memory/api_test31_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Memory/api_test31_out/MainSourceFiles.yaml | wc -l > %T/Memory/api_test31_out/count.txt
// RUN: FileCheck --input-file %T/Memory/api_test31_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Memory/api_test31_out

// CHECK: 35
// TEST_FEATURE: Memory_get_host_ptr

int main() {
  float* a;
  float b = 0;

  cudaMallocManaged(&a, 10 * sizeof(float));

  b = a[2];
  return 0;
}
