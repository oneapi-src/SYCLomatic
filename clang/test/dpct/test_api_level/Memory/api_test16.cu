// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Memory/api_test16_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Memory/api_test16_out/MainSourceFiles.yaml | wc -l > %T/Memory/api_test16_out/count.txt
// RUN: FileCheck --input-file %T/Memory/api_test16_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Memory/api_test16_out

// CHECK: 48
// TEST_FEATURE: Memory_constant_memory_alias

__constant__ float A[1024];

int main() {
  return 0;
}
