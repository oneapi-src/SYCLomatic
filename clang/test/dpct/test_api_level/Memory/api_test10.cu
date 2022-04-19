// RUN: c2s --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Memory/api_test10_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Memory/api_test10_out/MainSourceFiles.yaml | wc -l > %T/Memory/api_test10_out/count.txt
// RUN: FileCheck --input-file %T/Memory/api_test10_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Memory/api_test10_out

// CHECK: 3
// TEST_FEATURE: Memory_pitched_data

int main() {
  cudaPitchedPtr a;
  return 0;
}
