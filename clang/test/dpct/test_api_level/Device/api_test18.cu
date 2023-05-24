// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Device/api_test18_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Device/api_test18_out/MainSourceFiles.yaml | wc -l > %T/Device/api_test18_out/count.txt
// RUN: FileCheck --input-file %T/Device/api_test18_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Device/api_test18_out

// CHECK: 18
// TEST_FEATURE: Device_get_default_queue

__global__ void foo() {}

void f() { foo<<<1, 1>>>(); }

int main() {
  foo<<<1,1>>>();
  return 0;
}
