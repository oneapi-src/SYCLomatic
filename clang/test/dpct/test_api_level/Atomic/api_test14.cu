// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Atomic/api_test14_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Atomic/api_test14_out/MainSourceFiles.yaml | wc -l > %T/Atomic/api_test14_out/count.txt
// RUN: FileCheck --input-file %T/Atomic/api_test14_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Atomic/api_test14_out

// CHECK: 2
// TEST_FEATURE: Atomic_atomic_fetch_compare_dec

__global__ void test(int *data) {
  int inc = 1;
  atomicInc((unsigned int *)&data[5], (unsigned int)inc);
}
int main() {
  return 0;
}
