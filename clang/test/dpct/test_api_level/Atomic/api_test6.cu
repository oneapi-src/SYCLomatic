// RUN: c2s --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Atomic/api_test6_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Atomic/api_test6_out/MainSourceFiles.yaml | wc -l > %T/Atomic/api_test6_out/count.txt
// RUN: FileCheck --input-file %T/Atomic/api_test6_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Atomic/api_test6_out

// CHECK: 2
// TEST_FEATURE: Atomic_atomic_fetch_xor

__global__ void test(int *data) {
  int inc = 1;


  atomicXor(&data[10], inc);

}
int main() {
  return 0;
}
