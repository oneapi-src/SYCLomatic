// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Atomic/api_test8_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Atomic/api_test8_out/MainSourceFiles.yaml | wc -l > %T/Atomic/api_test8_out/count.txt
// RUN: FileCheck --input-file %T/Atomic/api_test8_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Atomic/api_test8_out

// CHECK: 2
// TEST_FEATURE: Atomic_atomic_fetch_max

__global__ void test(int *data) {
  int inc = 1;



  atomicMax(&data[3], inc);

}
int main() {
  return 0;
}
