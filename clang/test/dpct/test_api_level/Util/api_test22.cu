// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Util/api_test22_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Util/api_test22_out/MainSourceFiles.yaml | wc -l > %T/Util/api_test22_out/count.txt
// RUN: FileCheck --input-file %T/Util/api_test22_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Util/api_test22_out

// CHECK: 2
// TEST_FEATURE: Util_reverse_bits

__device__ void foo() {
  unsigned u;
  u = __brevll(u);
}

int main() {
  return 0;
}
