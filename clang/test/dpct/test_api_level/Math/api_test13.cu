// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Math/api_test13_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Math/api_test13_out/MainSourceFiles.yaml | wc -l > %T/Math/api_test13_out/count.txt
// RUN: FileCheck --input-file %T/Math/api_test13_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Math/api_test13_out

// CHECK: 2
// TEST_FEATURE: Math_ffs

__device__ void foo1() {
    int a;
    long long int b;
    int result;
    result = __ffs(a);
    result = __ffsll(b);
}

int main() {
    return 0;
}
  