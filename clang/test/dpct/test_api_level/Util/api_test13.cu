// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Util/api_test13_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Util/api_test13_out/MainSourceFiles.yaml | wc -l > %T/Util/api_test13_out/count.txt
// RUN: FileCheck --input-file %T/Util/api_test13_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Util/api_test13_out

// CHECK: 2
// TEST_FEATURE: Util_ffs

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
  