// RUN: c2s --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Util/api_test14_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Util/api_test14_out/MainSourceFiles.yaml | wc -l > %T/Util/api_test14_out/count.txt
// RUN: FileCheck --input-file %T/Util/api_test14_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Util/api_test14_out

// CHECK: 2
// TEST_FEATURE: Util_select_from_sub_group

__device__ void foo1() {
    int a;
    int result;
    unsigned mask;
    int src;
    result = __shfl(a, src);

    result = __shfl_sync(mask, a, src);
}

int main() {
    return 0;
}