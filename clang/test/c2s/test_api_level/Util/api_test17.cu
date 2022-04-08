// RUN: c2s --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Util/api_test17_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Util/api_test17_out/MainSourceFiles.yaml | wc -l > %T/Util/api_test17_out/count.txt
// RUN: FileCheck --input-file %T/Util/api_test17_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Util/api_test17_out

// CHECK: 2
// TEST_FEATURE: Util_permute_sub_group_by_xor

__device__ void foo1() {
    int a;
    int result;
    unsigned mask;
    int lanemask;
    result = __shfl_xor(a, lanemask);

    result = __shfl_xor_sync(mask, a, lanemask);
}

int main() {
    return 0;
}