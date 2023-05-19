// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none  --usm-level=none --use-experimental-features=masked_sub_group_operation --use-custom-helper=api -out-root %T/Util/api_test29_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Util/api_test29_out/MainSourceFiles.yaml | wc -l > %T/Util/api_test29_out/count.txt
// RUN: FileCheck --input-file %T/Util/api_test29_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Util/api_test29_out

// CHECK: 6
// TEST_FEATURE: Util_select_from_sub_group_mask
// TEST_FEATURE: Util_shift_sub_group_right_mask
// TEST_FEATURE: Util_shift_sub_group_left_mask
// TEST_FEATURE: Util_permute_sub_group_by_xor_mask

__device__ void foo1() {
    int a;
    int result;
    unsigned mask;
    int src;
    result = __shfl_sync(mask, a, src);
}

__device__ void foo2() {
    int a;
    int result;
    unsigned mask;
    int delta;
    result = __shfl_up_sync(mask, a, delta);
}

__device__ void foo3() {
    int a;
    int result;
    unsigned mask;
    int delta;
    result = __shfl_down_sync(mask, a, delta);
}

__device__ void foo4() {
    int a;
    int result;
    unsigned mask;
    int lanemask;
    result = __shfl_xor_sync(mask, a, lanemask);
}

int main() {
    return 0;
}
