// RUN: c2s --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Util/api_test16_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Util/api_test16_out/MainSourceFiles.yaml | wc -l > %T/Util/api_test16_out/count.txt
// RUN: FileCheck --input-file %T/Util/api_test16_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Util/api_test16_out

// CHECK: 2
// TEST_FEATURE: Util_shift_sub_group_left

__device__ void foo1() {
    int a;
    int result;
    unsigned mask;
    int delta;
    result = __shfl_down(a, delta);

    result = __shfl_down_sync(mask, a, delta);
}

int main() {
    return 0;
}