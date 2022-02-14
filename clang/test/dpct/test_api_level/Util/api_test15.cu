// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Util/api_test15_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Util/api_test15_out/MainSourceFiles.yaml | wc -l > %T/Util/api_test15_out/count.txt
// RUN: FileCheck --input-file %T/Util/api_test15_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Util/api_test15_out

// CHECK: 2
// TEST_FEATURE: Util_shift_sub_group_right

__device__ void foo1() {
    int a;
    int result;
    unsigned mask;
    int delta;
    result = __shfl_up(a, delta);

    result = __shfl_up_sync(mask, a, delta);
}

int main() {
    return 0;
}