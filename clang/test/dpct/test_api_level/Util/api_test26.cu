// RUN: rm -rf %T/Util/api_test26_out
// RUN: dpct --use-custom-helper=api --out-root=%T/Util/api_test26_out \
// RUN:      --cuda-include-path="%cuda-path/include" \
// RUN:      %s -- -x cuda -ptx
// RUN: [ 1 -eq $(grep "get_nth_parameter:" %T/Util/api_test26_out/MainSourceFiles.yaml | wc -l) ]
// RUN: [ 1 -eq $(grep "get_args_ptr:" %T/Util/api_test26_out/MainSourceFiles.yaml | wc -l) ]

// TEST_FEATURE: Util_get_args_ptr
// TEST_FEATURE: Util_get_nth_parameter

extern "C" __global__ void kernel(int *x, short y) {}
