// RUN: rm -rf %T/Util/api_test26_out
// RUN: dpct --use-custom-helper=api --out-root=%T/Util/api_test26_out \
// RUN:      --cuda-include-path="%cuda-path/include" \
// RUN:      %s -- -x cuda -ptx
// RUN: grep "args_selector:" %T/Util/api_test26_out/MainSourceFiles.yaml \
// RUN: | python -c "len(input().splitlines()) == 1"

// TEST_FEATURE: Util_args_selector

extern "C" __global__ void kernel(int *x, short y) {}
