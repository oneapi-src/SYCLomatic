// RUN: rm -rf %T/Util/api_test28_out
// RUN: dpct --use-custom-helper=api --out-root=%T/Util/api_test28_out \
// RUN:      --cuda-include-path="%cuda-path/include" \
// RUN:      %s -- -x cuda -ptx
// RUN: grep "err_types:" %T/Util/api_test28_out/MainSourceFiles.yaml \
// RUN: | python -c "assert len(input().splitlines()) == 1"


// TEST_FEATURE: Util_err_types

void f() {
  cudaError_t x;
}
