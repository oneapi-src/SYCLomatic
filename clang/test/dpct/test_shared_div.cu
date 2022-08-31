// RUN: dpct  -out-root %T/test_shared_div %s --cuda-include-path="%cuda-path/include" -extra-arg="-I  %S/inc"   -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/test_shared_div/test_shared_div.dp.cpp


__global__ void test() {
    __shared__ int idxInLev;
    // CHECK: int a = 1 / (*idxInLev);
    int a = 1/idxInLev;
}
