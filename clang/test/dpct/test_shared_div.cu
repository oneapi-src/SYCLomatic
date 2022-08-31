// RUN: dpct  -out-root %T/test_inc_included %s --cuda-include-path="%cuda-path/include" -extra-arg="-I  %S/inc"   -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/test_inc_included/test_inc_included.dp.cpp


__global__ void test() {
    __shared__ int idxInLev;
    int a = 1/idxInLev;
}
