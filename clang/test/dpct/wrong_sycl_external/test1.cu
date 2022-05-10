// RUN: dpct --format-range=none --usm-level=none -in-root %S -out-root %T/wrong_sycl_external %S/test1.cu %S/test2.cu -extra-arg="-I %S" --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %S/test1.cu --match-full-lines --input-file %T/wrong_sycl_external/test1.dp.cpp
// RUN: FileCheck %S/test2.cu --match-full-lines --input-file %T/wrong_sycl_external/test2.dp.cpp
#include "cuda_runtime.h"

namespace {
// CHECK: void test_device() {}
__device__ void test_device() {}
}

template<typename T>
__global__ void test_global1(T a) {

    test_device();

}

int host_func() {
    float a;
    test_global1<<<1,1>>>(a);
}