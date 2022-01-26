// RUN: echo
#include "cuda_runtime.h"

namespace {
// CHECK: void test_device() {}
__device__ void test_device() {}
}

template<typename T>
__global__ void test_global2(T a) {

    test_device();

}

int host_func() {
    int a;
    test_global2<<<1,1>>>(a);
}