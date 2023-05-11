// RUN: dpct --format-range=none -out-root %T/recursive_function %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/recursive_function/recursive_function.dp.cpp
#include <cuda.h>
// CHECK: /*
// CHECK-NEXT: DPCT1109:{{[0-9]+}}: Recursive functions cannot be called in a SYCL kernel or by functions called by the kernel. You may need to adjust the code.
// CHECK-NEXT: */
__device__ int factorial(int n) {
    if (n <= 1) {
        return 1;
    } else {
        // CHECK: /*
        // CHECK-NEXT: DPCT1109:{{[0-9]+}}: Recursive functions cannot be called in a SYCL kernel or by functions called by the kernel. You may need to adjust the code.
        // CHECK-NEXT: */
        return n * factorial(n - 1);
    }
}

__global__ void test_kernel() {
    factorial(10);
}


int factorial2(int n) {
    if (n <= 1) {
        return 1;
    } else {
        return n * factorial2(n - 1);
    }
}

int main() {
    test_kernel<<<1,1>>>();
}