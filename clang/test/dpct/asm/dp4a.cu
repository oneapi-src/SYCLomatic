// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -out-root %T/dp4a %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/dp4a/dp4a.dp.cpp

// clang-format off
#include <cuda_runtime.h>

__global__ void f() {
    int a = 0;
    // CHECK: a = dpct::dp4a<uint32_t, uint32_t>(UINT_MAX, UINT_MAX, 3);
    // CHECK: a = dpct::dp4a<int32_t, int32_t>(UINT_MAX, UINT_MAX, 3);
    // CHECK: a = dpct::dp4a<int32_t, uint32_t>(UINT_MAX, UINT_MAX, 3);
    // CHECK: a = dpct::dp4a<uint32_t, int32_t>(UINT_MAX, UINT_MAX, 3);
    asm("dp4a.u32.u32 %0, %1, %2, %3;" : "=r"(a) : "r"(UINT_MAX), "r"(UINT_MAX), "r"(3));
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(a) : "r"(UINT_MAX), "r"(UINT_MAX), "r"(3));
    asm("dp4a.s32.u32 %0, %1, %2, %3;" : "=r"(a) : "r"(UINT_MAX), "r"(UINT_MAX), "r"(3));
    asm("dp4a.u32.s32 %0, %1, %2, %3;" : "=r"(a) : "r"(UINT_MAX), "r"(UINT_MAX), "r"(3));
}
