// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -out-root %T/dp2a %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/dp2a/dp2a.dp.cpp

// clang-format off
#include <cuda_runtime.h>

__global__ void f() {
    int a = 0;
    // CHECK: a = dpct::dp2a_lo<uint32_t, uint32_t>(UINT_MAX, UINT_MAX, 3);
    // CHECK: a = dpct::dp2a_lo<int32_t, int32_t>(UINT_MAX, UINT_MAX, 3);
    // CHECK: a = dpct::dp2a_lo<int32_t, uint32_t>(UINT_MAX, UINT_MAX, 3);
    // CHECK: a = dpct::dp2a_lo<uint32_t, int32_t>(UINT_MAX, UINT_MAX, 3);
    // CHECK: a = dpct::dp2a_hi<uint32_t, uint32_t>(UINT_MAX, UINT_MAX, 3);
    // CHECK: a = dpct::dp2a_hi<int32_t, int32_t>(UINT_MAX, UINT_MAX, 3);
    // CHECK: a = dpct::dp2a_hi<int32_t, uint32_t>(UINT_MAX, UINT_MAX, 3);
    // CHECK: a = dpct::dp2a_hi<uint32_t, int32_t>(UINT_MAX, UINT_MAX, 3);
    asm("dp2a.lo.u32.u32 %0, %1, %2, %3;" : "=r"(a) : "r"(UINT_MAX), "r"(UINT_MAX), "r"(3));
    asm("dp2a.lo.s32.s32 %0, %1, %2, %3;" : "=r"(a) : "r"(UINT_MAX), "r"(UINT_MAX), "r"(3));
    asm("dp2a.lo.s32.u32 %0, %1, %2, %3;" : "=r"(a) : "r"(UINT_MAX), "r"(UINT_MAX), "r"(3));
    asm("dp2a.lo.u32.s32 %0, %1, %2, %3;" : "=r"(a) : "r"(UINT_MAX), "r"(UINT_MAX), "r"(3));
    asm("dp2a.hi.u32.u32 %0, %1, %2, %3;" : "=r"(a) : "r"(UINT_MAX), "r"(UINT_MAX), "r"(3));
    asm("dp2a.hi.s32.s32 %0, %1, %2, %3;" : "=r"(a) : "r"(UINT_MAX), "r"(UINT_MAX), "r"(3));
    asm("dp2a.hi.s32.u32 %0, %1, %2, %3;" : "=r"(a) : "r"(UINT_MAX), "r"(UINT_MAX), "r"(3));
    asm("dp2a.hi.u32.s32 %0, %1, %2, %3;" : "=r"(a) : "r"(UINT_MAX), "r"(UINT_MAX), "r"(3));
}
