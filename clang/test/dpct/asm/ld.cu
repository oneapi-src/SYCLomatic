// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -out-root %T/ld %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/ld/ld.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/ld/ld.dp.cpp -o %T/ld/ld.dp.o %}

// clang-format off
#include <cuda_runtime.h>

/*
.ss =                       { .const, .global, .local, .param, .shared };
.type =                     { .b8, .b16, .b32, .b64, .b128, 
                              .u8, .u16, .u32, .u64,
                              .s8, .s16, .s32, .s64,
                              .f32, .f64 };
Current only support the form likes "ld.ss.type" now.
*/

__global__ void ld(int *arr) {
  int a, b, c;
  unsigned long long d;
  // CHECK: a = *arr;
  asm volatile ("ld.global.s32 %0, [%1];" : "=r"(a) : "l"(arr));
  // CHECK: b = *((uint32_t *)(uintptr_t)arr);
  asm volatile ("ld.global.u32 %0, [%1];" : "=r"(b) : "l"(arr));
  // CHECK: c = *((uint32_t *)((uintptr_t)arr + 4));
  asm volatile ("ld.global.u32 %0, [%1 + 4];" : "=r"(c) : "l"(arr));
  // CHECK: d = *((uint64_t *)((uintptr_t)arr + 8));
  asm volatile ("ld.global.u64 %0, [%1 + 8];" : "=l"(d) : "l"(arr));
}

__device__ void shared_address_load32(uint32_t addr, uint32_t &val) {
  // CHECK: {
  // CHECK:   val = *((uint32_t *)(uintptr_t)addr);
  // CHECK: } 
  asm volatile("{ld.shared.b32 %0, [%1];}" : : "r"(val), "r"(addr) : "memory"); 
}

// CHECK: inline void load_global_short2(sycl::short2 &a, const sycl::short2 *addr) {
// CHECK-NEXT:  short x, y, z, w;
// CHECK-NEXT:  x  = *((int16_t *)(uintptr_t)addr + 0);
// CHECK-NEXT:  y  = *((int16_t *)(uintptr_t)addr + 1);
// CHECK-NEXT:  a.x() = x;
// CHECK-NEXT:  a.y() = y;
// CHECK-NEXT:}
__device__ inline void load_global_short2(short2 &a, const short2 *addr) {
  short x, y, z, w;
  asm("ld.cg.global.v2.s16 {%0, %1}, [%2+0];" : "=h"(x), "=h"(y) : "l"(addr));
  a.x = x;
  a.y = y;
}

// CHECK: inline void load_global_short4(sycl::short4 &a, const sycl::short4 *addr) {
// CHECK-NEXT:  short x, y, z, w;
// CHECK-NEXT:  x  = *((int16_t *)(uintptr_t)addr + 0);
// CHECK-NEXT:  y  = *((int16_t *)(uintptr_t)addr + 1);
// CHECK-NEXT:  z  = *((int16_t *)(uintptr_t)addr + 2);
// CHECK-NEXT:  w  = *((int16_t *)(uintptr_t)addr + 3);
// CHECK-NEXT:  a.x() = x;
// CHECK-NEXT:  a.y() = y;
// CHECK-NEXT:  a.z() = z;
// CHECK-NEXT:  a.w() = w;
// CHECK-NEXT:}
__device__ inline void load_global_short4(short4 &a, const short4 *addr) {
  short x, y, z, w;
  asm("ld.cg.global.v4.s16 {%0, %1, %2, %3}, [%4+0];" : "=h"(x), "=h"(y), "=h"(z), "=h"(w) : "l"(addr));
  a.x = x;
  a.y = y;
  a.z = z;
  a.w = w;
}

// clang-format on
