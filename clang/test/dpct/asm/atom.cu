// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -out-root %T/atom %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/atom/atom.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/atom/atom.dp.cpp -o %T/atom/atom.dp.o %}

// clang-format off
#include <cuda_runtime.h>
#include <cstdint>

/*
.space =              { .global, .shared{::cta, ::cluster} };
.type =               { .b32, .b64, .u32, .u64, .s32, .s64, .f32, .f64 };
.scope =              { .cta, .cluster, .gpu, .sys };

Current only support the form likes "atom.scope.space.type" now.

*/

__global__ void atom(int *a) {
  int d = 0;
  // CHECK: d = dpct::atomic_fetch_add(a, 1);
  asm volatile ("atom.s32.add %0, [%1], %2;" : "=r"(d) : "l"(a), "r"(1));

  // CHECK: d = dpct::atomic_fetch_min(a, 1);
  asm volatile ("atom.s32.min %0, [%1], %2;" : "=r"(d) : "l"(a), "r"(1));

  // CHECK: d = dpct::atomic_fetch_max(a, 1);
  asm volatile ("atom.s32.max %0, [%1], %2;" : "=r"(d) : "l"(a), "r"(1));

  // CHECK: d = dpct::atomic_fetch_add(a, 1);
  asm volatile ("atom.global.s32.add %0, [%1], %2;" : "=r"(d) : "l"(a), "r"(1));

  // CHECK: d = dpct::atomic_fetch_min(a, 1);
  asm volatile ("atom.global.s32.min %0, [%1], %2;" : "=r"(d) : "l"(a), "r"(1));

  // CHECK: d = dpct::atomic_fetch_max(a, 1);
  asm volatile ("atom.global.s32.max %0, [%1], %2;" : "=r"(d) : "l"(a), "r"(1));

  // CHECK: d = dpct::atomic_fetch_add(a, 1);
  asm volatile ("atom.shared.s32.add %0, [%1], %2;" : "=r"(d) : "l"(a), "r"(1));

  // CHECK: d = dpct::atomic_fetch_min(a, 1);
  asm volatile ("atom.shared.s32.min %0, [%1], %2;" : "=r"(d) : "l"(a), "r"(1));

  // CHECK: d = dpct::atomic_fetch_max(a, 1);
  asm volatile ("atom.shared.s32.max %0, [%1], %2;" : "=r"(d) : "l"(a), "r"(1));
}

__global__ void shared_address_atomic_fetch_add32(uint32_t addr, uint32_t n,
                                                  uint32_t o) {
  uint32_t __addr = (addr);
  uint32_t __n = (n);
  uint32_t __o = (o);
  // CHECK: {
  // CHECK-NEXT:   __o = dpct::atomic_fetch_add(((uint32_t *)(uintptr_t)__addr), __n);
  // CHECK-NEXT: }
  asm volatile("{atom.cta.shared.add.u32 %0, [%1], %2;}"
               : "=r"(__o)
               : "r"(__addr), "r"(__n)
               : "memory");
}

// clang-format on
