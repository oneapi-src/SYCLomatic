// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -out-root %T/ld %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/ld/ld.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/ld/ld.dp.cpp -o %T/ld/ld.dp.o %}

// clang-format off
#include <cuda_runtime.h>

/*

ld{.weak}{.ss}{.cop}{.level::cache_hint}{.level::prefetch_size}{.vec}.type  d, [a]{.unified}{, cache-policy};

ld{.weak}{.ss}{.level::eviction_priority}{.level::cache_hint}{.level::prefetch_size}{.vec}.type  d, [a]{.unified}{, cache-policy};

ld.volatile{.ss}{.level::prefetch_size}{.vec}.type  d, [a];

ld.relaxed.scope{.ss}{.level::eviction_priority}{.level::cache_hint}{.level::prefetch_size}{.vec}.type  d, [a]{, cache-policy};

ld.acquire.scope{.ss}{.level::eviction_priority}{.level::cache_hint}{.level::prefetch_size}{.vec}.type  d, [a]{, cache-policy};

ld.mmio.relaxed.sys{.global}.type  d, [a];

.ss =                       { .const, .global, .local, .param{::entry, ::func}, .shared{::cta, ::cluster} };
.cop =                      { .ca, .cg, .cs, .lu, .cv };
.level::eviction_priority = { .L1::evict_normal, .L1::evict_unchanged,
                              .L1::evict_first, .L1::evict_last, .L1::no_allocate };
.level::cache_hint =        { .L2::cache_hint };
.level::prefetch_size =     { .L2::64B, .L2::128B, .L2::256B }
.scope =                    { .cta, .cluster, .gpu, .sys };
.vec =                      { .v2, .v4 };
.type =                     { .b8, .b16, .b32, .b64, .b128,
                              .u8, .u16, .u32, .u64,
                              .s8, .s16, .s32, .s64,
                              .f32, .f64 };

We only support the form likes "ld.ss.type" now.

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

// clang-format on
