// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -out-root %T/st %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/st/st.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/st/st.dp.cpp -o %T/st/st.dp.o %}

// clang-format off
#include <cuda_runtime.h>

/*

st{.weak}{.ss}{.cop}{.level::cache_hint}{.vec}.type   [a], b{, cache-policy};
st{.weak}{.ss}{.level::eviction_priority}{.level::cache_hint}{.vec}.type
                                                      [a], b{, cache-policy};
st.volatile{.ss}{.vec}.type                           [a], b;
st.relaxed.scope{.ss}{.level::eviction_priority}{.level::cache_hint}{.vec}.type
                                                      [a], b{, cache-policy};
st.release.scope{.ss}{.level::eviction_priority}{.level::cache_hint}{.vec}.type
                                                      [a], b{, cache-policy};
st.mmio.relaxed.sys{.global}.type         [a], b;

.ss =                       { .global, .local, .param{::func}, .shared{::cta, ::cluster} };
.level::eviction_priority = { .L1::evict_normal, .L1::evict_unchanged,
                              .L1::evict_first, .L1::evict_last, .L1::no_allocate };
.level::cache_hint =        { .L2::cache_hint };
.cop =                      { .wb, .cg, .cs, .wt };
.sem =                      { .relaxed, .release };
.scope =                    { .cta, .cluster, .gpu, .sys };
.vec =                      { .v2, .v4 };
.type =                     { .b8, .b16, .b32, .b64, .b128,
                              .u8, .u16, .u32, .u64,
                              .s8, .s16, .s32, .s64,
                              .f32, .f64 };

We only support the form likes "st.ss.type" now.

*/

__global__ void st(int *a) {
  // CHECK: *a = 111;
  asm volatile ("st.global.s32 [%0], %1;" :: "l"(a), "r"(111));
  // CHECK: *((uint32_t *)(uintptr_t)a) = 111;
  asm volatile ("st.global.u32 [%0], %1;" :: "l"(a), "r"(111));
  // CHECK: *((uint32_t *)((uintptr_t)a + 4)) = 222;
  asm volatile ("st.global.u32 [%0 + 4], %1;" :: "l"(a), "r"(222));
  // CHECK: *((uint64_t *)((uintptr_t)a + 8)) = 0ull;
  asm volatile ("st.global.u64 [%0 + 8], %1;" :: "l"(a), "l"(0ull));
}

__device__ void shared_address_store32(uint32_t addr, uint32_t val) {
  uint32_t __addr = (addr);
  uint32_t __val = (val);
  // CHECK: {
  // CHECK:   *((uint32_t *)(uintptr_t)__addr) = __val;
  // CHECK: } 
  asm volatile("{st.shared.b32 [%0], %1;}" : : "r"(__addr), "r"(__val) : "memory"); 
}

// clang-format on
