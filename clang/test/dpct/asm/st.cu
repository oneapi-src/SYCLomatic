// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -out-root %T/st %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/st/st.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/st/st.dp.cpp -o %T/st/st.dp.o %}

// clang-format off
#include <cuda_runtime.h>
#include <cstdint>

__global__ void st(int *a) {
  // CHECK: *((uint32_t *)(uintptr_t)a) = 111;
  asm volatile ("st.global.u32 [%0], %1;" :: "l"(a), "r"(111));
  // CHECK: *((uint32_t *)((uintptr_t)a + 4)) = 222;
  asm volatile ("st.global.u32 [%0 + 4], %1;" :: "l"(a), "r"(222));
  // CHECK: *((uint64_t *)((uintptr_t)a + 8)) = 0ull;
  asm volatile ("st.global.u64 [%0 + 8], %1;" :: "l"(a), "l"(0ull));
}

// clang-format on
