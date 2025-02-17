// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none -out-root %T/prefetch %s --use-experimental-features=prefetch --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/prefetch/prefetch.dp.cpp
// RUN: %if BUILD_LIT %{icpx -c -DBUILD_TEST -fsycl %T/prefetch/prefetch.dp.cpp -o %T/prefetch/prefetch.dp.o %}

// clang-format off
#include <cuda_runtime.h>

/*
Supported syntax:
-----------------
prefetch.level [a];                            // prefetch to generic addr space cache
prefetch.global.level [a];                     // prefetch to global cache

Unsupported syntax:
-------------------
prefetch.local.level
prefetch.global.level::eviction_priority [a];   // prefetch to data cache
prefetch{.tensormap_space}.tensormap [a];       // prefetch the tensormap

.level =                    { .L1, .L2 };
.level::eviction_priority = { .L2::evict_last, .L2::evict_normal };
.tensormap_space =          { .const, .param };
*/

__global__ void prefetch(int *arr) {
  /* prefetch of no address space */
  // CHECK: sycl::ext::oneapi::experimental::prefetch(arr, sycl::ext::oneapi::experimental::properties{sycl::ext::oneapi::experimental::prefetch_hint_L1});
  asm volatile ("prefetch.L1 [%0];" : : "l"(arr));
  // CHECK: sycl::ext::oneapi::experimental::prefetch(arr, sycl::ext::oneapi::experimental::properties{sycl::ext::oneapi::experimental::prefetch_hint_L2});
  asm volatile ("prefetch.L2 [%0];" : : "l"(arr));

  /* prefetch of global address space */
  // CHECK: sycl::ext::oneapi::experimental::prefetch(arr, sycl::ext::oneapi::experimental::properties{sycl::ext::oneapi::experimental::prefetch_hint_L1});
  asm volatile ("prefetch.global.L1 [%0];" : : "l"(arr));
  // CHECK: sycl::ext::oneapi::experimental::prefetch(arr, sycl::ext::oneapi::experimental::properties{sycl::ext::oneapi::experimental::prefetch_hint_L2});
  asm volatile ("prefetch.global.L2 [%0];" : : "l"(arr));

  /* using Register-Immediate (Displacement) address mode */
  // CHECK: sycl::ext::oneapi::experimental::prefetch(((uint8_t *)((uintptr_t)arr + 2)), sycl::ext::oneapi::experimental::properties{sycl::ext::oneapi::experimental::prefetch_hint_L1});
  asm volatile("prefetch.global.L1 [%0 + 2];" :: "l"(arr));
}

// clang-format on
