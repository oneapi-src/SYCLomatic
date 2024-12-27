// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.1, cuda-11.2, cuda-11.3
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0, v11.1, v11.2, v11.3
// RUN: dpct --format-range=none -out-root %T/prefetch_eviction_priority %s --use-experimental-features=prefetch --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/prefetch_eviction_priority/prefetch_eviction_priority.dp.cpp
// RUN: %if BUILD_LIT %{icpx -c -DBUILD_TEST -fsycl %T/prefetch_eviction_priority/prefetch_eviction_priority.dp.cpp -o %T/prefetch_eviction_priority/prefetch_eviction_priority.dp.o %}

// clang-format off
#include <cuda_runtime.h>

// Unsupported syntax:
// prefetch.global.level::eviction_priority [a];   // prefetch to data cache
// .level::eviction_priority = { .L2::evict_last, .L2::evict_normal };

__global__ void prefetch(int *arr) {
#ifndef BUILD_TEST
  /* prefetch of global address space with eviction priority */
  // CHECK: /*
  // CHECK-NEXT: DPCT1007:{{.*}} Migration of prefetch.global.L2::evict_last [%0]; is not supported.
  // CHECK-NEXT: */
  asm volatile ("prefetch.global.L2::evict_last [%0];" : : "l"(arr));
  // CHECK: /*
  // CHECK-NEXT: DPCT1007:{{.*}} Migration of prefetch.global.L2::evict_normal [%0]; is not supported.
  // CHECK-NEXT: */
  asm volatile ("prefetch.global.L2::evict_normal [%0];" : : "l"(arr));
#endif // BUILD_TEST
}

// clang-format on
