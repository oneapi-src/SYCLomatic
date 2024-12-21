// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.1, cuda-11.2, cuda-11.3, cuda-11.4, cuda-11.5, cuda-11.6, cuda-11.7
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0, v11.1, v11.2, v11.3, v11.4, v11.5, v11.6, v11.7
// RUN: dpct --format-range=none -out-root %T/prefetch %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
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
prefetchu.L1 [a];                               // prefetch to uniform cache
prefetch{.tensormap_space}.tensormap [a];       // prefetch the tensormap

.level =                    { .L1, .L2 };
.level::eviction_priority = { .L2::evict_last, .L2::evict_normal };
.tensormap_space =          { .const, .param };
*/

__global__ void prefetch(int *arr) {
  // CHECK: sycl::ext::oneapi::experimental::prefetch(arr, sycl::ext::oneapi::experimental::properties{sycl::ext::oneapi::experimental::prefetch_hint_L1});
  asm volatile ("prefetch.L1 [%0];" : : "l"(arr));
  // CHECK: sycl::ext::oneapi::experimental::prefetch(arr, sycl::ext::oneapi::experimental::properties{sycl::ext::oneapi::experimental::prefetch_hint_L2});
  asm volatile ("prefetch.L2 [%0];" : : "l"(arr));
#ifndef BUILD_TEST
  // CHECK: /*
  // CHECK-NEXT: DPCT1053:{{.*}} Migration of device assembly code is not supported.
  // CHECK-NEXT: */
  asm volatile ("prefetch.L2::evict_last [%0];" : : "l"(arr));
  // CHECK: /*
  // CHECK-NEXT: DPCT1053:{{.*}} Migration of device assembly code is not supported.
  // CHECK-NEXT: */
  asm volatile ("prefetch.L2::evict_normal [%0];" : : "l"(arr));
#endif // BUILD_TEST

  // CHECK: sycl::ext::oneapi::experimental::prefetch(arr, sycl::ext::oneapi::experimental::properties{sycl::ext::oneapi::experimental::prefetch_hint_L1});
  asm volatile ("prefetch.global.L1 [%0];" : : "l"(arr));
  // CHECK: sycl::ext::oneapi::experimental::prefetch(arr, sycl::ext::oneapi::experimental::properties{sycl::ext::oneapi::experimental::prefetch_hint_L2});
  asm volatile ("prefetch.global.L2 [%0];" : : "l"(arr));
#ifndef BUILD_TEST
  // CHECK: /*
  // CHECK-NEXT: DPCT1053:{{.*}} Migration of device assembly code is not supported.
  // CHECK-NEXT: */
  asm volatile ("prefetch.global.L2::evict_last [%0];" : : "l"(arr));
  // CHECK: /*
  // CHECK-NEXT: DPCT1053:{{.*}} Migration of device assembly code is not supported.
  // CHECK-NEXT: */
  asm volatile ("prefetch.global.L2::evict_normal [%0];" : : "l"(arr));

  // CHECK: /*
  // CHECK-NEXT: DPCT1053:{{.*}} Migration of device assembly code is not supported.
  // CHECK-NEXT: */
  asm volatile ("prefetch.local.L1 [%0];" : : "l"(arr));
  // CHECK: /*
  // CHECK-NEXT: DPCT1053:{{.*}} Migration of device assembly code is not supported.
  // CHECK-NEXT: */
  asm volatile ("prefetch.local.L2 [%0];" : : "l"(arr));
  // CHECK: /*
  // CHECK-NEXT: DPCT1053:{{.*}} Migration of device assembly code is not supported.
  // CHECK-NEXT: */
  asm volatile ("prefetch.local.L2::evict_last [%0];" : : "l"(arr));
  // CHECK: /*
  // CHECK-NEXT: DPCT1053:{{.*}} Migration of device assembly code is not supported.
  // CHECK-NEXT: */
  asm volatile ("prefetch.local.L2::evict_normal [%0];" : : "l"(arr));

  // CHECK: /*
  // CHECK-NEXT: DPCT1053:{{.*}} Migration of device assembly code is not supported.
  // CHECK-NEXT: */
  asm volatile ("prefetchu.L1 [%0];" : : "l"(arr));

  // CHECK: /*
  // CHECK-NEXT: DPCT1053:{{.*}} Migration of device assembly code is not supported.
  // CHECK-NEXT: */
  asm volatile ("prefetch.const.tensormap [%0];" : : "l"(arr));
  // CHECK: /*
  // CHECK-NEXT: DPCT1053:{{.*}} Migration of device assembly code is not supported.
  // CHECK-NEXT: */
  asm volatile ("prefetch.param.tensormap [%0];" : : "l"(arr));
#endif // BUILD_TEST
}

// clang-format on
