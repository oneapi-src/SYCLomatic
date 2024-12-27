// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none -out-root %T/prefetchu_default %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/prefetchu_default/prefetchu_default.dp.cpp
// RUN: %if BUILD_LIT %{icpx -c -DBUILD_TEST -fsycl %T/prefetchu_default/prefetchu_default.dp.cpp -o %T/prefetchu_default/prefetchu_default.dp.o %}

// clang-format off
#include <cuda_runtime.h>

// Unsupported syntax:
// prefetchu.L1 [a];                               // prefetch to uniform cache

__global__ void prefetchu(int *arr) {
#ifndef BUILD_TEST
  /* prefetch of uniform address space */
  // CHECK: /*
  // CHECK-NEXT: DPCT1053:{{.*}} Migration of device assembly code is not supported.
  // CHECK-NEXT: */
  asm volatile ("prefetchu.L1 [%0];" : : "l"(arr));
#endif // BUILD_TEST
}

// clang-format on
