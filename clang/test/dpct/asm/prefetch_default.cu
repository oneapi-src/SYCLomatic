// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none -out-root %T/prefetch_default %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/prefetch_default/prefetch_default.dp.cpp
// RUN: %if BUILD_LIT %{icpx -c -DBUILD_TEST -fsycl %T/prefetch_default/prefetch_default.dp.cpp -o %T/prefetch_default/prefetch_default.dp.o %}

// clang-format off
#include <cuda_runtime.h>

__global__ void prefetch(int *arr) {
#ifndef BUILD_TEST
  /* prefetch of no address space */
  // CHECK: /*
  // CHECK-NEXT: DPCT1053:{{.*}} Migration of device assembly code is not supported.
  // CHECK-NEXT: */
  asm volatile ("prefetch.L1 [%0];" : : "l"(arr));
  // CHECK: /*
  // CHECK-NEXT: DPCT1053:{{.*}} Migration of device assembly code is not supported.
  // CHECK-NEXT: */
  asm volatile ("prefetch.L2 [%0];" : : "l"(arr));

  /* prefetch of global address space */
  // CHECK: /*
  // CHECK-NEXT: DPCT1053:{{.*}} Migration of device assembly code is not supported.
  // CHECK-NEXT: */
  asm volatile ("prefetch.global.L1 [%0];" : : "l"(arr));
  // CHECK: /*
  // CHECK-NEXT: DPCT1053:{{.*}} Migration of device assembly code is not supported.
  // CHECK-NEXT: */
  asm volatile ("prefetch.global.L2 [%0];" : : "l"(arr));

  /* using Register-Immediate (Displacement) address mode */
  // CHECK: /*
  // CHECK-NEXT: DPCT1053:{{.*}} Migration of device assembly code is not supported.
  // CHECK-NEXT: */
  asm volatile("prefetch.global.L1 [%0 + 2];" :: "l"(arr));
#endif // BUILD_TEST
}

// clang-format on
