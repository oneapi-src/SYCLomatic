// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.1, cuda-11.2, cuda-11.3, cuda-11.4, cuda-11.5, cuda-11.6, cuda-11.7
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0, v11.1, v11.2, v11.3, v11.4, v11.5, v11.6, v11.7
// RUN: dpct --format-range=none -out-root %T/prefetch_tensormap %s --use-experimental-features=prefetch --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/prefetch_tensormap/prefetch_tensormap.dp.cpp
// RUN: %if BUILD_LIT %{icpx -c -DBUILD_TEST -fsycl %T/prefetch_tensormap/prefetch_tensormap.dp.cpp -o %T/prefetch_tensormap/prefetch_tensormap.dp.o %}

// clang-format off
#include <cuda_runtime.h>

// Unsupported syntax:
// prefetch{.tensormap_space}.tensormap [a];       // prefetch the tensormap
// .tensormap_space =          { .const, .param };

__global__ void prefetch_tensormap(int *arr) {
#ifndef BUILD_TEST
  /* prefetch of tensormap space */
  // CHECK: /*
  // CHECK-NEXT: DPCT1007:{{.*}} Migration of prefetch.const.tensormap [%0]; is not supported.
  // CHECK-NEXT: */
  asm volatile ("prefetch.const.tensormap [%0];" : : "l"(arr));
  // CHECK: /*
  // CHECK-NEXT: DPCT1007:{{.*}} Migration of prefetch.param.tensormap [%0]; is not supported.
  // CHECK-NEXT: */
  asm volatile ("prefetch.param.tensormap [%0];" : : "l"(arr));
#endif // BUILD_TEST
}

// clang-format on
