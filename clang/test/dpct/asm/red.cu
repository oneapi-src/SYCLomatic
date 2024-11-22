// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -out-root %T/red %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/atom/red.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/red/red.dp.cpp -o %T/red/red.dp.o %}

// clang-format off
#include <cuda_runtime.h>
#include <cstdint>

/*
.space =              { .global, .shared{::cta, ::cluster} };
.type =               { .b32, .b64, .u32, .u64, .s32, .s64, .f32, .f64 };
.scope =              { .cta, .cluster, .gpu, .sys };
.sem =                { .release, .relaxed };

Current only support the form likes "red.sem.scope.space.type" now.

*/

__global__ void red(int *a) {
  int a = 0;
  
  // CHECK: d = dpct::atomic_fetch_add<sycl::access::address_space::global_space, sycl::memory_order::relaxed, sycl::memory_scope::device>(a, 1);
  asm volatile ("red.relaxed.gpu.global.add.s32 %0, [%1];" : "=r"(a) : "l"(a), "r"(1));

  // CHECK: d = dpct::atomic_fetch_min<sycl::access::address_space::global_space, sycl::memory_order::relaxed, sycl::memory_scope::device>(a, 1);
  asm volatile ("red.relaxed.gpu.global.min.s32 %0, [%1];" : "=r"(a) : "l"(a), "r"(1));
  
  // CHECK: d = dpct::atomic_fetch_max<sycl::access::address_space::global_space, sycl::memory_order::relaxed, sycl::memory_scope::device>(a, 1);
  asm volatile ("red.relaxed.gpu.global.max.s32 %0, [%1];" : "=r"(a) : "l"(a), "r"(1));

  // CHECK: d = dpct::atomic_fetch_or<sycl::access::address_space::global_space, sycl::memory_order::relaxed, sycl::memory_scope::device>(a, 1);
  asm volatile ("red.relaxed.gpu.global.or.s32 %0, [%1];" : "=r"(a) : "l"(a), "r"(1));

  // CHECK: d = dpct::atomic_fetch_xoe<sycl::access::address_space::global_space, sycl::memory_order::relaxed, sycl::memory_scope::device>(a, 1);
  asm volatile ("red.relaxed.gpu.global.xor.s32 %0, [%1];" : "=r"(a) : "l"(a), "r"(1));

  // CHECK: d = dpct::atomic_fetch_and<sycl::access::address_space::global_space, sycl::memory_order::relaxed, sycl::memory_scope::device>(a, 1);
  asm volatile ("red.relaxed.gpu.global.and.s32 %0, [%1];" : "=r"(a) : "l"(a), "r"(1));

  // CHECK: d = dpct::atomic_fetch_add<sycl::access::address_space::local_space, sycl::memory_order::relaxed, sycl::memory_scope::system>(a, 1);
  asm volatile ("red.relaxed.sys.shared.add.s32 %0, [%1];" : "=r"(a) : "l"(a), "r"(1));

  // CHECK: d = dpct::atomic_fetch_or<sycl::access::address_space::global_space, sycl::memory_order::release, sycl::memory_scope::device>(a, 1);
  asm volatile ("red.release.gpu.global.or.s32 %0, [%1];" : "=r"(a) : "l"(a), "r"(1));

  
}

// clang-format on
