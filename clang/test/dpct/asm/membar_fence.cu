// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -out-root %T/membar_fence %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/membar_fence/membar_fence.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/membar_fence/membar_fence.dp.cpp -o %T/membar_fence/membar_fence.dp.o %}

// clang-format off
#include <cuda_runtime.h>
#include <cstdint>

__global__ void fence() {
  // CHECK: sycl::atomic_fence(sycl::memory_order::seq_cst,sycl::memory_scope::work_group);
  asm volatile("fence.sc.cta; " : : : "memory");

  // CHECK: sycl::atomic_fence(sycl::memory_order::seq_cst,sycl::memory_scope::device);
  asm volatile("fence.sc.gpu; " : : : "memory");

  // CHECK: sycl::atomic_fence(sycl::memory_order::seq_cst,sycl::memory_scope::system);
  asm volatile("fence.sc.sys; " : : : "memory");

  // CHECK: sycl::atomic_fence(sycl::memory_order::acq_rel,sycl::memory_scope::work_group);
  asm volatile("fence.acq_rel.cta; " : : : "memory");

  // CHECK: sycl::atomic_fence(sycl::memory_order::acq_rel,sycl::memory_scope::device);
  asm volatile("fence.acq_rel.gpu; " : : : "memory");

  // CHECK: sycl::atomic_fence(sycl::memory_order::acq_rel,sycl::memory_scope::system);
  asm volatile("fence.acq_rel.sys; " : : : "memory");
}

// clang-format off
