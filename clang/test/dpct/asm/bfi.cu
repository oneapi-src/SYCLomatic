// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -out-root %T/bfi %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/bfi/bfi.dp.cpp

// clang-format off
#include <cuda_runtime.h>
#include <cstdint>

__global__ void bfi() {
  uint32_t b32;
  uint64_t b64;

  // CHECK: b32 = dpct::bfi<uint32_t>(10, 10, 1, 8);
  asm("bfi.b32 %0, %1, %2, %3, %4;" : "=r"(b32) : "r"(10), "r"(10), "r"(1), "r"(8));
  
  // CHECK: b64 = dpct::bfi<uint64_t>(10, 10, 1, 8);
  asm("bfi.b64 %0, %1, %2, %3, %4;" : "=r"(b64) : "r"(10), "r"(10), "r"(1), "r"(8));
}

// clang-format off
