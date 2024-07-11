// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -out-root %T/brev %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/brev/brev.dp.cpp

// clang-format off
#include <cuda_runtime.h>
#include <cstdint>

__global__ void brev() {
  uint32_t u32;
  uint64_t u64;

  // CHECK: u32 = dpct::reverse_bits<uint32_t>(0x80000000U);
  asm("brev.b32 %0, 0x80000000U;" : "=r"(u32));

  // CHECK: u64 = dpct::reverse_bits<uint64_t>(0x80000000U);
  asm("brev.b64 %0, 0x80000000U;" : "=r"(u64));
}

// clang-format on
