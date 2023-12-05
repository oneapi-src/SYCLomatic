// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -out-root %T/bfe %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/bfe/bfe.dp.cpp

// clang-format off
#include <cuda_runtime.h>
#include <cstdint>

__global__ void bfe() {
  int32_t i32;
  uint32_t u32;
  int64_t i64;
  uint64_t u64;

  // CHECK: i32 = dpct::bfe<int32_t>(10, 1, 8);
  asm("bfe.s32 %0, %1, %2, %3;" : "=r"(i32) : "r"(10), "r"(1), "r"(8));
  
  // CHECK: u32 = dpct::bfe<uint32_t>(10, 1, 8);
  asm("bfe.u32 %0, %1, %2, %3;" : "=r"(u32) : "r"(10), "r"(1), "r"(8));
  
  // CHECK: i64 = dpct::bfe<int64_t>(10, 1, 8);
  asm("bfe.s64 %0, %1, %2, %3;" : "=r"(i64) : "r"(10), "r"(1), "r"(8));
  
  // CHECK: u64 = dpct::bfe<uint64_t>(10, 1, 8);
  asm("bfe.u64 %0, %1, %2, %3;" : "=r"(u64) : "r"(10), "r"(1), "r"(8));
}

// clang-format off
