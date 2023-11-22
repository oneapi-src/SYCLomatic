// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -out-root %T/min %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/min/min.dp.cpp

// clang-format off
#include <cuda_runtime.h>
#include <cstdint>

__global__ void min() {
  int x = 1, y = 2;
  int16_t i16;
  uint16_t u16;
  int32_t i32;
  uint32_t u32;
  int64_t i64;
  uint64_t u64;
  short2 s16x2, sa{1, 2}, sb{1, 2};
  ushort2 u16x2, ua{1, 2}, ub{1, 2};

  // CHECK: i16 = sycl::min((int16_t)x, (int16_t)y);
  asm("min.s16 %0, %1, %2;" : "=r"(i16) : "r"(x), "r"(y));

  // CHECK: u16 = sycl::min((uint16_t)x, (uint16_t)y);
  asm("min.u16 %0, %1, %2;" : "=r"(u16) : "r"(x), "r"(y));

  // CHECK: i32 = sycl::min((int32_t)x, (int32_t)y);
  asm("min.s32 %0, %1, %2;" : "=r"(i32) : "r"(x), "r"(y));

  // CHECK: u32 = sycl::min((uint32_t)x, (uint32_t)y);
  asm("min.u32 %0, %1, %2;" : "=r"(u32) : "r"(x), "r"(y));

  // CHECK: i64 = sycl::min((int64_t)x, (int64_t)y);
  asm("min.s64 %0, %1, %2;" : "=r"(i64) : "r"(x), "r"(y));

  // CHECK: u64 = sycl::min((uint64_t)x, (uint64_t)y);
  asm("min.u64 %0, %1, %2;" : "=r"(u64) : "r"(x), "r"(y));

  // CHECK: s16x2 = sycl::min((sycl::short2)sa, (sycl::short2)sb);
  asm("min.s16x2 %0, %1, %2;" : "=r"(s16x2) : "r"(sa), "r"(sb));

  // CHECK: u16x2 = sycl::min((sycl::ushort2)ua, (sycl::ushort2)ub);
  asm("min.u16x2 %0, %1, %2;" : "=r"(u16x2) : "r"(ua), "r"(ub));

  // CHECK: i32 = dpct::relu(sycl::min((int32_t)x, (int32_t)y));
  asm("min.s32.relu %0, %1, %2;" : "=r"(i32) : "r"(x), "r"(y));

  // CHECK: s16x2 = dpct::relu(sycl::min((sycl::short2)sa, (sycl::short2)sb));
  asm("min.s16x2.relu %0, %1, %2;" : "=r"(s16x2) : "r"(sa), "r"(sb));
}


// clang-format on
