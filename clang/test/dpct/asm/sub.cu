// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -out-root %T/sub %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/sub/sub.dp.cpp

// clang-format off
#include <cuda_runtime.h>
#include <cstdint>

__global__ void sub() {
  int x = 1, y = 2;
  int16_t i16;
  uint16_t u16;
  int32_t i32;
  uint32_t u32;
  int64_t i64;
  uint64_t u64;
  short2 s16x2, sa{1, 2}, sb{1, 2};
  ushort2 u16x2, ua{1, 2}, ub{1, 2};

  // CHECK: i16 = x - y;
  asm("sub.s16 %0, %1, %2;" : "=r"(i16) : "r"(x), "r"(y));

  // CHECK: u16 = x - y;
  asm("sub.u16 %0, %1, %2;" : "=r"(u16) : "r"(x), "r"(y));

  // CHECK: i32 = x - y;
  asm("sub.s32 %0, %1, %2;" : "=r"(i32) : "r"(x), "r"(y));

  // CHECK: u32 = x - y;
  asm("sub.u32 %0, %1, %2;" : "=r"(u32) : "r"(x), "r"(y));

  // CHECK: i64 = x - y;
  asm("sub.s64 %0, %1, %2;" : "=r"(i64) : "r"(x), "r"(y));

  // CHECK: u64 = x - y;
  asm("sub.u64 %0, %1, %2;" : "=r"(u64) : "r"(x), "r"(y));

  // CHECK: i32 = sycl::sub_sat((int32_t)x, (int32_t)y);
  asm("sub.s32.sat %0, %1, %2;" : "=r"(i32) : "r"(x), "r"(y));

  // CHECK: s16x2 = sa - sb;
  asm("sub.s16x2 %0, %1, %2;" : "=r"(s16x2) : "r"(sa), "r"(sb));

  // CHECK: u16x2 = ua - ub;
  asm("sub.u16x2 %0, %1, %2;" : "=r"(u16x2) : "r"(ua), "r"(ub));

  // CHECK: s16x2 = sa - sycl::short2{1, 1};
  asm("sub.s16x2 %0, %1, {1, 1};" : "=r"(s16x2) : "r"(sa));

  // CHECK: u16x2 = ua - sycl::ushort2{1, 1};
  asm("sub.u16x2 %0, %1, {1, 1};" : "=r"(u16x2) : "r"(ua));

  // CHECK: s16x2 = sycl::short2{1, 1} - sa;
  asm("sub.s16x2 %0, {1, 1}, %1;" : "=r"(s16x2) : "r"(sa));

  // CHECK: u16x2 = sycl::ushort2{1, 1} - ua;
  asm("sub.u16x2 %0, {1, 1}, %1;" : "=r"(u16x2) : "r"(ua));
}

// clang-format on
