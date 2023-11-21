// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -out-root %T/mad %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/mad/mad.dp.cpp

// clang-format off
#include <cuda_runtime.h>
#include <cstdint>

__global__ void mad() {
  int x = 1, y = 2;
  int16_t i16;
  uint16_t u16;
  int32_t i32;
  uint32_t u32;
  int64_t i64;
  uint64_t u64;

  // CHECK: i16 = sycl::mad_hi((int16_t)x, (int16_t)y, (int16_t)3);
  asm("mad.hi.s16 %0, %1, %2, %3;" : "=r"(i16) : "r"(x), "r"(y), "r"(3));

  // CHECK: u16 = sycl::mad_hi((uint16_t)x, (uint16_t)y, (uint16_t)3);
  asm("mad.hi.u16 %0, %1, %2, %3;" : "=r"(u16) : "r"(x), "r"(y), "r"(3));

  // CHECK: i32 = sycl::mad_hi((int32_t)x, (int32_t)y, (int32_t)3);
  asm("mad.hi.s32 %0, %1, %2, %3;" : "=r"(i32) : "r"(x), "r"(y), "r"(3));

  // CHECK: u32 = sycl::mad_hi((uint32_t)x, (uint32_t)y, (uint32_t)3);
  asm("mad.hi.u32 %0, %1, %2, %3;" : "=r"(u32) : "r"(x), "r"(y), "r"(3));

  // CHECK: i64 = sycl::mad_hi((int64_t)x, (int64_t)y, (int64_t)3);
  asm("mad.hi.s64 %0, %1, %2, %3;" : "=r"(i64) : "r"(x), "r"(y), "r"(3));

  // CHECK: u64 = sycl::mad_hi((uint64_t)x, (uint64_t)y, (uint64_t)3);
  asm("mad.hi.u64 %0, %1, %2, %3;" : "=r"(u64) : "r"(x), "r"(y), "r"(3));

  // CHECK: i16 = (int32_t)x * (int32_t)y + (int32_t)3;
  asm("mad.wide.s16 %0, %1, %2, %3;" : "=r"(i16) : "r"(x), "r"(y), "r"(3));

  // CHECK: u16 = (uint32_t)x * (uint32_t)y + (uint32_t)3;
  asm("mad.wide.u16 %0, %1, %2, %3;" : "=r"(u16) : "r"(x), "r"(y), "r"(3));

  // CHECK: i32 = (int64_t)x * (int64_t)y + (int64_t)3;
  asm("mad.wide.s32 %0, %1, %2, %3;" : "=r"(i32) : "r"(x), "r"(y), "r"(3));

  // CHECK: u32 = (uint64_t)x * (uint64_t)y + (uint64_t)3;
  asm("mad.wide.u32 %0, %1, %2, %3;" : "=r"(u32) : "r"(x), "r"(y), "r"(3));

  // CHECK: DPCT1053:{{.*}}: Migration of device assembly code is not supported.
  asm("mad.wide.s64 %0, %1, %2, %3;" : "=r"(i64) : "r"(x), "r"(y), "r"(3));

  // CHECK: DPCT1053:{{.*}}: Migration of device assembly code is not supported.
  asm("mad.wide.u64 %0, %1, %2, %3;" : "=r"(u64) : "r"(x), "r"(y), "r"(3));
}

// clang-format on
