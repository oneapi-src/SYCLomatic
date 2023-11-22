// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -out-root %T/asm_arith %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/asm_arith/asm_arith.dp.cpp

// clang-format off
#include <cuda_runtime.h>
#include <cstdint>

__global__ void add() {
  int x = 1, y = 2;
  int16_t i16;
  uint16_t u16;
  int32_t i32;
  uint32_t u32;
  int64_t i64;
  uint64_t u64;
  short2 s16x2, sa{1, 2}, sb{1, 2};
  ushort2 u16x2, ua{1, 2}, ub{1, 2};

  // CHECK: i16 = x + y;
  asm("add.s16 %0, %1, %2;" : "=r"(i16) : "r"(x), "r"(y));

  // CHECK: u16 = x + y;
  asm("add.u16 %0, %1, %2;" : "=r"(u16) : "r"(x), "r"(y));

  // CHECK: i32 = x + y;
  asm("add.s32 %0, %1, %2;" : "=r"(i32) : "r"(x), "r"(y));

  // CHECK: u32 = x + y;
  asm("add.u32 %0, %1, %2;" : "=r"(u32) : "r"(x), "r"(y));

  // CHECK: i64 = x + y;
  asm("add.s64 %0, %1, %2;" : "=r"(i64) : "r"(x), "r"(y));

  // CHECK: u64 = x + y;
  asm("add.u64 %0, %1, %2;" : "=r"(u64) : "r"(x), "r"(y));

  // CHECK: i32 = sycl::add_sat((int32_t)x, (int32_t)y);
  asm("add.s32.sat %0, %1, %2;" : "=r"(i32) : "r"(x), "r"(y));

  // CHECK: s16x2 = sa + sb;
  asm("add.s16x2 %0, %1, %2;" : "=r"(s16x2) : "r"(sa), "r"(sb));

  // CHECK: u16x2 = ua + ub;
  asm("add.u16x2 %0, %1, %2;" : "=r"(u16x2) : "r"(ua), "r"(ub));

  // CHECK: s16x2 = sa + sycl::short2{1, 1};
  asm("add.s16x2 %0, %1, {1, 1};" : "=r"(s16x2) : "r"(sa));

  // CHECK: u16x2 = ua + sycl::ushort2{1, 1};
  asm("add.u16x2 %0, %1, {1, 1};" : "=r"(u16x2) : "r"(ua));

  // CHECK: s16x2 = sycl::short2{1, 1} + sa;
  asm("add.s16x2 %0, {1, 1}, %1;" : "=r"(s16x2) : "r"(sa));

  // CHECK: u16x2 = sycl::ushort2{1, 1} + ua;                                                                                                                                     
  asm("add.u16x2 %0, {1, 1}, %1;" : "=r"(u16x2) : "r"(ua));
}

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

__global__ void mul() {
  int x = 1, y = 2;
  int16_t i16;
  uint16_t u16;
  int32_t i32;
  uint32_t u32;
  int64_t i64;
  uint64_t u64;

  // CHECK: i16 = sycl::mul_hi((int16_t)x, (int16_t)y);
  asm("mul.hi.s16 %0, %1, %2;" : "=r"(i16) : "r"(x), "r"(y));

  // CHECK: u16 = sycl::mul_hi((uint16_t)x, (uint16_t)y);
  asm("mul.hi.u16 %0, %1, %2;" : "=r"(u16) : "r"(x), "r"(y));

  // CHECK: i32 = sycl::mul_hi((int32_t)x, (int32_t)y);
  asm("mul.hi.s32 %0, %1, %2;" : "=r"(i32) : "r"(x), "r"(y));

  // CHECK: u32 = sycl::mul_hi((uint32_t)x, (uint32_t)y);
  asm("mul.hi.u32 %0, %1, %2;" : "=r"(u32) : "r"(x), "r"(y));

  // CHECK: i64 = sycl::mul_hi((int64_t)x, (int64_t)y);
  asm("mul.hi.s64 %0, %1, %2;" : "=r"(i64) : "r"(x), "r"(y));

  // CHECK: u64 = sycl::mul_hi((uint64_t)x, (uint64_t)y);
  asm("mul.hi.u64 %0, %1, %2;" : "=r"(u64) : "r"(x), "r"(y));

  // CHECK: i16 = (int32_t)x * (int32_t)y;
  asm("mul.wide.s16 %0, %1, %2;" : "=r"(i16) : "r"(x), "r"(y));

  // CHECK: u16 = (uint32_t)x * (uint32_t)y;
  asm("mul.wide.u16 %0, %1, %2;" : "=r"(u16) : "r"(x), "r"(y));

  // CHECK: i32 = (int64_t)x * (int64_t)y;
  asm("mul.wide.s32 %0, %1, %2;" : "=r"(i32) : "r"(x), "r"(y));

  // CHECK: u32 = (uint64_t)x * (uint64_t)y;
  asm("mul.wide.u32 %0, %1, %2;" : "=r"(u32) : "r"(x), "r"(y));

  // CHECK: DPCT1053:{{.*}}: Migration of device assembly code is not supported.
  asm("mul.wide.s64 %0, %1, %2;" : "=r"(i64) : "r"(x), "r"(y));

  // CHECK: DPCT1053:{{.*}}: Migration of device assembly code is not supported.
  asm("mul.wide.u64 %0, %1, %2;" : "=r"(u64) : "r"(x), "r"(y));
}

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

__global__ void mul24() {
  int x = 1, y = 2;
  int32_t i32;
  uint32_t u32;

  // CHECK: i32 = sycl::mul24((int32_t)x, (int32_t)y);
  asm("mul24.lo.s32 %0, %1, %2;" : "=r"(i32) : "r"(x), "r"(y));

  // CHECK: u32 = sycl::mul24((uint32_t)x, (uint32_t)y);
  asm("mul24.lo.u32 %0, %1, %2;" : "=r"(u32) : "r"(x), "r"(y));

  // CHECK: DPCT1053:{{.*}}: Migration of device assembly code is not supported.
  asm("mul24.hi.s32 %0, %1, %2;" : "=r"(i32) : "r"(x), "r"(y));
  
  // CHECK: DPCT1053:{{.*}}: Migration of device assembly code is not supported.
  asm("mul24.hi.u32 %0, %1, %2;" : "=r"(u32) : "r"(x), "r"(y));
}

__global__ void mad24() {
  int x = 1, y = 2;
  int32_t i32;
  uint32_t u32;

  // CHECK: i32 = sycl::mad24((int32_t)x, (int32_t)y, (int32_t)3);
  asm("mad24.lo.s32 %0, %1, %2, %3;" : "=r"(i32) : "r"(x), "r"(y), "r"(3));

  // CHECK: u32 = sycl::mad24((uint32_t)x, (uint32_t)y, (uint32_t)3);
  asm("mad24.lo.u32 %0, %1, %2, %3;" : "=r"(u32) : "r"(x), "r"(y), "r"(3));

  // CHECK: DPCT1053:{{.*}}: Migration of device assembly code is not supported.
  asm("mad24.hi.s32 %0, %1, %2, %3;" : "=r"(i32) : "r"(x), "r"(y), "r"(3));

  // CHECK: DPCT1053:{{.*}}: Migration of device assembly code is not supported.
  asm("mad24.hi.u32 %0, %1, %2, %3;" : "=r"(u32) : "r"(x), "r"(y), "r"(3));
}

__global__ void div() {
  int x = 1, y = 2;
  int16_t i16;
  uint16_t u16;
  int32_t i32;
  uint32_t u32;
  int64_t i64;
  uint64_t u64;

  // CHECK: i16 = x / y;
  asm("div.s16 %0, %1, %2;" : "=r"(i16) : "r"(x), "r"(y));

  // CHECK: u16 = x / y;
  asm("div.u16 %0, %1, %2;" : "=r"(u16) : "r"(x), "r"(y));

  // CHECK: i32 = x / y;
  asm("div.s32 %0, %1, %2;" : "=r"(i32) : "r"(x), "r"(y));

  // CHECK: u32 = x / y;
  asm("div.u32 %0, %1, %2;" : "=r"(u32) : "r"(x), "r"(y));

  // CHECK: i64 = x / y;
  asm("div.s64 %0, %1, %2;" : "=r"(i64) : "r"(x), "r"(y));

  // CHECK: u64 = x / y;
  asm("div.u64 %0, %1, %2;" : "=r"(u64) : "r"(x), "r"(y));
}
__global__ void rem() {
  int x = 1, y = 2;
  int16_t i16;
  uint16_t u16;
  int32_t i32;
  uint32_t u32;
  int64_t i64;
  uint64_t u64;

  // CHECK: i16 = x % y;
  asm("rem.s16 %0, %1, %2;" : "=r"(i16) : "r"(x), "r"(y));

  // CHECK: u16 = x % y;
  asm("rem.u16 %0, %1, %2;" : "=r"(u16) : "r"(x), "r"(y));

  // CHECK: i32 = x % y;
  asm("rem.s32 %0, %1, %2;" : "=r"(i32) : "r"(x), "r"(y));

  // CHECK: u32 = x % y;
  asm("rem.u32 %0, %1, %2;" : "=r"(u32) : "r"(x), "r"(y));

  // CHECK: i64 = x % y;
  asm("rem.s64 %0, %1, %2;" : "=r"(i64) : "r"(x), "r"(y));

  // CHECK: u64 = x % y;
  asm("rem.u64 %0, %1, %2;" : "=r"(u64) : "r"(x), "r"(y));
}

__global__ void abs() {
  int x = 1;
  int16_t i16;
  int32_t i32;
  int64_t i64;

  // CHECK: i16 = sycl::abs(x);
  asm("abs.s16 %0, %1;" : "=r"(i16) : "r"(x));

  // CHECK: i32 = sycl::abs(x);
  asm("abs.s32 %0, %1;" : "=r"(i32) : "r"(x));

  // CHECK: i64 = sycl::abs(x);
  asm("abs.s64 %0, %1;" : "=r"(i64) : "r"(x));
}

__global__ void neg() {
  int x = 1;
  int16_t i16;
  int32_t i32;
  int64_t i64;

  // CHECK: i16 = -x;
  asm("neg.s16 %0, %1;" : "=r"(i16) : "r"(x));

  // CHECK: i32 = -x;
  asm("neg.s32 %0, %1;" : "=r"(i32) : "r"(x));

  // CHECK: i64 = -x;
  asm("neg.s64 %0, %1;" : "=r"(i64) : "r"(x));
}

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

__global__ void max() {
  int x = 1, y = 2;
  int16_t i16;
  uint16_t u16;
  int32_t i32;
  uint32_t u32;
  int64_t i64;
  uint64_t u64;
  short2 s16x2, sa{1, 2}, sb{1, 2};
  ushort2 u16x2, ua{1, 2}, ub{1, 2};

  // CHECK: i16 = sycl::max((int16_t)x, (int16_t)y);
  asm("max.s16 %0, %1, %2;" : "=r"(i16) : "r"(x), "r"(y));

  // CHECK: u16 = sycl::max((uint16_t)x, (uint16_t)y);
  asm("max.u16 %0, %1, %2;" : "=r"(u16) : "r"(x), "r"(y));

  // CHECK: i32 = sycl::max((int32_t)x, (int32_t)y);
  asm("max.s32 %0, %1, %2;" : "=r"(i32) : "r"(x), "r"(y));

  // CHECK: u32 = sycl::max((uint32_t)x, (uint32_t)y);
  asm("max.u32 %0, %1, %2;" : "=r"(u32) : "r"(x), "r"(y));

  // CHECK: i64 = sycl::max((int64_t)x, (int64_t)y);
  asm("max.s64 %0, %1, %2;" : "=r"(i64) : "r"(x), "r"(y));

  // CHECK: u64 = sycl::max((uint64_t)x, (uint64_t)y);
  asm("max.u64 %0, %1, %2;" : "=r"(u64) : "r"(x), "r"(y));

  // CHECK: s16x2 = sycl::max((sycl::short2)sa, (sycl::short2)sb);
  asm("max.s16x2 %0, %1, %2;" : "=r"(s16x2) : "r"(sa), "r"(sb));

  // CHECK: u16x2 = sycl::max((sycl::ushort2)ua, (sycl::ushort2)ub);
  asm("max.u16x2 %0, %1, %2;" : "=r"(u16x2) : "r"(ua), "r"(ub));

  // CHECK: i32 = dpct::relu(sycl::max((int32_t)x, (int32_t)y));
  asm("max.s32.relu %0, %1, %2;" : "=r"(i32) : "r"(x), "r"(y));

  // CHECK: s16x2 = dpct::relu(sycl::max((sycl::short2)sa, (sycl::short2)sb));
  asm("max.s16x2.relu %0, %1, %2;" : "=r"(s16x2) : "r"(sa), "r"(sb));
}

__global__ void popc() {
  int x = 1;
  int32_t i32;
  int64_t i64;
  // CHECK: i32 = sycl::popcount<uint32_t>(x);
  asm("popc.b32 %0, %1;" : "=r"(i32) : "r"(x));

  // CHECK: i64 = sycl::popcount<uint64_t>(x);
  asm("popc.b64 %0, %1;" : "=r"(i64) : "r"(x));
}

__global__ void clz() {
  int x = 1;
  int32_t i32;
  int64_t i64;

  // CHECK: i32 = sycl::clz<uint32_t>(x);
  asm("clz.b32 %0, %1;" : "=r"(i32) : "r"(x));

  // CHECK: i64 = sycl::clz<uint64_t>(x);
  asm("clz.b64 %0, %1;" : "=r"(i64) : "r"(x));
}

__global__ void bitwise_and() {
  int x = 1, y = 2;
  uint8_t u8;
  uint16_t u16;
  uint32_t u32;
  uint64_t u64;

  // CHECK: u8 = x & y;
  asm("and.pred %0, %1, %2;" : "=r"(u8) : "r"(x), "r"(y));

  // CHECK u16 = x & y;
  asm("and.b16 %0, %1, %2;" : "=r"(u16) : "r"(x), "r"(y));

  // CHECK: u32 = x & y;
  asm("and.b32 %0, %1, %2;" : "=r"(u32) : "r"(x), "r"(y));

  // CHECK: u64 = x & y;
  asm("and.b64 %0, %1, %2;" : "=r"(u64) : "r"(x), "r"(y));
}

__global__ void bitwise_or() {
  int x = 1, y = 2;
  uint8_t u8;
  uint16_t u16;
  uint32_t u32;
  uint64_t u64;

  // CHECK: u8 = x | y;
  asm("or.pred %0, %1, %2;" : "=r"(u8) : "r"(x), "r"(y));

  // CHECK: u16 = x | y;
  asm("or.b16 %0, %1, %2;" : "=r"(u16) : "r"(x), "r"(y));

  // CHECK: u32 = x | y;
  asm("or.b32 %0, %1, %2;" : "=r"(u32) : "r"(x), "r"(y));

  // CHECK: u64 = x | y;
  asm("or.b64 %0, %1, %2;" : "=r"(u64) : "r"(x), "r"(y));
}

__global__ void bitwise_xor() {
  int x = 1, y = 2;
  uint8_t u8;
  uint16_t u16;
  uint32_t u32;
  uint64_t u64;

  // CHECK: u8 = x ^ y;
  asm("xor.pred %0, %1, %2;" : "=r"(u8) : "r"(x), "r"(y));

  // CHECK: u16 = x ^ y;
  asm("xor.b16 %0, %1, %2;" : "=r"(u16) : "r"(x), "r"(y));

  // CHECK: u32 = x ^ y;
  asm("xor.b32 %0, %1, %2;" : "=r"(u32) : "r"(x), "r"(y));

  // CHECK: u64 = x ^ y;
  asm("xor.b64 %0, %1, %2;" : "=r"(u64) : "r"(x), "r"(y));
}

__global__ void bitwise_not() {
  int x = 1;
  uint8_t u8;
  uint16_t u16;
  uint32_t u32;
  uint64_t u64;

  // CHECK: u8 = ~x;
  asm("not.pred %0, %1;" : "=r"(u8) : "r"(x));

  // CHECK: u16 = ~x;
  asm("not.b16 %0, %1;" : "=r"(u16) : "r"(x));

  // CHECK: u32 = ~x;
  asm("not.b32 %0, %1;" : "=r"(u32) : "r"(x));

  // CHECK: u64 = ~x;
  asm("not.b64 %0, %1;" : "=r"(u64) : "r"(x));
}

__global__ void cnot() {
  int x = 1;
  uint16_t u16;
  uint32_t u32;
  uint64_t u64;

  // CHECK: u16 = x == 0;
  asm("cnot.b16 %0, %1;" : "=r"(u16) : "r"(x));

  // CHECK: u32 = x == 0;
  asm("cnot.b32 %0, %1;" : "=r"(u32) : "r"(x));

  // CHECK: u64 = x == 0;
  asm("cnot.b64 %0, %1;" : "=r"(u64) : "r"(x));
}

__global__ void shl() {
  int x = 1, y = 2;
  uint16_t u16;
  uint32_t u32;
  uint64_t u64;

  // CHECK: u16 = x << y;
  asm("shl.b16 %0, %1, %2;" : "=r"(u16) : "r"(x), "r"(y));

  // CHECK: u32 = x << y;
  asm("shl.b32 %0, %1, %2;" : "=r"(u32) : "r"(x), "r"(y));

  // CHECK: u64 = x << y;
  asm("shl.b64 %0, %1, %2;" : "=r"(u64) : "r"(x), "r"(y));
}

__global__ void shr() {
  int x = 1, y = 2;
  uint16_t u16;
  uint32_t u32;
  uint64_t u64;

  // CHECK: u16 = x >> y;
  asm("shr.b16 %0, %1, %2;" : "=r"(u16) : "r"(x), "r"(y));

  // CHECK: u32 = x >> y;
  asm("shr.b32 %0, %1, %2;" : "=r"(u32) : "r"(x), "r"(y));

  // CHECK: u64 = x >> y;
  asm("shr.b64 %0, %1, %2;" : "=r"(u64) : "r"(x), "r"(y));
}

// clang-format on
