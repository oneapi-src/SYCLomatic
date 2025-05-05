// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -out-root %T/neg %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/neg/neg.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/neg/neg.dp.cpp -o %T/neg/neg.dp.o %}

// clang-format off
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

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

// CHECK: inline void negate_half2(sycl::half2 *addr) {
// CHECK-NEXT:    unsigned reg[2];
// CHECK-NEXT:    reg[0] = *reinterpret_cast<unsigned int*>(addr);
// CHECK-NEXT:    reg[0] = (-sycl::vec<int, 1>(reg[0]).as<sycl::vec<sycl::half, 2>>()).as<sycl::vec<int, 1>>().x();
// CHECK-NEXT:    *reinterpret_cast<unsigned int*>(addr) = reg[0];
// CHECK-NEXT:}
__device__ inline void negate_half2(__half2 *addr) {
    unsigned reg[2];
    reg[0] = *reinterpret_cast<unsigned int*>(addr);
    asm volatile("neg.f16x2 %0, %0;" : "+r"(reg[0]));
    *reinterpret_cast<unsigned int*>(addr) = reg[0];
}

// clang-format on
