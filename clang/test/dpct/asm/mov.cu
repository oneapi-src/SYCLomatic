// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -out-root %T/mov %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/mov/mov.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/mov/mov.dp.cpp -o %T/mov/mov.dp.o %}

// clang-format off
#include <cuda_runtime.h>
#include <cstdint>

// CHECK:void mov() {
// CHECK-NEXT: unsigned p;
// CHECK-NEXT: double d;
// CHECK-NEXT: float f;
// CHECK-NEXT: p = 123 * 123U + 456 * ((4 ^ 7) + 2 ^ 3) | 777 & 128U == 2 != 3 > 4 < 5 <= 3 >= 5 >> 1 << 2 && 1 || 7 && !0;
// CHECK: f = sycl::bit_cast<float>(uint32_t(0x3f800000U));
// CHECK: f = sycl::bit_cast<float>(uint32_t(0x3f800000U));
// CHECK: d = sycl::bit_cast<double>(uint64_t(0x40091EB851EB851FULL));
// CHECK: d = sycl::bit_cast<double>(uint64_t(0x40091EB851EB851FULL));
// CHECK: }
__global__ void mov() {
  unsigned p;
  double d;
  float f;
  asm ("mov.s32 %0, 123 * 123U + 456 * ((4 ^7) + 2 ^ 3) | 777 & 128U == 2 != 3 > 4 < 5 <= 3 >= 5 >> 1 << 2 && 1 || 7 && !0;" : "=r"(p) );
  asm ("mov.f32 %0, 0F3f800000;" : "=f"(f));
  asm ("mov.f32 %0, 0f3f800000;" : "=f"(f));
  asm ("mov.f64 %0, 0D40091EB851EB851F;" : "=d"(d));
  asm ("mov.f64 %0, 0d40091EB851EB851F;" : "=d"(d));
  return;
}

inline __device__ float half_to_float(uint16_t h) {
  float f = 0;
  return f;
}

inline __device__ float2 half2_to_float2(uint32_t v) {
  uint16_t lo, hi;
  // CHECK: lo = sycl::vec<uint32_t, 1>(v).template as<sycl::vec<uint16_t, 2>>()[0];
  // CHECK: hi = sycl::vec<uint32_t, 1>(v).template as<sycl::vec<uint16_t, 2>>()[1];
  asm volatile("mov.b32 {%0, %1}, %2;\n" : "=h"(lo), "=h"(hi) : "r"(v));
  return make_float2(half_to_float(lo), half_to_float(hi));
}

__global__ void test() {
  {
    uint32_t v;
    uint16_t lo, hi;
    // CHECK: lo = sycl::vec<uint32_t, 1>(v).template as<sycl::vec<uint16_t, 2>>()[0];
    // CHECK: hi = sycl::vec<uint32_t, 1>(v).template as<sycl::vec<uint16_t, 2>>()[1];
    asm volatile("mov.b32 {%0, %1}, %2;\n" : "=h"(lo), "=h"(hi) : "h"(v));
  }
  {
    uint64_t v;
    uint32_t lo, hi;
    // CHECK: lo = sycl::vec<uint64_t, 1>(v).template as<sycl::vec<uint32_t, 2>>()[0];
    // CHECK: hi = sycl::vec<uint64_t, 1>(v).template as<sycl::vec<uint32_t, 2>>()[1];
    asm volatile("mov.b64 {%0, %1}, %2;\n" : "=r"(lo), "=r"(hi) : "l"(v));
  }
  {
    uint64_t v;
    uint16_t a, b, c, d;
    // CHECK: a = sycl::vec<uint64_t, 1>(v).template as<sycl::vec<uint16_t, 4>>()[0];
    // CHECK: b = sycl::vec<uint64_t, 1>(v).template as<sycl::vec<uint16_t, 4>>()[1];
    // CHECK: c = sycl::vec<uint64_t, 1>(v).template as<sycl::vec<uint16_t, 4>>()[2];
    // CHECK: d = sycl::vec<uint64_t, 1>(v).template as<sycl::vec<uint16_t, 4>>()[3];
    asm volatile("mov.b64 {%0, %1, %2, %3}, %4;\n" : "=h"(a), "=h"(b), "=h"(c), "=h"(d) : "l"(v));
  }
  {
    uint32_t v;
    uint16_t lo, hi;
    // CHECK: v = sycl::vec<uint16_t, 2>({lo, hi}).template as<sycl::vec<uint32_t, 1>>()[0];
    asm volatile("mov.b32 %0, {%1, %2};\n" : "=r"(v) : "h"(lo), "h"(hi));
  }
  {
    uint64_t v;
    uint32_t lo, hi;
    // CHECK: v = sycl::vec<uint32_t, 2>({lo, hi}).template as<sycl::vec<uint64_t, 1>>()[0];
    asm volatile("mov.b64 %0, {%1, %2};\n" : "=l"(v) : "r"(lo), "r"(hi));
  }
  {
    uint64_t v;
    uint16_t a, b, c, d;
    // CHECK: v = sycl::vec<uint16_t, 4>({a, b, c, d}).template as<sycl::vec<uint64_t, 1>>()[0];
    asm volatile("mov.b64 %0, {%1, %2, %3, %4};\n" : "=l"(v) : "h"(a), "h"(b), "h"(c), "h"(d));
  }
  
  {
    uint32_t v;
    uint16_t lo;
    // CHECK: lo = sycl::vec<uint32_t, 1>(v).template as<sycl::vec<uint16_t, 2>>()[0];
    asm volatile("mov.b32 {%0, _}, %1;\n" : "=h"(lo) : "h"(v));
  }
  {
    uint64_t v;
    uint32_t hi;
    // CHECK: hi = sycl::vec<uint64_t, 1>(v).template as<sycl::vec<uint32_t, 2>>()[1];
    asm volatile("mov.b64 {_, %0}, %1;\n" : "=r"(hi) : "l"(v));
  }
  {
    uint64_t v;
    uint16_t a, b, d;
    // CHECK: a = sycl::vec<uint64_t, 1>(v).template as<sycl::vec<uint16_t, 4>>()[0];
    // CHECK: b = sycl::vec<uint64_t, 1>(v).template as<sycl::vec<uint16_t, 4>>()[1];
    // CHECK: d = sycl::vec<uint64_t, 1>(v).template as<sycl::vec<uint16_t, 4>>()[3];
    asm volatile("mov.b64 {%0, %1, _, %2}, %3;\n" : "=h"(a), "=h"(b), "=h"(d) : "l"(v));
  }
}


// clang-format on
