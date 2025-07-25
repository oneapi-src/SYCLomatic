// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -out-root %T/ld %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/ld/ld.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/ld/ld.dp.cpp -o %T/ld/ld.dp.o %}

// clang-format off
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <stdio.h>

using bf16 = __nv_bfloat16;
using bf16_2 = __nv_bfloat162;
using half_2 = __half2;

/*
.ss =                       { .const, .global, .local, .param, .shared };
.type =                     { .b8, .b16, .b32, .b64, .b128, 
                              .u8, .u16, .u32, .u64,
                              .s8, .s16, .s32, .s64,
                              .f32, .f64 };
Current only support the form likes "ld.ss.type" now.
*/

__global__ void ld(int *arr) {
  int a, b, c;
  unsigned long long d;
  // CHECK: a = *arr;
  asm volatile ("ld.global.s32 %0, [%1];" : "=r"(a) : "l"(arr));
  // CHECK: b = *((uint32_t *)(uintptr_t)arr);
  asm volatile ("ld.global.u32 %0, [%1];" : "=r"(b) : "l"(arr));
  // CHECK: c = *((uint32_t *)((uintptr_t)arr + 4));
  asm volatile ("ld.global.u32 %0, [%1 + 4];" : "=r"(c) : "l"(arr));
  // CHECK: d = *((uint64_t *)((uintptr_t)arr + 8));
  asm volatile ("ld.global.u64 %0, [%1 + 8];" : "=l"(d) : "l"(arr));
}

__device__ void shared_address_load32(uint32_t addr, uint32_t &val) {
  // CHECK: {
  // CHECK:   val = *((uint32_t *)(uintptr_t)addr);
  // CHECK: } 
  asm volatile("{ld.shared.b32 %0, [%1];}" : : "r"(val), "r"(addr) : "memory"); 
}

// CHECK: inline void load_global_short2(sycl::short2 &a, const sycl::short2 *addr) {
// CHECK-NEXT:  short x, y, z, w;
// CHECK-NEXT:  x  = *((int16_t *)(uintptr_t)addr + 0);
// CHECK-NEXT:  y  = *((int16_t *)(uintptr_t)addr + 1);
// CHECK-NEXT:  a.x() = x;
// CHECK-NEXT:  a.y() = y;
// CHECK-NEXT:}
__device__ inline void load_global_short2(short2 &a, const short2 *addr) {
  short x, y, z, w;
  asm("ld.cg.global.v2.s16 {%0, %1}, [%2+0];" : "=h"(x), "=h"(y) : "l"(addr));
  a.x = x;
  a.y = y;
}

// CHECK: inline void load_global_short4(sycl::short4 &a, const sycl::short4 *addr) {
// CHECK-NEXT:  short x, y, z, w;
// CHECK-NEXT:  x  = *((int16_t *)(uintptr_t)addr + 0);
// CHECK-NEXT:  y  = *((int16_t *)(uintptr_t)addr + 1);
// CHECK-NEXT:  z  = *((int16_t *)(uintptr_t)addr + 2);
// CHECK-NEXT:  w  = *((int16_t *)(uintptr_t)addr + 3);
// CHECK-NEXT:  a.x() = x;
// CHECK-NEXT:  a.y() = y;
// CHECK-NEXT:  a.z() = z;
// CHECK-NEXT:  a.w() = w;
// CHECK-NEXT:}
__device__ inline void load_global_short4(short4 &a, const short4 *addr) {
  short x, y, z, w;
  asm("ld.cg.global.v4.s16 {%0, %1, %2, %3}, [%4+0];" : "=h"(x), "=h"(y), "=h"(z), "=h"(w) : "l"(addr));
  a.x = x;
  a.y = y;
  a.z = z;
  a.w = w;
}

// CHECK: __dpct_inline__ int ld_flag_volatile(int* flag_addr) {
// CHECK-NEXT:   int flag;
// CHECK-NEXT:   flag = *((uint32_t *)(uintptr_t)flag_addr);
// CHECK-NEXT:   sycl::atomic_fence(sycl::memory_order::seq_cst,sycl::memory_scope::device);
// CHECK-NEXT:   return flag;
// CHECK-NEXT: }
__device__ __forceinline__ int ld_flag_volatile(int* flag_addr) {
  int flag;
  asm volatile("ld.volatile.global.u32 %0, [%1]; membar.gl;" : "=r"(flag) : "l"(flag_addr));
  return flag;
}

// CHECK: __dpct_inline__ int ld_flag_acquire(int* flag_addr) {
// CHECK-NEXT:  int flag;
// CHECK-NEXT:  flag = *((uint32_t *)(uintptr_t)flag_addr);
// CHECK-NEXT:  return flag;
// CHECK-NEXT: }
__device__ __forceinline__ int ld_flag_acquire(int* flag_addr) {
  int flag;
  asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(flag) : "l"(flag_addr));
  return flag;
}

 // CHECK: static inline void lds(bf16& dst, uint32_t src) {
 // CHECK-NEXT:       *(uint16_t*)&dst = *((uint16_t *)(uintptr_t)src);
 // CHECK-NEXT: }
 __device__ static inline void lds(bf16& dst, uint32_t src) {
        asm volatile("ld.shared.b16 %0, [%1];" : "=h"(*(uint16_t*)&dst) : "r"(src));
 }

// CHECK: static inline void sts(uint32_t dst, const bf16& src) {
// CHECK-NEXT:    *((uint16_t *)(uintptr_t)dst) = *(uint16_t*)&src;
// CHECK-NEXT: }
__device__ static inline void sts(uint32_t dst, const bf16& src) {
    asm volatile("st.shared.b16 [%1], %0;\n" : : "h"(*(uint16_t*)&src), "r"(dst));
}

// CHECK: static inline void ldg(bf16& dst, bf16* src) {
// CHECK-NEXT:     *(uint16_t*)&dst = *((uint16_t *)(uintptr_t)src);
// CHECK-NEXT: }    
__device__ static inline void ldg(bf16& dst, bf16* src) {
    asm volatile("ld.global.b16 %0, [%1];\n" : "=h"(*(uint16_t*)&dst) : "l"(src));
}

// CHECK: static inline void stg(bf16* dst, const bf16& src) {
// CHECK-NEXT:     *((uint16_t *)(uintptr_t)dst) = *(uint16_t*)&src;
// CHECK-NEXT: }
__device__ static inline void stg(bf16* dst, const bf16& src) {
    asm volatile("st.global.b16 [%1], %0;\n" : : "h"(*(uint16_t*)&src), "l"(dst));
}

// CHECK: static inline void lds(sycl::half& dst, uint32_t src) {
// CHECK-NEXT:     *(uint16_t*)&dst = *((uint16_t *)(uintptr_t)src);
// CHECK-NEXT: }    
__device__ static inline void lds(half& dst, uint32_t src) {
    asm volatile("ld.shared.b16 %0, [%1];\n" : "=h"(*(uint16_t*)&dst) : "r"(src));
}

// CHECK: static inline void sts(uint32_t dst, const sycl::half& src) {
// CHECK-NEXT:     *((uint16_t *)(uintptr_t)dst) = *(uint16_t*)&src;
// CHECK-NEXT: }
__device__ static inline void sts(uint32_t dst, const half& src) {
    asm volatile("st.shared.b16 [%1], %0;\n" : : "h"(*(uint16_t*)&src), "r"(dst));
}

// CHECK: static inline void ldg(sycl::half& dst, sycl::half* src) {
// CHECK-NEXT:     *(uint16_t*)&dst = *((uint16_t *)(uintptr_t)src);
// CHECK-NEXT: }
__device__ static inline void ldg(half& dst, half* src) {
    asm volatile("ld.global.b16 %0, [%1];\n" : "=h"(*(uint16_t*)&dst) : "l"(src));
}

// CHECK: static inline void stg(sycl::half* dst, const sycl::half& src) {
// CHECK-NEXT:     *((uint16_t *)(uintptr_t)dst) = *(uint16_t*)&src;
// CHECK-NEXT: }
__device__ static inline void stg(half* dst, const half& src) {
    asm volatile("st.global.b16 [%1], %0;\n" : : "h"(*(uint16_t*)&src), "l"(dst));
}

// CHECK: static inline void lds(float& dst, uint32_t src) {
// CHECK-NEXT:     dst = *((float *)(uintptr_t)src);
// CHECK-NEXT: }
__device__ static inline void lds(float& dst, uint32_t src) {
    asm volatile("ld.shared.f32 %0, [%1];\n" : "=f"(dst) : "r"(src));
}

// CHECK: static inline void sts(uint32_t dst, const float& src) {
// CHECK-NEXT:     *((float *)(uintptr_t)dst) = src;
// CHECK-NEXT: }
__device__ static inline void sts(uint32_t dst, const float& src) {
    asm volatile("st.shared.f32 [%1], %0;\n" : : "f"(src), "r"(dst));
}

// CHECK: static inline void ldg(float& dst, float* src) {
// CHECK-NEXT:     dst = *src;
// CHECK-NEXT: }
__device__ static inline void ldg(float& dst, float* src) {
    asm volatile("ld.global.f32 %0, [%1];\n" : "=f"(dst) : "l"(src));
}

// CHECK: static inline void stg(float* dst, const float& src) {
// CHECK-NEXT:     *dst = src;
// CHECK-NEXT: }
__device__ static inline void stg(float* dst, const float& src) {
    asm volatile("st.global.f32 [%1], %0;\n" : : "f"(src), "l"(dst));
}

// CHECK: static inline void lds(bf16_2& dst, uint32_t src) {
// CHECK-NEXT:     *(uint32_t*)&dst = *((uint32_t *)(uintptr_t)src);
// CHECK-NEXT: }
__device__ static inline void lds(bf16_2& dst, uint32_t src) {
    asm volatile("ld.shared.b32 %0, [%1];\n" : "=r"(*(uint32_t*)&dst) : "r"(src));
}

// CHECK: static inline void sts(uint32_t dst, const bf16_2& src) {
// CHECK-NEXT:     *((uint32_t *)(uintptr_t)dst) = (*(uint32_t*)&src);
// CHECK-NEXT: }
__device__ static inline void sts(uint32_t dst, const bf16_2& src) {
    asm volatile("st.shared.b32 [%1], %0;\n" : : "r"(*(uint32_t*)&src), "r"(dst));
}

// CHECK: static inline void ldg(bf16_2& dst, bf16_2* src) {
// CHECK-NEXT:     *(uint32_t*)&dst = *((uint32_t *)(uintptr_t)src);
// CHECK-NEXT: }
__device__ static inline void ldg(bf16_2& dst, bf16_2* src) {
    asm volatile("ld.global.b32 %0, [%1];\n" : "=r"(*(uint32_t*)&dst) : "l"(src));
}

// CHECK: static inline void stg(bf16_2* dst, const bf16_2& src) {
// CHECK-NEXT:     *((uint32_t *)(uintptr_t)dst) = (*(uint32_t*)&src);
// CHECK-NEXT: }
__device__ static inline void stg(bf16_2* dst, const bf16_2& src) {
    asm volatile("st.global.b32 [%1], %0;\n" : : "r"(*(uint32_t*)&src), "l"(dst));
}

// CHECK: static inline void lds(half_2& dst, uint32_t src) {
// CHECK-NEXT:     *(uint32_t*)&dst = *((uint32_t *)(uintptr_t)src);
// CHECK-NEXT: }
__device__ static inline void lds(half_2& dst, uint32_t src) {
    asm volatile("ld.shared.b32 %0, [%1];\n" : "=r"(*(uint32_t*)&dst) : "r"(src));
}

// CHECK: static inline void sts(uint32_t dst, const half_2& src) {
// CHECK-NEXT:     *((uint32_t *)(uintptr_t)dst) = (*(uint32_t*)&src);
// CHECK-NEXT: }
__device__ static inline void sts(uint32_t dst, const half_2& src) {
    asm volatile("st.shared.b32 [%1], %0;\n" : : "r"(*(uint32_t*)&src), "r"(dst));
}

// CHECK: static inline void ldg(half_2& dst, half_2* src) {
// CHECK-NEXT:     *(uint32_t*)&dst = *((uint32_t *)(uintptr_t)src);
// CHECK-NEXT: }
__device__ static inline void ldg(half_2& dst, half_2* src) {
    asm volatile("ld.global.b32 %0, [%1];\n" : "=r"(*(uint32_t*)&dst) : "l"(src));
}

// CHECK: static inline void stg(half_2* dst, const half_2& src) {
// CHECK-NEXT:     *((uint32_t *)(uintptr_t)dst) = (*(uint32_t*)&src);
// CHECK-NEXT: }
__device__ static inline void stg(half_2* dst, const half_2& src) {
    asm volatile("st.global.b32 [%1], %0;\n" : : "r"(*(uint32_t*)&src), "l"(dst));
}

// CHECK: static inline void lds(sycl::float2& dst, uint32_t src) {
// CHECK-NEXT:     {dst.x(), dst.y()} = *((float *)(uintptr_t)src);
// CHECK-NEXT: }
__device__ static inline void lds(float2& dst, uint32_t src) {
    asm volatile("ld.shared.v2.f32 {%0, %1}, [%2];\n" : "=f"(dst.x), "=f"(dst.y) : "r"(src));
}

// CHECK: static inline void sts(uint32_t dst, const sycl::float2& src) {
// CHECK-NEXT:     *((float *)(uintptr_t)dst) = {src.x(), src.y()};
// CHECK-NEXT: }
__device__ static inline void sts(uint32_t dst, const float2& src) {
    asm volatile("st.shared.v2.f32 [%2], {%0, %1};\n" : : "f"(src.x), "f"(src.y), "r"(dst));
}

// CHECK: static inline void ldg(sycl::float2& dst, sycl::float2* src) {
// CHECK-NEXT:     {dst.x(), dst.y()} = *((float *)(uintptr_t)src);
// CHECK-NEXT: }
__device__ static inline void ldg(float2& dst, float2* src) {
    asm volatile("ld.global.v2.f32 {%0, %1}, [%2];\n" : "=f"(dst.x), "=f"(dst.y) : "l"(src));
}

// CHECK: static inline void stg(sycl::float2* dst, const sycl::float2& src) {
// CHECK-NEXT:     *((float *)(uintptr_t)dst) = {src.x(), src.y()};
// CHECK-NEXT: }
__device__ static inline void stg(float2* dst, const float2& src) {
    asm volatile("st.global.v2.f32 [%2], {%0, %1};\n" : : "f"(src.x), "f"(src.y), "l"(dst));
}

// CHECK: static inline void lds(sycl::float4& dst, uint32_t src) {
// CHECK-NEXT:     {dst.x(), dst.y(), dst.z(), dst.w()} = *((float *)(uintptr_t)src);
// CHECK-NEXT: }
__device__ static inline void lds(float4& dst, uint32_t src) {
    asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];\n" : "=f"(dst.x), "=f"(dst.y), "=f"(dst.z), "=f"(dst.w) : "r"(src));
}

// CHECK: static inline void sts(uint32_t dst, const sycl::float4& src) {
// CHECK-NEXT:     *((float *)(uintptr_t)dst) = {src.x(), src.y(), src.z(), src.w()};
// CHECK-NEXT: }
__device__ static inline void sts(uint32_t dst, const float4& src) {
    asm volatile("st.shared.v4.f32 [%4], {%0, %1, %2, %3};\n" : : "f"(src.x), "f"(src.y), "f"(src.z), "f"(src.w), "r"(dst));
}

// CHECK: static inline void ldg(sycl::float4& dst, sycl::float4* src) {
// CHECK-NEXT:     {dst.x(), dst.y(), dst.z(), dst.w()} = *((float *)(uintptr_t)src);
// CHECK-NEXT: }
__device__ static inline void ldg(float4& dst, float4* src) {
    asm volatile("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];\n" : "=f"(dst.x), "=f"(dst.y), "=f"(dst.z), "=f"(dst.w) : "l"(src));
}

// CHECK: static inline void stg(sycl::float4* dst, const sycl::float4& src) {
// CHECK-NEXT:     *((float *)(uintptr_t)dst) = {src.x(), src.y(), src.z(), src.w()};
// CHECK-NEXT: }
__device__ static inline void stg(float4* dst, const float4& src) {
    asm volatile("st.global.v4.f32 [%4], {%0, %1, %2, %3};\n" : : "f"(src.x), "f"(src.y), "f"(src.z), "f"(src.w), "l"(dst));
}
   

// clang-format on
