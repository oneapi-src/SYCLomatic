// RUN: dpct --format-range=none -out-root %T/asm %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/asm/asm.dp.cpp
// clang-format off
#include "cuda_runtime.h"

#include <stdio.h>

__global__ void gpu_ptx(int *d_ptr, int length) {
  int elemID = blockIdx.x * blockDim.x + threadIdx.x;

  for (int innerloops = 0; innerloops < 100000; innerloops++) {
    if (elemID < length) {
      unsigned int laneid;
      unsigned int warpid;
      unsigned int WARP_SZ;
      // CHECK: laneid = item_ct1.get_sub_group().get_local_linear_id();
      asm("mov.u32 %0, %%laneid;" : "=r"(laneid));

      // CHECK: warpid = item_ct1.get_sub_group().get_group_linear_id();
      asm("mov.u32 %0, %%warpid;" : "=r"(warpid));

      // CHECK: WARP_SZ = item_ct1.get_sub_group().get_local_range().get(0);
      asm("mov.u32 %0, WARP_SZ;" : "=r"(WARP_SZ));
      
      // ill-formed inline asm, missing ';' at the end of mov instruction.
      // CHECK: /*
      // CHECK-NEXT: DPCT1053:{{.*}} Migration of device assembly code is not supported.
      // CHECK-NEXT: */
      asm("mov.u32 %0, WARP_SZ" : "=r"(WARP_SZ));
      d_ptr[elemID] = laneid;
    }
  }
}

// CHECK: void asm_only(const sycl::nd_item<3> &item_ct1) {
// CHECK-NEXT:  unsigned laneid;
// CHECK-NEXT:  laneid = item_ct1.get_sub_group().get_local_linear_id();
// CHECK-MEXT: }
__global__ void asm_only() {
  unsigned laneid;
  asm volatile ("mov.u32 %0, %%laneid;" : "=r"(laneid));
}

// CHECK: void setp() {
// CHECK:   float x = 1.0, y;
// CHECK:   uint32_t a, b;
// CHECK:   {
// CHECK-NEXT:     bool _p_p[20], _p_q[20];
// CHECK-NEXT:     float fp;
// CHECK-NEXT:     fp = sycl::bit_cast<float>(uint32_t(0x3f800000U));
// CHECK-NEXT:     _p_p[1] = dpct::compare<float>(x, fp, std::equal_to<>());
// CHECK-NEXT:     _p_p[2] = dpct::compare<float>(x, fp, std::not_equal_to<>());
// CHECK-NEXT:     _p_p[3] = dpct::compare<float>(x, fp, std::less<>());
// CHECK-NEXT:     _p_p[4] = dpct::compare<float>(x, fp, std::less_equal<>());
// CHECK-NEXT:     _p_p[5] = dpct::compare<float>(x, fp, std::greater<>());
// CHECK-NEXT:     _p_p[6] = dpct::compare<float>(x, fp, std::greater_equal<>());
// CHECK-NEXT:     _p_p[7] = dpct::unordered_compare<float>(x, fp, std::equal_to<>());
// CHECK-NEXT:     _p_p[8] = dpct::unordered_compare<float>(x, fp, std::not_equal_to<>());
// CHECK-NEXT:     _p_p[9] = dpct::unordered_compare<float>(x, fp, std::less<>());
// CHECK-NEXT:     _p_p[10] = dpct::unordered_compare<float>(x, fp, std::less_equal<>());
// CHECK-NEXT:     _p_p[11] = dpct::unordered_compare<float>(x, fp, std::greater<>());
// CHECK-NEXT:     _p_p[12] = dpct::unordered_compare<float>(x, fp, std::greater_equal<>());
// CHECK-NEXT:     _p_p[13] = !sycl::isnan(x) && !sycl::isnan(fp);
// CHECK-NEXT:     _p_p[14] = sycl::isnan(x) || sycl::isnan(fp);
// CHECK-NEXT:     _p_p[1] = dpct::compare<double>(x, fp, std::equal_to<>());
// CHECK-NEXT:     _p_p[2] = dpct::compare<double>(x, fp, std::not_equal_to<>());
// CHECK-NEXT:     _p_p[3] = dpct::compare<double>(x, fp, std::less<>());
// CHECK-NEXT:     _p_p[4] = dpct::compare<double>(x, fp, std::less_equal<>());
// CHECK-NEXT:     _p_p[5] = dpct::compare<double>(x, fp, std::greater<>());
// CHECK-NEXT:     _p_p[6] = dpct::compare<double>(x, fp, std::greater_equal<>());
// CHECK-NEXT:     _p_p[7] = dpct::unordered_compare<double>(x, fp, std::equal_to<>());
// CHECK-NEXT:     _p_p[8] = dpct::unordered_compare<double>(x, fp, std::not_equal_to<>());
// CHECK-NEXT:     _p_p[9] = dpct::unordered_compare<double>(x, fp, std::less<>());
// CHECK-NEXT:     _p_p[10] = dpct::unordered_compare<double>(x, fp, std::less_equal<>());
// CHECK-NEXT:     _p_p[11] = dpct::unordered_compare<double>(x, fp, std::greater<>());
// CHECK-NEXT:     _p_p[12] = dpct::unordered_compare<double>(x, fp, std::greater_equal<>());
// CHECK-NEXT:     _p_p[13] = !sycl::isnan(x) && !sycl::isnan(fp);
// CHECK-NEXT:     _p_p[14] = sycl::isnan(x) || sycl::isnan(fp);
// CHECK-NEXT:     _p_q[1] = a == b;
// CHECK-NEXT:     _p_q[2] = a != b;
// CHECK-NEXT:     _p_q[3] = a < b;
// CHECK-NEXT:     _p_q[4] = a <= b;
// CHECK-NEXT:     _p_q[5] = a > b;
// CHECK-NEXT:     _p_q[6] = a >= b;
// CHECK-NEXT:     if (_p_p[0]) {
// CHECK-NEXT:       y = fp;
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK: }
__device__ void setp() {
  float x = 1.0, y;
  uint32_t a, b;
  asm("{\n\t"
      " .reg .pred %%p<20>, %%q<20>;\n\t"
      " .reg .f32 fp;\n\t"
      " mov.f32 fp, 0F3f800000;\n\t"
      " setp.eq.f32 %%p1, %1, fp;\n\t"
      " setp.ne.f32 %%p2, %1, fp;\n\t"
      " setp.lt.f32 %%p3, %1, fp;\n\t"
      " setp.le.f32 %%p4, %1, fp;\n\t"
      " setp.gt.f32 %%p5, %1, fp;\n\t"
      " setp.ge.f32 %%p6, %1, fp;\n\t"
      " setp.equ.f32 %%p7, %1, fp;\n\t"
      " setp.neu.f32 %%p8, %1, fp;\n\t"
      " setp.ltu.f32 %%p9, %1, fp;\n\t"
      " setp.leu.f32 %%p10, %1, fp;\n\t"
      " setp.gtu.f32 %%p11, %1, fp;\n\t"
      " setp.geu.f32 %%p12, %1, fp;\n\t"
      " setp.num.f32 %%p13, %1, fp;\n\t"
      " setp.nan.f32 %%p14, %1, fp;\n\t"
      " setp.eq.f64 %%p1, %1, fp;\n\t"
      " setp.ne.f64 %%p2, %1, fp;\n\t"
      " setp.lt.f64 %%p3, %1, fp;\n\t"
      " setp.le.f64 %%p4, %1, fp;\n\t"
      " setp.gt.f64 %%p5, %1, fp;\n\t"
      " setp.ge.f64 %%p6, %1, fp;\n\t"
      " setp.equ.f64 %%p7, %1, fp;\n\t"
      " setp.neu.f64 %%p8, %1, fp;\n\t"
      " setp.ltu.f64 %%p9, %1, fp;\n\t"
      " setp.leu.f64 %%p10, %1, fp;\n\t"
      " setp.gtu.f64 %%p11, %1, fp;\n\t"
      " setp.geu.f64 %%p12, %1, fp;\n\t"
      " setp.num.f64 %%p13, %1, fp;\n\t"
      " setp.nan.f64 %%p14, %1, fp;\n\t"
      " setp.eq.u32 %%q1, %2, %3;\n\t"
      " setp.ne.s32 %%q2, %2, %3;\n\t"
      " setp.lt.s32 %%q3, %2, %3;\n\t"
      " setp.le.s32 %%q4, %2, %3;\n\t"
      " setp.gt.s32 %%q5, %2, %3;\n\t"
      " setp.ge.s32 %%q6, %2, %3;\n\t"
      " @%%p mov.f32 %0, fp;\n\t"
      "}"
      : "=f"(y) : "f"(x), "r"(a), "r"(b));
}

// CHECK: void relax_asm() {
// CHECK:   float x = 1.0, y;
// CHECK:   uint32_t a, b;
// CHECK:   {
// CHECK-NEXT:     bool _p_p[20], _p_q[20];
// CHECK-NEXT:     float fp;
// CHECK-NEXT:     fp = sycl::bit_cast<float>(uint32_t(0x3f800000U));
// CHECK-NEXT:     _p_p[1] = dpct::compare<float>(x, fp, std::equal_to<>());
// CHECK-NEXT:     _p_p[2] = dpct::compare<float>(x, fp, std::not_equal_to<>());
// CHECK-NEXT:     _p_p[3] = dpct::compare<float>(x, fp, std::less<>());
// CHECK-NEXT:     _p_p[4] = dpct::compare<float>(x, fp, std::less_equal<>());
// CHECK-NEXT:     _p_p[5] = dpct::compare<float>(x, fp, std::greater<>());
// CHECK-NEXT:     _p_p[6] = dpct::compare<float>(x, fp, std::greater_equal<>());
// CHECK-NEXT:     _p_p[7] = dpct::unordered_compare<float>(x, fp, std::equal_to<>());
// CHECK-NEXT:     _p_p[8] = dpct::unordered_compare<float>(x, fp, std::not_equal_to<>());
// CHECK-NEXT:     _p_p[9] = dpct::unordered_compare<float>(x, fp, std::less<>());
// CHECK-NEXT:     _p_p[10] = dpct::unordered_compare<float>(x, fp, std::less_equal<>());
// CHECK-NEXT:     _p_p[11] = dpct::unordered_compare<float>(x, fp, std::greater<>());
// CHECK-NEXT:     _p_p[12] = dpct::unordered_compare<float>(x, fp, std::greater_equal<>());
// CHECK-NEXT:     _p_p[13] = !sycl::isnan(x) && !sycl::isnan(fp);
// CHECK-NEXT:     _p_p[14] = sycl::isnan(x) || sycl::isnan(fp);
// CHECK-NEXT:     _p_p[1] = dpct::compare<double>(x, fp, std::equal_to<>());
// CHECK-NEXT:     _p_p[2] = dpct::compare<double>(x, fp, std::not_equal_to<>());
// CHECK-NEXT:     _p_p[3] = dpct::compare<double>(x, fp, std::less<>());
// CHECK-NEXT:     _p_p[4] = dpct::compare<double>(x, fp, std::less_equal<>());
// CHECK-NEXT:     _p_p[5] = dpct::compare<double>(x, fp, std::greater<>());
// CHECK-NEXT:     _p_p[6] = dpct::compare<double>(x, fp, std::greater_equal<>());
// CHECK-NEXT:     _p_p[7] = dpct::unordered_compare<double>(x, fp, std::equal_to<>());
// CHECK-NEXT:     _p_p[8] = dpct::unordered_compare<double>(x, fp, std::not_equal_to<>());
// CHECK-NEXT:     _p_p[9] = dpct::unordered_compare<double>(x, fp, std::less<>());
// CHECK-NEXT:     _p_p[10] = dpct::unordered_compare<double>(x, fp, std::less_equal<>());
// CHECK-NEXT:     _p_p[11] = dpct::unordered_compare<double>(x, fp, std::greater<>());
// CHECK-NEXT:     _p_p[12] = dpct::unordered_compare<double>(x, fp, std::greater_equal<>());
// CHECK-NEXT:     _p_p[13] = !sycl::isnan(x) && !sycl::isnan(fp);
// CHECK-NEXT:     _p_p[14] = sycl::isnan(x) || sycl::isnan(fp);
// CHECK-NEXT:     _p_q[1] = a == b;
// CHECK-NEXT:     _p_q[2] = a != b;
// CHECK-NEXT:     _p_q[3] = a < b;
// CHECK-NEXT:     _p_q[4] = a <= b;
// CHECK-NEXT:     _p_q[5] = a > b;
// CHECK-NEXT:     _p_q[6] = a >= b;
// CHECK-NEXT:     if (_p_p[0]) {
// CHECK-NEXT:       y = fp;
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK: }
__device__ void relax_asm() {
  float x = 1.0, y;
  uint32_t a, b;
  asm("{\n\t"
      " .reg .pred %%p<20>, %%q<20>;\n\t"
      " .reg .f32 fp;\n\t"
      " mov.f32 fp, 0F3f800000;\n\t"
      " setp.eq.f32 %%p1, %1, fp;\n\t"
      " setp.ne.f32 %%p2, %1, fp;\n\t"
      " setp.lt.f32 %%p3, %1, fp;\n\t"
      " setp.le.f32 %%p4, %1, fp;\n\t"
      " setp.gt.f32 %%p5, %1, fp;\n\t"
      " setp.ge.f32 %%p6, %1, fp;\n\t"
      " setp.equ.f32 %%p7, %1, fp;\n\t"
      " setp.neu.f32 %%p8, %1, fp;\n\t"
      " setp.ltu.f32 %%p9, %1, fp;\n\t"
      " setp.leu.f32 %%p10, %1, fp;\n\t"
      " setp.gtu.f32 %%p11, %1, fp;\n\t"
      " setp.geu.f32 %%p12, %1, fp;\n\t"
      " setp.num.f32 %%p13, %1, fp;\n\t"
      " setp.nan.f32 %%p14, %1, fp;\n\t"
      " setp.eq.f64 %%p1, %1, fp;\n\t"
      " setp.ne.f64 %%p2, %1, fp;\n\t"
      " setp.lt.f64 %%p3, %1, fp;\n\t"
      " setp.le.f64 %%p4, %1, fp;\n\t"
      " setp.gt.f64 %%p5, %1, fp;\n\t"
      " setp.ge.f64 %%p6, %1, fp;\n\t"
      " setp.equ.f64 %%p7, %1, fp;\n\t"
      " setp.neu.f64 %%p8, %1, fp;\n\t"
      " setp.ltu.f64 %%p9, %1, fp;\n\t"
      " setp.leu.f64 %%p10, %1, fp;\n\t"
      " setp.gtu.f64 %%p11, %1, fp;\n\t"
      " setp.geu.f64 %%p12, %1, fp;\n\t"
      " setp.num.f64 %%p13, %1, fp;\n\t"
      " setp.nan.f64 %%p14, %1, fp;\n\t"
      " setp.eq.u32 %%q1, %2, %3;\n\t"
      " setp.ne.s32 %%q2, %2, %3;\n\t"
      " setp.lt.s32 %%q3, %2, %3;\n\t"
      " setp.le.s32 %%q4, %2, %3;\n\t"
      " setp.gt.s32 %%q5, %2, %3;\n\t"
      " setp.ge.s32 %%q6, %2, %3;\n\t"
      " @%%p mov.f32 %0, fp;\n\t"
      "}"
      : "=f"(y) : "f"(x) "r"(a) "r"(b));
}

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

// CHECK: void clean_specifial_character() {
// CHECK:   {
// CHECK-NEXT:     uint32_t _p_p_d_p, p_d___p_;
// CHECK-NEXT:     _p_p_d_p = 0x3F;
// CHECK-NEXT:     p_d___p_ = 0x3F;
// CHECK-NEXT:   }
// CHECK: }
__global__ void clean_specifial_character(void) {
  asm("{\n\t"
      " .reg .u32 %%p$p, p$_p_;\n\t"
      " mov.u32 %%p$p, 0x3F;\n\t"
      " mov.u32 p$_p_, 0x3F;\n\t"
      "}");
}

// CHECK: void variable_alignas() {
// CHECK:   {
// CHECK-NEXT:     alignas(8) uint32_t _p_p_d_p;
// CHECK-NEXT:     _p_p_d_p = 0x3F;
// CHECK-NEXT:   }
// CHECK: }
__global__ void variable_alignas(void) {
  asm("{\n\t"
      " .reg .align 8 .u32 %%p$p;\n\t"
      " mov.u32 %%p$p, 0x3F;\n\t"
      "}");
}

// CHECK: int cond(int x) {
// CHECK:   int y = 0;
// CHECK:   {
// CHECK-NEXT:     bool _p_p;
// CHECK-NEXT:     uint32_t _p_r[10];
// CHECK-NEXT:     _p_r[1] = 0x3F;
// CHECK-NEXT:     _p_r[2] = 2023;
// CHECK-NEXT:     _p_p = x == 34;
// CHECK-NEXT:     if (_p_p) {
// CHECK-NEXT:       y = 1;
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK:   return y;
// CHECK: }
__device__ int cond(int x) {
  int y = 0;
  asm("{\n\t"
      " .reg .pred %%p;\n\t"
      " .reg .u32 %%r<10>;\n\t"
      " mov.u32 %%r1, 0x3F;\n\t"
      " mov.u32 %%r2, 2023;\n\t"
      " setp.eq.s32 %%p, %1, 34;\n\t" // x == 34?
      " @%%p mov.s32 %0, 1;\n\t"      // set y to 1 if true
      "}"                             // conceptually y = (x==34)?1:y
      : "+r"(y) : "r" (x));
  return y;
}

// CHECK: void declaration(int *ec) {
// CHECK: #define COND(X, Y) \
// CHECK-NEXT: {\
// CHECK-NEXT: bool p;\
// CHECK-NEXT: p = X == 34;\
// CHECK-NEXT: if (p) {\
// CHECK-NEXT: Y = 1;\
// CHECK-NEXT: }\
// CHECK-NEXT: }
// CHECK: #define ARR(V) \
// CHECK-NEXT: {\
// CHECK-NEXT: uint32_t p[10];\
// CHECK-NEXT: p[0] = 2013;\
// CHECK-NEXT: p[1] = p[0];\
// CHECK-NEXT: p[2] = p[1];\
// CHECK-NEXT: p[3] = p[2];\
// CHECK-NEXT: p[4] = p[3];\
// CHECK-NEXT: p[5] = p[4];\
// CHECK-NEXT: p[6] = p[5];\
// CHECK-NEXT: p[7] = p[6];\
// CHECK-NEXT: p[8] = p[7];\
// CHECK-NEXT: p[9] = p[8];\
// CHECK-NEXT: V = p[9];\
// CHECK-NEXT: }
// CHECK: int x = 34;
// CHECK-NEXT: int y = 0;
// CHECK-NEXT: COND(x, y);
// CHECK: x = 33;
// CHECK-NEXT: y = 0;
// CHECK-NEXT: COND(x, y);
// CHECK: x = 0;
// CHECK-NEXT: ARR(x);
// CHECK: }
__global__ void declaration(int *ec) {
#define COND(X, Y)                  \
  asm("{\n\t"                       \
      " .reg .pred p;\n\t"          \
      " setp.eq.s32 p, %1, 34;\n\t" \
      " @p mov.s32 %0, 1;\n\t"      \
      "}"                           \
      : "+r"(Y) : "r" (X))

#define ARR(V)                      \
  asm("{\n\t"                       \
      " .reg .u32 p<10>;\n\t"       \
      " mov.u32 p0, 2013;\n\t"      \
      " mov.u32 p1, p0;\n\t"        \
      " mov.u32 p2, p1;\n\t"        \
      " mov.u32 p3, p2;\n\t"        \
      " mov.u32 p4, p3;\n\t"        \
      " mov.u32 p5, p4;\n\t"        \
      " mov.u32 p6, p5;\n\t"        \
      " mov.u32 p7, p6;\n\t"        \
      " mov.u32 p8, p7;\n\t"        \
      " mov.u32 p9, p8;\n\t"        \
      " mov.u32 %0, p9;\n\t"        \
      "}"                           \
      : "=r"(V))

  int x = 34;
  int y = 0;
  COND(x, y);
  if (y != 1) {
    *ec = 1;
    return;
  }

  x = 33;
  y = 0;
  COND(x, y);
  if (y == 1) {
    *ec = 2;
    return;
  }

  x = 0;
  ARR(x);
  if (x != 2013) {
    *ec = 3;
    return;
  }

  *ec = 0;
}

int main(int argc, char **argv) {
  const int N = 1000;
  int *d_ptr;
  cudaMalloc(&d_ptr, N * sizeof(int));
  int *h_ptr;
  cudaMallocHost(&h_ptr, N * sizeof(int));

  float time_elapsed = 0;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  dim3 cudaBlockSize(256, 1, 1);
  dim3 cudaGridSize((N + cudaBlockSize.x - 1) / cudaBlockSize.x, 1, 1);
  gpu_ptx<<<cudaGridSize, cudaBlockSize >>>(d_ptr, N);
  cudaGetLastError();
  cudaDeviceSynchronize();

  asm_only<<<1, 1>>>();

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(start);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time_elapsed, start, stop);

  printf("Time Used on GPU:%f(ms)\n", time_elapsed);

  __asm__("movl %esp,%eax");

  return 0;
  // CHECK: printf("Time Used on GPU:%f(ms)\n", time_elapsed);
  // CHECK-NOT: DPCT1053:{{[0-9]+}}: Migration of device assembly code is not supported.
  // CHECK: return 0;
}
__device__ void test(void* dst, const void* src) {
  uint4* data = reinterpret_cast<uint4*>(dst);
 // CHECK: /*
 // CHECK-NEXT: DPCT1053:{{.*}} Migration of device assembly code is not supported.
 // CHECK-NEXT: */
  asm volatile("ld.global.ca.v4.u32 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(data[0].x), "=r"(data[0].y), "=r"(data[0].z), "=r"(data[0].w)
      : "l"(src));

}
// clang-format on
