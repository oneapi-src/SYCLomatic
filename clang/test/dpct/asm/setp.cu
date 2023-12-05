// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -out-root %T/setp %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/setp/setp.dp.cpp

// clang-format off
#include <cuda_runtime.h>
#include <cstdint>

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

// clang-format on
