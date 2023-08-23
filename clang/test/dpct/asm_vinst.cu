// RUN: dpct -out-root %T/asm_vinst %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/asm_vinst/asm_vinst.dp.cpp


// clang-format off
#include <cstdint>

__global__ void vadd() {
  int a, b, c, d;

  // CHECK: a = dpct::extend_add<int32_t>(b, c);
  asm("vadd.s32.u32.s32 %0, %1, %2;" : "=r"(a) : "r"(b), "r"(c));

  // CHECK: a = dpct::extend_add<uint32_t>(b, c);
  asm("vadd.u32.u32.s32 %0, %1, %2;" : "=r"(a) : "r"(b), "r"(c));

  // CHECK: a = dpct::extend_add_sat<int32_t>(b, c);
  asm("vadd.s32.u32.s32.sat %0, %1, %2;" : "=r"(a) : "r"(b), "r"(c));

  // CHECK: a = dpct::extend_add_sat<uint32_t>(b, c);
  asm("vadd.u32.u32.s32.sat %0, %1, %2;" : "=r"(a) : "r"(b), "r"(c));

  // CHECK: a = dpct::extend_add_sat<int32_t>(b, c, d, sycl::plus<>());
  asm("vadd.s32.u32.s32.sat.add %0, %1, %2, %3;" : "=r"(a) : "r"(b), "r"(c), "r"(d));
  
  // CHECK: a = dpct::extend_add_sat<int32_t>(b, c, d, sycl::minimum<>());
  asm("vadd.s32.u32.s32.sat.min %0, %1, %2, %3;" : "=r"(a) : "r"(b), "r"(c), "r"(d));

  // CHECK: a = dpct::extend_add_sat<int32_t>(b, c, d, sycl::maximum<>());
  asm("vadd.s32.u32.s32.sat.max %0, %1, %2, %3;" : "=r"(a) : "r"(b), "r"(c), "r"(d));
}

__global__ void vsub() {
  int a, b, c, d;

  // CHECK: a = dpct::extend_sub<int32_t>(b, c);
  asm("vsub.s32.u32.s32 %0, %1, %2;" : "=r"(a) : "r"(b), "r"(c));

  // CHECK: a = dpct::extend_sub<uint32_t>(b, c);
  asm("vsub.u32.u32.s32 %0, %1, %2;" : "=r"(a) : "r"(b), "r"(c));

  // CHECK: a = dpct::extend_sub_sat<int32_t>(b, c);
  asm("vsub.s32.u32.s32.sat %0, %1, %2;" : "=r"(a) : "r"(b), "r"(c));

  // CHECK: a = dpct::extend_sub_sat<uint32_t>(b, c);
  asm("vsub.u32.u32.s32.sat %0, %1, %2;" : "=r"(a) : "r"(b), "r"(c));

  // CHECK: a = dpct::extend_sub_sat<int32_t>(b, c, d, sycl::plus<>());
  asm("vsub.s32.u32.s32.sat.add %0, %1, %2, %3;" : "=r"(a) : "r"(b), "r"(c), "r"(d));
  
  // CHECK: a = dpct::extend_sub_sat<int32_t>(b, c, d, sycl::minimum<>());
  asm("vsub.s32.u32.s32.sat.min %0, %1, %2, %3;" : "=r"(a) : "r"(b), "r"(c), "r"(d));

  // CHECK: a = dpct::extend_sub_sat<int32_t>(b, c, d, sycl::maximum<>());
  asm("vsub.s32.u32.s32.sat.max %0, %1, %2, %3;" : "=r"(a) : "r"(b), "r"(c), "r"(d));
}

__global__ void vabsdiff() {
  int a, b, c, d;

  // CHECK: a = dpct::extend_absdiff<int32_t>(b, c);
  asm("vabsdiff.s32.u32.s32 %0, %1, %2;" : "=r"(a) : "r"(b), "r"(c));

  // CHECK: a = dpct::extend_absdiff<uint32_t>(b, c);
  asm("vabsdiff.u32.u32.s32 %0, %1, %2;" : "=r"(a) : "r"(b), "r"(c));

  // CHECK: a = dpct::extend_absdiff_sat<int32_t>(b, c);
  asm("vabsdiff.s32.u32.s32.sat %0, %1, %2;" : "=r"(a) : "r"(b), "r"(c));

  // CHECK: a = dpct::extend_absdiff_sat<uint32_t>(b, c);
  asm("vabsdiff.u32.u32.s32.sat %0, %1, %2;" : "=r"(a) : "r"(b), "r"(c));

  // CHECK: a = dpct::extend_absdiff_sat<int32_t>(b, c, d, sycl::plus<>());
  asm("vabsdiff.s32.u32.s32.sat.add %0, %1, %2, %3;" : "=r"(a) : "r"(b), "r"(c), "r"(d));
  
  // CHECK: a = dpct::extend_absdiff_sat<int32_t>(b, c, d, sycl::minimum<>());
  asm("vabsdiff.s32.u32.s32.sat.min %0, %1, %2, %3;" : "=r"(a) : "r"(b), "r"(c), "r"(d));

  // CHECK: a = dpct::extend_absdiff_sat<int32_t>(b, c, d, sycl::maximum<>());
  asm("vabsdiff.s32.u32.s32.sat.max %0, %1, %2, %3;" : "=r"(a) : "r"(b), "r"(c), "r"(d));
}

__global__ void vmin() {
  int a, b, c, d;

  // CHECK: a = dpct::extend_min<int32_t>(b, c);
  asm("vmin.s32.u32.s32 %0, %1, %2;" : "=r"(a) : "r"(b), "r"(c));

  // CHECK: a = dpct::extend_min<uint32_t>(b, c);
  asm("vmin.u32.u32.s32 %0, %1, %2;" : "=r"(a) : "r"(b), "r"(c));

  // CHECK: a = dpct::extend_min_sat<int32_t>(b, c);
  asm("vmin.s32.u32.s32.sat %0, %1, %2;" : "=r"(a) : "r"(b), "r"(c));

  // CHECK: a = dpct::extend_min_sat<uint32_t>(b, c);
  asm("vmin.u32.u32.s32.sat %0, %1, %2;" : "=r"(a) : "r"(b), "r"(c));

  // CHECK: a = dpct::extend_min_sat<int32_t>(b, c, d, sycl::plus<>());
  asm("vmin.s32.u32.s32.sat.add %0, %1, %2, %3;" : "=r"(a) : "r"(b), "r"(c), "r"(d));
  
  // CHECK: a = dpct::extend_min_sat<int32_t>(b, c, d, sycl::minimum<>());
  asm("vmin.s32.u32.s32.sat.min %0, %1, %2, %3;" : "=r"(a) : "r"(b), "r"(c), "r"(d));

  // CHECK: a = dpct::extend_min_sat<int32_t>(b, c, d, sycl::maximum<>());
  asm("vmin.s32.u32.s32.sat.max %0, %1, %2, %3;" : "=r"(a) : "r"(b), "r"(c), "r"(d));
}

__global__ void vmax() {
  int a, b, c, d;

  // CHECK: a = dpct::extend_max<int32_t>(b, c);
  asm("vmax.s32.u32.s32 %0, %1, %2;" : "=r"(a) : "r"(b), "r"(c));

  // CHECK: a = dpct::extend_max<uint32_t>(b, c);
  asm("vmax.u32.u32.s32 %0, %1, %2;" : "=r"(a) : "r"(b), "r"(c));

  // CHECK: a = dpct::extend_max_sat<int32_t>(b, c);
  asm("vmax.s32.u32.s32.sat %0, %1, %2;" : "=r"(a) : "r"(b), "r"(c));

  // CHECK: a = dpct::extend_max_sat<uint32_t>(b, c);
  asm("vmax.u32.u32.s32.sat %0, %1, %2;" : "=r"(a) : "r"(b), "r"(c));

  // CHECK: a = dpct::extend_max_sat<int32_t>(b, c, d, sycl::plus<>());
  asm("vmax.s32.u32.s32.sat.add %0, %1, %2, %3;" : "=r"(a) : "r"(b), "r"(c), "r"(d));
  
  // CHECK: a = dpct::extend_max_sat<int32_t>(b, c, d, sycl::minimum<>());
  asm("vmax.s32.u32.s32.sat.min %0, %1, %2, %3;" : "=r"(a) : "r"(b), "r"(c), "r"(d));

  // CHECK: a = dpct::extend_max_sat<int32_t>(b, c, d, sycl::maximum<>());
  asm("vmax.s32.u32.s32.sat.max %0, %1, %2, %3;" : "=r"(a) : "r"(b), "r"(c), "r"(d));
}
// clang-format on
