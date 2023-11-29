// RUN: dpct --format-range=none -out-root %T/asm/vadd2 %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/asm/vadd2/vadd2.dp.cpp

__global__ void vadd2() {
  int a, b, c, d;

  // CHECK: d = dpct::extend_vadd2<int32_t, uint32_t, int32_t>(a, b, c);
  // CHECK-NEXT: d = dpct::extend_vadd2_sat<int32_t, uint32_t, int32_t>(a, b, c);
  // CHECK-NEXT: d = dpct::extend_vadd2_add<int32_t, uint32_t, int32_t>(a, b, c);
  // clang-format off
  asm("vadd2.s32.u32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  asm("vadd2.s32.u32.s32.sat %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  asm("vadd2.s32.u32.s32.add %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  // clang-format on
}

int main() {
  vadd2<<<1, 1>>>();
  return 0;
}
