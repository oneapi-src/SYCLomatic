// RUN: dpct --format-range=none -out-root %T/asm/vmin2 %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/asm/vmin2/vmin2.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/vmin2/vmin2.dp.cpp -o %T/vmin2/vmin2.dp.o %}

__global__ void vmin2() {
  int a, b, c, d;

  // CHECK: d = dpct::extend_vmin2<uint32_t, int32_t, uint32_t>(a, b, c);
  // CHECK-NEXT: d = dpct::extend_vmin2_sat<uint32_t, int32_t, uint32_t>(a, b, c);
  // CHECK-NEXT: d = dpct::extend_vmin2_add<uint32_t, int32_t, uint32_t>(a, b, c);
  // clang-format off
  asm("vmin2.u32.s32.u32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  asm("vmin2.u32.s32.u32.sat %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  asm("vmin2.u32.s32.u32.add %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  // clang-format on
}

int main() {
  vmin2<<<1, 1>>>();
  return 0;
}
