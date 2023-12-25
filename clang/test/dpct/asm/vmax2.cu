// RUN: dpct --format-range=none -out-root %T/asm/vmax2 %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/asm/vmax2/vmax2.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/vmax2/vmax2.dp.cpp -o %T/vmax2/vmax2.dp.o %}

__global__ void vmax2() {
  int a, b, c, d;

  // CHECK: d = dpct::extend_vmax2<uint32_t, uint32_t, uint32_t>(a, b, c);
  // CHECK-NEXT: d = dpct::extend_vmax2_sat<uint32_t, uint32_t, uint32_t>(a, b, c);
  // CHECK-NEXT: d = dpct::extend_vmax2_add<uint32_t, uint32_t, uint32_t>(a, b, c);
  // clang-format off
  asm("vmax2.u32.u32.u32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  asm("vmax2.u32.u32.u32.sat %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  asm("vmax2.u32.u32.u32.add %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  // clang-format on
}

int main() {
  vmax2<<<1, 1>>>();
  return 0;
}
