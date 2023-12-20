// RUN: dpct --format-range=none -out-root %T/asm/vabsdiff2 %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/asm/vabsdiff2/vabsdiff2.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/vabsdiff2/vabsdiff2.dp.cpp -o %T/vabsdiff2/vabsdiff2.dp.o %}

__global__ void vabsdiff2() {
  int a, b, c, d;

  // CHECK: d = dpct::extend_vabsdiff2<int32_t, int32_t, uint32_t>(a, b, c);
  // CHECK-NEXT: d = dpct::extend_vabsdiff2_sat<int32_t, int32_t, uint32_t>(a, b, c);
  // CHECK-NEXT: d = dpct::extend_vabsdiff2_add<int32_t, int32_t, uint32_t>(a, b, c);
  // clang-format off
  asm("vabsdiff2.s32.s32.u32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  asm("vabsdiff2.s32.s32.u32.sat %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  asm("vabsdiff2.s32.s32.u32.add %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  // clang-format on
}

int main() {
  vabsdiff2<<<1, 1>>>();
  return 0;
}
