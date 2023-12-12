// RUN: dpct --format-range=none -out-root %T/asm/vmin4 %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/asm/vmin4/vmin4.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/vmin4/vmin4.dp.cpp -o %T/vmin4/vmin4.dp.o %}

__global__ void vmin4() {
  int a, b, c, d;

  // CHECK: d = dpct::extend_vmin4<uint32_t, int32_t, uint32_t>(a, b, c);
  // CHECK-NEXT: d = dpct::extend_vmin4_sat<uint32_t, int32_t, uint32_t>(a, b, c);
  // CHECK-NEXT: d = dpct::extend_vmin4_add<uint32_t, int32_t, uint32_t>(a, b, c);
  // clang-format off
  asm("vmin4.u32.s32.u32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  asm("vmin4.u32.s32.u32.sat %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  asm("vmin4.u32.s32.u32.add %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  // clang-format on
}

int main() {
  vmin4<<<1, 1>>>();
  return 0;
}
