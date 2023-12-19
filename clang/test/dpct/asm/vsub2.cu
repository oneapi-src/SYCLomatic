// RUN: dpct --format-range=none -out-root %T/asm/vsub2 %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/asm/vsub2/vsub2.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/vsub2/vsub2.dp.cpp -o %T/vsub2/vsub2.dp.o %}

__global__ void vsub2() {
  int a, b, c, d;

  // CHECK: d = dpct::extend_vsub2<uint32_t, int32_t, int32_t>(a, b, c);
  // CHECK-NEXT: d = dpct::extend_vsub2_sat<uint32_t, int32_t, int32_t>(a, b, c);
  // CHECK-NEXT: d = dpct::extend_vsub2_add<uint32_t, int32_t, int32_t>(a, b, c);
  // clang-format off
  asm("vsub2.u32.s32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  asm("vsub2.u32.s32.s32.sat %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  asm("vsub2.u32.s32.s32.add %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  // clang-format on
}

int main() {
  vsub2<<<1, 1>>>();
  return 0;
}
