// RUN: dpct --format-range=none -out-root %T/asm/vsub4 %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/asm/vsub4/vsub4.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/vsub4/vsub4.dp.cpp -o %T/vsub4/vsub4.dp.o %}

__global__ void vsub4() {
  int a, b, c, d;

  // CHECK: d = dpct::extend_vsub4<uint32_t, int32_t, int32_t>(a, b, c);
  // CHECK-NEXT: d = dpct::extend_vsub4_sat<uint32_t, int32_t, int32_t>(a, b, c);
  // CHECK-NEXT: d = dpct::extend_vsub4_add<uint32_t, int32_t, int32_t>(a, b, c);
  // clang-format off
  asm("vsub4.u32.s32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  asm("vsub4.u32.s32.s32.sat %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  asm("vsub4.u32.s32.s32.add %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  // clang-format on
}

int main() {
  vsub4<<<1, 1>>>();
  return 0;
}
