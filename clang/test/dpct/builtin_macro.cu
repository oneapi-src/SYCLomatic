// RUN: dpct --format-range=none -out-root %T/builtin_macro %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/builtin_macro/builtin_macro.dp.cpp --match-full-lines %s

#define MACRO(ID, CMP, S) \
  {                       \
    S;                    \
    if (!(CMP)) {         \
      return ID;          \
    }                     \
  }

// clang-format off
__device__ int f() {
  int s32 = 0, s32x = 1;
  // CHECK: MACRO(7, s32 == INT_MAX, s32 = sycl::add_sat((int32_t)s32x, (int32_t)INT_MAX));
  MACRO(7, s32 == INT_MAX, asm("add.s32.sat %0, %1, %2;" : "=r"(s32) : "r"(s32x), "r"(INT_MAX)));

  return 0;
}
// clang-format on
