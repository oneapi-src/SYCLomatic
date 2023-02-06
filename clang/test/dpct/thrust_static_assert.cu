// RUN: dpct --format-range=none -out-root %T/thrust_static_assert %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/thrust_static_assert/thrust_static_assert.dp.cpp --match-full-lines %s

#include <thrust/detail/static_assert.h>

int main() {
  // CHECK: static_assert(sizeof(int) == 4);
  THRUST_STATIC_ASSERT(sizeof(int) == 4);
  return 0;
}
