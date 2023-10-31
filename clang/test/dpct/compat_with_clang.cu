// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1
// RUN: dpct --format-range=none -out-root %T/compat_with_clang %s --cuda-include-path="%cuda-path/include" --stop-on-parse-err
// RUN: FileCheck %s --match-full-lines --input-file %T/compat_with_clang/compat_with_clang.dp.cpp

#include <cstdint>

// CHECK: void foo1(int a, int b) {
// CHECK-NEXT:   sycl::range<3> block{1, 1, dpct::min(512, uint32_t(a * b))};
// CHECK-NEXT: }
void foo1(int a, int b) {
  dim3 block{min(512, uint32_t(a * b))};
}
