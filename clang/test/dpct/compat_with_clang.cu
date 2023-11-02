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


template <class T1, class T2> struct AAAAA {
  template <class T3> void foo(T3 x);
};

// CHECK: template <typename T4, typename T5>
// CHECK-NEXT: template <typename T6>
// CHECK-NEXT: void AAAAA<T4, T5>::foo(T6 x) {}
template <typename T4, typename T5>
template <typename T6>
void AAAAA<T4, T5>::foo<T6>(T6 x) {}
