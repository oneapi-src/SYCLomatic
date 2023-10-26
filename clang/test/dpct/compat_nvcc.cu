// RUN: dpct --format-range=none -out-root %T/compat_nvcc %s --cuda-include-path="%cuda-path/include" --stop-on-parse-err
// RUN: FileCheck %s --match-full-lines --input-file %T/compat_nvcc/compat_nvcc.dp.cpp

template <class T1, class T2> struct AAAAA {
  template <class T3> void foo(T3 x);
};

// CHECK: template <typename T4, typename T5>
// CHECK-NEXT: template <typename T6>
// CHECK-NEXT: void AAAAA<T4, T5>::foo(T6 x) {}
template <typename T4, typename T5>
template <typename T6>
void AAAAA<T4, T5>::foo<T6>(T6 x) {}
