// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0
// RUN: dpct -out-root %T/operator_overload/operator_overload_template %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck --input-file %T/operator_overload/operator_overload_template/operator_overload_template.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/operator_overload/operator_overload_template/operator_overload_template.dp.cpp -o %T/operator_overload/operator_overload_template/operator_overload_template.dp.o %}

#include "cuda_fp16.h"

// CHECK: template <typename T> struct S {};
// CHECK-NEXT: template <typename T> void operator*(T a, const S<T> &b) {}
template <typename T> struct S {};
template <typename T> void operator*(T a, const S<T> &b) {}

int main() {
  S<float> s1;
  S<__half> s2;
  // CHECK: (float)1 * s1;
  (float)1 * s1;
  // CHECK: sycl::half(1.0) * s2;
  __half(1.0) * s2;
  return 0;
}
