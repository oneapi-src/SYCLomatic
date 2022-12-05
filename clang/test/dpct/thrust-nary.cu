// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0
// RUN: dpct --format-range=none -out-root %T/thrust-nary %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only --std=c++14
// RUN: FileCheck --input-file %T/thrust-nary/thrust-nary.dp.cpp --match-full-lines %s
// CHECK: #include <sycl/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
#include <thrust/functional.h>

// CHECK: /*
// CHECK-NEXT: DPCT1044:{{[0-9]+}}: thrust::unary_function was removed because std::unary_function has been deprecated in C++11. You may need to remove references to typedefs from thrust::unary_function in the class definition.
// CHECK-NEXT: */
// CHECK-NEXT: struct uf {};
struct uf : public thrust::unary_function<int, int> {};

// CHECK: /*
// CHECK-NEXT: DPCT1044:{{[0-9]+}}: thrust::binary_function was removed because std::binary_function has been deprecated in C++11. You may need to remove references to typedefs from thrust::binary_function in the class definition.
// CHECK-NEXT: */
// CHECK-NEXT: struct bf {};
struct bf : public thrust::binary_function<int, int, int> {};

template <typename T> struct ST {};

// CHECK: /*
// CHECK-NEXT: DPCT1044:{{[0-9]+}}: thrust::unary_function was removed because std::unary_function has been deprecated in C++11. You may need to remove references to typedefs from thrust::unary_function in the class definition.
// CHECK-NEXT: */
// CHECK-NEXT: template <typename T> struct SSTU {};
template <typename T> struct SSTU : thrust::unary_function<ST<T>, T> {};

// CHECK: /*
// CHECK-NEXT: DPCT1044:{{[0-9]+}}: thrust::binary_function was removed because std::binary_function has been deprecated in C++11. You may need to remove references to typedefs from thrust::binary_function in the class definition.
// CHECK-NEXT: */
// CHECK-NEXT: template <typename T> struct SSTB {};
template <typename T> struct SSTB : thrust::binary_function<ST<T>, T, T> {};

