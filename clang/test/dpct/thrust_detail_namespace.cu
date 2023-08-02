// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0
// RUN: dpct --sycl-named-lambda --format-range=none --usm-level=none -out-root %T/thrust_detail_namespace -in-root=%S %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++17 -fsized-deallocation
// RUN: FileCheck --input-file %T/thrust_detail_namespace/thrust_detail_namespace.dp.cpp --match-full-lines %s

#include <iostream>
#include <thrust/detail/type_traits.h>

// CHECK:template <typename T>
// CHECK-NEXT:typename std::enable_if<std::is_integral<T>::value, void>::type
// CHECK-NEXT:print_info(T value) {
// CHECK-NEXT:  std::cout << "Integral value: " << value << std::endl;
// CHECK-NEXT:}
template <typename T>
typename thrust::detail::enable_if<std::is_integral<T>::value, void>::type
print_info(T value) {
  std::cout << "Integral value: " << value << std::endl;
}

// CHECK:template <typename T>
// CHECK-NEXT:typename std::enable_if<std::is_floating_point<T>::value, void>::type
// CHECK-NEXT:print_info(T value) {
// CHECK-NEXT:  std::cout << "Floating-point value: " << value << std::endl;
// CHECK-NEXT:}
template <typename T>
typename thrust::detail::enable_if<std::is_floating_point<T>::value, void>::type
print_info(T value) {
  std::cout << "Floating-point value: " << value << std::endl;
}

void foo() {
  int integer_val = 42;
  float float_val = 3.14f;
  double double_val = 2.71828;

  print_info(integer_val); // Output: Integral value: 42
  print_info(float_val);   // Output: Floating-point value: 3.14
  print_info(double_val);  // Output: Floating-point value: 2.71828
}