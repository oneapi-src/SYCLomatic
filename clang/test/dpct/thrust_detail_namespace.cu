// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-12.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v12.2
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

// CHECK:template <typename T>
// CHECK-NEXT:struct is_integer {
// CHECK-NEXT:    typedef std::false_type type;
// CHECK-NEXT:};
// CHECK-NEXT:template <>
// CHECK-NEXT:struct is_integer<int> {
// CHECK-NEXT:    typedef std::true_type type;
// CHECK-NEXT:};
// CHECK-NEXT:template <>
// CHECK-NEXT:struct is_integer<long> {
// CHECK-NEXT:    typedef std::true_type type;
// CHECK-NEXT:};
template <typename T>
struct is_integer {
  typedef thrust::detail::false_type type;
};
template <>
struct is_integer<int> {
  typedef thrust::detail::true_type type;
};
template <>
struct is_integer<long> {
  typedef thrust::detail::true_type type;
};

// CHECK:template <std::size_t Len, std::size_t Align>
// CHECK-NEXT:void test_aligned_storage_instantiation() {
// CHECK-NEXT:  typedef std::integral_constant<bool, false> ValidAlign;
// CHECK-NEXT:}
template <std::size_t Len, std::size_t Align>
void test_aligned_storage_instantiation() {
  typedef thrust::detail::integral_constant<bool, false> ValidAlign;
}

void foo() {
  int integer_val = 42;
  float float_val = 3.14f;
  double double_val = 2.71828;

  print_info(integer_val); // Output: Integral value: 42
  print_info(float_val);   // Output: Floating-point value: 3.14
  print_info(double_val);  // Output: Floating-point value: 2.71828
}