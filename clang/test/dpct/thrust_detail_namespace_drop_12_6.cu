// UNSUPPORTED: system-windows
// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-11.0, cuda-11.1, cuda-11.2, cuda-11.3, cuda-12.6
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v11.0, v11.1, v11.2, v11.3, v12.6
// RUN: dpct --sycl-named-lambda --format-range=none --usm-level=none -out-root %T/thrust_detail_namespace_drop_12_6 -in-root=%S %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++17 -fsized-deallocation
// RUN: FileCheck --input-file %T/thrust_detail_namespace_drop_12_6/thrust_detail_namespace_drop_12_6.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl -DBUILD_TEST  %T/thrust_detail_namespace_drop_12_6/thrust_detail_namespace_drop_12_6.dp.cpp -o %T/thrust_detail_namespace_drop_12_6/thrust_detail_namespace_drop_12_6.dp.o %}

#ifndef BUILD_TEST
#include <iostream>
#include <thrust/detail/type_traits.h>
#include <thrust/detail/allocator/allocator_traits.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/device_vector.h>


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

// CHECK:template <typename T, typename U>
// CHECK-NEXT:void checkTypes() {
// CHECK-NEXT:  if (std::is_same<T, U>::value) {
// CHECK-NEXT:    std::cout << "Types T and U are the same." << std::endl;
// CHECK-NEXT:  } else {
// CHECK-NEXT:    std::cout << "Types T and U are different." << std::endl;
// CHECK-NEXT:  }
// CHECK-NEXT:}
template <typename T, typename U>
void checkTypes() {
  if (thrust::detail::is_same<T, U>::value) {
    std::cout << "Types T and U are the same." << std::endl;
  } else {
    std::cout << "Types T and U are different." << std::endl;
  }
}

#endif
