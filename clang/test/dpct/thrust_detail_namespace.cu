// UNSUPPORTED: system-windows
// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-11.0, cuda-11.1, cuda-11.2, cuda-11.3
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v11.0, v11.1, v11.2, v11.3
// RUN: dpct --sycl-named-lambda --format-range=none --usm-level=none -out-root %T/thrust_detail_namespace -in-root=%S %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++17 -fsized-deallocation
// RUN: FileCheck --input-file %T/thrust_detail_namespace/thrust_detail_namespace.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl -DNO_BUILD_TEST  %T/thrust_detail_namespace/thrust_detail_namespace.dp.cpp -o %T/thrust_detail_namespace/thrust_detail_namespace.dp.o %}

#ifndef NO_BUILD_TEST
#include <iostream>
#include <thrust/detail/type_traits.h>
#include <thrust/detail/allocator/allocator_traits.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/device_vector.h>

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

// CHECK:template <class ExampleVector, typename NewType, typename new_alloc> struct vector_like {
// CHECK-NEXT:  typedef dpct::device_vector<NewType, new_alloc> type;
// CHECK-NEXT:};
template <class ExampleVector, typename NewType, typename new_alloc> struct vector_like {
  typedef thrust::detail::vector_base<NewType, new_alloc> type;
};

void foo() {
  int integer_val = 42;
  float float_val = 3.14f;
  double double_val = 2.71828;

  print_info(integer_val); // Output: Integral value: 42
  print_info(float_val);   // Output: Floating-point value: 3.14
  print_info(double_val);  // Output: Floating-point value: 2.71828

  // CHECK:  dpct::device_vector<int> Array1(7);
  // CHECK-NEXT:  dpct::device_vector<int> Array2(7);
  // CHECK-NEXT:  bool t = oneapi::dpl::equal(oneapi::dpl::execution::make_device_policy(dpct::get_out_of_order_queue()), Array1.begin(), Array1.end(), Array2.begin());
  thrust::device_vector<int> Array1(7);
  thrust::device_vector<int> Array2(7);
  bool t = thrust::detail::vector_equal(Array1.begin(), Array1.end(), Array2.begin());

  // CHECK:  typedef dpct::device_vector<int>::iterator Iterator;
  // CHECK-NEXT:  typedef oneapi::dpl::iterator_traits<Iterator>::value_type value_type;
  // CHECK-NEXT:  typedef oneapi::dpl::iterator_traits<Iterator>::difference_type difference_type;
  typedef thrust::device_vector<int>::iterator Iterator;
  typedef thrust::iterator_traits<Iterator>::value_type value_type;
  typedef thrust::iterator_traits<Iterator>::difference_type difference_type;

  // std::bad_alloc("bad_alloc");
  thrust::system::detail::bad_alloc("bad_alloc");
}
#endif
