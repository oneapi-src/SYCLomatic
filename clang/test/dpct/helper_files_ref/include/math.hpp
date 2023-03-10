//==---- math.hpp ---------------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __DPCT_MATH_HPP__
#define __DPCT_MATH_HPP__

#include <sycl/sycl.hpp>

namespace dpct {
/// Performs half comparison.
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] binary_op functor that implements the binary operation
/// \returns the comparison result
template <class BinaryOperation>
inline bool compare_half(const sycl::half a, const sycl::half b,
                         const BinaryOperation binary_op) {
  return binary_op(a, b);
}
inline bool compare_half(const sycl::half a, const sycl::half b,
                         const std::not_equal_to<> binary_op) {
  return !sycl::isnan(a) && !sycl::isnan(b) && a != b;
}

/// Performs half unordered comparison.
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] binary_op functor that implements the binary operation
/// \returns the comparison result
template <class BinaryOperation>
inline bool unordered_compare_half(const sycl::half a, const sycl::half b,
                                   const BinaryOperation binary_op) {
  return sycl::isnan(a) || sycl::isnan(b) || binary_op(a, b);
}

/// Performs half2 comparison and return a bool value.
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] binary_op functor that implements the binary operation
/// \returns the comparison result
template <class BinaryOperation>
inline bool compare_both_half2(const sycl::half2 a, const sycl::half2 b,
                               const BinaryOperation binary_op) {
  return compare_half(a.x(), b.x(), binary_op) &&
         compare_half(a.y(), b.y(), binary_op);
}

/// Performs half2 unordered comparison and return a bool value.
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] binary_op functor that implements the binary operation
/// \returns the comparison result
template <class BinaryOperation>
inline bool unordered_compare_both_half2(const sycl::half2 a,
                                         const sycl::half2 b,
                                         const BinaryOperation binary_op) {
  return unordered_compare_half(a.x(), b.x(), binary_op) &&
         unordered_compare_half(a.y(), b.y(), binary_op);
}

/// Performs half2 comparison and return a half2 value.
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] binary_op functor that implements the binary operation
/// \returns the comparison result
template <class BinaryOperation>
inline sycl::half2 compare_half2(const sycl::half2 a, const sycl::half2 b,
                                 const BinaryOperation binary_op) {
  return {compare_half(a.x(), b.x(), binary_op),
          compare_half(a.y(), b.y(), binary_op)};
}

/// Performs half2 unordered comparison and return a half2 value.
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] binary_op functor that implements the binary operation
/// \returns the comparison result
template <class BinaryOperation>
inline sycl::half2 unordered_compare_half2(const sycl::half2 a,
                                           const sycl::half2 b,
                                           const BinaryOperation binary_op) {
  return {unordered_compare_half(a.x(), b.x(), binary_op),
          unordered_compare_half(a.y(), b.y(), binary_op)};
}

/// Determine whether half2 is NaN and return a half2 value.
/// \param [in] h The half value
/// \returns the comparison result
inline sycl::half2 isnan(const sycl::half2 h) {
  return {sycl::isnan(h.x()), sycl::isnan(h.y())};
}
} // namespace dpct

#endif // __DPCT_MATH_HPP__
