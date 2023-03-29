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
/// Compute fast_length for variable-length array
/// \param [in] a The array
/// \param [in] len Length of the array
/// \returns The computed fast_length
inline float fast_length(const float *a, int len) {
  switch (len) {
  case 1:
    return a[0];
  case 2:
    return sycl::fast_length(sycl::float2(a[0], a[1]));
  case 3:
    return sycl::fast_length(sycl::float3(a[0], a[1], a[2]));
  case 4:
    return sycl::fast_length(sycl::float4(a[0], a[1], a[2], a[3]));
  case 0:
    return 0;
  default:
    float f = 0;
    for (int i = 0; i < len; ++i)
      f += a[i] * a[i];
    return sycl::sqrt(f);
  }
}

/// Calculate the square root of the input array.
/// \param [in] a The array pointer
/// \param [in] len Length of the array
/// \returns The square root
template <typename T> inline T length(const T *a, const int len) {
  switch (len) {
  case 1:
    return a[0];
  case 2:
    return sycl::length(sycl::vec<T, 2>(a[0], a[1]));
  case 3:
    return sycl::length(sycl::vec<T, 3>(a[0], a[1], a[2]));
  case 4:
    return sycl::length(sycl::vec<T, 4>(a[0], a[1], a[2], a[3]));
  default:
    T ret = 0;
    for (int i = 0; i < len; ++i)
      ret += a[i] * a[i];
    return sycl::sqrt(ret);
  }
}

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

#define DPCT_MATH_MIN_MAX_OVERLOAD_1(FUNC, IMPL_FUNC, TYPE, PROMOTED_TYPE)     \
  inline auto FUNC(const PROMOTED_TYPE a, const TYPE b) {                      \
    return IMPL_FUNC(a, static_cast<PROMOTED_TYPE>(b));                        \
  }                                                                            \
  inline auto FUNC(const TYPE a, const PROMOTED_TYPE b) {                      \
    return IMPL_FUNC(static_cast<PROMOTED_TYPE>(a), b);                        \
  }
#define DPCT_MATH_MIN_MAX_OVERLOAD_2(FUNC, IMPL_FUNC, TYPE)                    \
  inline auto FUNC(const TYPE a, const TYPE b) {                               \
    return IMPL_FUNC(a, b);                                                    \
  }

DPCT_MATH_MIN_MAX_OVERLOAD_1(min, sycl::fmin, float, double)
DPCT_MATH_MIN_MAX_OVERLOAD_2(min, sycl::fmin, float)
DPCT_MATH_MIN_MAX_OVERLOAD_2(min, sycl::fmin, double)
DPCT_MATH_MIN_MAX_OVERLOAD_1(min, sycl::min, std::int32_t, std::uint32_t)
DPCT_MATH_MIN_MAX_OVERLOAD_1(min, sycl::min, std::int64_t, std::uint64_t)
DPCT_MATH_MIN_MAX_OVERLOAD_1(min, sycl::min, std::int32_t, std::uint64_t)
DPCT_MATH_MIN_MAX_OVERLOAD_1(min, sycl::min, std::uint32_t, std::uint64_t)
DPCT_MATH_MIN_MAX_OVERLOAD_2(min, sycl::min, std::int32_t)
DPCT_MATH_MIN_MAX_OVERLOAD_2(min, sycl::min, std::uint32_t)
DPCT_MATH_MIN_MAX_OVERLOAD_2(min, sycl::min, std::int64_t)
DPCT_MATH_MIN_MAX_OVERLOAD_2(min, sycl::min, std::uint64_t)
DPCT_MATH_MIN_MAX_OVERLOAD_1(max, sycl::fmax, float, double)
DPCT_MATH_MIN_MAX_OVERLOAD_2(max, sycl::fmax, float)
DPCT_MATH_MIN_MAX_OVERLOAD_2(max, sycl::fmax, double)
DPCT_MATH_MIN_MAX_OVERLOAD_1(max, sycl::max, std::int32_t, std::uint32_t)
DPCT_MATH_MIN_MAX_OVERLOAD_1(max, sycl::max, std::int64_t, std::uint64_t)
DPCT_MATH_MIN_MAX_OVERLOAD_1(max, sycl::max, std::int32_t, std::uint64_t)
DPCT_MATH_MIN_MAX_OVERLOAD_1(max, sycl::max, std::uint32_t, std::uint64_t)
DPCT_MATH_MIN_MAX_OVERLOAD_2(max, sycl::max, std::int32_t)
DPCT_MATH_MIN_MAX_OVERLOAD_2(max, sycl::max, std::uint32_t)
DPCT_MATH_MIN_MAX_OVERLOAD_2(max, sycl::max, std::int64_t)
DPCT_MATH_MIN_MAX_OVERLOAD_2(max, sycl::max, std::uint64_t)
#undef DPCT_MATH_MIN_MAX_OVERLOAD_1
#undef DPCT_MATH_MIN_MAX_OVERLOAD_2
} // namespace dpct

#endif // __DPCT_MATH_HPP__
