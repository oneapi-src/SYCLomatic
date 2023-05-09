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
namespace detail {
template <typename VecT, class BinaryOperation, class = void>
class vectorized_binary {
public:
  inline VecT operator()(VecT a, VecT b, const BinaryOperation binary_op) {
    VecT v4;
    for (size_t i = 0; i < v4.size(); ++i) {
      v4[i] = binary_op(a[i], b[i]);
    }
    return v4;
  }
};
template <typename VecT, class BinaryOperation>
class vectorized_binary<
    VecT, BinaryOperation,
    std::void_t<std::invoke_result_t<BinaryOperation, VecT, VecT>>> {
public:
  inline VecT operator()(VecT a, VecT b, const BinaryOperation binary_op) {
    return binary_op(a, b).template as<VecT>();
  }
};

template <typename T> bool isnan(const T a) { return sycl::isnan(a); }
// TODO: Need add more specialization such as bfloat16 version.
} // namespace detail

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

/// Performs comparison.
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] binary_op functor that implements the binary operation
/// \returns the comparison result
template <typename T, class BinaryOperation>
inline std::enable_if_t<
    std::is_same_v<std::invoke_result_t<BinaryOperation, T, T>, bool>, bool>
compare(const T a, const T b, const BinaryOperation binary_op) {
  return binary_op(a, b);
}
template <typename T>
inline std::enable_if_t<
    std::is_same_v<std::invoke_result_t<std::not_equal_to<>, T, T>, bool>, bool>
compare(const T a, const T b, const std::not_equal_to<> binary_op) {
  return !detail::isnan(a) && !detail::isnan(b) && binary_op(a, b);
}

/// Performs unordered comparison.
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] binary_op functor that implements the binary operation
/// \returns the comparison result
template <typename T, class BinaryOperation>
inline std::enable_if_t<
    std::is_same_v<std::invoke_result_t<BinaryOperation, T, T>, bool>, bool>
unordered_compare(const T a, const T b, const BinaryOperation binary_op) {
  return detail::isnan(a) || detail::isnan(b) || binary_op(a, b);
}

/// Performs 2 element comparison and return true if both results are true.
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] binary_op functor that implements the binary operation
/// \returns the comparison result
template <typename T, class BinaryOperation>
inline std::enable_if_t<T::size() == 2, bool>
compare_both(const T a, const T b, const BinaryOperation binary_op) {
  return compare(a[0], b[0], binary_op) && compare(a[1], b[1], binary_op);
}

/// Performs 2 element unordered comparison and return true if both results are
/// true.
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] binary_op functor that implements the binary operation
/// \returns the comparison result
template <typename T, class BinaryOperation>
inline std::enable_if_t<T::size() == 2, bool>
unordered_compare_both(const T a, const T b, const BinaryOperation binary_op) {
  return unordered_compare(a[0], b[0], binary_op) &&
         unordered_compare(a[1], b[1], binary_op);
}

/// Performs 2 element comparison.
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] binary_op functor that implements the binary operation
/// \returns the comparison result
template <typename T, class BinaryOperation>
inline std::enable_if_t<T::size() == 2, T>
compare(const T a, const T b, const BinaryOperation binary_op) {
  return {compare(a[0], b[0], binary_op), compare(a[1], b[1], binary_op)};
}

/// Performs 2 element unordered comparison.
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] binary_op functor that implements the binary operation
/// \returns the comparison result
template <typename T, class BinaryOperation>
inline std::enable_if_t<T::size() == 2, T>
unordered_compare(const T a, const T b, const BinaryOperation binary_op) {
  return {unordered_compare(a[0], b[0], binary_op),
          unordered_compare(a[1], b[1], binary_op)};
}

/// Determine whether 2 element value is NaN.
/// \param [in] a The input value
/// \returns the comparison result
template <typename T>
inline std::enable_if_t<T::size() == 2, T> isnan(const T a) {
  return {detail::isnan(a[0]), detail::isnan(a[1])};
}

// min function overloads.
// For floating-point types, `float` or `double` arguments are acceptable.
// For integer types, `std::uint32_t`, `std::int32_t`, `std::uint64_t` or
// `std::int64_t` type arguments are acceptable.
inline double min(const double a, const float b) {
  return sycl::fmin(a, static_cast<double>(b));
}
inline double min(const float a, const double b) {
  return sycl::fmin(static_cast<double>(a), b);
}
inline float min(const float a, const float b) { return sycl::fmin(a, b); }
inline double min(const double a, const double b) { return sycl::fmin(a, b); }
inline std::uint32_t min(const std::uint32_t a, const std::int32_t b) {
  return sycl::min(a, static_cast<std::uint32_t>(b));
}
inline std::uint32_t min(const std::int32_t a, const std::uint32_t b) {
  return sycl::min(static_cast<std::uint32_t>(a), b);
}
inline std::int32_t min(const std::int32_t a, const std::int32_t b) {
  return sycl::min(a, b);
}
inline std::uint32_t min(const std::uint32_t a, const std::uint32_t b) {
  return sycl::min(a, b);
}
inline std::uint64_t min(const std::uint64_t a, const std::int64_t b) {
  return sycl::min(a, static_cast<std::uint64_t>(b));
}
inline std::uint64_t min(const std::int64_t a, const std::uint64_t b) {
  return sycl::min(static_cast<std::uint64_t>(a), b);
}
inline std::int64_t min(const std::int64_t a, const std::int64_t b) {
  return sycl::min(a, b);
}
inline std::uint64_t min(const std::uint64_t a, const std::uint64_t b) {
  return sycl::min(a, b);
}
inline std::uint64_t min(const std::uint64_t a, const std::int32_t b) {
  return sycl::min(a, static_cast<std::uint64_t>(b));
}
inline std::uint64_t min(const std::int32_t a, const std::uint64_t b) {
  return sycl::min(static_cast<std::uint64_t>(a), b);
}
inline std::uint64_t min(const std::uint64_t a, const std::uint32_t b) {
  return sycl::min(a, static_cast<std::uint64_t>(b));
}
inline std::uint64_t min(const std::uint32_t a, const std::uint64_t b) {
  return sycl::min(static_cast<std::uint64_t>(a), b);
}
// max function overloads.
// For floating-point types, `float` or `double` arguments are acceptable.
// For integer types, `std::uint32_t`, `std::int32_t`, `std::uint64_t` or
// `std::int64_t` type arguments are acceptable.
inline double max(const double a, const float b) {
  return sycl::fmax(a, static_cast<double>(b));
}
inline double max(const float a, const double b) {
  return sycl::fmax(static_cast<double>(a), b);
}
inline float max(const float a, const float b) { return sycl::fmax(a, b); }
inline double max(const double a, const double b) { return sycl::fmax(a, b); }
inline std::uint32_t max(const std::uint32_t a, const std::int32_t b) {
  return sycl::max(a, static_cast<std::uint32_t>(b));
}
inline std::uint32_t max(const std::int32_t a, const std::uint32_t b) {
  return sycl::max(static_cast<std::uint32_t>(a), b);
}
inline std::int32_t max(const std::int32_t a, const std::int32_t b) {
  return sycl::max(a, b);
}
inline std::uint32_t max(const std::uint32_t a, const std::uint32_t b) {
  return sycl::max(a, b);
}
inline std::uint64_t max(const std::uint64_t a, const std::int64_t b) {
  return sycl::max(a, static_cast<std::uint64_t>(b));
}
inline std::uint64_t max(const std::int64_t a, const std::uint64_t b) {
  return sycl::max(static_cast<std::uint64_t>(a), b);
}
inline std::int64_t max(const std::int64_t a, const std::int64_t b) {
  return sycl::max(a, b);
}
inline std::uint64_t max(const std::uint64_t a, const std::uint64_t b) {
  return sycl::max(a, b);
}
inline std::uint64_t max(const std::uint64_t a, const std::int32_t b) {
  return sycl::max(a, static_cast<std::uint64_t>(b));
}
inline std::uint64_t max(const std::int32_t a, const std::uint64_t b) {
  return sycl::max(static_cast<std::uint64_t>(a), b);
}
inline std::uint64_t max(const std::uint64_t a, const std::uint32_t b) {
  return sycl::max(a, static_cast<std::uint64_t>(b));
}
inline std::uint64_t max(const std::uint32_t a, const std::uint64_t b) {
  return sycl::max(static_cast<std::uint64_t>(a), b);
}

/// A sycl::abs wrapper functors.
struct abs {
  template <typename T> auto operator()(const T x) const {
    return sycl::abs(x);
  }
};

/// A sycl::abs_diff wrapper functors.
struct abs_diff {
  template <typename T> auto operator()(const T x, const T y) const {
    return sycl::abs_diff(x, y);
  }
};

/// A sycl::add_sat wrapper functors.
struct add_sat {
  template <typename T> auto operator()(const T x, const T y) const {
    return sycl::add_sat(x, y);
  }
};

/// A sycl::rhadd wrapper functors.
struct rhadd {
  template <typename T> auto operator()(const T x, const T y) const {
    return sycl::rhadd(x, y);
  }
};

/// A sycl::hadd wrapper functors.
struct hadd {
  template <typename T> auto operator()(const T x, const T y) const {
    return sycl::hadd(x, y);
  }
};

/// A sycl::max wrapper functors.
struct maximum {
  template <typename T> auto operator()(const T x, const T y) const {
    return sycl::max(x, y);
  }
};

/// A sycl::min wrapper functors.
struct minimum {
  template <typename T> auto operator()(const T x, const T y) const {
    return sycl::min(x, y);
  }
};

/// A sycl::sub_sat wrapper functors.
struct sub_sat {
  template <typename T> auto operator()(const T x, const T y) const {
    return sycl::sub_sat(x, y);
  }
};

/// Compute vectorized binary operation value for two values, with each value
/// treated as a vector type \p VecT.
/// \tparam [in] VecT The type of the vector
/// \tparam [in] BinaryOperation The binary operation class
/// \param [in] a The first value
/// \param [in] b The second value
/// \returns The vectorized binary operation value of the two values
template <typename VecT, class BinaryOperation>
inline unsigned vectorized_binary(unsigned a, unsigned b,
                                  const BinaryOperation binary_op) {
  sycl::vec<unsigned, 1> v0{a}, v1{b};
  auto v2 = v0.as<VecT>();
  auto v3 = v1.as<VecT>();
  auto v4 =
      detail::vectorized_binary<VecT, BinaryOperation>()(v2, v3, binary_op);
  v0 = v4.template as<sycl::vec<unsigned, 1>>();
  return v0;
}

/// Compute vectorized isgreater for two values, with each value treated as a
/// vector type \p S.
/// \tparam [in] S The type of the vector
/// \tparam [in] T The type of the original values
/// \param [in] a The first value
/// \param [in] b The second value
/// \returns The vectorized greater than of the two values
template <typename S, typename T> inline T vectorized_isgreater(T a, T b) {
  sycl::vec<T, 1> v0{a}, v1{b};
  auto v2 = v0.template as<S>();
  auto v3 = v1.template as<S>();
  auto v4 = v2 > v3;
  v0 = v4.template as<sycl::vec<T, 1>>();
  return v0;
}

/// Compute vectorized max for two values, with each value treated as a vector
/// type \p S.
/// \tparam [in] S The type of the vector
/// \tparam [in] T The type of the original values
/// \param [in] a The first value
/// \param [in] b The second value
/// \returns The vectorized max of the two values
template <typename S, typename T> inline T vectorized_max(T a, T b) {
  sycl::vec<T, 1> v0{a}, v1{b};
  auto v2 = v0.template as<S>();
  auto v3 = v1.template as<S>();
  auto v4 = sycl::max(v2, v3);
  v0 = v4.template as<sycl::vec<T, 1>>();
  return v0;
}

/// Compute vectorized min for two values, with each value treated as a vector
/// type \p S.
/// \tparam [in] S The type of the vector
/// \tparam [in] T The type of the original values
/// \param [in] a The first value
/// \param [in] b The second value
/// \returns The vectorized min of the two values
template <typename S, typename T> inline T vectorized_min(T a, T b) {
  sycl::vec<T, 1> v0{a}, v1{b};
  auto v2 = v0.template as<S>();
  auto v3 = v1.template as<S>();
  auto v4 = sycl::min(v2, v3);
  v0 = v4.template as<sycl::vec<T, 1>>();
  return v0;
}

/// Compute vectorized unary operation for a value, with the value treated as a
/// vector type \p VecT.
/// \tparam [in] VecT The type of the vector
/// \tparam [in] UnaryOperation The unary operation class
/// \param [in] a The input value
/// \returns The vectorized unary operation value of the input value
template <typename VecT, class UnaryOperation>
inline unsigned vectorized_unary(unsigned a, const UnaryOperation unary_op) {
  sycl::vec<unsigned, 1> v0{a};
  auto v1 = v0.as<VecT>();
  auto v2 = unary_op(v1);
  v0 = v2.template as<sycl::vec<unsigned, 1>>();
  return v0;
}

/// Compute vectorized absolute difference for two values without modulo
/// overflow, with each value treated as a vector type \p VecT.
/// \tparam [in] VecT The type of the vector
/// \param [in] a The first value
/// \param [in] b The second value
/// \returns The vectorized absolute difference of the two values
template <typename VecT>
inline unsigned vectorized_sum_abs_diff(unsigned a, unsigned b) {
  sycl::vec<unsigned, 1> v0{a}, v1{b};
  auto v2 = v0.as<VecT>();
  auto v3 = v1.as<VecT>();
  auto v4 = sycl::abs_diff(v2, v3);
  unsigned sum = 0;
  for (size_t i = 0; i < v4.size(); ++i) {
    sum += v4[i];
  }
  return sum;
}
} // namespace dpct

#endif // __DPCT_MATH_HPP__
