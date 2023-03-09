// DPCT_LABEL_BEGIN|License|
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
//==---- math.hpp ---------------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
// DPCT_LABEL_END

#ifndef __DPCT_MATH_HPP__
#define __DPCT_MATH_HPP__
// DPCT_COMMENT
// DPCT_COMMENT Example1:
// DPCT_COMMENT // DPCT_LABEL_BEGIN|FeatureNameDef|[Namespace]
// DPCT_COMMENT // DPCT_DEPENDENCY_EMPTY
// DPCT_COMMENT // DPCT_CODE
// DPCT_COMMENT some code
// DPCT_COMMENT // DPCT_LABEL_END
// DPCT_COMMENT
// DPCT_COMMENT Example2:
// DPCT_COMMENT // DPCT_LABEL_BEGIN|FeatureNameDef|[Namespace]
// DPCT_COMMENT // DPCT_DEPENDENCY_BEGIN
// DPCT_COMMENT // FileID|FeatureNameRef
// DPCT_COMMENT [// FileID|FeatureNameRef]
// DPCT_COMMENT ...
// DPCT_COMMENT // DPCT_DEPENDENCY_END
// DPCT_COMMENT // DPCT_CODE
// DPCT_COMMENT some code
// DPCT_COMMENT // DPCT_LABEL_END
// DPCT_COMMENT
// DPCT_COMMENT For header file including dependency, please use predefined
// DPCT_COMMENT   local_include_dependency: dpct helper files
// DPCT_COMMENT   non_local_include_dependency: other header files

// DPCT_LABEL_BEGIN|non_local_include_dependency|
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
#include <sycl/sycl.hpp>
// DPCT_LABEL_END

namespace dpct {
// DPCT_LABEL_BEGIN|cast_double_to_int|dpct
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
/// Cast the high or low 32 bits of a double to an integer.
/// \param [in] d The double value.
/// \param [in] use_high32 Cast the high 32 bits of the double if true;
/// otherwise cast the low 32 bits.
inline int cast_double_to_int(double d, bool use_high32 = true) {
  sycl::vec<double, 1> v0{d};
  auto v1 = v0.as<sycl::int2>();
  if (use_high32)
    return v1[1];
  return v1[0];
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|cast_ints_to_double|dpct
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
/// Combine two integers, the first as the high 32 bits and the second
/// as the low 32 bits, into a double.
/// \param [in] high32 The integer as the high 32 bits
/// \param [in] low32 The integer as the low 32 bits
inline double cast_ints_to_double(int high32, int low32) {
  sycl::int2 v0{low32, high32};
  auto v1 = v0.as<sycl::vec<double, 1>>();
  return v1;
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|fast_length|dpct
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
/// Compute fast_length for variable-length array
/// \param [in] a The array
/// \param [in] len Length of the array
/// \returns The computed fast_length
inline float fast_length(const float *a, int len) {
  switch (len) {
  case 1:
    return sycl::fast_length(a[0]);
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
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|vectorized_max|dpct
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
/// Compute vectorized max for two values, with each value treated as a vector
/// type \p S
/// \param [in] S The type of the vector
/// \param [in] T The type of the original values
/// \param [in] a The first value
/// \param [in] b The second value
/// \returns The vectorized max of the two values
template <typename S, typename T> inline T vectorized_max(T a, T b) {
  sycl::vec<T, 1> v0{a}, v1{b};
  auto v2 = v0.template as<S>();
  auto v3 = v1.template as<S>();
  v2 = sycl::max(v2, v3);
  v0 = v2.template as<sycl::vec<T, 1>>();
  return v0;
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|vectorized_min|dpct
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
/// Compute vectorized min for two values, with each value treated as a vector
/// type \p S
/// \param [in] S The type of the vector
/// \param [in] T The type of the original values
/// \param [in] a The first value
/// \param [in] b The second value
/// \returns The vectorized min of the two values
template <typename S, typename T> inline T vectorized_min(T a, T b) {
  sycl::vec<T, 1> v0{a}, v1{b};
  auto v2 = v0.template as<S>();
  auto v3 = v1.template as<S>();
  v2 = sycl::min(v2, v3);
  v0 = v2.template as<sycl::vec<T, 1>>();
  return v0;
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|vectorized_isgreater_T|dpct
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
/// Compute vectorized isgreater for two values, with each value treated as a
/// vector type \p S
/// \param [in] S The type of the vector
/// \param [in] T The type of the original values
/// \param [in] a The first value
/// \param [in] b The second value
/// \returns The vectorized greater than of the two values
template <typename S, typename T> inline T vectorized_isgreater(T a, T b) {
  sycl::vec<T, 1> v0{a}, v1{b};
  auto v2 = v0.template as<S>();
  auto v3 = v1.template as<S>();
  auto v4 = sycl::isgreater(v2, v3);
  v0 = v4.template as<sycl::vec<T, 1>>();
  return v0;
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|vectorized_isgreater_unsigned|dpct
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
/// Compute vectorized isgreater for two unsigned int values, with each value
/// treated as a vector of two unsigned short
/// \param [in] a The first value
/// \param [in] b The second value
/// \returns The vectorized greater than of the two values
template <>
inline unsigned vectorized_isgreater<sycl::ushort2, unsigned>(unsigned a,
                                                              unsigned b) {
  sycl::vec<unsigned, 1> v0{a}, v1{b};
  auto v2 = v0.template as<sycl::ushort2>();
  auto v3 = v1.template as<sycl::ushort2>();
  sycl::ushort2 v4;
  v4[0] = v2[0] > v3[0] ? 0xffff : 0;
  v4[1] = v2[1] > v3[1] ? 0xffff : 0;
  v0 = v4.template as<sycl::vec<unsigned, 1>>();
  return v0;
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|vectorized_isgreater_uchar4|dpct
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
/// Compute vectorized isgreater for two unsigned int values, with each value
/// treated as a vector of four unsigned char.
/// \param [in] a The first value
/// \param [in] b The second value
/// \returns The vectorized greater than of the two values
template <>
inline unsigned vectorized_isgreater<sycl::uchar4, unsigned>(unsigned a,
                                                             unsigned b) {
  sycl::vec<unsigned, 1> v0{a}, v1{b};
  auto v2 = v0.template as<sycl::uchar4>();
  auto v3 = v1.template as<sycl::uchar4>();
  sycl::uchar4 v4;
  v4[0] = v2[0] > v3[0] ? 0xff : 0;
  v4[1] = v2[1] > v3[1] ? 0xff : 0;
  v4[2] = v2[2] > v3[2] ? 0xff : 0;
  v4[3] = v2[3] > v3[3] ? 0xff : 0;
  v0 = v4.template as<sycl::vec<unsigned, 1>>();
  return v0;
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|reverse_bits|dpct
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
/// Reverse the bit order of an unsigned integer
/// \param [in] a Input unsigned integer value
/// \returns Value of a with the bit order reversed
template <typename T> inline T reverse_bits(T a) {
  static_assert(std::is_unsigned<T>::value && std::is_integral<T>::value,
                "unsigned integer required");
  if (!a)
    return 0;
  T mask = 0;
  size_t count = 4 * sizeof(T);
  mask = ~mask >> count;
  while (count) {
    a = ((a & mask) << count) | ((a & ~mask) >> count);
    count = count >> 1;
    mask = mask ^ (mask << count);
  }
  return a;
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|byte_level_permute|dpct
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
/// \param [in] a The first value contains 4 bytes
/// \param [in] b The second value contains 4 bytes
/// \param [in] s The selector value, only lower 16bit used
/// \returns the permutation result of 4 bytes selected in the way
/// specified by \p s from \p a and \p b
inline unsigned int byte_level_permute(unsigned int a, unsigned int b,
                                       unsigned int s) {
  unsigned int ret = 0;
  std::uint64_t temp = (std::uint64_t)b << 32 | a;
  // (temp >> (s & 0x7) * 8) & 0xff
  ret |= (temp >> (s << 3 & 0x38)) & 0xff;
  // ((temp >> ((s >> 4) & 0x7) * 8) & 0xff) << 8
  ret |= (temp >> (s >> 1 & 0x38) << 8) & 0xff00;
  // ((temp >> ((s >> 8) & 0x7) * 8) & 0xff) << 16
  ret |= (temp >> (s >> 5 & 0x38) << 16) & 0xff0000;
  // ((temp >> ((s >> 12) & 0x7) * 8) & 0xff) << 24
  ret |= (temp >> (s >> 9 & 0x38) << 24) & 0xff000000;
  return ret;
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|ffs|dpct
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
/// Find position of first least significant set bit in an integer.
/// ffs(0) returns 0.
///
/// \param [in] a Input integer value
/// \returns The position
template <typename T> inline int ffs(T a) {
  static_assert(std::is_integral<T>::value, "integer required");
  return (sycl::ctz(a) + 1) % (sizeof(T) * 8 + 1);
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|compare_half|dpct
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
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
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|unordered_compare_half|dpct
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
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
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|compare_both_half2|dpct
// DPCT_DEPENDENCY_BEGIN
// Math|compare_half
// DPCT_DEPENDENCY_END
// DPCT_CODE
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
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|unordered_compare_both_half2|dpct
// DPCT_DEPENDENCY_BEGIN
// Math|unordered_compare_half
// DPCT_DEPENDENCY_END
// DPCT_CODE
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
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|compare_half2|dpct
// DPCT_DEPENDENCY_BEGIN
// Math|compare_half
// DPCT_DEPENDENCY_END
// DPCT_CODE
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
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|unordered_compare_half2|dpct
// DPCT_DEPENDENCY_BEGIN
// Math|unordered_compare_half
// DPCT_DEPENDENCY_END
// DPCT_CODE
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
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|isnan|dpct
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
/// Determine whether half2 is NaN and return a half2 value.
/// \param [in] h The half value
/// \returns the comparison result
inline sycl::half2 isnan(const sycl::half2 h) {
  return {sycl::isnan(h.x()), sycl::isnan(h.y())};
}
// DPCT_LABEL_END
} // namespace dpct

#endif // __DPCT_MATH_HPP__
