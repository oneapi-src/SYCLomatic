// DPCT_LABEL_BEGIN|License|
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
//==---- lib_common_utils.hpp ---------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
// DPCT_LABEL_END

#ifndef __DPCT_LIB_COMMON_UTILS_HPP__
#define __DPCT_LIB_COMMON_UTILS_HPP__
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
// DPCT_COMMENT For header file including dependency, please use predefined feature name:
// DPCT_COMMENT   local_include_dependency: dpct helper files
// DPCT_COMMENT   non_local_include_dependency: other header files

// DPCT_LABEL_BEGIN|non_local_include_dependency|
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|local_include_dependency|
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
// DPCT_LABEL_END
#include "memory.hpp"
#include "util.hpp"

namespace dpct {
namespace detail {
// DPCT_LABEL_BEGIN|get_memory|dpct::detail
// DPCT_DEPENDENCY_BEGIN
// Memory|get_buffer_T|UsmNone
// DPCT_DEPENDENCY_END
// DPCT_CODE
template <typename T> inline auto get_memory(T *x) {
#ifdef DPCT_USM_LEVEL_NONE
  return dpct::get_buffer<std::remove_cv_t<T>>(x);
#else
  return x;
#endif
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|get_value|dpct::detail
// DPCT_DEPENDENCY_BEGIN
// Memory|dpct_memcpy_detail
// Util|DataType
// DPCT_DEPENDENCY_END
// DPCT_CODE
template <typename T>
inline typename DataType<T>::T2 get_value(const T *s, sycl::queue &q) {
  using Ty = typename DataType<T>::T2;
  Ty s_h;
  detail::dpct_memcpy(q, (void *)&s_h, (void *)s, sizeof(T), automatic).wait();
  return s_h;
}
// DPCT_LABEL_END
}
// DPCT_LABEL_BEGIN|version_field|dpct
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
enum class version_field : int {
  major,
  minor,
  update,
  patch
};
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|mkl_get_version|dpct
// DPCT_DEPENDENCY_BEGIN
// LibCommonUtils|version_field
// DPCT_DEPENDENCY_END
// DPCT_CODE
/// Returns the requested field of Intel(R) oneAPI Math Kernel Library version.
/// \param field The version information field (major, minor, update or patch).
/// \param result The result value.
inline void mkl_get_version(version_field field, int *result) {
#ifndef __INTEL_MKL__
  throw std::runtime_error("The oneAPI Math Kernel Library (oneMKL) Interfaces "
                           "Project does not support this API.");
#else
  MKLVersion version;
  mkl_get_version(&version);
  if (version_field::major == field) {
    *result = version.MajorVersion;
  } else if (version_field::minor == field) {
    *result = version.MinorVersion;
  } else if (version_field::update == field) {
    *result = version.UpdateVersion;
  } else if (version_field::patch == field) {
    *result = 0;
  } else {
    throw std::runtime_error("unknown field");
  }
#endif
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|library_data_t|dpct
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
enum class library_data_t : unsigned char {
  real_float = 0,
  complex_float,
  real_double,
  complex_double,
  real_half,
  complex_half,
  real_bfloat16,
  complex_bfloat16,
  real_int4,
  complex_int4,
  real_uint4,
  complex_uint4,
  real_int8,
  complex_int8,
  real_uint8,
  complex_uint8,
  real_int16,
  complex_int16,
  real_uint16,
  complex_uint16,
  real_int32,
  complex_int32,
  real_uint32,
  complex_uint32,
  real_int64,
  complex_int64,
  real_uint64,
  complex_uint64,
  real_int8_4,
  real_int8_32,
  real_uint8_4,
  library_data_t_size
};
// DPCT_LABEL_END

namespace detail {
// DPCT_LABEL_BEGIN|get_type_combination_id_T|dpct::detail
// DPCT_DEPENDENCY_BEGIN
// LibCommonUtils|get_type_combination_id_Ts
// LibCommonUtils|library_data_t
// DPCT_DEPENDENCY_END
// DPCT_CODE
template <typename ArgT>
inline constexpr std::uint64_t get_type_combination_id(ArgT Val) {
  static_assert((unsigned char)library_data_t::library_data_t_size <=
                    std::numeric_limits<unsigned char>::max() &&
                "library_data_t size exceeds limit.");
  static_assert(std::is_same_v<ArgT, library_data_t>, "Unsupported ArgT");
  return (std::uint64_t)Val;
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|get_type_combination_id_Ts|dpct::detail
// DPCT_DEPENDENCY_BEGIN
// LibCommonUtils|get_type_combination_id_T
// LibCommonUtils|library_data_t
// DPCT_DEPENDENCY_END
// DPCT_CODE
template <typename FirstT, typename... RestT>
inline constexpr std::uint64_t get_type_combination_id(FirstT FirstVal,
                                                       RestT... RestVal) {
  static_assert((std::uint8_t)library_data_t::library_data_t_size <=
                    std::numeric_limits<unsigned char>::max() &&
                "library_data_t size exceeds limit.");
  static_assert(sizeof...(RestT) <= 8 && "Too many parameters");
  static_assert(std::is_same_v<FirstT, library_data_t>, "Unsupported FirstT");
  return get_type_combination_id(RestVal...) << 8 | ((std::uint64_t)FirstVal);
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|library_data_size|dpct::detail
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
inline constexpr std::size_t library_data_size[] = {
    8 * sizeof(float),                    // real_float
    8 * sizeof(std::complex<float>),      // complex_float
    8 * sizeof(double),                   // real_double
    8 * sizeof(std::complex<double>),     // complex_double
    8 * sizeof(sycl::half),               // real_half
    8 * sizeof(std::complex<sycl::half>), // complex_half
    16,                                   // real_bfloat16
    16 * 2,                               // complex_bfloat16
    4,                                    // real_int4
    4 * 2,                                // complex_int4
    4,                                    // real_uint4
    4 * 2,                                // complex_uint4
    8,                                    // real_int8
    8 * 2,                                // complex_int8
    8,                                    // real_uint8
    8 * 2,                                // complex_uint8
    16,                                   // real_int16
    16 * 2,                               // complex_int16
    16,                                   // real_uint16
    16 * 2,                               // complex_uint16
    32,                                   // real_int32
    32 * 2,                               // complex_int32
    32,                                   // real_uint32
    32 * 2,                               // complex_uint32
    64,                                   // real_int64
    64 * 2,                               // complex_int64
    64,                                   // real_uint64
    64 * 2,                               // complex_uint64
    8,                                    // real_int8_4
    8,                                    // real_int8_32
    8                                     // real_uint8_4
};
// DPCT_LABEL_END
} // namespace detail
} // namespace dpct

#endif // __DPCT_LIB_COMMON_UTILS_HPP__
