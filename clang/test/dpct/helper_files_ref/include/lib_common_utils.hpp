//==---- lib_common_utils.hpp ---------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __DPCT_LIB_COMMON_UTILS_HPP__
#define __DPCT_LIB_COMMON_UTILS_HPP__

#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>
#include "memory.hpp"

namespace dpct {
namespace detail {
template <typename T> inline auto get_memory(T *x) {
#ifdef DPCT_USM_LEVEL_NONE
  return dpct::get_buffer<std::remove_cv_t<T>>(x);
#else
  return x;
#endif
}
}
enum class version_field : int {
  major,
  minor,
  update,
  patch
};

/// Returns the requested field of Intel(R) oneAPI Math Kernel Library version.
/// \param field The version information field (major, minor, update or patch).
/// \param result The result value.
inline void mkl_get_version(version_field field, int *result) {
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
}

enum library_data_t {
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

namespace detail {
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
}
} // namespace dpct

#endif // __DPCT_LIB_COMMON_UTILS_HPP__
