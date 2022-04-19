//==---- lib_common_utils.hpp ---------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __C2S_LIB_COMMON_UTILS_HPP__
#define __C2S_LIB_COMMON_UTILS_HPP__

#include <CL/sycl.hpp>
#include <oneapi/mkl.hpp>

namespace c2s {
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
} // namespace c2s

#endif // __C2S_LIB_COMMON_UTILS_HPP__
