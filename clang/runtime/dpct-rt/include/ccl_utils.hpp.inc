// DPCT_LABEL_BEGIN|License|
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
//==---- ccl_utils.hpp----------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
// DPCT_LABEL_END

#ifndef __DPCT_CCL_UTILS_HPP__
#define __DPCT_CCL_UTILS_HPP__
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
// DPCT_COMMENT   non_local_include_dependency: other headler files

// DPCT_LABEL_BEGIN|non_local_include_dependency|
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
#include <sycl/sycl.hpp>
#include <oneapi/ccl.hpp>
#include <unordered_map>
#include <memory>
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|local_include_dependency|
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
// DPCT_LABEL_END
#include "lib_common_utils.hpp"

namespace dpct {
namespace ccl {
namespace detail {

// DPCT_LABEL_BEGIN|get_kvs_detail|dpct::ccl::detail
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
/// Get stored kvs with specified kvs address.
inline std::shared_ptr<oneapi::ccl::kvs> &
get_kvs(const oneapi::ccl::kvs::address_type &addr) {
  struct hash {
    std::size_t operator()(const oneapi::ccl::kvs::address_type &in) const {
      return std::hash<std::string_view>()(std::string_view(in.data(), in.size()));
    }
  };
  static std::unordered_map<oneapi::ccl::kvs::address_type,
                            std::shared_ptr<oneapi::ccl::kvs>, hash>
      kvs_map;
  return kvs_map[addr];
}
// DPCT_LABEL_END

} // namespace detail

// DPCT_LABEL_BEGIN|ccl_datatype_convert|dpct::ccl
// DPCT_DEPENDENCY_BEGIN
// LibCommonUtils|library_data_t
// DPCT_DEPENDENCY_END
// DPCT_CODE
/// Convert dpct::library_data_t to oneapi::ccl::datatype.
inline oneapi::ccl::datatype to_ccl_datatype(dpct::library_data_t dt) {
  switch (dt) {
  case dpct::library_data_t::real_int8:
    return oneapi::ccl::datatype::int8;
  case dpct::library_data_t::real_uint8:
    return oneapi::ccl::datatype::uint8;
  case dpct::library_data_t::real_int32:
    return oneapi::ccl::datatype::int32;
  case dpct::library_data_t::real_uint32:
    return oneapi::ccl::datatype::uint32;
  case dpct::library_data_t::real_int64:
    return oneapi::ccl::datatype::int64;
  case dpct::library_data_t::real_half:
    return oneapi::ccl::datatype::float16;
  case dpct::library_data_t::real_float:
    return oneapi::ccl::datatype::float32;
  case dpct::library_data_t::real_double:
    return oneapi::ccl::datatype::float64;
  case dpct::library_data_t::real_bfloat16:
    return oneapi::ccl::datatype::bfloat16;
  default:
    throw std::runtime_error("to_dnnl_data_type: unsupported data type.");
  }
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|get_version|dpct::ccl
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
/// Get concatenated library version as an integer.
static inline int get_version() {
  auto ver = oneapi::ccl::get_library_version();
  return ver.major * 10000 + ver.minor * 100 + ver.update;
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|create_kvs_address|dpct::ccl
// DPCT_DEPENDENCY_BEGIN
// CclUtils|get_kvs_detail
// DPCT_DEPENDENCY_END
// DPCT_CODE
/// Create main kvs and return its address.
static inline oneapi::ccl::kvs::address_type create_kvs_address() {
  auto ptr = oneapi::ccl::create_main_kvs();
  auto addr = ptr->get_address();
  detail::get_kvs(addr) = ptr;
  return addr;
}
// DPCT_LABEL_END


// DPCT_LABEL_BEGIN|create_kvs|dpct::ccl
// DPCT_DEPENDENCY_BEGIN
// CclUtils|get_kvs_detail
// DPCT_DEPENDENCY_END
// DPCT_CODE
/// Get stored kvs with /p addr if exist. Otherwise, create kvs with /p addr.
static inline std::shared_ptr<oneapi::ccl::kvs>
create_kvs(const oneapi::ccl::kvs::address_type &addr) {
  auto &ptr = detail::get_kvs(addr);
  if (!ptr)
    ptr = oneapi::ccl::create_kvs(addr);
  return ptr;
}
// DPCT_LABEL_END

} // namespace ccl
} // namespace dpct

#endif // __DPCT_CCL_UTILS_HPP__