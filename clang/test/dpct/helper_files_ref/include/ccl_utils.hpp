//==---- ccl_utils.hpp----------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __DPCT_CCL_UTILS_HPP__
#define __DPCT_CCL_UTILS_HPP__

#include <sycl/sycl.hpp>
#include <oneapi/ccl.hpp>
#include <unordered_map>
#include <memory>

#include "lib_common_utils.hpp"

namespace dpct {
namespace ccl {
namespace detail {

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

} // namespace detail

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

/// Get concatenated library version as an integer.
static inline int get_version() {
  auto ver = oneapi::ccl::get_library_version();
  return ver.major * 10000 + ver.minor * 100 + ver.update;
}

/// Create main kvs and return its address.
static inline oneapi::ccl::kvs::address_type create_kvs_address() {
  auto ptr = oneapi::ccl::create_main_kvs();
  auto addr = ptr->get_address();
  detail::get_kvs(addr) = ptr;
  return addr;
}


/// Get stored kvs with /p addr if exist. Otherwise, create kvs with /p addr.
static inline std::shared_ptr<oneapi::ccl::kvs>
create_kvs(const oneapi::ccl::kvs::address_type &addr) {
  auto &ptr = detail::get_kvs(addr);
  if (!ptr)
    ptr = oneapi::ccl::create_kvs(addr);
  return ptr;
}

} // namespace ccl
} // namespace dpct

#endif // __DPCT_CCL_UTILS_HPP__