//==---- ccl_utils.hpp----------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __DPCT_CCL_UTILS_HPP__
#define __DPCT_CCL_UTILS_HPP__

#include <CL/sycl.hpp>
#include <oneapi/ccl.hpp>
#include <unordered_map>
#include <memory>

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