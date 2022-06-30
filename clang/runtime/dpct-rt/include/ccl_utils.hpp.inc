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
#include <CL/sycl.hpp>
#include <oneapi/ccl.hpp>
#include <unordered_map>
#include <memory>
// DPCT_LABEL_END

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