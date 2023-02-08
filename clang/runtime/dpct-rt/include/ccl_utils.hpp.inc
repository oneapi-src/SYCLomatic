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


// DPCT_LABEL_BEGIN|communicator_ext|dpct::ccl
// DPCT_DEPENDENCY_BEGIN
// CclUtils|communicator_ext_2
// DPCT_DEPENDENCY_END
// DPCT_CODE
/// dpct communicator extension
class communicator_ext
{
public:
communicator_ext(int size,
                 int rank,
                 std::shared_ptr<oneapi::ccl::device> device_ptr_comm,
                 const oneapi::ccl::context& context_comm,
                 oneapi::ccl::shared_ptr_class<oneapi::ccl::kvs_interface> kvs,
                 const oneapi::ccl::comm_attr &attr = oneapi::ccl::default_comm_attr) : _device_ptr_comm(device_ptr_comm)
{
  _comm_ptr = std::make_shared<oneapi::ccl::communicator>(
    oneapi::ccl::create_communicator(size, rank, *device_ptr_comm, context_comm, kvs));
}


~communicator_ext(){};
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|communicator_ext_rank|dpct::ccl
// DPCT_DEPENDENCY_BEGIN
// CclUtils|communicator_ext
// DPCT_DEPENDENCY_END
// DPCT_CODE
/**
 * Return the rank in a oneapi::ccl::communicator
 * @return rank corresponding to communicator object
 */
inline int rank() const{
  return _comm_ptr->rank();
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|communicator_ext_size|dpct::ccl
// DPCT_DEPENDENCY_BEGIN
// CclUtils|communicator_ext
// DPCT_DEPENDENCY_END
// DPCT_CODE
/**
 * Return the number of rank in oneapi::ccl::communicator
 * @return number of the ranks
 */
inline int size() const{
return _comm_ptr->size();
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|communicator_ext_get_device|dpct::ccl
// DPCT_DEPENDENCY_BEGIN
// CclUtils|communicator_ext
// DPCT_DEPENDENCY_END
// DPCT_CODE
/**
 * Return underlying device, which was used in oneapi::ccl::communicator 
 */
inline oneapi::ccl::device get_device() const{
return _comm_ptr->get_device();
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|communicator_ext_get_context|dpct::ccl
// DPCT_DEPENDENCY_BEGIN
// CclUtils|communicator_ext
// DPCT_DEPENDENCY_END
// DPCT_CODE
/**
 * Return underlying context, which was used in oneapi::ccl::communicator
 */
inline oneapi::ccl::context get_context() const{
  return _comm_ptr->get_context();
};
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|communicator_ext_get_ccl_communicator_ptr|dpct::ccl
// DPCT_DEPENDENCY_BEGIN
// CclUtils|communicator_ext
// DPCT_DEPENDENCY_END
// DPCT_CODE
/**
 * Return oneapi::ccl::communicator pointer
 */
inline std::shared_ptr<oneapi::ccl::communicator> get_ccl_communicator_ptr() const{
  return _comm_ptr;
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|communicator_ext_get_ccl_communicator|dpct::ccl
// DPCT_DEPENDENCY_BEGIN
// CclUtils|communicator_ext
// DPCT_DEPENDENCY_END
// DPCT_CODE
/**
 * Return oneapi::ccl::communicator rvalue
 */
inline oneapi::ccl::communicator get_ccl_communicator() const{
  return static_cast<oneapi::ccl::communicator&&>(*_comm_ptr);
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|communicator_ext_2|dpct::ccl
// DPCT_DEPENDENCY_BEGIN
// CclUtils|communicator_ext
// DPCT_DEPENDENCY_END
// DPCT_CODE
private:
std::shared_ptr<oneapi::ccl::device> _device_ptr_comm;
std::shared_ptr<oneapi::ccl::communicator> _comm_ptr;
};
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|create_communicator_ext|dpct::ccl
// DPCT_DEPENDENCY_BEGIN
// CclUtils|communicator_ext
// DPCT_DEPENDENCY_END
// DPCT_CODE
inline communicator_ext create_communicator_ext(int size,
                                                int rank,
                                                const sycl::device &dev_,
                                                const sycl::context &cont,
                                                oneapi::ccl::shared_ptr_class<oneapi::ccl::kvs_interface> kvs)
{
  auto dev_ptr = std::make_shared<oneapi::ccl::device>(oneapi::ccl::create_device(dev_));                                        
  auto ctx = oneapi::ccl::create_context(cont);
  communicator_ext comm_ext(size,rank,dev_ptr,ctx,kvs);
  return std::move(comm_ext);
}
// DPCT_LABEL_END

} // namespace ccl
} // namespace dpct

#endif // __DPCT_CCL_UTILS_HPP__