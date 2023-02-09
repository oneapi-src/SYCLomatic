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

#include "device.hpp"

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


/// dpct communicator extension
class communicator_ext
{
public:
  communicator_ext(int size,
                    int rank,
                    oneapi::ccl::shared_ptr_class<oneapi::ccl::kvs_interface> kvs,
                    const oneapi::ccl::comm_attr &attr = oneapi::ccl::default_comm_attr) 
                  :_device_comm(oneapi::ccl::create_device(static_cast<sycl::device>(
                      dpct::get_current_device()))), 
                  _context_comm(oneapi::ccl::create_context(dpct::get_default_context())),
                  _comm(oneapi::ccl::create_communicator(size, rank, _device_comm, _context_comm, kvs)){}

  ~communicator_ext(){};

  /**
   * Return the rank in a oneapi::ccl::communicator
   * @return rank corresponding to communicator object
   */
  inline int rank() const{
    return _comm.rank();
  }

  /**
   * Return the number of rank in oneapi::ccl::communicator
   * @return number of the ranks
   */
  inline int size() const{
  return _comm.size();
  }

  /**
   * Return underlying device, which was used in oneapi::ccl::communicator 
   */
  inline oneapi::ccl::device get_device() const{
  return _comm.get_device();
  }

  /**
   * Return underlying context, which was used in oneapi::ccl::communicator
   */
  inline oneapi::ccl::context get_context() const{
    return _comm.get_context();
  };

  /**
  * Return oneapi::ccl::communicator
  */
  inline const oneapi::ccl::communicator &get_ccl_communicator()
  {
    return _comm;
  }

private:
  oneapi::ccl::device _device_comm;
  oneapi::ccl::context _context_comm;
  oneapi::ccl::communicator _comm;
};

} // namespace ccl
} // namespace dpct

#endif // __DPCT_CCL_UTILS_HPP__