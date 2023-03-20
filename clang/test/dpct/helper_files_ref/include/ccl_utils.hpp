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
    throw std::runtime_error("to_ccl_datatype: unsupported data type.");
  }
}

/// helper class to make sure ccl::init() be called before other oneCCL API
class ccl_init_helper {
public:
  ccl_init_helper() { oneapi::ccl::init(); }
};

} // namespace detail

/// Get concatenated library version as an integer.
static inline int get_version() {
  oneapi::ccl::init();
  auto ver = oneapi::ccl::get_library_version();
  return ver.major * 10000 + ver.minor * 100 + ver.update;
}

/// Create main kvs and return its address.
static inline oneapi::ccl::kvs::address_type create_kvs_address() {
  oneapi::ccl::init();
  auto ptr = oneapi::ccl::create_main_kvs();
  auto addr = ptr->get_address();
  detail::get_kvs(addr) = ptr;
  return addr;
}

/// Get stored kvs with /p addr if exist. Otherwise, create kvs with /p addr.
static inline std::shared_ptr<oneapi::ccl::kvs>
create_kvs(const oneapi::ccl::kvs::address_type &addr) {
  oneapi::ccl::init();
  auto &ptr = detail::get_kvs(addr);
  if (!ptr)
    ptr = oneapi::ccl::create_kvs(addr);
  return ptr;
}

/// dpct communicator extension
class communicator_ext:public dpct::ccl::detail::ccl_init_helper {
public:
  communicator_ext(
      int size, int rank, oneapi::ccl::kvs::address_type id,
      const oneapi::ccl::comm_attr &attr = oneapi::ccl::default_comm_attr)
      : _device_comm(oneapi::ccl::create_device(
            static_cast<sycl::device &>(dpct::get_current_device()))),
        _context_comm(oneapi::ccl::create_context(dpct::get_default_context())),
        _comm(oneapi::ccl::create_communicator(
            size, rank, _device_comm, _context_comm, dpct::ccl::create_kvs(id),
            attr)) {}

  ~communicator_ext(){};

  /// Return the rank in a oneapi::ccl::communicator
  /// \returns The rank corresponding to communicator object
  int rank() const {
    return _comm.rank();
  }

  /// Retrieves the number of rank in oneapi::ccl::communicator
  /// \returns The number of the ranks
  int size() const {
    return _comm.size();
  }

  /// Return underlying native device, which was used in oneapi::ccl::communicator
  sycl::device get_device() const {
    return _comm.get_device().get_native();
  }

  /// Return underlying native context, which was used in oneapi::ccl::communicator
  sycl::context get_context() const {
    return _comm.get_context().get_native();
  };

  /// \brief Allreduce is a collective communication operation that performs the global reduction operation
  ///       on values from all ranks of communicator and distributes the result back to all ranks.
  /// \param send_buf the buffer with @c count elements of @c dtype that stores local data to be reduced
  /// \param recv_buf [out] the buffer to store reduced result, must have the same dimension as @c send_buf
  /// \param count the number of elements of type @c dtype in @c send_buf and @c recv_buf
  /// \param dtype the datatype of elements in @c send_buf and @c recv_buf
  /// \param rtype the type of the reduction operation to be applied
  /// \param stream a oneapi::ccl::stream associated with the operation
  /// \return @ref return oneapi::ccl::event to track the progress of the operation
  oneapi::ccl::event allreduce(const void *sendbuff, void *recvbuff,
                               size_t count, dpct::library_data_t dtype,
                               oneapi::ccl::reduction rtype,
                               const oneapi::ccl::stream &stream) const {
    return oneapi::ccl::allreduce(sendbuff, recvbuff, count,
                                  dpct::ccl::detail::to_ccl_datatype(dtype),
                                  rtype, _comm, stream);
  };

private:
  oneapi::ccl::device _device_comm;
  oneapi::ccl::context _context_comm;
  oneapi::ccl::communicator _comm;
};

typedef dpct::ccl::communicator_ext * comm_ptr;

} // namespace ccl
} // namespace dpct

#endif // __DPCT_CCL_UTILS_HPP__