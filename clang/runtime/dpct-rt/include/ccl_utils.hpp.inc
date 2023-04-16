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
#include "device.hpp"
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

// DPCT_LABEL_BEGIN|ccl_init_helper|dpct::ccl::detail
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
/// Help class to init ccl environment. 
class ccl_init_helper {
public:
  ccl_init_helper() { oneapi::ccl::init(); }
};
// DPCT_LABEL_END

} // namespace detail

// DPCT_LABEL_BEGIN|get_version|dpct::ccl
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
/// Get concatenated library version as an integer.
static inline int get_version() {
  oneapi::ccl::init();
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
  oneapi::ccl::init();
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
  oneapi::ccl::init();
  auto &ptr = detail::get_kvs(addr);
  if (!ptr)
    ptr = oneapi::ccl::create_kvs(addr);
  return ptr;
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|communicator_wrapper|dpct::ccl
// DPCT_DEPENDENCY_BEGIN
// CclUtils|communicator_wrapper_2
// CclUtils|create_kvs
// CclUtils|ccl_init_helper
// Device|get_default_context
// Device|get_current_device
// DPCT_DEPENDENCY_END
// DPCT_CODE
/// dpct communicator extension
class communicator_wrapper : public dpct::ccl::detail::ccl_init_helper {
public:
  communicator_wrapper(
      int size, int rank, oneapi::ccl::kvs::address_type id,
      const oneapi::ccl::comm_attr &attr = oneapi::ccl::default_comm_attr)
      : _device_comm(oneapi::ccl::create_device(
            static_cast<sycl::device &>(dpct::get_current_device()))),
        _context_comm(oneapi::ccl::create_context(dpct::get_default_context())),
        _comm(oneapi::ccl::create_communicator(
            size, rank, _device_comm, _context_comm, dpct::ccl::create_kvs(id),
            attr)) {
    _queue_init = false;
    _ccl_stream_ptr = nullptr;
  }

  ~communicator_wrapper() {
    delete _ccl_stream_ptr;
  };
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|communicator_wrapper_rank|dpct::ccl
// DPCT_DEPENDENCY_BEGIN
// CclUtils|communicator_wrapper
// DPCT_DEPENDENCY_END
// DPCT_CODE
  /// Return the rank in a oneapi::ccl::communicator
  /// \returns The rank corresponding to communicator object
  int rank() const {
    return _comm.rank();
  }
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|communicator_wrapper_size|dpct::ccl
// DPCT_DEPENDENCY_BEGIN
// CclUtils|communicator_wrapper
// DPCT_DEPENDENCY_END
// DPCT_CODE
  /// Retrieves the number of rank in oneapi::ccl::communicator
  /// \returns The number of the ranks
  int size() const {
    return _comm.size();
  }
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|communicator_wrapper_get_device|dpct::ccl
// DPCT_DEPENDENCY_BEGIN
// CclUtils|communicator_wrapper
// DPCT_DEPENDENCY_END
// DPCT_CODE
  /// Return underlying native device, which was used in oneapi::ccl::communicator
  sycl::device get_device() const {
    return _comm.get_device().get_native();
  }
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|communicator_wrapper_allreduce|dpct::ccl
// DPCT_DEPENDENCY_BEGIN
// CclUtils|communicator_wrapper
// DPCT_DEPENDENCY_END
// DPCT_CODE
  /// \brief allreduce is a collective communication operation that performs the global reduction operation
  ///       on values from all ranks of communicator and distributes the result back to all ranks.
  /// \param send_buf the buffer with @c count elements of @c dtype that stores local data to be reduced
  /// \param recv_buf [out] the buffer to store reduced result, must have the same dimension as @c send_buf
  /// \param count the number of elements of type @c dtype in @c send_buf and @c recv_buf
  /// \param dtype the datatype of elements in @c send_buf and @c recv_buf
  /// \param rtype the type of the reduction operation to be applied
  /// \param queue_ptr a sycl::queue ptr associated with the operation
  /// \return @ref void
  void allreduce(const void *sendbuff, void *recvbuff, size_t count,
                 oneapi::ccl::datatype dtype, oneapi::ccl::reduction rtype,
                 sycl::queue *queue_ptr) {
    call_func_wrapper(
        [=](const oneapi::ccl::stream &stream) {
          return oneapi::ccl::allreduce(sendbuff, recvbuff, count, dtype, rtype,
                                        _comm, stream);
        },
        queue_ptr);
  }
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|communicator_wrapper_reduce|dpct::ccl
// DPCT_DEPENDENCY_BEGIN
// CclUtils|communicator_wrapper
// DPCT_DEPENDENCY_END
// DPCT_CODE
  /// \brief reduce is a collective communication operation that performs the
  ///        global reduction operation on values from all ranks of the communicator
  ///        and returns the result to the root rank.
  /// \param send_buf the buffer with @c count elements of @c dtype that stores
  ///        local data to be reduced 
  /// \param recv_buf [out] the buffer to store reduced result, 
  ///        must have the same dimension as @c send_buf 
  /// \param count the number of elements of type @c dtype in @c send_buf and @c recv_buf 
  /// \param dtype the datatype of elements in @c send_buf and @c recv_buf 
  /// \param root the rank that gets the result of reduction 
  /// \param rtype the type of the reduction operation to be applied 
  /// \param queue_ptr a sycl::queue ptr associated with the operation 
  /// \return @ref void
  void reduce(const void *sendbuff, void *recvbuff, size_t count,
              oneapi::ccl::datatype dtype, oneapi::ccl::reduction rtype,
              int root, sycl::queue *queue_ptr) {
    call_func_wrapper(
        [=](const oneapi::ccl::stream &stream) {
          return oneapi::ccl::reduce(sendbuff, recvbuff, count, dtype, rtype,
                                     root, _comm, stream);
        },
        queue_ptr);
  }
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|communicator_wrapper_broadcast|dpct::ccl
// DPCT_DEPENDENCY_BEGIN
// CclUtils|communicator_wrapper
// DPCT_DEPENDENCY_END
// DPCT_CODE
  /// \brief broadcast is a collective communication operation that broadcasts data
  ///        from one rank of communicator (denoted as root) to all other ranks.
  ///        Only support in-place operation
  /// \param send_buf the buffer with @c count elements of @c dtype that stores
  ///        local data to be reduced 
  /// \param recv_buf [out] the buffer to store reduced result
  /// \param count the number of elements of type @c dtype in @c buf 
  /// \param dtype thedatatype of elements in @c buf 
  /// \param root the rank that broadcasts @c buf
  /// \param queue_ptr a sycl::queue ptr associated with the operation
  /// \return @ref void
  void broadcast(void *sendbuff, void *recvbuff, size_t count,
                 oneapi::ccl::datatype dtype, int root,
                 sycl::queue *queue_ptr) {
    if (sendbuff != recvbuff) {
      throw std::runtime_error(
          "oneCCL broadcast only support in-place operation. "
          "send_buf and recv_buf must be same.");
      return;
    }
    call_func_wrapper(
        [=](const oneapi::ccl::stream &stream) {
          return oneapi::ccl::broadcast(recvbuff, count, dtype, root, _comm,
                                        stream);
        },
        queue_ptr);
  }
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|communicator_wrapper_reduce_scatter|dpct::ccl
// DPCT_DEPENDENCY_BEGIN
// CclUtils|communicator_wrapper
// DPCT_DEPENDENCY_END
// DPCT_CODE
  /// \brief reduce_scatter is a collective communication operation that performs the global reduction operation
  ///        on values from all ranks of the communicator and scatters the result in blocks back to all ranks.
  /// \param send_buf the buffer with @c count elements of @c dtype that stores local data to be reduced
  /// \param recv_buf [out] the buffer to store reduced result, must have the same dimension as @c send_buf
  /// \param recv_count the number of elements of type @c dtype in receive block
  /// \param dtype the datatype of elements in @c send_buf and @c recv_buf
  /// \param rtype the type of the reduction operation to be applied
  /// \param queue_ptr a sycl::queue ptr associated with the operation
  /// \return @ref void
  void reduce_scatter(const void *sendbuff, void *recvbuff, size_t recv_count,
                      oneapi::ccl::datatype dtype, oneapi::ccl::reduction rtype,
                      sycl::queue *queue_ptr) {
    call_func_wrapper(
        [=](const oneapi::ccl::stream &stream) {
          return oneapi::ccl::reduce_scatter(sendbuff, recvbuff, recv_count,
                                             dtype, rtype, _comm, stream);
        },
        queue_ptr);
  }
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|communicator_wrapper_2|dpct::ccl
// DPCT_DEPENDENCY_BEGIN
// CclUtils|communicator_wrapper
// DPCT_DEPENDENCY_END
// DPCT_CODE
private:
  oneapi::ccl::device _device_comm;
  oneapi::ccl::context _context_comm;
  oneapi::ccl::communicator _comm;
  sycl::queue _queue;
  bool _queue_init;
  oneapi::ccl::stream *_ccl_stream_ptr;

  template <class Fn>
  void call_func_wrapper(Fn func, sycl::queue *qptr) {
    if (_queue_init && *qptr != _queue) {
      call_func_async(func, qptr);
    } else {
      if(!_queue_init) {
        _queue = *qptr;
        _queue_init = true;
        _ccl_stream_ptr = new oneapi::ccl::stream(oneapi::ccl::create_stream(_queue));
      }
      std::invoke(func, *_ccl_stream_ptr);
    }
  }

  class call_func_async {
    sycl::queue *_q_ptr;
    struct call_async_impl {
      oneapi::ccl::stream _ccl_stream_impl;
      oneapi::ccl::event _ccl_event_impl;
      template <class Fn>
      explicit call_async_impl(Fn func, sycl::queue *qptr)
          : _ccl_stream_impl(oneapi::ccl::create_stream(*qptr)),
            _ccl_event_impl(std::invoke(func, _ccl_stream_impl)) {}
    };
    call_async_impl *_imp;

  public:
    template <class Fn>
    explicit call_func_async(Fn func, sycl::queue *qptr)
        : _q_ptr(qptr),
          _imp(new call_async_impl(func, qptr)) {}
    ~call_func_async() {
      _q_ptr->submit([&](sycl::handler &cgh)
                     { cgh.host_task([=]
                                     {
        _imp->_ccl_event_impl.wait();
        delete _imp; }); });
    }
  };
};
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|typedef_comm_ptr|dpct::ccl
// DPCT_DEPENDENCY_BEGIN
// CclUtils|communicator_wrapper
// DPCT_DEPENDENCY_END
// DPCT_CODE
typedef dpct::ccl::communicator_wrapper *comm_ptr;
// DPCT_LABEL_END

} // namespace ccl
} // namespace dpct

#endif // __DPCT_CCL_UTILS_HPP__