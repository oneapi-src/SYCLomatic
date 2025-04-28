//==---- ze_utils.hpp ---------------------------------*- C++
//-*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __ZE_UTILS_HPP__
#define __ZE_UTILS_HPP__

#ifdef ONEAPI_BACKEND_LEVEL_ZERO_EXT
#if defined(__linux__)
#include <level_zero/ze_api.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/sycl.hpp>
namespace dpct {
namespace experimental {

namespace detail {

/// Covert remote fd to the local fd through IPC handle extension.
/// \param [in] ipc_ext_handle The extension of the IPC handle
/// \returns Local process file descriptor
template <class T> int convert_fd_from_handle(T ipc_ext_handle) {
  int pidfd = syscall(434, ipc_ext_handle.pid,
                      0); // obtain a file descriptor that refers to a
                          // process(requires kernel 5.6+).
  if (pidfd < 0)
    return -1;
  return syscall(438, pidfd, *(int *)ipc_ext_handle.handle.data,
                 0); // obtain a duplicate of another process's file
                     // descriptor(requires kernel 5.6+).
}

} // namespace detail

struct ipc_mem_handle_ext_t {
  pid_t pid;
  ze_ipc_mem_handle_t handle;
};

/// Acquires IPC handle for shared memory region.
/// \param [in] ptr Pointer to shared memory region
/// \param [out] handle_ptr Output IPC handle
/// \returns Level Zero operation status code
inline ze_result_t get_mem_ipc_handle(const void *ptr,
                                      ipc_mem_handle_ext_t *handle_ptr) {
  handle_ptr->pid = getpid();
  return zeMemGetIpcHandle(
      sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
          dpct::get_current_device().get_context()),
      ptr, &handle_ptr->handle);
}

/// Maps remote IPC memory to local address space.
/// \param [in] ipc_ext_handle The extension of the IPC handle
/// \param [out] ptr Mapped memory pointer in local process
/// \returns Level Zero operation status code
inline ze_result_t open_mem_ipc_handle(ipc_mem_handle_ext_t ipc_ext_handle,
                                       void **ptr) {
  int newfd = detail::convert_fd_from_handle(ipc_ext_handle);
  if (newfd < 0)
    throw std::runtime_error("Cannot convert fd from handle");
  *((int *)ipc_ext_handle.handle.data) = newfd;
  return zeMemOpenIpcHandle(
      sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
          dpct::get_current_device().get_context()),
      sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
          (sycl::device)dpct::get_current_device()),
      ipc_ext_handle.handle, 0u, ptr);
}

} // namespace experimental
} // namespace dpct

#endif // __linux__
#endif // ONEAPI_BACKEND_LEVEL_ZERO_EXT

#endif // ! __ZE_UTILS_HPP__