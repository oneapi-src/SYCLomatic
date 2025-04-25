//==---- ze_util.hpp ---------------------------------*- C++ -*----------------==//
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
#include "level_zero/ze_api.h"
#include "sycl/ext/oneapi/backend/level_zero.hpp"
#include <sycl/sycl.hpp>
namespace dpct {
namespace experimental {

///  System call number definitions for kernel compatibility.
///  SYS_pidfd_open: Process file descriptor opener (requires kernel 5.6+).
///  SYS_pidfd_getfd: Cross-process FD fetcher system call.
#ifndef SYS_pidfd_open
#define SYS_pidfd_open 434
#endif

#ifndef SYS_pidfd_getfd
#define SYS_pidfd_getfd 438
#endif

/// Process id and IPC memory handle structure for cross-process sharing.
struct ipc_mem_handle_ext_t {
  pid_t pid;
  ze_ipc_mem_handle_t handle;
};

/// Acquires IPC handle for shared memory region.
/// \param [in] ptr Pointer to shared memory region
/// \param [out] phipc Output IPC handle
/// \returns Level Zero operation status code
ze_result_t get_mem_ipc_handle(const void *ptr,
                               ipc_mem_handle_ext_t *ipc_ext_handle) {
  ipc_ext_handle->pid = getpid();
  return zeMemGetIpcHandle(
      sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
          dpct::get_current_device().get_context()),
      ptr, &ipc_ext_handle->handle);
}

/// Releases resources associated with IPC handle.
/// \param [in] ptr Pointer to shared memory region
/// \returns Level Zero operation status code
ze_result_t close_mem_ipc_handle(const void *ptr) {
  if (ptr == nullptr) {
    return ZE_RESULT_SUCCESS;
  }
  return zeMemCloseIpcHandle(
      sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
          dpct::get_current_device().get_context()),
      (char *)ptr);
}

/// Covert remote fd to the local fd through the IPC handle extension.
/// \param [in] ipc_ext_handle The extension of the IPC handle
/// \returns Local process file descriptor
template <class T> int convert_fd_pidfd_from_handle(T ipc_ext_handle) {
  int pidfd = syscall(SYS_pidfd_open, ipc_ext_handle.pid, 0);
  int fd;
  memcpy(&fd, (void *)&ipc_ext_handle.handle.data, sizeof(int));
  return syscall(SYS_pidfd_getfd, pidfd, fd, 0);
}

/// Maps remote IPC memory to local address space.
/// \param [in] ipc_ext_handle The extension of the IPC handle
/// \param [out] ptr Mapped memory pointer in local process
/// \returns Level Zero operation status code
ze_result_t open_mem_ipc_handle(ipc_mem_handle_ext_t ipc_ext_handle,
                                void **ptr) {
  int newfd = convert_fd_pidfd_from_handle(ipc_ext_handle);
  memcpy(&ipc_ext_handle.handle.data, &newfd, sizeof(newfd));
  return zeMemOpenIpcHandle(
      sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
          dpct::get_current_device().get_context()),
      sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
          (sycl::device)dpct::get_current_device()),
      ipc_ext_handle.handle, 0u, ptr);
}

} // namespace experimental
#endif // __linux__
#endif // ONEAPI_BACKEND_LEVEL_ZERO_EXT

} // namespace dpct
#endif // ! __ZE_UTILS_HPP__