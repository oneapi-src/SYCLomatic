//==---- dispatch.hpp -----------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __DPCT_PROXY_HPP__
#define __DPCT_PROXY_HPP__

// Only for testing
#include <syclcompat.hpp>

namespace dpct {
namespace detail {
namespace dispatch {
#if USE_DPCT_HELPER
template <typename T> using DataType = ::dpct::DataType<T>;
using pointer_access_attribute = ::dpct::detail::pointer_access_attribute;
using memcpy_direction = ::dpct::memcpy_direction;
inline ::dpct::device_ext &get_current_device() {
  return ::dpct::get_current_device();
}
using queue_ptr = ::dpct::queue_ptr;
template <class... Args> using kernel_name = dpct_kernel_name<Args...>;
#else
template <typename T> using DataType = ::syclcompat::detail::DataType<T>;
using pointer_access_attribute = ::syclcompat::detail::pointer_access_attribute;
using memcpy_direction = ::syclcompat::experimental::memcpy_direction;
inline ::syclcompat::device_ext &get_current_device() {
  return ::syclcompat::get_current_device();
}
using queue_ptr = ::syclcompat::queue_ptr;
template <class... Args> using kernel_name = syclcompat_kernel_name<Args...>;
#endif

inline sycl::queue &get_default_queue() {
#if USE_DPCT_HELPER
  return ::dpct::get_default_queue();
#else
  return *::syclcompat::detail::dev_mgr::instance()
              .current_device()
              .default_queue();
#endif
}

inline sycl::context get_default_context() {
#if USE_DPCT_HELPER
  return ::dpct::get_default_context();
#else
  return ::syclcompat::get_default_context();
#endif
}

inline sycl::event
memcpy(sycl::queue &q, void *to_ptr, const void *from_ptr, size_t size,
       memcpy_direction direction = memcpy_direction::automatic,
       const std::vector<sycl::event> &dep_events = {}) {
#if USE_DPCT_HELPER
  return ::dpct::detail::dpct_memcpy(q, to_ptr, from_ptr, size, direction,
                                     dep_events);
#else
  return ::syclcompat::detail::memcpy(q, to_ptr, from_ptr, size, dep_events);
#endif
}

inline std::vector<sycl::event>
memcpy(sycl::queue &q, void *to_ptr, const void *from_ptr, size_t to_pitch,
       size_t from_pitch, size_t x, size_t y,
       memcpy_direction direction = memcpy_direction::automatic) {
#if USE_DPCT_HELPER
  return ::dpct::detail::dpct_memcpy(q, to_ptr, from_ptr, to_pitch, from_pitch,
                                     x, y, direction);
#else
  return ::syclcompat::detail::memcpy(q, to_ptr, from_ptr, to_pitch, from_pitch,
                                      x, y);
#endif
}

inline void *malloc(size_t size, sycl::queue q = get_default_queue()) {
#if USE_DPCT_HELPER
  return ::dpct::dpct_malloc(size, q);
#else
  return ::syclcompat::malloc(size, q);
#endif
}

template <typename valueT>
inline sycl::event fill(sycl::queue &q, void *dev_ptr, valueT value,
                        size_t size) {
#if USE_DPCT_HELPER
  return ::dpct::detail::dpct_memset<valueT>(q, dev_ptr, value, size);
#else
  return ::syclcompat::detail::fill<valueT>(q, dev_ptr, value, size);
#endif
}

inline void free(void *to_ptr, sycl::queue q) {
#if USE_DPCT_HELPER
  return ::dpct::detail::dpct_free(to_ptr, q);
#else
  return ::syclcompat::free(to_ptr, q);
#endif
}

inline sycl::event enqueue_free(const std::vector<void *> &pointers,
                                const std::vector<sycl::event> &events,
                                sycl::queue q = get_default_queue()) {
#if USE_DPCT_HELPER
  return ::dpct::detail::async_dpct_free(pointers, events, q);
#else
  return ::syclcompat::enqueue_free(pointers, events, q);
#endif
}

inline pointer_access_attribute get_pointer_attribute(sycl::queue q,
                                                      const void *ptr) {
#if USE_DPCT_HELPER
  return ::dpct::detail::get_pointer_attribute(q, ptr);
#else
  return ::syclcompat::detail::get_pointer_attribute(q, ptr);
#endif
}

} // namespace dispatch
} // namespace detail
} // namespace dpct

#endif // __DPCT_PROXY_HPP__
