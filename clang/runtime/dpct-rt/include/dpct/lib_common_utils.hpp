//==---- lib_common_utils.hpp ---------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __DPCT_LIB_COMMON_UTILS_HPP__
#define __DPCT_LIB_COMMON_UTILS_HPP__

#if USE_DPCT_HELPER
#include "memory.hpp"
#include "util.hpp"
#else
#include <syclcompat.hpp>
#endif

#if defined(__has_include) && __has_include(<oneapi/mkl.hpp>)
#include <oneapi/mkl.hpp>
#endif
#include <sycl/sycl.hpp>

namespace dpct {
namespace detail {
namespace proxy {
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

inline void *malloc(size_t size, sycl::queue &q) {
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

inline void free(void *to_ptr, sycl::queue &q) {
#if USE_DPCT_HELPER
  return ::dpct::detail::dpct_free(to_ptr, q);
#else
  return ::syclcompat::free(to_ptr, q);
#endif
}

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

} // namespace proxy

template <typename T> inline auto get_memory(const void *x) {
  T *new_x = reinterpret_cast<T *>(const_cast<void *>(x));
#ifdef DPCT_USM_LEVEL_NONE
  return dpct::get_buffer<std::remove_cv_t<T>>(new_x);
#else
  return new_x;
#endif
}

template <typename T>
inline typename ::dpct::detail::proxy::DataType<T>::T2
get_value(const T *s, sycl::queue &q) {
  using Ty = typename ::dpct::detail::proxy::DataType<T>::T2;
  Ty s_h;
  if (::dpct::detail::proxy::get_pointer_attribute(q, s) ==
      ::dpct::detail::proxy::pointer_access_attribute::device_only)
    ::dpct::detail::proxy::memcpy(
        q, (void *)&s_h, (void *)s, sizeof(T),
        ::dpct::detail::proxy::memcpy_direction::device_to_host)
        .wait();
  else
    s_h = *reinterpret_cast<const Ty *>(s);
  return s_h;
}
} // namespace detail

enum class version_field : int { major, minor, update, patch };

#if defined(__has_include) && __has_include(<oneapi/mkl.hpp>)
/// Returns the requested field of Intel(R) oneAPI Math Kernel Library version.
/// \param field The version information field (major, minor, update or patch).
/// \param result The result value.
inline void mkl_get_version(version_field field, int *result) {
#ifndef __INTEL_MKL__
  throw std::runtime_error("The oneAPI Math Kernel Library (oneMKL) Interfaces "
                           "Project does not support this API.");
#else
  MKLVersion version;
  mkl_get_version(&version);
  if (version_field::major == field) {
    *result = version.MajorVersion;
  } else if (version_field::minor == field) {
    *result = version.MinorVersion;
  } else if (version_field::update == field) {
    *result = version.UpdateVersion;
  } else if (version_field::patch == field) {
    *result = 0;
  } else {
    throw std::runtime_error("unknown field");
  }
#endif
}
#endif

enum class library_data_t : unsigned char {
  real_float = 0,
  complex_float,
  real_double,
  complex_double,
  real_half,
  complex_half,
  real_bfloat16,
  complex_bfloat16,
  real_int4,
  complex_int4,
  real_uint4,
  complex_uint4,
  real_int8,
  complex_int8,
  real_uint8,
  complex_uint8,
  real_int16,
  complex_int16,
  real_uint16,
  complex_uint16,
  real_int32,
  complex_int32,
  real_uint32,
  complex_uint32,
  real_int64,
  complex_int64,
  real_uint64,
  complex_uint64,
  real_int8_4,
  real_int8_32,
  real_uint8_4,
  real_f8_e4m3,
  real_f8_e5m2,
  library_data_t_size
};

enum class compute_type : int {
  f16,
  f16_standard,
  f32,
  f32_standard,
  f32_fast_bf16,
  f32_fast_tf32,
  f64,
  f64_standard,
  i32,
  i32_standard,
};

namespace detail {
template <typename ArgT>
inline constexpr std::uint64_t get_type_combination_id(ArgT Val) {
  static_assert((unsigned char)library_data_t::library_data_t_size <=
                    std::numeric_limits<unsigned char>::max() &&
                "library_data_t size exceeds limit.");
  static_assert(std::is_same_v<ArgT, library_data_t>, "Unsupported ArgT");
  return (std::uint64_t)Val;
}

template <typename FirstT, typename... RestT>
inline constexpr std::uint64_t get_type_combination_id(FirstT FirstVal,
                                                       RestT... RestVal) {
  static_assert((std::uint8_t)library_data_t::library_data_t_size <=
                    std::numeric_limits<unsigned char>::max() &&
                "library_data_t size exceeds limit.");
  static_assert(sizeof...(RestT) <= 8 && "Too many parameters");
  static_assert(std::is_same_v<FirstT, library_data_t>, "Unsupported FirstT");
  return get_type_combination_id(RestVal...) << 8 | ((std::uint64_t)FirstVal);
}

inline constexpr std::size_t library_data_size[] = {
    8 * sizeof(float),                    // real_float
    8 * sizeof(std::complex<float>),      // complex_float
    8 * sizeof(double),                   // real_double
    8 * sizeof(std::complex<double>),     // complex_double
    8 * sizeof(sycl::half),               // real_half
    8 * sizeof(std::complex<sycl::half>), // complex_half
    16,                                   // real_bfloat16
    16 * 2,                               // complex_bfloat16
    4,                                    // real_int4
    4 * 2,                                // complex_int4
    4,                                    // real_uint4
    4 * 2,                                // complex_uint4
    8,                                    // real_int8
    8 * 2,                                // complex_int8
    8,                                    // real_uint8
    8 * 2,                                // complex_uint8
    16,                                   // real_int16
    16 * 2,                               // complex_int16
    16,                                   // real_uint16
    16 * 2,                               // complex_uint16
    32,                                   // real_int32
    32 * 2,                               // complex_int32
    32,                                   // real_uint32
    32 * 2,                               // complex_uint32
    64,                                   // real_int64
    64 * 2,                               // complex_int64
    64,                                   // real_uint64
    64 * 2,                               // complex_uint64
    8,                                    // real_int8_4
    8,                                    // real_int8_32
    8                                     // real_uint8_4
};
} // namespace detail

#ifdef DPCT_USM_LEVEL_NONE
/// Cast a "rvalue reference to a temporary object" to an "lvalue reference to
/// that temporary object".
/// CAUTION:
/// The returned lvalue reference is available only before the last step in
/// evaluating the full-expression that contains this function call.
/// \param [in] temporary_object The rvalue reference to a temporary object.
/// \returns The lvalue reference to that temporary object.
template <typename T>
inline typename std::enable_if_t<std::is_rvalue_reference_v<T &&>, T &>
rvalue_ref_to_lvalue_ref(T &&temporary_object) {
  return temporary_object;
}
#endif
} // namespace dpct

#endif // __DPCT_LIB_COMMON_UTILS_HPP__
