/******************************************************************************
*
* Copyright 2018 - 2019 Intel Corporation.
*
* This software and the related documents are Intel copyrighted materials,
* and your use of them is governed by the express license under which they
* were provided to you ("License"). Unless the License provides otherwise,
* you may not use, modify, copy, publish, distribute, disclose or transmit
* this software or the related documents without Intel's prior written
* permission.

* This software and the related documents are provided as is, with no express
* or implied warranties, other than those that are expressly stated in the
* License.
*****************************************************************************/

//===--- util.hpp -------------------------------*- C++ -*-----===//

#ifndef __DPCT_UTIL_HPP__
#define __DPCT_UTIL_HPP__

#include <CL/sycl.hpp>
#include <complex>

namespace dpct {

template <typename I, typename O> inline O bit_cast(I i) {
  return *reinterpret_cast<O *>(&i);
}
namespace internal {
template <int... Ints> struct integer_sequence {};
template <int Size, int... Ints>
struct make_index_sequence
    : public make_index_sequence<Size - 1, Size - 1, Ints...> {};
template <int... Ints>
struct make_index_sequence<0, Ints...> : public integer_sequence<Ints...> {};

template <class T>
static inline T *compute_offset(T *ptr, const cl::sycl::range<3> &range,
                                const cl::sycl::id<3> &offset) {
  return ptr + (offset.get(0) + offset.get(1) * range.get(0) +
                offset.get(2) * range.get(0) * range.get(1));
}
static inline void *compute_offset(void *ptr, const cl::sycl::range<3> &range,
                                   const cl::sycl::id<3> &offset) {
  return compute_offset((char *)ptr, range, offset);
}
} // namespace internal

template <typename T> struct DataType { using T2 = T; };
template <typename T> struct DataType<cl::sycl::vec<T, 2>> {
  using T2 = std::complex<T>;
};
} // namespace dpct

#endif // __DPCT_UTIL_HPP__
