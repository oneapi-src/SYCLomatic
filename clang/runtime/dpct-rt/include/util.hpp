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

template <int... Ints> struct integer_sequence {};
template <int Size, int... Ints>
struct make_index_sequence
    : public make_index_sequence<Size - 1, Size - 1, Ints...> {};
template <int... Ints>
struct make_index_sequence<0, Ints...> : public integer_sequence<Ints...> {};

template <typename T> struct DataType { using T2 = T; };
template <typename T> struct DataType<cl::sycl::vec<T, 2>> {
  using T2 = std::complex<T>;
};

/// Copy matrix data. The default leading dimension is column.
/// \param [out] to_ptr A poniter points to the destination location.
/// \param [in] from_ptr A poniter points to the source location.
/// \param [in] to_ld The leading dimension the destination matrix.
/// \param [in] from_ld The leading dimension the source matrix.
/// \param [in] rows The number of rows of the source matrix.
/// \param [in] cols The number of columns of the source matrix.
/// \param [in] direction The direction of the data copy.
/// \param [in] queue The queue where the routine should be executed.
template <typename T>
inline void matrix_mem_copy(T *to_ptr, const T *from_ptr, int to_ld,
                            int from_ld, int rows, int cols,
                            memcpy_direction direction, cl::sycl::queue &queue) {
  using Ty = typename DataType<T>::T2;
  if (to_ptr == from_ptr && to_ld == from_ld) {
    return;
  }
  if (to_ld == from_ld) {
    dpct_memcpy(queue, (void *)to_ptr, (void *)from_ptr,
                sizeof(Ty) * to_ld * cols, direction);
  } else {
    auto to_ptr_t = to_ptr;
    auto from_ptr_t = from_ptr;
    to_ptr_t = to_ptr_t - to_ld;
    from_ptr_t = from_ptr_t - from_ld;
    for (int c = 0; c < cols; ++c) {
      to_ptr_t = to_ptr_t + to_ld;
      from_ptr_t = from_ptr_t + from_ld;
      dpct_memcpy(queue, (void *)(to_ptr_t), (void *)(from_ptr_t),
                  sizeof(Ty) * rows, direction);
    }
  }
}

} // namespace dpct

#endif // __DPCT_UTIL_HPP__
