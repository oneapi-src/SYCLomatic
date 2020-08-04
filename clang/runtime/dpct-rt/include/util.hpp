/******************************************************************************
*
* Copyright 2018 - 2020 Intel Corporation.
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

/// Obtain a value of type O by reinterpreting the object representation of I
/// \param i input data with type I
/// \return The value of type O
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

inline void matrix_mem_copy(void *to_ptr, const void *from_ptr, int to_ld,
                            int from_ld, int rows, int cols, int elem_size,
                            memcpy_direction direction = automatic,
                            cl::sycl::queue &queue = dpct::get_default_queue(),
                            bool async = false) {
  if (to_ptr == from_ptr && to_ld == from_ld) {
    return;
  }

  if (to_ld == from_ld) {
    size_t cpoy_size = elem_size * ((cols - 1) * to_ld + rows);
    if (async)
      detail::dpct_memcpy(queue, (void *)to_ptr, (void *)from_ptr,
                          cpoy_size, direction);
    else
      detail::dpct_memcpy(queue, (void *)to_ptr, (void *)from_ptr,
                          cpoy_size, direction).wait();
  } else {
    if (async)
      detail::dpct_memcpy(queue, to_ptr, from_ptr, elem_size * to_ld,
                          elem_size * from_ld, elem_size * rows, cols,
                          direction);
    else
      cl::sycl::event::wait(detail::dpct_memcpy(
          queue, to_ptr, from_ptr, elem_size * to_ld, elem_size * from_ld,
          elem_size * rows, cols, direction));
  }
}

/// Copy matrix data. The default leading dimension is column.
/// \param [out] to_ptr A poniter points to the destination location.
/// \param [in] from_ptr A poniter points to the source location.
/// \param [in] to_ld The leading dimension the destination matrix.
/// \param [in] from_ld The leading dimension the source matrix.
/// \param [in] rows The number of rows of the source matrix.
/// \param [in] cols The number of columns of the source matrix.
/// \param [in] direction The direction of the data copy.
/// \param [in] queue The queue where the routine should be executed.
/// \param [in] async If this argument is true, the return of the function
/// does NOT guarantee the copy is completed.
template <typename T>
inline void matrix_mem_copy(T *to_ptr, const T *from_ptr, int to_ld,
                            int from_ld, int rows, int cols,
                            memcpy_direction direction = automatic,
                            cl::sycl::queue &queue = dpct::get_default_queue(),
                            bool async = false) {
  using Ty = typename DataType<T>::T2;
  matrix_mem_copy((void *)to_ptr, (void *)from_ptr, to_ld, from_ld, rows, cols,
                  sizeof(Ty), direction, queue, async);
}

/// Cast the high or low 32 bits of a double to an integer.
/// \param [in] d The double value.
/// \param [in] use_high32 Cast the high 32 bits of the double if true;
/// otherwise cast the low 32 bits.
inline int cast_double_to_int(double d, bool use_high32 = true) {
  cl::sycl::vec<double, 1> v0{d};
  auto v1 = v0.as<cl::sycl::int2>();
  if (use_high32)
    return v1[0];
  return v1[1];
}

/// Combine two integers, the first as the high 32 bits and the second
/// as the low 32 bits, into a double.
/// \param [in] high32 The integer as the high 32 bits
/// \param [in] low32 The integer as the low 32 bits
inline double cast_ints_to_double(int high32, int low32) {
  cl::sycl::int2 v0{high32, low32};
  auto v1 = v0.as<cl::sycl::vec<double, 1>>();
  return v1;
}

/// Compute fast_length for variable-length array
/// \param [in] a The array
/// \param [in] len Length of the array
/// \return The computed fast_length
inline float fast_length(const float *a, int len) {
  switch (len) {
  case 1:
    return cl::sycl::fast_length(a[0]);
  case 2:
    return cl::sycl::fast_length(cl::sycl::float2(a[0], a[1]));
  case 3:
    return cl::sycl::fast_length(cl::sycl::float3(a[0], a[1], a[2]));
  case 4:
    return cl::sycl::fast_length(cl::sycl::float4(a[0], a[1], a[2], a[3]));
  case 0:
    return 0;
  default:
    float f = 0;
    for (int i = 0; i < len; ++i)
      f += a[i] * a[i];
    return cl::sycl::sqrt(f);
  }
}

/// Compute vectorized max for two values, with each value treated as a vector
/// type S
/// \param [in] S The type of the vector
/// \param [in] T The type of the original values
/// \param [in] a The first value
/// \param [in] b The second value
/// \return The vectorize max of the two values
template <typename S, typename T>
inline T vectorized_max(T a, T b) {
  cl::sycl::vec<T, 1> v0{a}, v1{b};
  auto v2 = v0.template as<S>();
  auto v3 = v1.template as<S>();
  v2 = cl::sycl::max(v2, v3);
  v0 = v2.template as<cl::sycl::vec<T, 1>>();
  return v0;
}

/// Compute vectorized min for two values, with each value treated as a vector
/// type S
/// \param [in] S The type of the vector
/// \param [in] T The type of the original values
/// \param [in] a The first value
/// \param [in] b The second value
/// \return The vectorized min of the two values
template <typename S, typename T>
inline T vectorized_min(T a, T b) {
  cl::sycl::vec<T, 1> v0{a}, v1{b};
  auto v2 = v0.template as<S>();
  auto v3 = v1.template as<S>();
  v2 = cl::sycl::min(v2, v3);
  v0 = v2.template as<cl::sycl::vec<T, 1>>();
  return v0;
}

/// Compute vectorized isgreater for two values, with each value treated as a
/// vector type S
/// \param [in] S The type of the vector
/// \param [in] T The type of the original values
/// \param [in] a The first value
/// \param [in] b The second value
/// \return The vectorized greater than of the two values
template <typename S, typename T>
inline T vectorized_isgreater(T a, T b) {
  cl::sycl::vec<T, 1> v0{a}, v1{b};
  auto v2 = v0.template as<S>();
  auto v3 = v1.template as<S>();
  auto v4 = cl::sycl::isgreater(v2, v3);
  v0 = v4.template as<cl::sycl::vec<T, 1>>();
  return v0;
}

/// Compute vectorized isgreater for two unsigned int values, with each value
/// treated as a vector of two unsigned short
/// \param [in] a The first value
/// \param [in] b The second value
/// \return The vectorized greater than of the two values
template<>
inline unsigned
vectorized_isgreater<cl::sycl::ushort2, unsigned>(unsigned a, unsigned b) {
  cl::sycl::vec<unsigned, 1> v0{a}, v1{b};
  auto v2 = v0.template as<cl::sycl::ushort2>();
  auto v3 = v1.template as<cl::sycl::ushort2>();
  cl::sycl::ushort2 v4;
  v4[0] = v2[0] > v3[0];
  v4[1] = v2[1] > v3[1];
  v0 = v4.template as<cl::sycl::vec<unsigned, 1>>();
  return v0;
}

} // namespace dpct

#endif // __DPCT_UTIL_HPP__
