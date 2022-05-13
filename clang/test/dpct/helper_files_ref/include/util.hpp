//==---- util.hpp ---------------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __DPCT_UTIL_HPP__
#define __DPCT_UTIL_HPP__

#include <CL/sycl.hpp>
#include <complex>
#include <type_traits>
#include <cassert>

namespace dpct {

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
/// \param [out] to_ptr A pointer points to the destination location.
/// \param [in] from_ptr A pointer points to the source location.
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
/// \returns The computed fast_length
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
/// type \p S
/// \param [in] S The type of the vector
/// \param [in] T The type of the original values
/// \param [in] a The first value
/// \param [in] b The second value
/// \returns The vectorized max of the two values
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
/// type \p S
/// \param [in] S The type of the vector
/// \param [in] T The type of the original values
/// \param [in] a The first value
/// \param [in] b The second value
/// \returns The vectorized min of the two values
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
/// vector type \p S
/// \param [in] S The type of the vector
/// \param [in] T The type of the original values
/// \param [in] a The first value
/// \param [in] b The second value
/// \returns The vectorized greater than of the two values
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
/// \returns The vectorized greater than of the two values
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

/// Reverse the bit order of an unsigned integer
/// \param [in] a Input unsigned integer value
/// \returns Value of a with the bit order reversed
template <typename T> inline T reverse_bits(T a) {
  static_assert(std::is_unsigned<T>::value && std::is_integral<T>::value,
                "unsigned integer required");
  if (!a)
    return 0;
  T mask = 0;
  size_t count = 4 * sizeof(T);
  mask = ~mask >> count;
  while (count) {
    a = ((a & mask) << count) | ((a & ~mask) >> count);
    count = count >> 1;
    mask = mask ^ (mask << count);
  }
  return a;
}

/// \param [in] a The first value contains 4 bytes
/// \param [in] b The second value contains 4 bytes
/// \param [in] s The selector value, only lower 16bit used
/// \returns the permutation result of 4 bytes selected in the way
/// specified by \p s from \p a and \p b
inline unsigned int byte_level_permute(unsigned int a, unsigned int b,
                                       unsigned int s) {
  unsigned int ret;
  ret =
      ((((std::uint64_t)b << 32 | a) >> (s & 0x7) * 8) & 0xff) |
      (((((std::uint64_t)b << 32 | a) >> ((s >> 4) & 0x7) * 8) & 0xff) << 8) |
      (((((std::uint64_t)b << 32 | a) >> ((s >> 8) & 0x7) * 8) & 0xff) << 16) |
      (((((std::uint64_t)b << 32 | a) >> ((s >> 12) & 0x7) * 8) & 0xff) << 24);
  return ret;
}

/// Find position of first least significant set bit in an integer.
/// ffs(0) returns 0.
///
/// \param [in] a Input integer value
/// \returns The position
template <typename T> inline int ffs(T a) {
  static_assert(std::is_integral<T>::value, "integer required");
  return (sycl::ext::intel::ctz(a) + 1) % (sizeof(T) * 8 + 1);
}

/// select_from_sub_group allows work-items to obtain a copy of a value held by
/// any other work-item in the sub_group. The input sub_group will be divided
/// into several logical sub_groups with id range [0, \p logical_sub_group_size
/// - 1]. Each work-item in logical sub_group gets value from another work-item
/// whose id is \p remote_local_id. If \p remote_local_id is outside the
/// logical sub_group id range, \p remote_local_id will modulo with \p
/// logical_sub_group_size. The \p logical_sub_group_size must be a power of 2
/// and not exceed input sub_group size.
/// \tparam T Input value type
/// \param [in] g Input sub_group
/// \param [in] x Input value
/// \param [in] remote_local_id Input source work item id
/// \param [in] logical_sub_group_size Input logical sub_group size
/// \returns The result
template <typename T>
T select_from_sub_group(sycl::sub_group g, T x, int remote_local_id,
                        int logical_sub_group_size = 32) {
  unsigned int start_index =
      g.get_local_linear_id() / logical_sub_group_size * logical_sub_group_size;
  return sycl::select_from_group(
      g, x, start_index + remote_local_id % logical_sub_group_size);
}

/// shift_sub_group_left move values held by the work-items in a sub_group
/// directly to another work-item in the sub_group, by shifting values a fixed
/// number of work-items to the left. The input sub_group will be divided into
/// several logical sub_groups with id range [0, \p logical_sub_group_size - 1].
/// Each work-item in logical sub_group gets value from another work-item whose
/// id is caller's id adds \p delta. If calculated id is outside the logical
/// sub_group id range, the work-item will get value from itself. The \p
/// logical_sub_group_size must be a power of 2 and not exceed input sub_group
/// size.
/// \tparam T Input value type
/// \param [in] g Input sub_group
/// \param [in] x Input value
/// \param [in] delta Input delta
/// \param [in] logical_sub_group_size Input logical sub_group size
/// \returns The result
template <typename T>
T shift_sub_group_left(sycl::sub_group g, T x, unsigned int delta,
                       int logical_sub_group_size = 32) {
  unsigned int id = g.get_local_linear_id();
  unsigned int end_index =
      (id / logical_sub_group_size + 1) * logical_sub_group_size;
  T result = sycl::shift_group_left(g, x, delta);
  if ((id + delta) >= end_index) {
    result = x;
  }
  return result;
}

/// shift_sub_group_right move values held by the work-items in a sub_group
/// directly to another work-item in the sub_group, by shifting values a fixed
/// number of work-items to the right. The input sub_group will be divided into
/// several logical_sub_groups with id range [0, \p logical_sub_group_size - 1].
/// Each work-item in logical_sub_group gets value from another work-item whose
/// id is caller's id subtracts \p delta. If calculated id is outside the
/// logical sub_group id range, the work-item will get value from itself. The \p
/// logical_sub_group_size must be a power of 2 and not exceed input sub_group
/// size.
/// \tparam T Input value type
/// \param [in] g Input sub_group
/// \param [in] x Input value
/// \param [in] delta Input delta
/// \param [in] logical_sub_group_size Input logical sub_group size
/// \returns The result
template <typename T>
T shift_sub_group_right(sycl::sub_group g, T x, unsigned int delta,
                        int logical_sub_group_size = 32) {
  unsigned int id = g.get_local_linear_id();
  unsigned int start_index =
      id / logical_sub_group_size * logical_sub_group_size;
  T result = sycl::shift_group_right(g, x, delta);
  if ((id - start_index) < delta) {
    result = x;
  }
  return result;
}

/// permute_sub_group_by_xor permutes values by exchanging values held by pairs
/// of work-items identified by computing the bitwise exclusive OR of the
/// work-item id and some fixed mask. The input sub_group will be divided into
/// several logical sub_groups with id range [0, \p logical_sub_group_size - 1].
/// Each work-item in logical sub_group gets value from another work-item whose
/// id is bitwise exclusive OR of the caller's id and \p mask. If calculated id
/// is outside the logical sub_group id range, the work-item will get value from
/// itself. The \p logical_sub_group_size must be a power of 2 and not exceed
/// input sub_group size.
/// \tparam T Input value type
/// \param [in] g Input sub_group
/// \param [in] x Input value
/// \param [in] mask Input mask
/// \param [in] logical_sub_group_size Input logical sub_group size
/// \returns The result
template <typename T>
T permute_sub_group_by_xor(sycl::sub_group g, T x, unsigned int mask,
                           int logical_sub_group_size = 32) {
  unsigned int id = g.get_local_linear_id();
  unsigned int start_index =
      id / logical_sub_group_size * logical_sub_group_size;
  unsigned int target_offset = (id % logical_sub_group_size) ^ mask;
  return sycl::select_from_group(g, x,
                                 target_offset < logical_sub_group_size
                                     ? start_index + target_offset
                                     : id);
}

/// The logical-group is the group of some work-items.
/// Currently only cl::sycl::group<3> is supported to be the parent group of
/// the logical-group.
class logical_group {
  cl::sycl::group<3> _g;
  uint32_t _logical_group_size;
  uint32_t _group_linear_range_in_parent;
public:
  /// Dividing \p parent_group into several logical-groups.
  /// \param [in] parent_group The group to be divided.
  /// \param [in] size The logical-group size.
  logical_group(cl::sycl::group<3> parent_group, uint32_t size)
      : _g(parent_group), _logical_group_size(size) {
    _group_linear_range_in_parent =
        (_g.get_local_linear_range() - 1) / _logical_group_size + 1;
  }
  /// Returns the index of the work-item within the logical-group.
  uint32_t get_local_linear_id() const {
    return _g.get_local_linear_id() % _logical_group_size;
  }
  /// Returns the index of the logical-group in the parent group.
  uint32_t get_group_linear_id() const {
    return _g.get_local_linear_id() / _logical_group_size;
  }
  /// Returns the number of work-items in the logical-group.
  uint32_t get_local_linear_range() const {
    if (_g.get_local_linear_range() % _logical_group_size == 0) {
      return _logical_group_size;
    }
    uint32_t last_item_group_id =
        _g.get_local_linear_range() / _logical_group_size;
    if (_g.get_local_linear_id() >=
        last_item_group_id * _logical_group_size) {
      return _g.get_local_linear_range() -
             last_item_group_id * _logical_group_size;
    } else {
      return _logical_group_size;
    }
  }
  /// Returns the number of logical-group in the parent group.
  uint32_t get_group_linear_range() const {
    return _group_linear_range_in_parent;
  }
};

/// Computes the multiplication of two complex numbers.
/// \tparam T Complex element type
/// \param [in] x The first input complex number
/// \param [in] y The second input complex number
/// \returns The result
template <typename T>
sycl::vec<T, 2> cmul(sycl::vec<T, 2> x, sycl::vec<T, 2> y) {
  std::complex<T> t1(x[0], x[1]), t2(y[0], y[1]);
  t1 = t1 * t2;
  return sycl::vec<T, 2>(t1.real(), t1.imag());
}

/// Computes the division of two complex numbers.
/// \tparam T Complex element type
/// \param [in] x The first input complex number
/// \param [in] y The second input complex number
/// \returns The result
template <typename T>
sycl::vec<T, 2> cdiv(sycl::vec<T, 2> x, sycl::vec<T, 2> y) {
  std::complex<T> t1(x[0], x[1]), t2(y[0], y[1]);
  t1 = t1 / t2;
  return sycl::vec<T, 2>(t1.real(), t1.imag());
}

/// Computes the magnitude of a complex number.
/// \tparam T Complex element type
/// \param [in] x The input complex number
/// \returns The result
template <typename T>
T cabs(sycl::vec<T, 2> x) {
  std::complex<T> t(x[0], x[1]);
  return std::abs(t);
}

/// Computes the complex conjugate of a complex number.
/// \tparam T Complex element type
/// \param [in] x The input complex number
/// \returns The result
template <typename T>
sycl::vec<T, 2> conj(sycl::vec<T, 2> x) {
  std::complex<T> t(x[0], x[1]);
  t = std::conj(t);
  return sycl::vec<T, 2>(t.real(), t.imag());
}

namespace experimental {
/// Synchronize work items from all work groups within a DPC++ kernel.
/// \param [in] item:  Represents a work group.
/// \param [in] counter: An atomic object defined on a device memory which can
/// be accessed by work items in all work groups. The initial value of the
/// counter should be zero.
/// Note: Please make sure that all the work items of all work groups within
/// a DPC++ kernel can be scheduled actively at the same time on a device.
template <int dimensions = 3>
inline void
nd_range_barrier(cl::sycl::nd_item<dimensions> item,
                 cl::sycl::atomic_ref<
                     unsigned int, cl::sycl::memory_order::seq_cst,
                     cl::sycl::memory_scope::device,
                     cl::sycl::access::address_space::global_space> &counter) {

  static_assert(dimensions == 3, "dimensions must be 3.");

  unsigned int num_groups = item.get_group_range(2) * item.get_group_range(1) *
                            item.get_group_range(0);

  item.barrier();

  if (item.get_local_linear_id() == 0) {
    unsigned int inc = 1;
    unsigned int old_arrive = 0;
    bool is_group0 =
        (item.get_group(2) + item.get_group(1) + item.get_group(0) == 0);
    if (is_group0) {
      inc = 0x80000000 - (num_groups - 1);
    }

    old_arrive = counter.fetch_add(inc);
    // Synchronize all the work groups
    while (((old_arrive ^ counter.load()) & 0x80000000) == 0)
      ;
  }

  item.barrier();
}

/// Synchronize work items from all work groups within a DPC++ kernel.
/// \param [in] item:  Represents a work group.
/// \param [in] counter: An atomic object defined on a device memory which can
/// be accessed by work items in all work groups. The initial value of the
/// counter should be zero.
/// Note: Please make sure that all the work items of all work groups within
/// a DPC++ kernel can be scheduled actively at the same time on a device.
template <>
inline void
nd_range_barrier(cl::sycl::nd_item<1> item,
                 cl::sycl::atomic_ref<
                     unsigned int, cl::sycl::memory_order::seq_cst,
                     cl::sycl::memory_scope::device,
                     cl::sycl::access::address_space::global_space> &counter) {
  unsigned int num_groups = item.get_group_range(0);

  item.barrier();

  if (item.get_local_linear_id() == 0) {
    unsigned int inc = 1;
    unsigned int old_arrive = 0;
    bool is_group0 = (item.get_group(0) == 0);
    if (is_group0) {
      inc = 0x80000000 - (num_groups - 1);
    }

    old_arrive = counter.fetch_add(inc);
    // Synchronize all the work groups
    while (((old_arrive ^ counter.load()) & 0x80000000) == 0)
      ;
  }

  item.barrier();
}
}
} // namespace dpct

#endif // __DPCT_UTIL_HPP__
