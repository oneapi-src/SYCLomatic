//==---- mma_utils.hpp ----------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __DPCT_MMA_UTILS_HPP__
#define __DPCT_MMA_UTILS_HPP__

#include <sycl/sycl.hpp>

namespace dpct {
namespace experimental {
namespace matrix {

namespace sycl_matrix = sycl::ext::oneapi::experimental::matrix;

struct row_major
    : public std::integral_constant<sycl_matrix::layout,
                                    sycl_matrix::layout::row_major> {};
struct col_major
    : public std::integral_constant<sycl_matrix::layout,
                                    sycl_matrix::layout::col_major> {};
struct a
    : public std::integral_constant<sycl_matrix::use, sycl_matrix::use::a> {};
struct b
    : public std::integral_constant<sycl_matrix::use, sycl_matrix::use::b> {};
struct accumulator
    : public std::integral_constant<sycl_matrix::use,
                                    sycl_matrix::use::accumulator> {};
enum layout_t { m_row_major, m_col_major };

template <typename use, int m, int n, int k, typename T,
          typename layout = std::integral_constant<
              sycl_matrix::layout, sycl_matrix::layout::dynamic>>
class joint_matrix {
  static constexpr int rows = (use::value == sycl_matrix::use::b) ? n : m;
  static constexpr int cols =
      (use::value == sycl_matrix::use::accumulator) ? m : k;
  using joint_matrix_type =
      sycl_matrix::joint_matrix<sycl::sub_group, T, use::value, rows, cols,
                                layout::value>;

public:
  joint_matrix()
      : matrix(), g(sycl::ext::oneapi::experimental::this_sub_group()) {}
  joint_matrix(joint_matrix &other) {
    sycl_matrix::joint_matrix_copy(g, other.get(), matrix);
  }
  joint_matrix &operator=(joint_matrix &other) {
    if (this != &other) {
      sycl_matrix::joint_matrix_copy(g, other.get(), matrix);
    }
    return *this;
  }

  joint_matrix_type &get() { return matrix; }

  const joint_matrix_type &get() const { return matrix; }

private:
  sycl::sub_group g;
  joint_matrix_type matrix;
};

template <typename MT, typename T>
void joint_matrix_load(sycl::sub_group g, MT &res, const T *src,
                       unsigned stride) {
  sycl_matrix::joint_matrix_load(
      g, res.get(),
      sycl::address_space_cast<sycl::access::address_space::generic_space,
                               sycl::access::decorated::no, const T>(src),
      stride);
}

template <typename MT, typename T>
void joint_matrix_load(sycl::sub_group g, MT &res, const T *src,
                       unsigned stride, layout_t layout) {
  sycl_matrix::joint_matrix_load(
      g, res.get(),
      sycl::address_space_cast<sycl::access::address_space::generic_space,
                               sycl::access::decorated::no, const T>(src),
      stride,
      layout == layout_t::m_row_major ? sycl_matrix::layout::row_major
                                      : sycl_matrix::layout::col_major);
}

template <typename MT, typename T>
void joint_matrix_store(sycl::sub_group g, T *dest, const MT &res,
                        unsigned stride, layout_t layout) {
  sycl_matrix::joint_matrix_store(
      g, res.get(),
      sycl::address_space_cast<sycl::access::address_space::generic_space,
                               sycl::access::decorated::no, T>(dest),
      stride,
      layout == layout_t::m_row_major ? sycl_matrix::layout::row_major
                                      : sycl_matrix::layout::col_major);
}

template <typename MT, typename T>
void joint_matrix_fill(sycl::sub_group g, MT &m, const T &v) {
  sycl_matrix::joint_matrix_fill(g, m.get(), v);
}

template <typename Td, typename Ta, typename Tb, typename Tc>
void joint_matrix_mad(sycl::sub_group g, Td &d, const Ta &a, const Tb &b,
                      const Tc &c) {
  sycl_matrix::joint_matrix_mad(g, d.get(), a.get(), b.get(), c.get());
};

} // namespace matrix
} // namespace experimental
} // namespace dpct

#endif // __DPCT_MMA_UTILS_HPP__
