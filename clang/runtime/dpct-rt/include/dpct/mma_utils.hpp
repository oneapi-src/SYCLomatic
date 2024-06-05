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
struct row_major;
struct col_major;
struct a;
struct b;
struct accumulator;
enum layout_t { m_row_major, m_col_major };

namespace sycl_matrix = sycl::ext::oneapi::experimental::matrix;

template <typename use, int m, int n, int k, typename T, typename layout = void>
class joint_matrix {
  template <sycl_matrix::use v>
  using UseEnum = std::integral_constant<sycl_matrix::use, v>;
  template <sycl_matrix::layout v>
  using LayoutEnum = std::integral_constant<sycl_matrix::layout, v>;
  using rows =
      std::conditional_t<std::is_same_v<use, b>, std::integral_constant<int, n>,
                         std::integral_constant<int, m>>;
  using cols = std::conditional_t<std::is_same_v<use, accumulator>,
                                  std::integral_constant<int, m>,
                                  std::integral_constant<int, k>>;
  using joint_matrix_sycl_use = std::conditional_t<
      std::is_same_v<use, a>, UseEnum<sycl_matrix::use::a>,
      std::conditional_t<std::is_same_v<use, b>, UseEnum<sycl_matrix::use::b>,
                         UseEnum<sycl_matrix::use::accumulator>>>;
  using joint_matrix_sycl_layout = std::conditional_t<
      std::conjunction_v<std::is_same<use, accumulator>,
                         std::is_same<layout, void>>,
      LayoutEnum<sycl_matrix::layout::dynamic>,
      std::conditional_t<std::is_same_v<layout, row_major>,
                         LayoutEnum<sycl_matrix::layout::row_major>,
                         LayoutEnum<sycl_matrix::layout::col_major>>>;
  using JointMatrixType =
      sycl_matrix::joint_matrix<sycl::sub_group, T,
                                joint_matrix_sycl_use::value, rows::value,
                                cols::value, joint_matrix_sycl_layout::value>;

public:
  joint_matrix() : matrix() {}
  joint_matrix(joint_matrix &other) {
    sycl_matrix::joint_matrix_copy(
        sycl::ext::oneapi::experimental::this_sub_group(), other.get(), matrix);
  }
  joint_matrix &operator=(joint_matrix &other) {
    if (this != &other) {
      sycl_matrix::joint_matrix_copy(
          sycl::ext::oneapi::experimental::this_sub_group(), other.get(),
          matrix);
    }
    return *this;
  }

  JointMatrixType &get() { return matrix; }

  const JointMatrixType &get() const { return matrix; }

private:
  JointMatrixType matrix;
};

template <typename MT, typename T>
void joint_matrix_load(MT &res, const T *src, unsigned stride) {
  sycl_matrix::joint_matrix_load(
      sycl::ext::oneapi::experimental::this_sub_group(), res.get(),
      sycl::address_space_cast<sycl::access::address_space::generic_space,
                               sycl::access::decorated::no, const T>(src),
      stride);
}

template <typename MT, typename T>
void joint_matrix_load(MT &res, const T *src, unsigned stride,
                       layout_t layout) {
  sycl_matrix::joint_matrix_load(
      sycl::ext::oneapi::experimental::this_sub_group(), res.get(),
      sycl::address_space_cast<sycl::access::address_space::generic_space,
                               sycl::access::decorated::no, const T>(src),
      stride,
      layout == layout_t::m_row_major ? sycl_matrix::layout::row_major
                                      : sycl_matrix::layout::col_major);
}

template <typename MT, typename T>
void joint_matrix_store(T *dest, const MT &res, unsigned stride,
                        layout_t layout) {
  sycl_matrix::joint_matrix_store(
      sycl::ext::oneapi::experimental::this_sub_group(), res.get(),
      sycl::address_space_cast<sycl::access::address_space::generic_space,
                               sycl::access::decorated::no, T>(dest),
      stride,
      layout == layout_t::m_row_major ? sycl_matrix::layout::row_major
                                      : sycl_matrix::layout::col_major);
}

template <typename MT, typename T> void joint_matrix_fill(MT &m, const T &v) {
  sycl_matrix::joint_matrix_fill(
      sycl::ext::oneapi::experimental::this_sub_group(), m.get(), v);
}

template <typename Td, typename Ta, typename Tb, typename Tc>
void joint_matrix_mad(Td &d, const Ta &a, const Tb &b, const Tc &c) {
  sycl_matrix::joint_matrix_mad(
      sycl::ext::oneapi::experimental::this_sub_group(), d.get(), a.get(),
      b.get(), c.get());
};

} // namespace matrix
} // namespace experimental
} // namespace dpct

#endif // __DPCT_MMA_UTILS_HPP__
