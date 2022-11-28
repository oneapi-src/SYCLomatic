//==---- sparse_utils.hpp -------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __DPCT_SPARSE_UTILS_HPP__
#define __DPCT_SPARSE_UTILS_HPP__

#include <oneapi/mkl.hpp>
#include <sycl/sycl.hpp>

namespace dpct {
namespace sparse {
// Describes some properties of a sparse matrix.
// The properties are matrix type, diag, uplo and index base.
class sparse_matrix_info {
public:
  // Matrix types are:
  // ge: General matrix
  // sy: Symmetric matrix
  // he: Hermitian matrix
  // tr: Triangular matrix
  enum class matrix_type { ge, sy, he, tr };

  template <typename T> auto get() { return _mat_type; }
  template <> auto get<oneapi::mkl::diag>() { return _unit_diag; }
  template <> auto get<oneapi::mkl::uplo>() { return _upper_lower; }
  template <> auto get<oneapi::mkl::index_base>() { return _index; }
  void set(matrix_type mat_type) { _mat_type = mat_type; }
  void set(oneapi::mkl::diag unit_diag) { _unit_diag = unit_diag; }
  void set(oneapi::mkl::uplo upper_lower) { _upper_lower = upper_lower; }
  void set(oneapi::mkl::index_base index) { _index = index; }

private:
  matrix_type _mat_type = matrix_type::ge;
  oneapi::mkl::diag _unit_diag = oneapi::mkl::diag::nonunit;
  oneapi::mkl::uplo _upper_lower = oneapi::mkl::uplo::upper;
  oneapi::mkl::index_base _index = oneapi::mkl::index_base::zero;
};
} // namespace sparse
} // namespace dpct

#endif // __DPCT_SPARSE_UTILS_HPP__
