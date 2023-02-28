// DPCT_LABEL_BEGIN|License|
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
//==---- sparse_utils.hpp -------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
// DPCT_LABEL_END

#ifndef __DPCT_SPARSE_UTILS_HPP__
#define __DPCT_SPARSE_UTILS_HPP__
// DPCT_COMMENT
// DPCT_COMMENT Example1:
// DPCT_COMMENT // DPCT_LABEL_BEGIN|FeatureNameDef|[Namespace]
// DPCT_COMMENT // DPCT_DEPENDENCY_EMPTY
// DPCT_COMMENT // DPCT_CODE
// DPCT_COMMENT some code
// DPCT_COMMENT // DPCT_LABEL_END
// DPCT_COMMENT
// DPCT_COMMENT Example2:
// DPCT_COMMENT // DPCT_LABEL_BEGIN|FeatureNameDef|[Namespace]
// DPCT_COMMENT // DPCT_DEPENDENCY_BEGIN
// DPCT_COMMENT // FileID|FeatureNameRef
// DPCT_COMMENT [// FileID|FeatureNameRef]
// DPCT_COMMENT ...
// DPCT_COMMENT // DPCT_DEPENDENCY_END
// DPCT_COMMENT // DPCT_CODE
// DPCT_COMMENT some code
// DPCT_COMMENT // DPCT_LABEL_END
// DPCT_COMMENT
// DPCT_COMMENT For header file including dependency, please use predefined feature name:
// DPCT_COMMENT   local_include_dependency: dpct helper files
// DPCT_COMMENT   non_local_include_dependency: other header files

// DPCT_LABEL_BEGIN|local_include_dependency|
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
#include "lib_common_utils.hpp"
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|non_local_include_dependency|
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
#include <oneapi/mkl.hpp>
#include <sycl/sycl.hpp>
// DPCT_LABEL_END

namespace dpct {
namespace sparse {
// DPCT_LABEL_BEGIN|matrix_info|dpct::sparse
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
/// Describes properties of a sparse matrix.
/// The properties are matrix type, diag, uplo and index base.
class matrix_info {
public:
  /// Matrix types are:
  /// ge: General matrix
  /// sy: Symmetric matrix
  /// he: Hermitian matrix
  /// tr: Triangular matrix
  enum class matrix_type : int { ge = 0, sy, he, tr };

  auto get_matrix_type() const { return _matrix_type; }
  auto get_diag() const { return _diag; }
  auto get_uplo() const { return _uplo; }
  auto get_index_base() const { return _index_base; }
  void set_matrix_type(matrix_type mt) { _matrix_type = mt; }
  void set_diag(oneapi::mkl::diag d) { _diag = d; }
  void set_uplo(oneapi::mkl::uplo u) { _uplo = u; }
  void set_index_base(oneapi::mkl::index_base ib) { _index_base = ib; }

private:
  matrix_type _matrix_type = matrix_type::ge;
  oneapi::mkl::diag _diag = oneapi::mkl::diag::nonunit;
  oneapi::mkl::uplo _uplo = oneapi::mkl::uplo::upper;
  oneapi::mkl::index_base _index_base = oneapi::mkl::index_base::zero;
};
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|csrmv|dpct::sparse
// DPCT_DEPENDENCY_BEGIN
// DPCT_DEPENDENCY_END
// DPCT_CODE
template <typename T>
void csrmv(sycl::queue &queue, oneapi::mkl::transpose trans, int num_rows,
           int num_cols, const T *alpha,
           const std::shared_ptr<matrix_info> info, const T *val,
           const int *row_ptr, const int *col_ind, const T *x, const T *beta,
           T *y) {
  using Ty = typename dpct::DataType<T>::T2;
  static_assert(
      std::disjunction_v<std::is_same<Ty, std::complex<float>>,
                         std::is_same<Ty, std::complex<double>>>,
      "Unimplemented functionality: oneapi::mkl::sparse::optimize_gemv/gemv, "
      "oneapi::mkl::sparse::symv and oneapi::mkl::sparse::optimize_trmv/trmv "
      "currently do not support complex data types");
  auto alpha_value =
      detail::get_value(reinterpret_cast<const Ty *>(alpha), queue);
  auto beta_value =
      detail::get_value(reinterpret_cast<const Ty *>(beta), queue);

  oneapi::mkl::sparse::matrix_handle_t *sparse_matrix_handle =
      new oneapi::mkl::sparse::matrix_handle_t;
  oneapi::mkl::sparse::init_matrix_handle(sparse_matrix_handle);
  auto data_row_ptr = detail::get_memory(const_cast<int *>(row_ptr));
  auto data_col_ind = detail::get_memory(const_cast<int *>(col_ind));
  auto data_val =
      detail::get_memory(reinterpret_cast<Ty *>(const_cast<T *>(val)));
  oneapi::mkl::sparse::set_csr_data(queue, *sparse_matrix_handle, num_rows,
                                    num_cols, info->get_index_base(),
                                    data_row_ptr, data_col_ind, data_val);

  auto data_x = detail::get_memory(reinterpret_cast<const Ty *>(x));
  auto data_y = detail::get_memory(reinterpret_cast<Ty *>(y));
  switch (info->get_matrix_type()) {
  case matrix_info::matrix_type::ge: {
    oneapi::mkl::sparse::optimize_gemv(queue, trans, *sparse_matrix_handle);
    oneapi::mkl::sparse::gemv(queue, trans, alpha_value, *sparse_matrix_handle,
                              data_x, beta_value, data_y);
    break;
  }
  case matrix_info::matrix_type::sy: {
    oneapi::mkl::sparse::symv(queue, info->get_uplo(), alpha_value,
                              *sparse_matrix_handle, const_cast<Ty *>(data_x),
                              beta_value, data_y);
    break;
  }
  case matrix_info::matrix_type::tr: {
    oneapi::mkl::sparse::optimize_trmv(queue, info->get_uplo(), trans,
                                       info->get_diag(), *sparse_matrix_handle);
    oneapi::mkl::sparse::trmv(queue, info->get_uplo(), trans, info->get_diag(),
                              alpha_value, *sparse_matrix_handle,
                              const_cast<Ty *>(data_x), beta_value, data_y);
    break;
  }
  default:
    throw std::runtime_error(
        "the spmv does not support matrix_info::matrix_type::he");
  }

  sycl::event e =
      oneapi::mkl::sparse::release_matrix_handle(queue, sparse_matrix_handle);
  queue.submit([&](sycl::handler &cgh) {
    cgh.depends_on(e);
    cgh.host_task([=] { delete sparse_matrix_handle; });
  });
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|csrmm|dpct::sparse
// DPCT_DEPENDENCY_BEGIN
// DPCT_DEPENDENCY_END
// DPCT_CODE
template <typename T>
void csrmm(sycl::queue &queue, oneapi::mkl::transpose trans, int sparse_rows,
           int dense_cols, int sparse_cols, const T *alpha,
           const std::shared_ptr<matrix_info> info, const T *val,
           const int *row_ptr, const int *col_ind, T *b, int ldb, const T *beta,
           T *c, int ldc) {
  using Ty = typename dpct::DataType<T>::T2;
  static_assert(std::disjunction_v<std::is_same<Ty, std::complex<float>>,
                                   std::is_same<Ty, std::complex<double>>>,
                "Unimplemented functionality: oneapi::mkl::sparse::gemm "
                "currently does not support complex data types");
  auto alpha_value =
      detail::get_value(reinterpret_cast<const Ty *>(alpha), queue);
  auto beta_value =
      detail::get_value(reinterpret_cast<const Ty *>(beta), queue);

  oneapi::mkl::sparse::matrix_handle_t *sparse_matrix_handle =
      new oneapi::mkl::sparse::matrix_handle_t;
  oneapi::mkl::sparse::init_matrix_handle(sparse_matrix_handle);
  auto data_row_ptr = detail::get_memory(const_cast<int *>(row_ptr));
  auto data_col_ind = detail::get_memory(const_cast<int *>(col_ind));
  auto data_val =
      detail::get_memory(reinterpret_cast<Ty *>(const_cast<T *>(val)));
  oneapi::mkl::sparse::set_csr_data(queue, *sparse_matrix_handle, sparse_rows,
                                    sparse_cols, info->get_index_base(),
                                    data_row_ptr, data_col_ind, data_val);

  auto data_b = detail::get_memory(reinterpret_cast<Ty *>(const_cast<T *>(b)));
  auto data_c = detail::get_memory(reinterpret_cast<Ty *>(c));
  switch (info->get_matrix_type()) {
  case matrix_info::matrix_type::ge: {
    oneapi::mkl::sparse::gemm(queue, oneapi::mkl::layout::row_major, trans,
                              oneapi::mkl::transpose::nontrans, alpha_value,
                              *sparse_matrix_handle, data_b, dense_cols, ldb,
                              beta_value, data_c, ldc);
    break;
  }
  default:
    throw std::runtime_error(
        "the csrmm does not support matrix_info::matrix_type::sy, "
        "matrix_info::matrix_type::tr amd matrix_info::matrix_type::he");
  }

  sycl::event e =
      oneapi::mkl::sparse::release_matrix_handle(queue, sparse_matrix_handle);
  queue.submit([&](sycl::handler &cgh) {
    cgh.depends_on(e);
    cgh.host_task([=] { delete sparse_matrix_handle; });
  });
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|optimize_info|dpct::sparse
// DPCT_DEPENDENCY_BEGIN
// DPCT_DEPENDENCY_END
// DPCT_CODE
class optimize_info {
public:
  optimize_info() { oneapi::mkl::sparse::init_matrix_handle(&_matrix_handle); }
  ~optimize_info() {
    oneapi::mkl::sparse::release_matrix_handle(get_default_queue(),
                                               &_matrix_handle, _deps);
  }
  void add_dependency(sycl::event e) { _deps.push_back(e); }
  oneapi::mkl::sparse::matrix_handle_t get_matrix_handle() {
    return _matrix_handle;
  }

private:
  oneapi::mkl::sparse::matrix_handle_t _matrix_handle = nullptr;
  std::vector<sycl::event> _deps;
};
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|optimize_csrsv|dpct::sparse
// DPCT_DEPENDENCY_BEGIN
// DPCT_DEPENDENCY_END
// DPCT_CODE
template <typename T>
void optimize_csrsv(sycl::queue &queue, oneapi::mkl::transpose trans,
                    int row_col, const std::shared_ptr<matrix_info> info,
                    const T *val, const int *row_ptr, const int *col_ind,
                    std::shared_ptr<optimize_info> optimize_info_ptr) {
  using Ty = typename dpct::DataType<T>::T2;
  static_assert(
      std::disjunction_v<std::is_same<Ty, std::complex<float>>,
                         std::is_same<Ty, std::complex<double>>>,
      "Unimplemented functionality: oneapi::mkl::sparse::optimize_trsv "
      "currently does not support complex data types");
  auto data_row_ptr = detail::get_memory(const_cast<int *>(row_ptr));
  auto data_col_ind = detail::get_memory(const_cast<int *>(col_ind));
  auto data_val =
      detail::get_memory(reinterpret_cast<Ty *>(const_cast<T *>(val)));
  oneapi::mkl::sparse::set_csr_data(
      queue, optimize_info_ptr->get_matrix_handle(), row_col, row_col,
      info->get_index_base(), data_row_ptr, data_col_ind, data_val);
  sycl::event e;
#ifndef DPCT_USM_LEVEL_NONE
  e =
#endif
      oneapi::mkl::sparse::optimize_trsv(
          queue, info->get_uplo(), trans, info->get_diag(),
          optimize_info_ptr->get_matrix_handle());
  optimize_info_ptr->add_dependency(e);
}
// DPCT_LABEL_END

} // namespace sparse
} // namespace dpct

#endif // __DPCT_SPARSE_UTILS_HPP__
