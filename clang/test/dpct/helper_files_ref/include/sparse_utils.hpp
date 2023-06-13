//==---- sparse_utils.hpp -------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __DPCT_SPARSE_UTILS_HPP__
#define __DPCT_SPARSE_UTILS_HPP__

#include "lib_common_utils.hpp"
#include <oneapi/mkl.hpp>
#include <sycl/sycl.hpp>

namespace dpct {
namespace sparse {
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

/// Computes a CSR format sparse matrix-dense vector product.
/// y = alpha * op(A) * x + beta * y
/// \param [in] queue The queue where the routine should be executed. It must
/// have the in_order property when using the USM mode.
/// \param [in] trans The operation applied to the matrix A.
/// \param [in] num_rows Number of rows of the matrix A.
/// \param [in] num_cols Number of columns of the matrix A.
/// \param [in] alpha Scaling factor for the matrix A.
/// \param [in] info Matrix info of the matrix A.
/// \param [in] val An array containing the non-zero elements of the matrix A.
/// \param [in] row_ptr An array of length \p num_rows + 1.
/// \param [in] col_ind An array containing the column indices in index-based
/// numbering.
/// \param [in] x Data of the vector x.
/// \param [in] beta Scaling factor for the vector x.
/// \param [in, out] y Data of the vector y.
template <typename T>
void csrmv(sycl::queue &queue, oneapi::mkl::transpose trans, int num_rows,
           int num_cols, const T *alpha,
           const std::shared_ptr<matrix_info> info, const T *val,
           const int *row_ptr, const int *col_ind, const T *x, const T *beta,
           T *y) {
#ifndef __INTEL_MKL__
  throw std::runtime_error("The oneAPI Math Kernel Library (oneMKL) Interfaces "
                           "Project does not support this API.");
#else
  using Ty = typename dpct::DataType<T>::T2;
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

  auto data_x = detail::get_memory(reinterpret_cast<Ty *>(const_cast<T *>(x)));
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
                              *sparse_matrix_handle, data_x, beta_value,
                              data_y);
    break;
  }
  case matrix_info::matrix_type::tr: {
    oneapi::mkl::sparse::optimize_trmv(queue, info->get_uplo(), trans,
                                       info->get_diag(), *sparse_matrix_handle);
    oneapi::mkl::sparse::trmv(queue, info->get_uplo(), trans, info->get_diag(),
                              alpha_value, *sparse_matrix_handle, data_x,
                              beta_value, data_y);
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
#endif
}

/// Computes a CSR format sparse matrix-dense matrix product.
/// C = alpha * op(A) * B + beta * C
/// \param [in] queue The queue where the routine should be executed. It must
/// have the in_order property when using the USM mode.
/// \param [in] trans The operation applied to the matrix A.
/// \param [in] sparse_rows Number of rows of the matrix A.
/// \param [in] dense_cols Number of columns of the matrix B or C.
/// \param [in] sparse_cols Number of columns of the matrix A.
/// \param [in] alpha Scaling factor for the matrix A.
/// \param [in] info Matrix info of the matrix A.
/// \param [in] val An array containing the non-zero elements of the matrix A.
/// \param [in] row_ptr An array of length \p num_rows + 1.
/// \param [in] col_ind An array containing the column indices in index-based
/// numbering.
/// \param [in] b Data of the matrix B.
/// \param [in] ldb Leading dimension of the matrix B.
/// \param [in] beta Scaling factor for the matrix B.
/// \param [in, out] c Data of the matrix C.
/// \param [in] ldc Leading dimension of the matrix C.
template <typename T>
void csrmm(sycl::queue &queue, oneapi::mkl::transpose trans, int sparse_rows,
           int dense_cols, int sparse_cols, const T *alpha,
           const std::shared_ptr<matrix_info> info, const T *val,
           const int *row_ptr, const int *col_ind, const T *b, int ldb,
           const T *beta, T *c, int ldc) {
#ifndef __INTEL_MKL__
  throw std::runtime_error("The oneAPI Math Kernel Library (oneMKL) Interfaces "
                           "Project does not support this API.");
#else
  using Ty = typename dpct::DataType<T>::T2;
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
#endif
}

#ifdef __INTEL_MKL__ // The oneMKL Interfaces Project does not support this.
/// Saving the optimization information for solving a system of linear
/// equations.
class optimize_info {
public:
  /// Constructor
  optimize_info() { oneapi::mkl::sparse::init_matrix_handle(&_matrix_handle); }
  /// Destructor
  ~optimize_info() {
    oneapi::mkl::sparse::release_matrix_handle(get_default_queue(),
                                               &_matrix_handle, _deps)
        .wait();
  }
  /// Add dependency for the destructor.
  /// \param [in] e The event which the destructor depends on.
  void add_dependency(sycl::event e) { _deps.push_back(e); }
  /// Get the internal saved matrix handle.
  /// \return Returns the matrix handle.
  oneapi::mkl::sparse::matrix_handle_t get_matrix_handle() const noexcept {
    return _matrix_handle;
  }

private:
  oneapi::mkl::sparse::matrix_handle_t _matrix_handle = nullptr;
  std::vector<sycl::event> _deps;
};
#endif

#ifdef __INTEL_MKL__ // The oneMKL Interfaces Project does not support this.
/// Performs internal optimizations for solving a system of linear equations for
/// a CSR format sparse matrix.
/// \param [in] queue The queue where the routine should be executed. It must
/// have the in_order property when using the USM mode.
/// \param [in] trans The operation applied to the sparse matrix.
/// \param [in] row_col Number of rows of the sparse matrix.
/// \param [in] info Matrix info of the sparse matrix.
/// \param [in] val An array containing the non-zero elements of the sparse matrix.
/// \param [in] row_ptr An array of length \p num_rows + 1.
/// \param [in] col_ind An array containing the column indices in index-based
/// numbering.
/// \param [out] optimize_info The result of the optimizations.
template <typename T>
void optimize_csrsv(sycl::queue &queue, oneapi::mkl::transpose trans,
                    int row_col, const std::shared_ptr<matrix_info> info,
                    const T *val, const int *row_ptr, const int *col_ind,
                    std::shared_ptr<optimize_info> optimize_info) {
  using Ty = typename dpct::DataType<T>::T2;
  auto data_row_ptr = detail::get_memory(const_cast<int *>(row_ptr));
  auto data_col_ind = detail::get_memory(const_cast<int *>(col_ind));
  auto data_val =
      detail::get_memory(reinterpret_cast<Ty *>(const_cast<T *>(val)));
  oneapi::mkl::sparse::set_csr_data(queue, optimize_info->get_matrix_handle(),
                                    row_col, row_col, info->get_index_base(),
                                    data_row_ptr, data_col_ind, data_val);
  if (info->get_matrix_type() != matrix_info::matrix_type::tr)
    return;
#ifndef DPCT_USM_LEVEL_NONE
  sycl::event e;
  e =
#endif
      oneapi::mkl::sparse::optimize_trsv(queue, info->get_uplo(), trans,
                                         info->get_diag(),
                                         optimize_info->get_matrix_handle());
#ifndef DPCT_USM_LEVEL_NONE
  optimize_info->add_dependency(e);
#endif
}
#endif

struct dense_vector_desc {
  dense_vector_desc(std::int64_t ele_num, void *value,
                    library_data_t value_type)
      : _ele_num(ele_num), _value(value), _value_type(value_type) {}
  void get_desc(std::int64_t *ele_num, const void **value,
                library_data_t *value_type) {
    *ele_num = _ele_num;
    *value = _value;
    *value_type = _value_type;
  }
  void get_desc(std::int64_t *ele_num, void **value,
                library_data_t *value_type) {
    get_desc(ele_num, const_cast<const void **>(value), value_type);
  }
  std::int64_t _ele_num;
  void *_value;
  library_data_t _value_type;
};

struct dense_matrix_desc {
  dense_matrix_desc(std::int64_t row_num, std::int64_t col_num,
                    std::int64_t leading_dim, void *value,
                    library_data_t value_type, oneapi::mkl::layout layout)
      : _row_num(row_num), _col_num(col_num), _leading_dim(leading_dim),
        _value(value), _value_type(value_type), _layout(layout) {}
  void get_desc(std::int64_t *row_num, std::int64_t *col_num,
                std::int64_t *leading_dim, void **value,
                library_data_t *value_type, oneapi::mkl::layout *layout) {
    *row_num = _row_num;
    *col_num = _col_num;
    *leading_dim = _leading_dim;
    *value = _value;
    *value_type = _value_type;
    *layout = _layout;
  }
  std::int64_t _row_num;
  std::int64_t _col_num;
  std::int64_t _leading_dim;
  void *_value;
  library_data_t _value_type;
  oneapi::mkl::layout _layout;
};

enum matrix_format : int {
  csr = 1,
};

enum matrix_attribute : int { uplo = 0, diag };

class sparse_matrix_desc;

using sparse_matrix_desc_t = sparse_matrix_desc *;

class sparse_matrix_desc {
public:
  static void create_csr(sparse_matrix_desc_t *desc, std::int64_t row_num,
                         std::int64_t col_num, std::int64_t nnz, void *row_ptr,
                         void *col_ind, void *value,
                         library_data_t row_ptr_type,
                         library_data_t col_ind_type,
                         oneapi::mkl::index_base base,
                         library_data_t value_type) {
    *desc =
        new sparse_matrix_desc(row_num, col_num, nnz, row_ptr, col_ind, value,
                               row_ptr_type, col_ind_type, base, value_type);
    (*desc)->_format = matrix_format::csr;
  }
  static void destroy(sparse_matrix_desc_t desc) {
    desc->~sparse_matrix_desc();
  }

  /// Add dependency for the destructor.
  /// \param [in] e The event which the destructor depends on.
  void add_dependency(sycl::event e) { _deps.push_back(e); }
  /// Get the internal saved matrix handle.
  /// \return Returns the matrix handle.
  oneapi::mkl::sparse::matrix_handle_t get_matrix_handle() const noexcept {
    return _matrix_handle;
  }
  void get_desc(int64_t *row_num, int64_t *col_num, int64_t *nnz,
                void **row_ptr, void **col_ind, void **value,
                library_data_t *row_ptr_type, library_data_t *col_ind_type,
                oneapi::mkl::index_base *base,
                library_data_t *value_type) const noexcept {
    *row_num = _row_num;
    *col_num = _col_num;
    *nnz = _nnz;
    *row_ptr = _row_ptr;
    *col_ind = _col_ind;
    *value = _value;
    *row_ptr_type = _row_ptr_type;
    *col_ind_type = _col_ind_type;
    *base = _base;
    *value_type = _value_type;
  }
  void get_format(matrix_format *format) const noexcept { *format = _format; }
  void get_base(oneapi::mkl::index_base *base) const noexcept { *base = _base; }
  void get_value(void **value) const noexcept { *value = _value; }
  void set_value(void *value) { _value = value; }
  void get_size(int64_t *row_num, int64_t *col_num,
                int64_t *nnz) const noexcept {
    *row_num = _row_num;
    *col_num = _col_num;
    *nnz = _nnz;
  }
  void set_attribute(matrix_attribute attribute, const void *data,
                     size_t data_size) {
    if (attribute == matrix_attribute::diag) {
      const oneapi::mkl::diag *diag_ptr =
          reinterpret_cast<const oneapi::mkl::diag *>(data);
      if (*diag_ptr == oneapi::mkl::diag::unit) {
        _diag = oneapi::mkl::diag::unit;
      } else if (*diag_ptr == oneapi::mkl::diag::nonunit) {
        _diag = oneapi::mkl::diag::nonunit;
      } else {
        throw std::runtime_error("unsupported diag value");
      }
    } else if (attribute == matrix_attribute::uplo) {
      const oneapi::mkl::uplo *uplo_ptr =
          reinterpret_cast<const oneapi::mkl::uplo *>(data);
      if (*uplo_ptr == oneapi::mkl::uplo::upper) {
        _uplo = oneapi::mkl::uplo::upper;
      } else if (*uplo_ptr == oneapi::mkl::uplo::lower) {
        _uplo = oneapi::mkl::uplo::lower;
      } else {
        throw std::runtime_error("unsupported uplo value");
      }
    } else {
      throw std::runtime_error("unsupported attribute");
    }
  }
  void get_attribute(matrix_attribute attribute, void *data, size_t data_size) {
    if (attribute == matrix_attribute::diag) {
      oneapi::mkl::diag *diag_ptr = reinterpret_cast<oneapi::mkl::diag *>(data);
      if (_diag.has_value()) {
        *diag_ptr = _diag.value();
      } else {
        *diag_ptr = oneapi::mkl::diag::nonunit;
      }
    } else if (attribute == matrix_attribute::uplo) {
      oneapi::mkl::uplo *uplo_ptr = reinterpret_cast<oneapi::mkl::uplo *>(data);
      if (_uplo.has_value()) {
        *uplo_ptr = _uplo.value();
      } else {
        *uplo_ptr = oneapi::mkl::uplo::lower;
      }
    } else {
      throw std::runtime_error("unsupported attribute");
    }
  }
  void set_pointers(void *row_ptr, void *col_ind, void *value) {
    _row_ptr = row_ptr;
    _col_ind = col_ind;
    _value = value;
  }

private:
  sparse_matrix_desc(std::int64_t row_num, std::int64_t col_num,
                     std::int64_t nnz, void *row_ptr, void *col_ind,
                     void *value, library_data_t row_ptr_type,
                     library_data_t col_ind_type, oneapi::mkl::index_base base,
                     library_data_t value_type)
      : _row_num(row_num), _col_num(col_num), _nnz(nnz), _row_ptr(row_ptr),
        _col_ind(col_ind), _value(value), _row_ptr_type(row_ptr_type),
        _col_ind_type(col_ind_type), _base(base), _value_type(value_type) {
    oneapi::mkl::sparse::init_matrix_handle(&_matrix_handle);
#define SET_DATA(INDEX_TYPE, VALUE_TYPE)                                       \
  do {                                                                         \
    auto data_row_ptr =                                                        \
        detail::get_memory(reinterpret_cast<INDEX_TYPE *>(_row_ptr));          \
    auto data_col_ind =                                                        \
        detail::get_memory(reinterpret_cast<INDEX_TYPE *>(_col_ind));          \
    auto data_value =                                                          \
        detail::get_memory(reinterpret_cast<VALUE_TYPE *>(_value));            \
    oneapi::mkl::sparse::set_csr_data(get_default_queue(), _matrix_handle,     \
                                      _row_num, _col_num, _base, data_row_ptr, \
                                      data_col_ind, data_value)                \
        .wait();                                                               \
  } while (0)
    std::uint64_t key = detail::get_type_combination_id(
        _row_ptr_type, _col_ind_type, _value_type);
    switch (key) {
    case detail::get_type_combination_id(library_data_t::real_int32,
                                         library_data_t::real_int32,
                                         library_data_t::real_float): {
      SET_DATA(std::int32_t, float);
      break;
    }
    case detail::get_type_combination_id(library_data_t::real_int32,
                                         library_data_t::real_int32,
                                         library_data_t::real_double): {
      SET_DATA(std::int32_t, double);
      break;
    }
    case detail::get_type_combination_id(library_data_t::real_int32,
                                         library_data_t::real_int32,
                                         library_data_t::complex_float): {
      SET_DATA(std::int32_t, std::complex<float>);
      break;
    }
    case detail::get_type_combination_id(library_data_t::real_int32,
                                         library_data_t::real_int32,
                                         library_data_t::complex_double): {
      SET_DATA(std::int32_t, std::complex<double>);
      break;
    }
    case detail::get_type_combination_id(library_data_t::real_int64,
                                         library_data_t::real_int64,
                                         library_data_t::real_float): {
      SET_DATA(std::int64_t, float);
      break;
    }
    case detail::get_type_combination_id(library_data_t::real_int64,
                                         library_data_t::real_int64,
                                         library_data_t::real_double): {
      SET_DATA(std::int64_t, double);
      break;
    }
    case detail::get_type_combination_id(library_data_t::real_int64,
                                         library_data_t::real_int64,
                                         library_data_t::complex_float): {
      SET_DATA(std::int64_t, std::complex<float>);
      break;
    }
    case detail::get_type_combination_id(library_data_t::real_int64,
                                         library_data_t::real_int64,
                                         library_data_t::complex_double): {
      SET_DATA(std::int64_t, std::complex<double>);
      break;
    }
    default:
      throw std::runtime_error("the combination of data type is unsupported");
    }
#undef SET_DATA
  }
  /// Destructor
  ~sparse_matrix_desc() {
    oneapi::mkl::sparse::release_matrix_handle(get_default_queue(),
                                               &_matrix_handle, _deps)
        .wait();
  }

  std::int64_t _row_num;
  std::int64_t _col_num;
  std::int64_t _nnz;
  void *_row_ptr;
  void *_col_ind;
  void *_value;
  library_data_t _row_ptr_type;
  library_data_t _col_ind_type;
  oneapi::mkl::index_base _base;
  library_data_t _value_type;
  oneapi::mkl::sparse::matrix_handle_t _matrix_handle = nullptr;
  std::vector<sycl::event> _deps;
  matrix_format _format;
  std::optional<oneapi::mkl::diag> _diag;
  std::optional<oneapi::mkl::uplo> _uplo;
};

void spmv(sycl::queue queue, oneapi::mkl::transpose trans, const void *alpha,
          sparse_matrix_desc_t a, std::shared_ptr<dense_vector_desc> x,
          const void *beta, std::shared_ptr<dense_vector_desc> y,
          library_data_t compute_type) {
#define SPMV(Ty)                                                               \
  do {                                                                         \
    auto alpha_value =                                                         \
        detail::get_value(reinterpret_cast<const Ty *>(alpha), queue);         \
    auto beta_value =                                                          \
        detail::get_value(reinterpret_cast<const Ty *>(beta), queue);          \
    oneapi::mkl::sparse::optimize_gemv(queue, trans, a->get_matrix_handle());  \
    auto data_x = detail::get_memory(reinterpret_cast<Ty *>(x->_value));       \
    auto data_y = detail::get_memory(reinterpret_cast<Ty *>(y->_value));       \
    oneapi::mkl::sparse::gemv(queue, trans, alpha_value,                       \
                              a->get_matrix_handle(), data_x, beta_value,      \
                              data_y);                                         \
  } while (0)

  switch (compute_type) {
  case library_data_t::real_int32: {
    SPMV(float);
    break;
  }
  case library_data_t::real_double: {
    SPMV(double);
    break;
  }
  case library_data_t::complex_float: {
    SPMV(std::complex<float>);
    break;
  }
  case library_data_t::complex_double: {
    SPMV(std::complex<double>);
    break;
  }
  default:
    throw std::runtime_error("the combination of data type is unsupported");
  }
#undef SPMV
}

void spmm(sycl::queue queue, oneapi::mkl::transpose trans_a,
          oneapi::mkl::transpose trans_b, const void *alpha,
          sparse_matrix_desc_t a, std::shared_ptr<dense_matrix_desc> b,
          const void *beta, std::shared_ptr<dense_matrix_desc> c,
          library_data_t compute_type) {
  if (b->_layout != c->_layout)
    throw std::runtime_error("the layout of b and c are different");
#define SPMM(Ty)                                                               \
  do {                                                                         \
    auto alpha_value =                                                         \
        detail::get_value(reinterpret_cast<const Ty *>(alpha), queue);         \
    auto beta_value =                                                          \
        detail::get_value(reinterpret_cast<const Ty *>(beta), queue);          \
    auto data_b = detail::get_memory(reinterpret_cast<Ty *>(b->_value));       \
    auto data_c = detail::get_memory(reinterpret_cast<Ty *>(c->_value));       \
    if (b->_layout == oneapi::mkl::layout::row_major)                          \
      oneapi::mkl::sparse::gemm(                                               \
          queue, oneapi::mkl::layout::row_major, trans_a, trans_b,             \
          alpha_value, a->get_matrix_handle(), data_b, b->_col_num,            \
          b->_leading_dim, beta_value, data_c, c->_leading_dim);               \
    else                                                                       \
      oneapi::mkl::sparse::gemm(                                               \
          queue, oneapi::mkl::layout::col_major, trans_a, trans_b,             \
          alpha_value, a->get_matrix_handle(), data_b, b->_col_num,            \
          b->_leading_dim, beta_value, data_c, c->_leading_dim);               \
  } while (0)

  switch (compute_type) {
  case library_data_t::real_int32: {
    SPMM(float);
    break;
  }
  case library_data_t::real_double: {
    SPMM(double);
    break;
  }
  case library_data_t::complex_float: {
    SPMM(std::complex<float>);
    break;
  }
  case library_data_t::complex_double: {
    SPMM(std::complex<double>);
    break;
  }
  default:
    throw std::runtime_error("the combination of data type is unsupported");
  }
#undef SPMM
}

} // namespace sparse
} // namespace dpct

#endif // __DPCT_SPARSE_UTILS_HPP__
