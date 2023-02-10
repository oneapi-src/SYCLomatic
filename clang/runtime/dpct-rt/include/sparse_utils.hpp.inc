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

// DPCT_LABEL_BEGIN|dense_vector_handle_t|dpct::sparse
// DPCT_DEPENDENCY_BEGIN
// LibCommonUtils|library_data_t
// DPCT_DEPENDENCY_END
// DPCT_CODE
using dense_vector_handle_t =
    std::shared_ptr<std::tuple<std::int64_t, void *, library_data_t>>;
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|create_dense_vector_handle|dpct::sparse
// DPCT_DEPENDENCY_BEGIN
// SparseUtils|dense_vector_handle_t
// DPCT_DEPENDENCY_END
// DPCT_CODE
dense_vector_handle_t create_dense_vector_handle(std::int64_t n, void *ptr,
                                                 library_data_t type) {
  return std::make_shared<
      std::tuple<std::int64_t, void *, dpct::library_data_t>>(n, ptr, type);
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|destroy_dense_vector_handle|dpct::sparse
// DPCT_DEPENDENCY_BEGIN
// SparseUtils|dense_vector_handle_t
// DPCT_DEPENDENCY_END
// DPCT_CODE
void destroy_dense_vector_handle(dense_vector_handle_t handle) {
  handle = nullptr;
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|sparse_matrix_handle_t|dpct::sparse
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
using sparse_matrix_handle_t = oneapi::mkl::sparse::matrix_handle_t;
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|destroy_dense_vector_handle|dpct::sparse
// DPCT_DEPENDENCY_BEGIN
// SparseUtils|sparse_matrix_handle_t
// LibCommonUtils|library_data_t
// LibCommonUtils|get_type_combination_id_Ts
// LibCommonUtils|get_memory
// DPCT_DEPENDENCY_END
// DPCT_CODE
void create_csr_matrix_handle(sycl::queue &queue,
                              sparse_matrix_handle_t &mat_handle,
                              std::int64_t num_rows, std::int64_t num_cols,
                              const void *row_ptr, const void *col_ind,
                              const void *val, library_data_t row_ptr_type,
                              library_data_t col_ind_type,
                              oneapi::mkl::index_base index,
                              library_data_t value_type) {
  oneapi::mkl::sparse::init_matrix_handle(&mat_handle);
  std::uint64_t key =
      detail::get_type_combination_id(row_ptr_type, col_ind_type, value_type);

#define CASE(row_ptr_type_enum, col_ind_type_enum, value_type_enum,            \
             row_ptr_type, col_ind_type, value_type)                           \
  case detail::get_type_combination_id(row_ptr_type_enum, col_ind_type_enum,   \
                                       value_type_enum): {                     \
    auto data_row_ptr = detail::get_memory((row_ptr_type *)row_ptr);           \
    auto data_col_ind = detail::get_memory((col_ind_type *)col_ind);           \
    auto data_val = detail::get_memory((value_type *)val);                     \
    oneapi::mkl::sparse::set_csr_data(queue, mat_handle, num_rows, num_cols,   \
                                      index, data_row_ptr, data_col_ind,       \
                                      data_val);                               \
    break;                                                                     \
  }
  switch (key) {
    CASE(library_data_t::real_int32, library_data_t::real_int32,
         library_data_t::real_float, std::int32_t, std::int32_t, float)
    CASE(library_data_t::real_int32, library_data_t::real_int32,
         library_data_t::real_double, std::int32_t, std::int32_t, double)
    CASE(library_data_t::real_int32, library_data_t::real_int32,
         library_data_t::complex_float, std::int32_t, std::int32_t,
         std::complex<float>)
    CASE(library_data_t::real_int32, library_data_t::real_int32,
         library_data_t::complex_double, std::int32_t, std::int32_t,
         std::complex<double>)
    CASE(library_data_t::real_int64, library_data_t::real_int64,
         library_data_t::real_float, std::int64_t, std::int64_t, float)
    CASE(library_data_t::real_int64, library_data_t::real_int64,
         library_data_t::real_double, std::int64_t, std::int64_t, double)
    CASE(library_data_t::real_int64, library_data_t::real_int64,
         library_data_t::complex_float, std::int64_t, std::int64_t,
         std::complex<float>)
    CASE(library_data_t::real_int64, library_data_t::real_int64,
         library_data_t::complex_double, std::int64_t, std::int64_t,
         std::complex<double>)
  default:
    throw std::runtime_error("the combination of data type is unsupported");
  }
#undef CASE
#ifndef DPCT_USM_LEVEL_NONE
  queue.wait();
#endif
}
// DPCT_LABEL_END

namespace detail {
// DPCT_LABEL_BEGIN|destroy_csr_matrix_handle_async|dpct::sparse::detail
// DPCT_DEPENDENCY_BEGIN
// SparseUtils|sparse_matrix_handle_t
// DPCT_DEPENDENCY_END
// DPCT_CODE
sycl::event
destroy_csr_matrix_handle_async(sycl::queue &queue,
                                sparse_matrix_handle_t &mat_handle) {
  return oneapi::mkl::sparse::release_matrix_handle(queue, &mat_handle);
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|get_type_as_library_data|dpct::sparse::detail
// DPCT_DEPENDENCY_BEGIN
// LibCommonUtils|library_data_t
// DPCT_DEPENDENCY_END
// DPCT_CODE
template <typename T>
inline constexpr library_data_t get_type_as_library_data(
    typename std::enable_if<std::is_same<T, float>::value>::type * = 0) {
  return library_data_t::real_float;
}
template <typename T>
inline constexpr library_data_t get_type_as_library_data(
    typename std::enable_if<std::is_same<T, double>::value>::type * = 0) {
  return library_data_t::real_double;
}
template <typename T>
inline constexpr library_data_t get_type_as_library_data(
    typename std::enable_if<std::is_same<T, sycl::float2>::value>::type * = 0) {
  return library_data_t::complex_float;
}
template <typename T>
inline constexpr library_data_t get_type_as_library_data(
    typename std::enable_if<std::is_same<T, sycl::double2>::value>::type * =
        0) {
  return library_data_t::complex_double;
}
template <typename T>
inline constexpr library_data_t get_type_as_library_data(
    typename std::enable_if<std::is_same<T, std::complex<float>>::value>::type
        * = 0) {
  return library_data_t::complex_float;
}
template <typename T>
inline constexpr library_data_t get_type_as_library_data(
    typename std::enable_if<std::is_same<T, std::complex<double>>::value>::type
        * = 0) {
  return library_data_t::complex_double;
}
template <typename T>
inline constexpr library_data_t get_type_as_library_data(
    typename std::enable_if<std::is_same<T, std::int32_t>::value>::type * = 0) {
  return library_data_t::real_int32;
}
template <typename T>
inline constexpr library_data_t get_type_as_library_data(
    typename std::enable_if<std::is_same<T, std::int64_t>::value>::type * = 0) {
  return library_data_t::real_int64;
}
// DPCT_LABEL_END
} // namespace detail

// DPCT_LABEL_BEGIN|destroy_csr_matrix_handle|dpct::sparse
// DPCT_DEPENDENCY_BEGIN
// SparseUtils|sparse_matrix_handle_t
// SparseUtils|destroy_csr_matrix_handle_async
// DPCT_DEPENDENCY_END
// DPCT_CODE
void destroy_csr_matrix_handle(sycl::queue &queue,
                               sparse_matrix_handle_t &mat_handle) {
  detail::destroy_csr_matrix_handle_async(queue, mat_handle).wait();
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|spmv|dpct::sparse
// DPCT_DEPENDENCY_BEGIN
// SparseUtils|sparse_matrix_handle_t
// SparseUtils|dense_vector_handle_t
// LibCommonUtils|library_data_t
// SparseUtils|matrix_info
// LibCommonUtils|get_value
// LibCommonUtils|get_memory
// DPCT_DEPENDENCY_END
// DPCT_CODE
void spmv(sycl::queue &queue, oneapi::mkl::transpose trans, const void *alpha,
          sparse_matrix_handle_t mat_handle, dense_vector_handle_t x,
          const void *beta, dense_vector_handle_t y,
          library_data_t compute_type,
          matrix_info::matrix_type mt = matrix_info::matrix_type::ge) {

#define CASE(TYPE)                                                             \
  auto alpha_value = get_value(reinterpret_cast<const TYPE *>(alpha), queue);  \
  auto beta_value = get_value(reinterpret_cast<const TYPE *>(beta), queue);    \
  auto data_x = dpct::detail::get_memory((const TYPE *)std::get<1>(*x));       \
  auto data_y = dpct::detail::get_memory((TYPE *)std::get<1>(*y));             \
  oneapi::mkl::sparse::optimize_gemv(queue, trans, mat_handle);                \
  oneapi::mkl::sparse::gemv(queue, trans, alpha_value, mat_handle, data_x,     \
                            beta_value, data_y);                               \
  break;

  switch (compute_type) {
  case library_data_t::real_float: {
    CASE(float)
  }
  case library_data_t::real_double: {
    CASE(double)
  }
  case library_data_t::complex_float: {
    CASE(std::complex<float>)
  }
  case library_data_t::complex_double: {
    CASE(std::complex<double>)
  }
  default:
    throw std::runtime_error("the data type is unsupported");
  }
#undef CASE
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|spmv|dpct::sparse
// DPCT_DEPENDENCY_BEGIN
// SparseUtils|sparse_matrix_handle_t
// SparseUtils|create_csr_matrix_handle
// SparseUtils|matrix_info
// LibCommonUtils|get_value
// LibCommonUtils|get_memory
// SparseUtils|get_type_as_library_data
// SparseUtils|destroy_csr_matrix_handle_async
// Util|DataType
// DPCT_DEPENDENCY_END
// DPCT_CODE
template <typename T>
void csrmv(sycl::queue &queue, oneapi::mkl::transpose trans, int num_rows,
           int num_cols, const T *alpha, const matrix_info info, const T *val,
           const int *row_ptr, const int *col_ind, const T *x, const T *beta,
           T *y) {
  sparse_matrix_handle_t *sparse_matrix_handle = new sparse_matrix_handle_t;
  create_csr_matrix_handle(queue, *sparse_matrix_handle, num_rows, num_cols,
                           row_ptr, col_ind, val,
                           detail::get_type_as_library_data<int>(),
                           detail::get_type_as_library_data<int>(),
                           info.get<oneapi::mkl::index_base>(),
                           detail::get_type_as_library_data<T>());

  using Ty = typename dpct::DataType<T>::T2;
  auto alpha_value = get_value(reinterpret_cast<const Ty *>(alpha), queue);
  auto beta_value = get_value(reinterpret_cast<const Ty *>(beta), queue);
  auto data_x = dpct::detail::get_memory(reinterpret_cast<const Ty *>(x));
  auto data_y = dpct::detail::get_memory(reinterpret_cast<Ty *>(y));

  switch (info.get<matrix_info::matrix_type>()) {
  case matrix_info::matrix_type::ge: {
    oneapi::mkl::sparse::optimize_gemv(queue, trans, *sparse_matrix_handle);
    oneapi::mkl::sparse::gemv(queue, trans, alpha_value, *sparse_matrix_handle,
                              data_x, beta_value, data_y);
    break;
  }
  case matrix_info::matrix_type::sy: {
    oneapi::mkl::sparse::symv(queue, info.get<oneapi::mkl::uplo>(), alpha_value,
                              *sparse_matrix_handle, const_cast<Ty *>(data_x),
                              beta_value, data_y);
    break;
  }
  case matrix_info::matrix_type::tr: {
    oneapi::mkl::sparse::optimize_trmv(queue, info.get<oneapi::mkl::uplo>(),
                                       trans, info.get<oneapi::mkl::diag>(),
                                       *sparse_matrix_handle);
    oneapi::mkl::sparse::trmv(queue, info.get<oneapi::mkl::uplo>(), trans,
                              info.get<oneapi::mkl::diag>(), alpha_value,
                              *sparse_matrix_handle, const_cast<Ty *>(data_x),
                              beta_value, data_y);
    break;
  }
  default:
    throw std::runtime_error(
        "the spmv does not support matrix_info::matrix_type::he");
  }

  sycl::event e =
      detail::destroy_csr_matrix_handle_async(queue, *sparse_matrix_handle);
  queue.submit([&](sycl::handler &cgh) {
    cgh.depends_on(e);
    cgh.host_task([=] { delete sparse_matrix_handle; });
  });
}
// DPCT_LABEL_END

} // namespace sparse
} // namespace dpct

#endif // __DPCT_SPARSE_UTILS_HPP__
