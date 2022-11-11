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

using dense_vec_descr =
    std::shared_ptr<std::tuple<std::int64_t, void *, library_data_t>>;

void create_csr_matrix_handle(
    sycl::queue &queue, oneapi::mkl::sparse::matrix_handle_t handle,
    std::int64_t num_rows, std::int64_t num_cols, void *row_ptr, void *col_ind,
    void *val, library_data_t row_ptr_type, library_data_t col_ind_type,
    oneapi::mkl::index_base index, library_data_t value_type,
    const std::vector<sycl::event> &dependencies = {}) {
  oneapi::mkl::sparse::init_matrix_handle(&handle);
  std::uint64_t key =
      detail::get_type_combination_id(row_ptr_type, col_ind_type, value_type);

#define CASE(row_ptr_type_enum, col_ind_type_enum, value_type_enum,            \
             row_ptr_type, col_ind_type, value_type)                           \
  case detail::get_type_combination_id(row_ptr_type_enum, col_ind_type_enum,   \
                                       value_type_enum): {                     \
    auto data_row_ptr = get_memory((row_ptr_type *)row_ptr);                   \
    auto data_col_ind = get_memory((col_ind_type *)col_ind);                   \
    auto data_val = get_memory((value_type *)val);                             \
    oneapi::mkl::sparse::set_csr_data(queue, handle, num_rows, num_cols,       \
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
  queue.wait();
}

void destroy_matrix_handle(sycl::queue &queue,
                           oneapi::mkl::sparse::matrix_handle_t handle) {
  oneapi::mkl::sparse::release_matrix_handle(queue, &handle).wait();
}

void spmv(sycl::queue &handle, oneapi::mkl::transpose uplo_val,
          const void *alpha, oneapi::mkl::sparse::matrix_handle_t mat_handle,
          dense_vec_descr x, const void *beta, dense_vec_descr y,
          library_data_t compute_type) {
  std::uint64_t key = detail::get_type_combination_id(compute_type);
#define CASE(compute_type_enum, compute_type)                                  \
  case detail::get_type_combination_id(compute_type_enum): {                   \
    float alpha_value =                                                        \
        get_value(reinterpret_cast<const compute_type *>(alpha), handle);      \
    float beta_value =                                                         \
        get_value(reinterpret_cast<const compute_type *>(beta), handle);       \
    auto data_x = get_memory((const compute_type *)std::get<1>(*x));           \
    auto data_y = get_memory((compute_type *)std::get<1>(*y));                 \
    oneapi::mkl::sparse::optimize_gemv(handle, uplo_val, mat_handle);          \
    oneapi::mkl::sparse::gemv(handle, uplo_val, alpha_value, mat_handle,       \
                              data_x, beta_value, data_y);                     \
    break;                                                                     \
  }
  switch (key) {
    CASE(library_data_t::real_float, float)
    CASE(library_data_t::real_double, double)
    CASE(library_data_t::complex_float, std::complex<float>)
    CASE(library_data_t::complex_double, std::complex<double>)
  default:
    throw std::runtime_error("the combination of data type is unsupported");
  }
#undef CASE
}
} // namespace sparse
} // namespace dpct

#endif // __DPCT_SPARSE_UTILS_HPP__
