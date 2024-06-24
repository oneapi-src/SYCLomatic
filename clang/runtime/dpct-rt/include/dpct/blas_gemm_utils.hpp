//==---- blas_gemm_utils.hpp ----------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __DPCT_BLAS_GEMM_UTILS_HPP__
#define __DPCT_BLAS_GEMM_UTILS_HPP__

#include "dnnl_utils.hpp"
#include "lib_common_utils.hpp"
#include "memory.hpp"

#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>

#include <sycl/sycl.hpp>

namespace dpct {
namespace blas_gemm {
namespace experimental {
enum class order_t : std::uint8_t {
  col,
  row,
  col32,
  col4_4r2_8c,
  col32_2r_4r4
};
enum class pointer_mode_t {
  host,
  device,
  device_vector,
  alpha_device_vector_beta_zero,
  alpha_device_vector_beta_host
};

class descriptor;
using descriptor_ptr = descriptor *;
class matrix_layout_t;
using matrix_layout_ptr = matrix_layout_t *;
class matmul_desc_t;
using matmul_desc_ptr = matmul_desc_t *;
class transform_desc_t;
using transform_desc_ptr = transform_desc_t *;
class transform_desc_t;
using transform_desc_ptr = transform_desc_t *;

class descriptor {
public:
  descriptor() {}
  void init(sycl::queue *q_ptr) {
    _engine = ::dnnl::sycl_interop::make_engine(q_ptr->get_device(),
                                                q_ptr->get_context());
    _engine_stream = ::dnnl::sycl_interop::make_stream(_engine, *q_ptr);
  }
  ::dnnl::engine get_engine() const noexcept { return _engine; }
  ::dnnl::stream get_engine_stream() const noexcept { return _engine_stream; };

private:
  ::dnnl::engine _engine;
  ::dnnl::stream _engine_stream;
};

class matrix_layout_t {
public:
  enum class attribute { type, order, rows, cols, ld };

  matrix_layout_t(library_data_t type, std::uint64_t rows, std::uint64_t cols,
                  std::int64_t ld)
      : _type(type), _rows(rows), _cols(cols), _ld(ld) {}

  void set_attribute(attribute attr, const void *mem) {
    get_set_attr<true>(attr, const_cast<void *>(mem));
  }
  void get_attribute(attribute attr, void *mem) {
    get_set_attr<false>(attr, mem);
  }

private:
  template <bool is_set> void get_set_attr(attribute attr, void *mem) {
#define CASE(tag)                                                              \
  case attribute::tag:                                                         \
    if constexpr (is_set) {                                                    \
      _##tag = *static_cast<decltype(_##tag) *>(mem);                          \
    } else {                                                                   \
      *static_cast<decltype(_##tag) *>(mem) = _##tag;                          \
    }                                                                          \
    break;
    switch (attr) {
      CASE(type)
      CASE(order)
      CASE(rows)
      CASE(cols)
      CASE(ld)
    }
#undef CASE
  }

  library_data_t _type;
  order_t _order = order_t::col;
  std::uint64_t _rows;
  std::uint64_t _cols;
  std::int64_t _ld;

  friend sycl::event matmul(descriptor_ptr handle, matmul_desc_ptr computeDesc,
                            const void *alpha, const void *a,
                            matrix_layout_ptr a_desc, const void *b,
                            matrix_layout_ptr b_desc, const void *beta,
                            const void *c, matrix_layout_ptr c_desc, void *d,
                            matrix_layout_ptr d_desc, dpct::queue_ptr q_ptr);
  friend sycl::event
  matrix_transform(transform_desc_ptr transform_desc, const void *alpha,
                   const void *a, matrix_layout_ptr a_desc, const void *beta,
                   const void *b, matrix_layout_ptr b_desc, void *c,
                   matrix_layout_ptr c_desc, queue_ptr q_ptr);
};

class matmul_desc_t {
public:
  enum class attribute {
    compute_type,
    scale_type,
    pointer_mode,
    trans_a,
    trans_b,
    trans_c,
    epilogue
  };

  matmul_desc_t(compute_type compute_type, library_data_t scale_type)
      : _compute_type(compute_type), _scale_type(scale_type) {}

  void set_attribute(attribute attr, const void *mem) {
    get_set_attr<true>(attr, const_cast<void *>(mem));
  }
  void get_attribute(attribute attr, void *mem) {
    get_set_attr<false>(attr, mem);
  }

private:
  template <bool is_set> void get_set_attr(attribute attr, void *mem) {
#define CASE(tag)                                                              \
  case attribute::tag:                                                         \
    if constexpr (is_set) {                                                    \
      _##tag = *static_cast<decltype(_##tag) *>(mem);                          \
    } else {                                                                   \
      *static_cast<decltype(_##tag) *>(mem) = _##tag;                          \
    }                                                                          \
    break;
    switch (attr) {
      CASE(compute_type)
      CASE(scale_type)
      CASE(pointer_mode)
      CASE(trans_a)
      CASE(trans_b)
      CASE(trans_c)
      CASE(epilogue)
    }
#undef CASE
  }

  compute_type _compute_type;
  library_data_t _scale_type;
  pointer_mode_t _pointer_mode = pointer_mode_t::host;
  oneapi::mkl::transpose _trans_a = oneapi::mkl::transpose::nontrans;
  oneapi::mkl::transpose _trans_b = oneapi::mkl::transpose::nontrans;
  oneapi::mkl::transpose _trans_c = oneapi::mkl::transpose::nontrans;
  int _epilogue = 1;

  friend sycl::event matmul(descriptor_ptr handle, matmul_desc_ptr computeDesc,
                            const void *alpha, const void *a,
                            matrix_layout_ptr a_desc, const void *b,
                            matrix_layout_ptr b_desc, const void *beta,
                            const void *c, matrix_layout_ptr c_desc, void *d,
                            matrix_layout_ptr d_desc, dpct::queue_ptr q_ptr);
};

namespace detail {
/// Sacling each row of matrix D with the corresponding element of vector alpha.
template <class T, class Talpha>
sycl::event scale_d_with_vector_alpha_impl(queue_ptr q_ptr, int rows, int cols,
                                           T *d, const Talpha *alpha,
                                           std::vector<sycl::event> deps) {
  return q_ptr->submit([&](sycl::handler &cgh) {
    cgh.depends_on(deps);
#ifdef DPCT_USM_LEVEL_NONE
    access_wrapper<T *> d_acc(d, cgh);
    access_wrapper<const Talpha *> alpha_acc(alpha, cgh);
#endif
    cgh.parallel_for<
        dpct_kernel_name<class scale_with_vector_alpha, T, Talpha>>(
        sycl::range<2>(rows, cols), [=](sycl::id<2> index) {
#ifdef DPCT_USM_LEVEL_NONE
          auto d_data = d_acc.get_raw_pointer();
          auto alpha_data = alpha_acc.get_raw_pointer();
#else
            auto d_data = d;
            auto alpha_data = alpha;
#endif
          size_t row_idx = index.get(0);
          size_t col_idx = index.get(1);
          size_t idx = rows * col_idx + row_idx;
          d_data[idx] = d_data[idx] * alpha_data[row_idx];
        });
  });
}

// d is col major without padding
inline sycl::event scale_d_with_vector_alpha(queue_ptr q_ptr, int rows,
                                             int cols, void *d,
                                             library_data_t d_type,
                                             const void *alpha,
                                             library_data_t alpha_type,
                                             std::vector<sycl::event> deps) {
  std::uint64_t key = dpct::detail::get_type_combination_id(d_type, alpha_type);
  sycl::event e;
  switch (key) {
  case dpct::detail::get_type_combination_id(library_data_t::real_int8,
                                             library_data_t::real_float): {
    e = scale_d_with_vector_alpha_impl<std::int8_t, float>(
        q_ptr, rows, cols, (std::int8_t *)d, (const float *)alpha, deps);
    break;
  }
  case dpct::detail::get_type_combination_id(library_data_t::real_int32,
                                             library_data_t::real_float): {
    e = scale_d_with_vector_alpha_impl<int, float>(q_ptr, rows, cols, (int *)d,
                                                   (const float *)alpha, deps);
    break;
  }
  default:
    throw std::runtime_error("dpct::blas_gemm::experimental::detail::scale_d_"
                             "with_vector_alpha() does not support the data "
                             "type combination currently.");
  }
  return e;
}

/// Get a linear idx map for a 2D point (row_idx, col_idx) between src_order and
/// dst_order.
inline std::tuple<size_t, size_t>
get_linear_idx_map(size_t rows, size_t cols, size_t src_ld, order_t src_order,
                   size_t dst_ld, order_t dst_order, size_t row_idx,
                   size_t col_idx) {
#define COMBINE(from, to)                                                      \
  static_cast<std::uint16_t>(from) << 8 | static_cast<std::uint8_t>(to)

  size_t from_linear_idx, to_linear_idx;
  switch (COMBINE(src_order, dst_order)) {
  case COMBINE(order_t::col, order_t::row): {
    from_linear_idx = src_ld * col_idx + row_idx;
    to_linear_idx = dst_ld * row_idx + col_idx;
    break;
  }
  case COMBINE(order_t::row, order_t::col): {
    from_linear_idx = src_ld * row_idx + col_idx;
    to_linear_idx = dst_ld * col_idx + row_idx;
    break;
  }
  case COMBINE(order_t::col, order_t::col32): {
    from_linear_idx = src_ld * col_idx + row_idx;
    to_linear_idx = dst_ld * (col_idx / 32) + 32 * row_idx + col_idx % 32;
    break;
  }
  case COMBINE(order_t::col32, order_t::col): {
    from_linear_idx = src_ld * (col_idx / 32) + 32 * row_idx + col_idx % 32;
    to_linear_idx = dst_ld * col_idx + row_idx;
    break;
  }
  case COMBINE(order_t::col, order_t::col4_4r2_8c): {
    from_linear_idx = src_ld * col_idx + row_idx;

    size_t from_row_in_row8_col32 = row_idx % 8;
    size_t from_col_in_row8_col32 = col_idx % 32;

    size_t to_row_in_row8_col32 =
        4 * (from_row_in_row8_col32 % 2) + from_col_in_row8_col32 / 8;
    size_t to_col_in_row8_col32 = 16 * ((from_col_in_row8_col32 / 4) % 2) +
                                  4 * (from_row_in_row8_col32 / 2) +
                                  from_col_in_row8_col32 % 4;
    size_t to_linear_idx_in_row8_col32 =
        to_row_in_row8_col32 * 32 + to_col_in_row8_col32;

    to_linear_idx = dst_ld * (col_idx / 32) + (row_idx / 8) * (32 * 8) +
                    to_linear_idx_in_row8_col32;
    break;
  }
  case COMBINE(order_t::col4_4r2_8c, order_t::col): {
    to_linear_idx = dst_ld * col_idx + row_idx;

    size_t to_row_in_row8_col32 = row_idx % 8;
    size_t to_col_in_row8_col32 = col_idx % 32;

    size_t from_row_in_row8_col32 =
        4 * (to_row_in_row8_col32 % 2) + to_col_in_row8_col32 / 8;
    size_t from_col_in_row8_col32 = 16 * ((to_col_in_row8_col32 / 4) % 2) +
                                    4 * (to_row_in_row8_col32 / 2) +
                                    to_col_in_row8_col32 % 4;
    size_t from_linear_idx_in_row8_col32 =
        from_row_in_row8_col32 * 32 + from_col_in_row8_col32;

    from_linear_idx = src_ld * (col_idx / 32) + (row_idx / 8) * (32 * 8) +
                      from_linear_idx_in_row8_col32;
    break;
  }
  case COMBINE(order_t::col, order_t::col32_2r_4r4): {
    from_linear_idx = src_ld * col_idx + row_idx;

    size_t from_row_in_row32_col32 = row_idx % 32;
    size_t from_col_in_row32_col32 = col_idx % 32;

    size_t to_row_in_row32_col32 = 8 * ((from_row_in_row32_col32 % 8) / 2) +
                                   (from_row_in_row32_col32 / 8) * 2 +
                                   from_row_in_row32_col32 % 2;
    size_t to_col_in_row32_col32 = from_col_in_row32_col32;
    size_t to_linear_idx_in_row32_col32 =
        to_row_in_row32_col32 * 32 + to_col_in_row32_col32;

    to_linear_idx = dst_ld * (col_idx / 32) + (row_idx / 32) * (32 * 32) +
                    to_linear_idx_in_row32_col32;
    break;
  }
  case COMBINE(order_t::col32_2r_4r4, order_t::col): {
    to_linear_idx = dst_ld * col_idx + row_idx;

    size_t to_row_in_row32_col32 = row_idx % 32;
    size_t to_col_in_row32_col32 = col_idx % 32;

    size_t from_row_in_row32_col32 = 8 * ((to_row_in_row32_col32 % 8) / 2) +
                                     (to_row_in_row32_col32 / 8) * 2 +
                                     to_row_in_row32_col32 % 2;
    size_t from_col_in_row32_col32 = to_col_in_row32_col32;
    size_t from_linear_idx_in_row32_col32 =
        from_row_in_row32_col32 * 32 + from_col_in_row32_col32;

    from_linear_idx = src_ld * (col_idx / 32) + (row_idx / 32) * (32 * 32) +
                      from_linear_idx_in_row32_col32;
    break;
  }
  }
#undef COMBINE
  return std::make_tuple(from_linear_idx, to_linear_idx);
}

template <class T>
sycl::event matrix_transform(queue_ptr q_ptr, size_t rows, size_t cols,
                             size_t a_ld, order_t a_order, const T *a,
                             size_t c_ld, order_t c_order, T *c,
                             std::vector<sycl::event> deps) {
  if ((a_order != order_t::col && c_order != order_t::col) ||
      (a_order == order_t::col && c_order == order_t::col)) {
    throw std::runtime_error(
        "dpct::blas_gemm::experimental::detail::matrix_transform() does not "
        "support the order combination currently.");
  }

  return q_ptr->submit([&](sycl::handler &cgh) {
    cgh.depends_on(deps);
#ifdef DPCT_USM_LEVEL_NONE
    access_wrapper<const T *> a_acc(a, cgh);
    access_wrapper<T *> c_acc(c, cgh);
#endif
    cgh.parallel_for<dpct_kernel_name<class matrix_transform_col_to_row, T>>(
        sycl::range<2>(a_ld, cols), [=](sycl::id<2> index) {
#ifdef DPCT_USM_LEVEL_NONE
          auto a_data = a_acc.get_raw_pointer();
          auto c_data = c_acc.get_raw_pointer();
#else
            auto a_data = a;
            auto c_data = c;
#endif
          size_t row_idx = index.get(0);
          size_t col_idx = index.get(1);
          if (row_idx < rows) {
            size_t from_linear_idx, to_linear_idx;
            std::tie(from_linear_idx, to_linear_idx) = get_linear_idx_map(
                rows, cols, a_ld, a_order, c_ld, c_order, row_idx, col_idx);
            c_data[to_linear_idx] = a_data[from_linear_idx];
          }
        });
  });
}
} // namespace detail

/// TODO: Impl row-major matmul without layout conversion
/// This function does operation: D = alpha*(A*B) + beta*(C).
/// Currently it only supports beta==0.
/// NOTE: Non-col-major matrix will be converted to col-major matrix before
/// multiplication and converted back after multiplication.
/// \param [in] handle A handle containing context info.
/// \param [in] compute_desc Describe the computation.
/// \param [in] alpha Scaling factor alpha.
/// \param [in] a Input matrix A.
/// \param [in] a_desc Describe the matrix A.
/// \param [in] b Input matrix B.
/// \param [in] b_desc Describe the matrix B.
/// \param [in] beta Scaling factor beta.
/// \param [in] c Input matrix C.
/// \param [in] c_desc Describe the matrix C.
/// \param [out] d Output matrix D.
/// \param [in] d_desc Describe the matrix D.
/// \param [in] q_ptr The queue where the routine should be executed.
inline sycl::event matmul(descriptor_ptr handle, matmul_desc_ptr compute_desc,
                          const void *alpha, const void *a,
                          matrix_layout_ptr a_desc, const void *b,
                          matrix_layout_ptr b_desc, const void *beta,
                          const void *c, matrix_layout_ptr c_desc, void *d,
                          matrix_layout_ptr d_desc, dpct::queue_ptr q_ptr) {
  if (!q_ptr)
    q_ptr = &get_default_queue();
  handle->init(q_ptr);
  bool vector_alpha = false;
  if (compute_desc->_pointer_mode == pointer_mode_t::device_vector ||
      compute_desc->_pointer_mode ==
          pointer_mode_t::alpha_device_vector_beta_zero ||
      compute_desc->_pointer_mode ==
          pointer_mode_t::alpha_device_vector_beta_host) {
    vector_alpha = true;
  }

  if (beta != nullptr) {
    size_t beta_size =
        dpct::detail::library_data_size[static_cast<unsigned int>(
            compute_desc->_scale_type)] /
        8;
    void *beta_host = std::malloc(beta_size);
    void *beta_zero = std::malloc(beta_size);
    std::memset(beta_zero, 0, beta_size);
    q_ptr->memcpy(beta_host, beta, beta_size).wait();
    if (std::memcmp(beta_host, beta_zero, beta_size))
      throw std::runtime_error("dpct::blas_gemm::experimental::matmul() does "
                               "not support non-zero beta currently.");
  }

  if (compute_desc->_epilogue != 1) {
    throw std::runtime_error("dpct::blas_gemm::experimental::matmul() does "
                             "not support epilogue currently.");
  }

  if (compute_desc->_trans_a != oneapi::mkl::transpose::nontrans) {
    throw std::runtime_error("dpct::blas_gemm::experimental::matmul() only "
                             "supports non-transposed matrix A currently.");
  }
  if (compute_desc->_trans_b != oneapi::mkl::transpose::trans) {
    throw std::runtime_error("dpct::blas_gemm::experimental::matmul() only "
                             "supports transposed matrix B currently.");
  }

  // For non-col_major matrix, convert it to col_major.
  const void *new_a = a;
  const void *new_b = b;
  void *new_d = d;
  size_t new_lda = a_desc->_ld, new_ldb = b_desc->_ld, new_ldd = d_desc->_ld;
  std::vector<sycl::event> transform_events;
  if (a_desc->_order != order_t::col) {
    new_lda = a_desc->_rows;
    if (a_desc->_type == library_data_t::real_int8) {
      new_a =
          dpct_malloc(sizeof(std::int8_t) * a_desc->_cols * new_lda, *q_ptr);
      sycl::event e = detail::matrix_transform<std::int8_t>(
          q_ptr, a_desc->_rows, a_desc->_cols, a_desc->_ld, a_desc->_order,
          (const std::int8_t *)a, new_lda, order_t::col, (std::int8_t *)new_a,
          {});
      transform_events.push_back(e);
    } else {
      new_a = dpct_malloc(sizeof(int) * a_desc->_cols * new_lda, *q_ptr);
      sycl::event e = detail::matrix_transform<int>(
          q_ptr, a_desc->_rows, a_desc->_cols, a_desc->_ld, a_desc->_order,
          (const int *)a, new_lda, order_t::col, (int *)new_a, {});
      transform_events.push_back(e);
    }
  }
  if (b_desc->_order != order_t::col) {
    new_ldb = b_desc->_rows;
    if (b_desc->_type == library_data_t::real_int8) {
      new_b =
          dpct_malloc(sizeof(std::int8_t) * b_desc->_cols * new_ldb, *q_ptr);
      sycl::event e = detail::matrix_transform<std::int8_t>(
          q_ptr, b_desc->_rows, b_desc->_cols, b_desc->_ld, b_desc->_order,
          (const std::int8_t *)b, new_ldb, order_t::col, (std::int8_t *)new_b,
          {});
      transform_events.push_back(e);
    } else {
      new_b = dpct_malloc(sizeof(int) * b_desc->_cols * new_ldb, *q_ptr);
      sycl::event e = detail::matrix_transform<int>(
          q_ptr, b_desc->_rows, b_desc->_cols, b_desc->_ld, b_desc->_order,
          (const int *)b, new_ldb, order_t::col, (int *)new_b, {});
      transform_events.push_back(e);
    }
  }
  if (d_desc->_order != order_t::col) {
    new_ldd = d_desc->_rows;
    if (d_desc->_type == library_data_t::real_int8) {
      new_d =
          dpct_malloc(sizeof(std::int8_t) * d_desc->_cols * new_ldd, *q_ptr);
    } else {
      new_d = dpct_malloc(sizeof(int) * d_desc->_cols * new_ldd, *q_ptr);
    }
  }

  // start to call oneDNN matmul primitive
  // a,d are col_major, b is row_major
  const size_t m = a_desc->_rows;
  const size_t n = d_desc->_cols;
  const size_t k = b_desc->_cols;
  const ::dnnl::memory::dim M = m;
  const ::dnnl::memory::dim N = n;
  const ::dnnl::memory::dim K = k;
  const library_data_t a_type = a_desc->_type;
  const library_data_t b_type = b_desc->_type;
  const library_data_t d_type = d_desc->_type;
  const library_data_t scale_type = compute_desc->_scale_type;

  ::dnnl::memory::dims src_dims = {M, K};
  ::dnnl::memory::dims weights_dims = {K, N};
  ::dnnl::memory::dims dst_dims = {M, N};

  const ::dnnl::memory::dims src_strides =
      ::dnnl::memory::dims{1, static_cast<long>(new_lda)};
  const ::dnnl::memory::dims weights_strides =
      ::dnnl::memory::dims{static_cast<long>(new_ldb), 1};
  const ::dnnl::memory::dims dst_strides =
      ::dnnl::memory::dims{1, static_cast<long>(new_ldd)};

  auto src_md = ::dnnl::memory::desc(
      src_dims, dpct::dnnl::memory_desc_ext::to_dnnl_data_type(a_type),
      src_strides);
  auto weights_md = ::dnnl::memory::desc(
      weights_dims, dpct::dnnl::memory_desc_ext::to_dnnl_data_type(b_type),
      weights_strides);
  auto dst_md = ::dnnl::memory::desc(
      dst_dims, dpct::dnnl::memory_desc_ext::to_dnnl_data_type(d_type),
      dst_strides);

  auto *src_mem =
      new ::dnnl::memory(src_md, handle->get_engine(), DNNL_MEMORY_NONE);
  auto *weights_mem =
      new ::dnnl::memory(weights_md, handle->get_engine(), DNNL_MEMORY_NONE);
  auto *dst_mem =
      new ::dnnl::memory(dst_md, handle->get_engine(), DNNL_MEMORY_NONE);

#ifdef DPCT_USM_LEVEL_NONE
#define SET_BUFFER(DST, TYPE, SRC)                                             \
  {                                                                            \
    switch (TYPE) {                                                            \
    case library_data_t::real_int8: {                                          \
      auto buf = get_buffer<std::int8_t>(SRC);                                 \
      ::dnnl::sycl_interop::set_buffer(*DST, buf);                             \
      break;                                                                   \
    }                                                                          \
    case library_data_t::real_int32: {                                         \
      auto buf = get_buffer<int>(SRC);                                         \
      ::dnnl::sycl_interop::set_buffer(*DST, buf);                             \
      break;                                                                   \
    }                                                                          \
    default:                                                                   \
      throw std::runtime_error("dpct::blas_gemm::experimental::matmul() does " \
                               "not support type (dpct::library_data_t) " +    \
                               std::to_string((std::uint8_t)TYPE) +            \
                               " currently.");                                 \
    }                                                                          \
  }

  SET_BUFFER(src_mem, a_type, new_a);
  SET_BUFFER(weights_mem, b_type, new_b);
  SET_BUFFER(dst_mem, d_type, new_d);
#undef SET_BUFFER
#else
  src_mem->set_data_handle(const_cast<void *>(new_a));
  weights_mem->set_data_handle(const_cast<void *>(new_b));
  dst_mem->set_data_handle(new_d);
#endif

  std::unordered_map<int, ::dnnl::memory> matmul_args;
  matmul_args.insert({DNNL_ARG_SRC, *src_mem});
  matmul_args.insert({DNNL_ARG_WEIGHTS, *weights_mem});
  matmul_args.insert({DNNL_ARG_DST, *dst_mem});
  ::dnnl::primitive_attr matmul_attr;
  ::dnnl::memory *scales_alpha = nullptr;
  if (!vector_alpha) {
    matmul_attr.set_scales_mask(DNNL_ARG_WEIGHTS, 0);
    std::size_t Size =
        dpct::detail::library_data_size[static_cast<unsigned int>(scale_type)] /
        8;
    scales_alpha = new ::dnnl::memory(
        {{1}, dpct::dnnl::memory_desc_ext::to_dnnl_data_type(scale_type), {1}},
        handle->get_engine());
    dpct::dpct_memcpy(scales_alpha->get_data_handle(), alpha, Size, automatic,
                      *q_ptr);
    matmul_args.insert(
        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, *scales_alpha});
  }

  auto matmul_pd = ::dnnl::matmul::primitive_desc(
      handle->get_engine(), src_md, weights_md, dst_md, matmul_attr);
  auto matmul_prim = ::dnnl::matmul(matmul_pd);
  sycl::event matmul_prim_event = ::dnnl::sycl_interop::execute(
      matmul_prim, handle->get_engine_stream(), matmul_args, transform_events);

  sycl::event scale_d_event;
  if (vector_alpha)
    scale_d_event = detail::scale_d_with_vector_alpha(
        q_ptr, m, n, new_d, d_type, alpha, scale_type, {matmul_prim_event});
  // end of calling oneDNN

  sycl::event transform_d_event;
  if (d_desc->_order != order_t::col) {
    if (d_desc->_type == library_data_t::real_int8) {
      transform_d_event = detail::matrix_transform<std::int8_t>(
          q_ptr, d_desc->_rows, d_desc->_cols, new_ldd, order_t::col,
          (const std::int8_t *)new_d, d_desc->_ld, d_desc->_order,
          (std::int8_t *)d, {scale_d_event, matmul_prim_event});
    } else {
      transform_d_event = detail::matrix_transform<int>(
          q_ptr, d_desc->_rows, d_desc->_cols, new_ldd, order_t::col,
          (const int *)new_d, d_desc->_ld, d_desc->_order, (int *)d,
          {scale_d_event, matmul_prim_event});
    }
  }

  sycl::event free_event = q_ptr->submit([&](sycl::handler &cgh) {
    cgh.depends_on({transform_d_event, scale_d_event, matmul_prim_event});
    cgh.host_task([=] {
      delete src_mem;
      delete weights_mem;
      delete dst_mem;
      if (!vector_alpha)
        delete scales_alpha;
      dpct::detail::dpct_free((void *)new_a, *q_ptr);
      dpct::detail::dpct_free((void *)new_b, *q_ptr);
      dpct::detail::dpct_free((void *)new_d, *q_ptr);
    });
  });
  return free_event;
}

class transform_desc_t {
public:
  enum class attribute { scale_type, pointer_mode, trans_a, trans_b };

  transform_desc_t(library_data_t scale_type) : _scale_type(scale_type) {}
  void set_attribute(attribute attr, const void *mem) {
    get_set_attr<true>(attr, const_cast<void *>(mem));
  }
  void get_attribute(attribute attr, void *mem) {
    get_set_attr<false>(attr, mem);
  }

private:
  template <bool is_set> void get_set_attr(attribute attr, void *mem) {
#define CASE(tag)                                                              \
  case attribute::tag:                                                         \
    if constexpr (is_set) {                                                    \
      _##tag = *static_cast<decltype(_##tag) *>(mem);                          \
    } else {                                                                   \
      *static_cast<decltype(_##tag) *>(mem) = _##tag;                          \
    }                                                                          \
    break;
    switch (attr) {
      CASE(scale_type)
      CASE(pointer_mode)
      CASE(trans_a)
      CASE(trans_b)
    }
#undef CASE
  }

  library_data_t _scale_type;
  pointer_mode_t _pointer_mode = pointer_mode_t::host;
  oneapi::mkl::transpose _trans_a = oneapi::mkl::transpose::nontrans;
  oneapi::mkl::transpose _trans_b = oneapi::mkl::transpose::nontrans;

  friend sycl::event
  matrix_transform(transform_desc_ptr transform_desc, const void *alpha,
                   const void *a, matrix_layout_ptr a_desc, const void *beta,
                   const void *b, matrix_layout_ptr b_desc, void *c,
                   matrix_layout_ptr c_desc, queue_ptr q_ptr);
};

/// This function does operation:
/// C = alpha*transformation(A) + beta*transformation(B).
/// Currently it only supports alpha==1 && beta==0.
/// \param [in] transform_desc Describe the transformation.
/// \param [in] alpha Scaling factor alpha.
/// \param [in] a Input matrix A.
/// \param [in] a_desc Describe the matrix A.
/// \param [in] beta Scaling factor beta.
/// \param [in] b Input matrix B.
/// \param [in] b_desc Describe the matrix B.
/// \param [out] c Output matrix C.
/// \param [in] c_desc Describe the matrix C.
/// \param [in] q_ptr The queue where the routine should be executed.
inline sycl::event matrix_transform(transform_desc_ptr transform_desc,
                                    const void *alpha, const void *a,
                                    matrix_layout_ptr a_desc, const void *beta,
                                    const void *b, matrix_layout_ptr b_desc,
                                    void *c, matrix_layout_ptr c_desc,
                                    queue_ptr q_ptr) {
  if (!q_ptr)
    q_ptr = &get_default_queue();

  if (transform_desc->_pointer_mode != pointer_mode_t::host) {
    throw std::runtime_error(
        "dpct::blas_gemm::experimental::matrix_transform() "
        "only supports pointer_mode_t::host as pointer_mode currently.");
  }
  if (transform_desc->_scale_type != library_data_t::real_float) {
    throw std::runtime_error(
        "dpct::blas_gemm::experimental::matrix_transform() "
        "only supports library_data_t::real_float as scale_type currently.");
  }

  if (alpha != nullptr) {
    if (1.0f != *reinterpret_cast<const float *>(alpha))
      throw std::runtime_error(
          "dpct::blas_gemm::experimental::matrix_transform() does not "
          "support non-one alpha currently.");
  }

  if (beta != nullptr) {
    if (0.0f != *reinterpret_cast<const float *>(beta))
      throw std::runtime_error(
          "dpct::blas_gemm::experimental::matrix_transform() does not "
          "support non-zero beta currently.");
  }

  if (b != nullptr) {
    throw std::runtime_error(
        "dpct::blas_gemm::experimental::matrix_transform() does not "
        "support matrix B currently.");
  }

  if ((a_desc->_type != library_data_t::real_int8 ||
       c_desc->_type != library_data_t::real_int8) &&
      (a_desc->_type != library_data_t::real_int32 ||
       c_desc->_type != library_data_t::real_int32)) {
    throw std::runtime_error(
        "dpct::blas_gemm::experimental::matrix_transform() only supports "
        "combinations of data types: a_type==real_int8&&c_type==real_int8, "
        "a_type==real_int32&&c_type==real_int32.");
  }

  if (a_desc->_type == library_data_t::real_int8) {
    return detail::matrix_transform<std::int8_t>(
        q_ptr, a_desc->_rows, a_desc->_cols, a_desc->_ld, a_desc->_order,
        (const std::int8_t *)a, c_desc->_ld, c_desc->_order, (std::int8_t *)c,
        {});
  } else {
    return detail::matrix_transform<int>(
        q_ptr, a_desc->_rows, a_desc->_cols, a_desc->_ld, a_desc->_order,
        (const int *)a, c_desc->_ld, c_desc->_order, (int *)c, {});
  }
}
} // namespace experimental
} // namespace blas_gemm
} // namespace dpct
#endif // __DPCT_BLAS_GEMM_UTILS_HPP__
