//==---- lapack_utils.hpp -------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __DPCT_LAPACK_UTILS_HPP__
#define __DPCT_LAPACK_UTILS_HPP__

#include "memory.hpp"
#include "util.hpp"
#include "lib_common_utils.hpp"

#include <oneapi/mkl.hpp>
#include <sycl/sycl.hpp>

namespace dpct {
namespace lapack {
/// Computes all eigenvalues and, optionally, eigenvectors of a real generalized
/// symmetric definite eigenproblem using a divide and conquer method.
/// \return Returns 0 if no synchronous exception, otherwise returns 1.
/// \param [in] queue Device queue where calculations will be performed.
/// \param [in] itype Must be 1 or 2 or 3. Specifies the problem type to be solved.
/// \param [in] jobz Must be job::novec or job::vec.
/// \param [in] uplo Must be uplo::upper or uplo::lower.
/// \param [in] n The order of the matrices A and B.
/// \param [in,out] a The symmetric matrix A.
/// \param [in] lda The leading dimension of matrix A.
/// \param [in,out] b The symmetric matrix B.
/// \param [in] ldb The leading dimension of matrix B.
/// \param [out] w Eigenvalues.
/// \param [in] scratchpad Scratchpad memory to be used by the routine
/// for storing intermediate results.
/// \param [in] scratchpad_size Size of scratchpad memory as a number of
/// floating point elements of type T.
/// \param [out] info If lapack synchronous exception is caught, the value
/// returned from info() method of the exception is set to \p info.
template <typename T>
inline int sygvd(sycl::queue &queue, std::int64_t itype, oneapi::mkl::job jobz,
                 oneapi::mkl::uplo uplo, int n, T *a, int lda, T *b, int ldb,
                 T *w, T *scratchpad, int scratchpad_size, int *info) {
#ifdef DPCT_USM_LEVEL_NONE
  auto info_buf = get_buffer<int>(info);
  auto a_buffer = get_buffer<T>(a);
  auto b_buffer = get_buffer<T>(b);
  auto w_buffer = get_buffer<T>(w);
  auto scratchpad_buffer = get_buffer<T>(scratchpad);
  int info_val = 0;
  int ret_val = 0;
  try {
    oneapi::mkl::lapack::sygvd(queue, itype, jobz, uplo, n, a_buffer, lda,
                               b_buffer, ldb, w_buffer, scratchpad_buffer,
                               scratchpad_size);
  } catch (oneapi::mkl::lapack::exception const& e) {
    std::cerr << "Unexpected exception caught during call to LAPACK API: sygvd"
              << std::endl
              << "reason: " << e.what() << std::endl
              << "info: " << e.info() << std::endl;
    info_val = static_cast<int>(e.info());
    ret_val = 1;
  } catch (sycl::exception const& e) {
    std::cerr << "Caught synchronous SYCL exception:" << std::endl
              << "reason: " << e.what() << std::endl;
    ret_val = 1;
  }
  queue.submit([&, info_val](sycl::handler &cgh) {
    auto info_acc = info_buf.get_access<sycl::access_mode::write>(cgh);
    cgh.single_task<dpct_kernel_name<class sygvd_set_info, T>>(
        [=]() { info_acc[0] = info_val; });
  });
  return ret_val;
#else
  try {
    oneapi::mkl::lapack::sygvd(queue, itype, jobz, uplo, n, a, lda, b, ldb, w,
                               scratchpad, scratchpad_size);
  } catch (oneapi::mkl::lapack::exception const& e) {
    std::cerr << "Unexpected exception caught during call to LAPACK API: sygvd"
              << std::endl
              << "reason: " << e.what() << std::endl
              << "info: " << e.info() << std::endl;
    int info_val = static_cast<int>(e.info());
    queue.memcpy(info, &info_val, sizeof(int)).wait();
    return 1;
  } catch (sycl::exception const& e) {
    std::cerr << "Caught synchronous SYCL exception:" << std::endl
              << "reason: " << e.what() << std::endl;
    queue.memset(info, 0, sizeof(int)).wait();
    return 1;
  }
  queue.memset(info, 0, sizeof(int));
  return 0;
#endif
}
/// Computes all the eigenvalues, and optionally, the eigenvectors of a complex
/// generalized Hermitian positive-definite eigenproblem using a divide and
/// conquer method.
/// \return Returns 0 if no synchronous exception, otherwise returns 1.
/// \param [in] queue Device queue where calculations will be performed.
/// \param [in] itype Must be 1 or 2 or 3. Specifies the problem type to be solved.
/// \param [in] jobz Must be job::novec or job::vec.
/// \param [in] uplo Must be uplo::upper or uplo::lower.
/// \param [in] n The order of the matrices A and B.
/// \param [in,out] a The Hermitian matrix A.
/// \param [in] lda The leading dimension of matrix A.
/// \param [in,out] b The Hermitian matrix B.
/// \param [in] ldb The leading dimension of matrix B.
/// \param [in] w Eigenvalues.
/// \param [in] scratchpad Scratchpad memory to be used by the routine
/// for storing intermediate results.
/// \param [in] scratchpad_size Size of scratchpad memory as a number of
/// floating point elements of type T.
/// \param [out] info If lapack synchronous exception is caught, the value
/// returned from info() method of the exception is set to \p info.
template <typename T, typename Tw>
inline int hegvd(sycl::queue &queue, std::int64_t itype, oneapi::mkl::job jobz,
                 oneapi::mkl::uplo uplo, int n, T *a, int lda, T *b, int ldb,
                 Tw *w, T *scratchpad, int scratchpad_size, int *info) {
  using Ty = typename DataType<T>::T2;
#ifdef DPCT_USM_LEVEL_NONE
  auto info_buf = get_buffer<int>(info);
  auto a_buffer = get_buffer<Ty>(a);
  auto b_buffer = get_buffer<Ty>(b);
  auto w_buffer = get_buffer<Tw>(w);
  auto scratchpad_buffer = get_buffer<Ty>(scratchpad);
  int info_val = 0;
  int ret_val = 0;
  try {
    oneapi::mkl::lapack::hegvd(queue, itype, jobz, uplo, n, a_buffer, lda,
                               b_buffer, ldb, w_buffer, scratchpad_buffer,
                               scratchpad_size);
  } catch (oneapi::mkl::lapack::exception const& e) {
    std::cerr << "Unexpected exception caught during call to LAPACK API: hegvd"
              << std::endl
              << "reason: " << e.what() << std::endl
              << "info: " << e.info() << std::endl;
    info_val = static_cast<int>(e.info());
    ret_val = 1;
  } catch (sycl::exception const& e) {
    std::cerr << "Caught synchronous SYCL exception:" << std::endl
              << "reason: " << e.what() << std::endl;
    ret_val = 1;
  }
  queue.submit([&, info_val](sycl::handler &cgh) {
    auto info_acc = info_buf.get_access<sycl::access_mode::write>(cgh);
    cgh.single_task<dpct_kernel_name<class hegvd_set_info, T>>(
        [=]() { info_acc[0] = info_val; });
  });
  return ret_val;
#else
  try {
    oneapi::mkl::lapack::hegvd(queue, itype, jobz, uplo, n, (Ty *)a, lda, (Ty *)b,
                               ldb, w, (Ty *)scratchpad, scratchpad_size);
  } catch (oneapi::mkl::lapack::exception const& e) {
    std::cerr << "Unexpected exception caught during call to LAPACK API: hegvd"
              << std::endl
              << "reason: " << e.what() << std::endl
              << "info: " << e.info() << std::endl;
    int info_val = static_cast<int>(e.info());
    queue.memcpy(info, &info_val, sizeof(int)).wait();
    return 1;
  } catch (sycl::exception const& e) {
    std::cerr << "Caught synchronous SYCL exception:" << std::endl
              << "reason: " << e.what() << std::endl;
    queue.memset(info, 0, sizeof(int)).wait();
    return 1;
  }
  queue.memset(info, 0, sizeof(int));
  return 0;
#endif
}
/// Computes the Cholesky factorizations of a batch of symmetric (or Hermitian,
/// for complex data) positive-definite matrices.
/// \return Returns 0 if no synchronous exception, otherwise returns 1.
/// \param [in] queue Device queue where calculations will be performed.
/// \param [in] uplo Must be uplo::upper or uplo::lower.
/// \param [in] n The order of the matrix A.
/// \param [in,out] a Array of pointers to matrix A.
/// \param [in] lda The leading dimension of matrix A.
/// \param [out] info If lapack synchronous exception is caught, the value
/// returned from info() method of the exception is set to \p info.
/// \param [in] group_size The batch size.
template <typename T>
inline int potrf_batch(sycl::queue &queue, oneapi::mkl::uplo uplo, int n,
                       T *a[], int lda, int *info, int group_size) {
#ifdef DPCT_USM_LEVEL_NONE
  throw std::runtime_error("this API is unsupported when USM level is none");
#else
  using Ty = typename DataType<T>::T2;
  struct matrix_info_t {
    oneapi::mkl::uplo uplo_info;
    std::int64_t n_info;
    std::int64_t lda_info;
    std::int64_t group_size_info;
  };
  matrix_info_t *matrix_info =
      (matrix_info_t *)std::malloc(sizeof(matrix_info_t));
  matrix_info->uplo_info = uplo;
  matrix_info->n_info = n;
  matrix_info->lda_info = lda;
  matrix_info->group_size_info = group_size;
  std::int64_t scratchpad_size = 0;
  sycl::event e;
  Ty *scratchpad = nullptr;
  try {
    scratchpad_size = oneapi::mkl::lapack::potrf_batch_scratchpad_size<Ty>(
        queue, &(matrix_info->uplo_info), &(matrix_info->n_info),
        &(matrix_info->lda_info), 1, &(matrix_info->group_size_info));
    scratchpad = sycl::malloc_device<Ty>(scratchpad_size, queue);
    e = oneapi::mkl::lapack::potrf_batch(
        queue, &(matrix_info->uplo_info), &(matrix_info->n_info), (Ty **)a,
        &(matrix_info->lda_info), 1, &(matrix_info->group_size_info),
        scratchpad, scratchpad_size);
  } catch (oneapi::mkl::lapack::batch_error const &be) {
    std::cerr << "Unexpected exception caught during call to LAPACK API: "
                 "potrf_batch_scratchpad_size/potrf_batch"
              << std::endl
              << "reason: " << be.what() << std::endl
              << "number: " << be.info() << std::endl;
    int i = 0;
    auto &ids = be.ids();
    std::vector<int> info_vec(group_size);
    for (auto const &e : be.exceptions()) {
      try {
        std::rethrow_exception(e);
      } catch (oneapi::mkl::lapack::exception &e) {
        std::cerr << "Exception " << ids[i] << std::endl
                  << "reason: " << e.what() << std::endl
                  << "info: " << e.info() << std::endl;
        info_vec[i] = e.info();
        i++;
      }
    }
    queue.memcpy(info, info_vec.data(), group_size * sizeof(int)).wait();
    std::free(matrix_info);
    if (scratchpad)
      sycl::free(scratchpad, queue);
    return 1;
  } catch (sycl::exception const &e) {
    std::cerr << "Caught synchronous SYCL exception:" << std::endl
              << "reason: " << e.what() << std::endl;
    queue.memset(info, 0, group_size * sizeof(int)).wait();
    std::free(matrix_info);
    if (scratchpad)
      sycl::free(scratchpad, queue);
    return 1;
  }
  queue.submit([&](sycl::handler &cgh) {
    cgh.depends_on(e);
    cgh.host_task([=] {
      std::free(matrix_info);
      sycl::free(scratchpad, queue);
    });
  });
  queue.memset(info, 0, group_size * sizeof(int));
  return 0;
#endif
}
/// Solves a batch of systems of linear equations with a Cholesky-factored
/// symmetric (Hermitian) positive-definite coefficient matrices.
/// \return Returns 0 if no synchronous exception, otherwise returns 1.
/// \param [in] queue Device queue where calculations will be performed.
/// \param [in] uplo Must be uplo::upper or uplo::lower.
/// \param [in] n The order of the matrix A.
/// \param [in] nrhs The number of right-hand sides.
/// \param [in,out] a Array of pointers to matrix A.
/// \param [in] lda The leading dimension of matrix A.
/// \param [in,out] b Array of pointers to matrix B.
/// \param [in] ldb The leading dimension of matrix B.
/// \param [out] info If lapack synchronous exception is caught, the value
/// returned from info() method of the exception is set to \p info.
/// \param [in] group_size The batch size.
template <typename T>
inline int potrs_batch(sycl::queue &queue, oneapi::mkl::uplo uplo, int n,
                       int nrhs, T *a[], int lda, T *b[], int ldb, int *info,
                       int group_size) {
#ifdef DPCT_USM_LEVEL_NONE
  throw std::runtime_error("this API is unsupported when USM level is none");
#else
  using Ty = typename DataType<T>::T2;
  struct matrix_info_t {
    oneapi::mkl::uplo uplo_info;
    std::int64_t n_info;
    std::int64_t nrhs_info;
    std::int64_t lda_info;
    std::int64_t ldb_info;
    std::int64_t group_size_info;
  };
  matrix_info_t *matrix_info =
      (matrix_info_t *)std::malloc(sizeof(matrix_info_t));
  matrix_info->uplo_info = uplo;
  matrix_info->n_info = n;
  matrix_info->nrhs_info = nrhs;
  matrix_info->lda_info = lda;
  matrix_info->ldb_info = ldb;
  matrix_info->group_size_info = group_size;
  std::int64_t scratchpad_size = 0;
  sycl::event e;
  Ty *scratchpad = nullptr;
  try {
    scratchpad_size = oneapi::mkl::lapack::potrs_batch_scratchpad_size<Ty>(
        queue, &(matrix_info->uplo_info), &(matrix_info->n_info),
        &(matrix_info->nrhs_info), &(matrix_info->lda_info),
        &(matrix_info->ldb_info), 1, &(matrix_info->group_size_info));
    scratchpad = sycl::malloc_device<Ty>(scratchpad_size, queue);
    e = oneapi::mkl::lapack::potrs_batch(
        queue, &(matrix_info->uplo_info), &(matrix_info->n_info),
        &(matrix_info->nrhs_info), (Ty **)a, &(matrix_info->lda_info), (Ty **)b,
        &(matrix_info->ldb_info), 1, &(matrix_info->group_size_info),
        scratchpad, scratchpad_size);
  } catch (oneapi::mkl::lapack::batch_error const &be) {
    std::cerr << "Unexpected exception caught during call to LAPACK API: "
                 "potrs_batch_scratchpad_size/potrs_batch"
              << std::endl
              << "reason: " << be.what() << std::endl
              << "number: " << be.info() << std::endl;
    int i = 0;
    auto &ids = be.ids();
    std::vector<int> info_vec(group_size);
    for (auto const &e : be.exceptions()) {
      try {
        std::rethrow_exception(e);
      } catch (oneapi::mkl::lapack::exception &e) {
        std::cerr << "Exception " << ids[i] << std::endl
                  << "reason: " << e.what() << std::endl
                  << "info: " << e.info() << std::endl;
        info_vec[i] = e.info();
        i++;
      }
    }
    queue.memcpy(info, info_vec.data(), group_size * sizeof(int)).wait();
    std::free(matrix_info);
    if (scratchpad)
      sycl::free(scratchpad, queue);
    return 1;
  } catch (sycl::exception const &e) {
    std::cerr << "Caught synchronous SYCL exception:" << std::endl
              << "reason: " << e.what() << std::endl;
    queue.memset(info, 0, group_size * sizeof(int)).wait();
    std::free(matrix_info);
    if (scratchpad)
      sycl::free(scratchpad, queue);
    return 1;
  }
  queue.submit([&](sycl::handler &cgh) {
    cgh.depends_on(e);
    cgh.host_task([=] {
      std::free(matrix_info);
      sycl::free(scratchpad, queue);
    });
  });
  queue.memset(info, 0, group_size * sizeof(int));
  return 0;
#endif
}

namespace detail {
template <bool free_ws_in_catch, typename func_t, typename... args_t>
inline int handle_sync_exception(sycl::queue &q, void *const &device_ws,
                                 int *info, std::string lapack_api_name,
                                 func_t func, args_t... args) {
  try {
    func(args...);
  } catch (oneapi::mkl::lapack::exception const &e) {
    std::cerr << "Unexpected exception caught during call to LAPACK API: "
              << lapack_api_name << std::endl
              << "reason: " << e.what() << std::endl
              << "info: " << e.info() << std::endl
              << "detail: " << e.detail() << std::endl;
    int info_val = static_cast<int>(e.info());
    if (info)
      dpct::detail::dpct_memcpy(q, info, &info_val, sizeof(int),
                                memcpy_direction::host_to_device)
          .wait();
    if constexpr (free_ws_in_catch) {
      if (device_ws)
        dpct::dpct_free(device_ws, q);
    }
    return 1;
  } catch (sycl::exception const &e) {
    std::cerr << "Caught synchronous SYCL exception:" << std::endl
              << "reason: " << e.what() << std::endl;
    if (info)
      dpct::detail::dpct_memset(q, info, 0, sizeof(int)).wait();
    if constexpr (free_ws_in_catch) {
      if (device_ws)
        dpct::dpct_free(device_ws, q);
    }
    return 1;
  }
  return 0;
}

inline void getrf_scratchpad_size_impl(sycl::queue &q, std::int64_t m,
                                       std::int64_t n, library_data_t a_type,
                                       std::int64_t lda,
                                       size_t *device_ws_size) {
#define CASE(TYPE_NAME, TYPE)                                                  \
  case TYPE_NAME: {                                                            \
    *device_ws_size =                                                          \
        oneapi::mkl::lapack::getrf_scratchpad_size<TYPE>(q, m, n, lda) *       \
        sizeof(TYPE);                                                          \
    break;                                                                     \
  }
  switch (a_type) {
    CASE(library_data_t::real_float, float)
    CASE(library_data_t::real_double, double)
    CASE(library_data_t::complex_float, std::complex<float>)
    CASE(library_data_t::complex_double, std::complex<double>)
  default:
    throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                          "the data type is unsupported");
  }
#undef CASE
}

inline void getrf_impl(sycl::queue &q, std::int64_t m, std::int64_t n,
                       library_data_t a_type, void *a, std::int64_t lda,
                       std::int64_t *ipiv, void *device_ws,
                       size_t device_ws_size, int *info) {
  auto ipiv_data = dpct::detail::get_memory(ipiv);

#define CASE(TYPE_NAME, TYPE)                                                  \
  case TYPE_NAME: {                                                            \
    auto a_data = dpct::detail::get_memory(reinterpret_cast<TYPE *>(a));       \
    auto device_ws_data =                                                      \
        dpct::detail::get_memory(reinterpret_cast<TYPE *>(device_ws));         \
    oneapi::mkl::lapack::getrf(q, m, n, a_data, lda, ipiv_data,                \
                               device_ws_data, device_ws_size / sizeof(TYPE)); \
    break;                                                                     \
  }

  switch (a_type) {
    CASE(library_data_t::real_float, float)
    CASE(library_data_t::real_double, double)
    CASE(library_data_t::complex_float, std::complex<float>)
    CASE(library_data_t::complex_double, std::complex<double>)
  default:
    throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                          "the data type is unsupported");
  }

#undef CASE
  dpct::detail::dpct_memset(q, info, 0, sizeof(int));
}

inline void getrs_impl(sycl::queue &q, oneapi::mkl::transpose trans,
                       std::int64_t n, std::int64_t nrhs, library_data_t a_type,
                       void *a, std::int64_t lda, std::int64_t *ipiv,
                       library_data_t b_type, void *b, std::int64_t ldb,
                       void *&device_ws, int *info) {
  auto ipiv_data = dpct::detail::get_memory(ipiv);

#define CASE(TYPE_NAME, TYPE)                                                  \
  case TYPE_NAME: {                                                            \
    std::int64_t device_ws_size =                                              \
        oneapi::mkl::lapack::getrs_scratchpad_size<TYPE>(q, trans, n, nrhs,    \
                                                         lda, ldb);            \
    device_ws = dpct::detail::dpct_malloc(device_ws_size * sizeof(TYPE), q);   \
    auto device_ws_data =                                                      \
        dpct::detail::get_memory(reinterpret_cast<TYPE *>(device_ws));         \
    auto a_data = dpct::detail::get_memory(reinterpret_cast<TYPE *>(a));       \
    auto b_data = dpct::detail::get_memory(reinterpret_cast<TYPE *>(b));       \
    oneapi::mkl::lapack::getrs(q, trans, n, nrhs, a_data, lda, ipiv_data,      \
                               b_data, ldb, device_ws_data, device_ws_size);   \
    break;                                                                     \
  }

  switch (a_type) {
    CASE(library_data_t::real_float, float)
    CASE(library_data_t::real_double, double)
    CASE(library_data_t::complex_float, std::complex<float>)
    CASE(library_data_t::complex_double, std::complex<double>)
  default:
    throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                          "the data type is unsupported");
  }

#undef CASE
  sycl::event e = dpct::detail::dpct_memset(q, info, 0, sizeof(int));
  if (device_ws) {
#ifdef DPCT_USM_LEVEL_NONE
    dpct::detail::mem_mgr::instance().mem_free(device_ws);
#else
    dpct::async_dpct_free({device_ws}, {e}, q);
#endif
  }
}

inline void geqrf_scratchpad_size_impl(sycl::queue &q, std::int64_t m,
                                       std::int64_t n, library_data_t a_type,
                                       std::int64_t lda,
                                       size_t *device_ws_size) {
#define CASE(TYPE_NAME, TYPE)                                                  \
  case TYPE_NAME: {                                                            \
    *device_ws_size =                                                          \
        oneapi::mkl::lapack::geqrf_scratchpad_size<TYPE>(q, m, n, lda) *       \
        sizeof(TYPE);                                                          \
    break;                                                                     \
  }
  switch (a_type) {
    CASE(library_data_t::real_float, float)
    CASE(library_data_t::real_double, double)
    CASE(library_data_t::complex_float, std::complex<float>)
    CASE(library_data_t::complex_double, std::complex<double>)
  default:
    throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                          "the data type is unsupported");
  }
#undef CASE
}

inline void geqrf_impl(sycl::queue &q, std::int64_t m, std::int64_t n,
                       library_data_t a_type, void *a, std::int64_t lda,
                       library_data_t tau_type, void *tau, void *device_ws,
                       size_t device_ws_size, int *info) {
#define CASE(TYPE_NAME, TYPE)                                                  \
  case TYPE_NAME: {                                                            \
    auto a_data = dpct::detail::get_memory(reinterpret_cast<TYPE *>(a));       \
    auto tau_data = dpct::detail::get_memory(reinterpret_cast<TYPE *>(tau));   \
    auto device_ws_data =                                                      \
        dpct::detail::get_memory(reinterpret_cast<TYPE *>(device_ws));         \
    oneapi::mkl::lapack::geqrf(q, m, n, a_data, lda, tau_data, device_ws_data, \
                               device_ws_size / sizeof(TYPE));                 \
    break;                                                                     \
  }
  switch (a_type) {
    CASE(library_data_t::real_float, float)
    CASE(library_data_t::real_double, double)
    CASE(library_data_t::complex_float, std::complex<float>)
    CASE(library_data_t::complex_double, std::complex<double>)
  default:
    throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                          "the data type is unsupported");
  }
#undef CASE
  dpct::detail::dpct_memset(q, info, 0, sizeof(int));
}

inline int getrfnp_impl(sycl::queue &q, std::int64_t m, std::int64_t n,
                        library_data_t a_type, void *a, std::int64_t lda,
                        void *device_ws, size_t device_ws_size, int *info) {
#define CASE(TYPE_NAME, TYPE)                                                  \
  case TYPE_NAME: {                                                            \
    std::int64_t a_stride = m * lda;                                           \
    auto a_data = dpct::detail::get_memory(reinterpret_cast<TYPE *>(a));       \
    auto device_ws_data =                                                      \
        dpct::detail::get_memory(reinterpret_cast<TYPE *>(device_ws));         \
    oneapi::mkl::lapack::getrfnp_batch(q, m, n, a_data, lda, a_stride, 1,      \
                                       device_ws_data,                         \
                                       device_ws_size / sizeof(TYPE));         \
    break;                                                                     \
  }

  try {
    switch (a_type) {
      CASE(library_data_t::real_float, float)
      CASE(library_data_t::real_double, double)
      CASE(library_data_t::complex_float, std::complex<float>)
      CASE(library_data_t::complex_double, std::complex<double>)
    default:
      throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                            "the data type is unsupported");
    }
  } catch (oneapi::mkl::lapack::batch_error const &be) {
    try {
      std::rethrow_exception(be.exceptions()[0]);
    } catch (oneapi::mkl::lapack::exception &e) {
      std::cerr << "Unexpected exception caught during call to LAPACK API: "
                   "getrfnp_batch"
                << std::endl
                << "reason: " << e.what() << std::endl
                << "number: " << e.info() << std::endl;
      int info_val = static_cast<int>(e.info());
      dpct::detail::dpct_memcpy(q, info, &info_val, sizeof(int),
                                memcpy_direction::host_to_device)
          .wait();
      return 1;
    }
  } catch (sycl::exception const &e) {
    std::cerr << "Caught synchronous SYCL exception:" << std::endl
              << "reason: " << e.what() << std::endl;
    dpct::detail::dpct_memset(q, info, 0, sizeof(int)).wait();
    return 1;
  }

#undef CASE
  dpct::detail::dpct_memset(q, info, 0, sizeof(int));
  return 0;
}
} // namespace detail

/// Computes the size of workspace memory of getrf function.
/// \return Returns 0 if no synchronous exception, otherwise returns 1.
/// \param [in] q Device queue where computation will be performed.
/// \param [in] m The number of rows in the matrix A.
/// \param [in] n The number of columns in the matrix A.
/// \param [in] a_type The data type of the matrix A.
/// \param [in] lda The leading dimension of the matrix A.
/// \param [out] device_ws_size The workspace size in bytes.
inline int getrf_scratchpad_size(sycl::queue &q, std::int64_t m, std::int64_t n,
                                 library_data_t a_type, std::int64_t lda,
                                 size_t *device_ws_size) {
  return detail::handle_sync_exception<false>(
      q, nullptr, nullptr, "getrf_scratchpad_size",
      detail::getrf_scratchpad_size_impl, q, m, n, a_type, lda, device_ws_size);
}

/// Computes the LU factorization of a general m-by-n matrix.
/// \param [in] q The queue where the routine should be executed.
/// \param [in] m The number of rows in the matrix A.
/// \param [in] n The number of columns in the matrix A.
/// \param [in] a_type The data type of the matrix A.
/// \param [in, out] a The input matrix A. Overwritten by L and U. The unit
/// diagonal elements of L are not stored.
/// \param [in] lda The leading dimension of the matrix A.
/// \param [out] ipiv The pivot indices. If \p ipiv is nullptr, non-pivoting
/// LU factorization is computed.
/// \param [in] device_ws The workspace.
/// \param [in] device_ws_size The workspace size in bytes.
/// \param [out] info If lapack synchronous exception is caught, the value
/// returned from info() method of the exception is set to \p info.
inline int getrf(sycl::queue &q, std::int64_t m, std::int64_t n,
                 library_data_t a_type, void *a, std::int64_t lda,
                 std::int64_t *ipiv, void *device_ws, size_t device_ws_size,
                 int *info) {
  if (ipiv == nullptr) {
    return detail::getrfnp_impl(q, m, n, a_type, a, lda, device_ws,
                                device_ws_size, info);
  }
  return detail::handle_sync_exception<false>(
      q, nullptr, info, "getrf", detail::getrf_impl, q, m, n, a_type, a, lda,
      ipiv, device_ws, device_ws_size, info);
}

/// Solves a system of linear equations with a LU-factored square coefficient
/// matrix, with multiple right-hand sides.
/// \param [in] q The queue where the routine should be executed.
/// \param [in] trans Indicates the form of the linear equation.
/// \param [in] n The order of the matrix A and the number of rows in matrix B.
/// \param [in] nrhs The number of right hand sides.
/// \param [in] a_type The data type of the matrix A.
/// \param [in] a The input matrix A.
/// \param [in] lda The leading dimension of the matrix A.
/// \param [in] ipiv The pivot indices.
/// \param [in] b_type The data type of the matrix B.
/// \param [in, out] b The matrix B, whose columns are the right-hand sides
/// for the systems of equations.
/// \param [in] ldb The leading dimension of the matrix B.
/// \param [out] info If lapack synchronous exception is caught, the value
/// returned from info() method of the exception is set to \p info.
inline int getrs(sycl::queue &q, oneapi::mkl::transpose trans, std::int64_t n,
                 std::int64_t nrhs, library_data_t a_type, void *a,
                 std::int64_t lda, std::int64_t *ipiv, library_data_t b_type,
                 void *b, std::int64_t ldb, int *info) {
  void *device_ws = nullptr;
  return detail::handle_sync_exception<true>(
      q, device_ws, info, "getrs_scratchpad_size/getrs", detail::getrs_impl, q,
      trans, n, nrhs, a_type, a, lda, ipiv, b_type, b, ldb, device_ws, info);
}

/// Computes the size of workspace memory of geqrf function.
/// \return Returns 0 if no synchronous exception, otherwise returns 1.
/// \param [in] q Device queue where computation will be performed.
/// \param [in] m The number of rows in the matrix A.
/// \param [in] n The number of columns in the matrix A.
/// \param [in] a_type The data type of the matrix A.
/// \param [in] lda The leading dimension of the matrix A.
/// \param [out] device_ws_size The workspace size in bytes.
inline int geqrf_scratchpad_size(sycl::queue &q, std::int64_t m, std::int64_t n,
                                 library_data_t a_type, std::int64_t lda,
                                 size_t *device_ws_size) {
  return detail::handle_sync_exception<false>(
      q, nullptr, nullptr, "geqrf_scratchpad_size",
      detail::geqrf_scratchpad_size_impl, q, m, n, a_type, lda, device_ws_size);
}

/// Computes the QR factorization of a general m-by-n matrix.
/// \param [in] q The queue where the routine should be executed.
/// \param [in] m The number of rows in the matrix A.
/// \param [in] n The number of columns in the matrix A.
/// \param [in] a_type The data type of the matrix A.
/// \param [in, out] a The input matrix A. Overwritten by the factorization data.
/// \param [in] lda The leading dimension of the matrix A.
/// \param [in] tau_type The data type of the array tau.
/// \param [in] tau The array contains scalars that define elementary reflectors
/// for the matrix Q in its decomposition in a product of elementary reflectors.
/// \param [in] device_ws The workspace.
/// \param [in] device_ws_size The workspace size in bytes.
/// \param [out] info If lapack synchronous exception is caught, the value
/// returned from info() method of the exception is set to \p info.
inline int geqrf(sycl::queue &q, std::int64_t m, std::int64_t n,
                 library_data_t a_type, void *a, std::int64_t lda,
                 library_data_t tau_type, void *tau, void *device_ws,
                 size_t device_ws_size, int *info) {
  return detail::handle_sync_exception<false>(
      q, nullptr, info, "geqrf", detail::geqrf_impl, q, m, n, a_type, a, lda,
      tau_type, tau, device_ws, device_ws_size, info);
}

} // namespace lapack
} // namespace dpct

#endif // __DPCT_LAPACK_UTILS_HPP__
