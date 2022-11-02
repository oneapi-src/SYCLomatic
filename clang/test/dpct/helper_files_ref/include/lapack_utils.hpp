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

#include <oneapi/mkl.hpp>
#include <sycl/sycl.hpp>

namespace dpct {
namespace lapack {
/// Computes all eigenvalues and, optionally, eigenvectors of a real generalized
/// symmetric definite eigenproblem using a divide and conquer method.
/// \param [in] queue Device queue where calculations will be performed.
/// \param [in] itype Must be 1 or 2 or 3. Specifies the problem type to be solved.
/// \param [in] jobz Must be job::novec or job::vec.
/// \param [in] uplo Must be uplo::upper or uplo::lower.
/// \param [in] n The order of the matrices A and B.
/// \param [in,out] a The symmetric matrix A.
/// \param [in] lda The leading dimension of matrix A.
/// \param [in,out] b The symmetric matrix B.
/// \param [in] ldb The leading dimension of matrix B.
/// \param [out] w Eigen values.
/// \param [in] scratchpad Scratchpad memory to be used by the routine
/// for storing intermediate results.
/// \param [in] scratchpad_size Size of scratchpad memory as a number of
/// floating point elements of type T.
/// \param [out] info The memory pointed by \p info is set to 0.
template <typename T>
inline void sygvd(sycl::queue &queue, std::int64_t itype, oneapi::mkl::job jobz,
                  oneapi::mkl::uplo uplo, int n, T *a, int lda, T *b, int ldb,
                  T *w, T *scratchpad, int scratchpad_size, int *info) {
#ifdef DPCT_USM_LEVEL_NONE
  auto a_buffer = get_buffer<T>(a);
  auto b_buffer = get_buffer<T>(b);
  auto w_buffer = get_buffer<T>(w);
  auto scratchpad_buffer = get_buffer<T>(scratchpad);
  oneapi::mkl::lapack::sygvd(queue, itype, jobz, uplo, n, a_buffer, lda,
                             b_buffer, ldb, w_buffer, scratchpad_buffer,
                             scratchpad_size);
  auto info_buf = get_buffer<int>(info);
  queue.submit([&](sycl::handler &cgh) {
    auto info_acc = info_buf.get_access<sycl::access_mode::write>(cgh);
    cgh.single_task<dpct_kernel_name<class sygvd, T>>(
        [=]() { info_acc[0] = 0; });
  });
#else
  oneapi::mkl::lapack::sygvd(queue, itype, jobz, uplo, n, a, lda, b, ldb, w,
                             scratchpad, scratchpad_size);
  queue.memset(info, 0, sizeof(int));
#endif
}

/// Computes all the eigenvalues, and optionally, the eigenvectors of a complex
/// generalized Hermitian positive-definite eigenproblem using a divide and
/// conquer method.
/// \param [in] queue Device queue where calculations will be performed.
/// \param [in] itype Must be 1 or 2 or 3. Specifies the problem type to be solved.
/// \param [in] jobz Must be job::novec or job::vec.
/// \param [in] uplo Must be uplo::upper or uplo::lower.
/// \param [in] n The order of the matrices A and B.
/// \param [in,out] a The Hermitian matrix A.
/// \param [in] lda The leading dimension of matrix A.
/// \param [in,out] b The Hermitian matrix B.
/// \param [in] ldb The leading dimension of matrix B.
/// \param [out] w Eigen values.
/// \param [in] scratchpad Scratchpad memory to be used by the routine
/// for storing intermediate results.
/// \param [in] scratchpad_size Size of scratchpad memory as a number of
/// floating point elements of type T.
/// \param [out] info The memory pointed by \p info is set to 0.
template <typename T, typename Tw>
inline void hegvd(sycl::queue &queue, std::int64_t itype, oneapi::mkl::job jobz,
                  oneapi::mkl::uplo uplo, int n, T *a, int lda, T *b, int ldb,
                  Tw *w, T *scratchpad, int scratchpad_size, int *info) {
  using Ty = typename DataType<T>::T2;
#ifdef DPCT_USM_LEVEL_NONE
  auto a_buffer = get_buffer<Ty>(a);
  auto b_buffer = get_buffer<Ty>(b);
  auto w_buffer = get_buffer<Tw>(w);
  auto scratchpad_buffer = get_buffer<Ty>(scratchpad);
  oneapi::mkl::lapack::hegvd(queue, itype, jobz, uplo, n, a_buffer, lda,
                             b_buffer, ldb, w_buffer, scratchpad_buffer,
                             scratchpad_size);
  auto info_buf = get_buffer<int>(info);
  queue.submit([&](sycl::handler &cgh) {
    auto info_acc = info_buf.get_access<sycl::access_mode::write>(cgh);
    cgh.single_task<dpct_kernel_name<class hegvd, Ty>>(
        [=]() { info_acc[0] = 0; });
  });
#else
  oneapi::mkl::lapack::hegvd(queue, itype, jobz, uplo, n, (Ty *)a, lda, (Ty *)b,
                             ldb, w, (Ty *)scratchpad, scratchpad_size);
  queue.memset(info, 0, sizeof(int));
#endif
}

/// Computes the Cholesky factorizations of a batch of symmetric (or Hermitian,
/// for complex data) positive-definite matrices.
/// \param [in] queue Device queue where calculations will be performed.
/// \param [in] uplo Must be uplo::upper or uplo::lower.
/// \param [in] n The order of the matrix A.
/// \param [in,out] a Array of pointers to matrix A.
/// \param [in] lda The leading dimension of matrix A.
/// \param [out] info The memory pointed by \p info is set to 0.
/// \param [in] group_size The batch size.
template <typename T>
inline void potrf_batch(sycl::queue &queue, oneapi::mkl::uplo uplo, int n,
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
  std::int64_t scratchpad_size =
      oneapi::mkl::lapack::potrf_batch_scratchpad_size<Ty>(
          queue, &(matrix_info->uplo_info), &(matrix_info->n_info),
          &(matrix_info->lda_info), 1, &(matrix_info->group_size_info));
  Ty *scratchpad = sycl::malloc_device<Ty>(scratchpad_size, queue);
  sycl::event e = oneapi::mkl::lapack::potrf_batch(
      queue, &(matrix_info->uplo_info), &(matrix_info->n_info), (Ty **)a,
      &(matrix_info->lda_info), 1, &(matrix_info->group_size_info), scratchpad,
      scratchpad_size);
  queue.submit([&](sycl::handler &cgh) {
    cgh.depends_on(e);
    cgh.host_task([=] { std::free(matrix_info); });
  });
  queue.memset(info, 0, group_size * sizeof(int));
#endif
}

/// Solves a batch of systems of linear equations with a Cholesky-factored
/// symmetric (Hermitian) positive-definite coefficient matrices.
/// \param [in] queue Device queue where calculations will be performed.
/// \param [in] uplo Must be uplo::upper or uplo::lower.
/// \param [in] n The order of the matrix A.
/// \param [in] nrhs The number of right-hand sides.
/// \param [in,out] a Array of pointers to matrix A.
/// \param [in] lda The leading dimension of matrix A.
/// \param [in,out] b Array of pointers to matrix B.
/// \param [in] ldb The leading dimension of matrix B.
/// \param [out] info The memory pointed by \p info is set to 0.
/// \param [in] group_size The batch size.
template <typename T>
inline void potrs_batch(sycl::queue &queue, oneapi::mkl::uplo uplo, int n,
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
  std::int64_t scratchpad_size =
      oneapi::mkl::lapack::potrs_batch_scratchpad_size<Ty>(
          queue, &(matrix_info->uplo_info), &(matrix_info->n_info),
          &(matrix_info->nrhs_info), &(matrix_info->lda_info),
          &(matrix_info->ldb_info), 1, &(matrix_info->group_size_info));
  Ty *scratchpad = sycl::malloc_device<Ty>(scratchpad_size, queue);
  sycl::event e = oneapi::mkl::lapack::potrs_batch(
      queue, &(matrix_info->uplo_info), &(matrix_info->n_info),
      &(matrix_info->nrhs_info), (Ty **)a, &(matrix_info->lda_info), (Ty **)b,
      &(matrix_info->ldb_info), 1, &(matrix_info->group_size_info), scratchpad,
      scratchpad_size);
  queue.submit([&](sycl::handler &cgh) {
    cgh.depends_on(e);
    cgh.host_task([=] { std::free(matrix_info); });
  });
  queue.memset(info, 0, group_size * sizeof(int));
#endif
}
} // namespace lapack
} // namespace dpct

#endif // __DPCT_LAPACK_UTILS_HPP__
