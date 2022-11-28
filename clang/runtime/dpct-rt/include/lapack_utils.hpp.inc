// DPCT_LABEL_BEGIN|License|
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
//==---- lapack_utils.hpp -------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
// DPCT_LABEL_END

#ifndef __DPCT_LAPACK_UTILS_HPP__
#define __DPCT_LAPACK_UTILS_HPP__
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
#include "memory.hpp"
#include "util.hpp"

// DPCT_LABEL_BEGIN|non_local_include_dependency|
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
#include <oneapi/mkl.hpp>
#include <sycl/sycl.hpp>
// DPCT_LABEL_END

namespace dpct {
namespace lapack {
// DPCT_LABEL_BEGIN|sygvd|dpct::lapack
// DPCT_DEPENDENCY_BEGIN
// Memory|get_buffer_T|UsmNone
// DPCT_DEPENDENCY_END
// DPCT_CODE
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
/// returned from info() method of the exception is set to \p info. If other
/// SYCL synchronous exception is caught, -1 is set to \p info.
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
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|hegvd|dpct::lapack
// DPCT_DEPENDENCY_BEGIN
// Memory|get_buffer_T|UsmNone
// Util|DataType
// DPCT_DEPENDENCY_END
// DPCT_CODE
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
/// returned from info() method of the exception is set to \p info. If other
/// SYCL synchronous exception is caught, -1 is set to \p info.
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
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|potrf_batch|dpct::lapack
// DPCT_DEPENDENCY_BEGIN
// Util|DataType|UsmRestricted
// DPCT_DEPENDENCY_END
// DPCT_CODE
/// Computes the Cholesky factorizations of a batch of symmetric (or Hermitian,
/// for complex data) positive-definite matrices.
/// \return Returns 0 if no synchronous exception, otherwise returns 1.
/// \param [in] queue Device queue where calculations will be performed.
/// \param [in] uplo Must be uplo::upper or uplo::lower.
/// \param [in] n The order of the matrix A.
/// \param [in,out] a Array of pointers to matrix A.
/// \param [in] lda The leading dimension of matrix A.
/// \param [out] info If lapack synchronous exception is caught, the value
/// returned from info() method of the exception is set to \p info. If other
/// SYCL synchronous exception is caught, -1 is set to \p info.
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
  } catch (oneapi::mkl::lapack::exception const &e) {
    std::cerr << "Unexpected exception caught during call to LAPACK API: "
                 "potrf_batch_scratchpad_size/potrf_batch"
              << std::endl
              << "reason: " << e.what() << std::endl
              << "info: " << e.info() << std::endl;
    std::vector<int> info_vec(group_size, static_cast<int>(e.info()));
    queue.memcpy(info, info_vec.data(), group_size * sizeof(int)).wait();
    if (scratchpad)
      sycl::free(scratchpad, queue);
    return 1;
  } catch (sycl::exception const &e) {
    std::cerr << "Caught synchronous SYCL exception:" << std::endl
              << "reason: " << e.what() << std::endl;
    queue.memset(info, 0, group_size * sizeof(int)).wait();
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
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|potrs_batch|dpct::lapack
// DPCT_DEPENDENCY_BEGIN
// Util|DataType|UsmRestricted
// DPCT_DEPENDENCY_END
// DPCT_CODE
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
/// returned from info() method of the exception is set to \p info. If other
/// SYCL synchronous exception is caught, -1 is set to \p info.
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
  } catch (oneapi::mkl::lapack::exception const &e) {
    std::cerr << "Unexpected exception caught during call to LAPACK API: "
                 "potrs_batch_scratchpad_size/potrs_batch"
              << std::endl
              << "reason: " << e.what() << std::endl
              << "info: " << e.info() << std::endl;
    std::vector<int> info_vec(group_size, static_cast<int>(e.info()));
    queue.memcpy(info, info_vec.data(), group_size * sizeof(int)).wait();
    if (scratchpad)
      sycl::free(scratchpad, queue);
    return 1;
  } catch (sycl::exception const &e) {
    std::cerr << "Caught synchronous SYCL exception:" << std::endl
              << "reason: " << e.what() << std::endl;
    queue.memset(info, 0, group_size * sizeof(int)).wait();
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
// DPCT_LABEL_END
} // namespace lapack
} // namespace dpct

#endif // __DPCT_LAPACK_UTILS_HPP__
