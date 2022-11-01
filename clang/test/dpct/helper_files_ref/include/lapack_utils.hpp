//==---- lapack_utils.hpp -------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#include "memory.hpp"
#include "util.hpp"

#include <oneapi/mkl.hpp>
#include <sycl/sycl.hpp>

namespace dpct {
namespace lapack {

template <typename T>
inline void sygvd(sycl::queue &handle, std::int64_t itype,
                  oneapi::mkl::job jobz, oneapi::mkl::uplo uplo, int n, T *A,
                  int lda, T *B, int ldb, T *W, T *work, int lwork,
                  int *devInfo) {
#ifdef DPCT_USM_LEVEL_NONE
  auto A_buffer = get_buffer<T>(A);
  auto B_buffer = get_buffer<T>(B);
  auto W_buffer = get_buffer<T>(W);
  auto work_buffer = get_buffer<T>(work);
  oneapi::mkl::lapack::sygvd(handle, itype, jobz, uplo, n, A_buffer, lda,
                             B_buffer, ldb, W_buffer, work_buffer, lwork);
  auto devInfo_buf = get_buffer<int>(devInfo);
  handle.submit([&](sycl::handler &cgh) {
    auto devInfo_acc = devInfo_buf.get_access<sycl::access_mode::write>(cgh);
    cgh.single_task<dpct_kernel_name<class sygvd, T>>(
        [=]() { devInfo_acc[0] = 0; });
  });
#else
  oneapi::mkl::lapack::sygvd(handle, itype, jobz, uplo, n, A, lda, B, ldb, W,
                             work, lwork);
  handle.memset(devInfo, 0, sizeof(int));
#endif
}

template <typename T>
inline void hegvd(sycl::queue &handle, std::int64_t itype,
                  oneapi::mkl::job jobz, oneapi::mkl::uplo uplo, int n, T *A,
                  int lda, T *B, int ldb, float *W, T *work, int lwork,
                  int *devInfo) {
  using Ty = typename DataType<T>::T2;
#ifdef DPCT_USM_LEVEL_NONE
  auto A_buffer = get_buffer<Ty>(A);
  auto B_buffer = get_buffer<Ty>(B);
  auto W_buffer = get_buffer<Ty>(W);
  auto work_buffer = get_buffer<Ty>(work);
  oneapi::mkl::lapack::hegvd(handle, itype, jobz, uplo, n, A_buffer, lda,
                             B_buffer, ldb, W_buffer, work_buffer, lwork);
  auto devInfo_buf = get_buffer<int>(devInfo);
  handle.submit([&](sycl::handler &cgh) {
    auto devInfo_acc = devInfo_buf.get_access<sycl::access_mode::write>(cgh);
    cgh.single_task<dpct_kernel_name<class hegvd, Ty>>(
        [=]() { devInfo_acc[0] = 0; });
  });
#else
  oneapi::mkl::lapack::hegvd(handle, itype, jobz, uplo, n, (Ty *)A, lda,
                             (Ty *)B, ldb, (Ty *)W, (Ty *)work, lwork);
  handle.memset(devInfo, 0, sizeof(int));
#endif
}

template <typename T>
inline void potrf_batch(sycl::queue &handle, oneapi::mkl::uplo uplo, int n,
                        T *Aarray[], int lda, int *infoArray, int batchSize) {
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
  matrix_info->group_size_info = batchSize;
  std::int64_t ws_size = oneapi::mkl::lapack::potrf_batch_scratchpad_size<Ty>(
      handle, &(matrix_info->uplo_info), &(matrix_info->n_info),
      &(matrix_info->lda_info), 1, &(matrix_info->group_size_info));
  Ty *ws = sycl::malloc_device<Ty>(ws_size, handle);
  sycl::event e = oneapi::mkl::lapack::potrf_batch(
      handle, &(matrix_info->uplo_info), &(matrix_info->n_info), (Ty **)Aarray,
      &(matrix_info->lda_info), 1, &(matrix_info->group_size_info), ws,
      ws_size);
  handle.submit([&](sycl::handler &cgh) {
    cgh.depends_on(e);
    cgh.host_task([=] { std::free(matrix_info); });
  });
  handle.memset(infoArray, 0, batchSize * sizeof(int));
#endif
}

template <typename T>
inline void potrs_batch(sycl::queue &handle, oneapi::mkl::uplo uplo, int n,
                        int nrhs, T *Aarray[], int lda, T *Barray[], int ldb,
                        int *info, int batchSize) {
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
  matrix_info->group_size_info = batchSize;
  std::int64_t ws_size = oneapi::mkl::lapack::potrs_batch_scratchpad_size<Ty>(
      handle, &(matrix_info->uplo_info), &(matrix_info->n_info),
      &(matrix_info->nrhs_info), &(matrix_info->lda_info),
      &(matrix_info->ldb_info), 1, &(matrix_info->group_size_info));
  Ty *ws = sycl::malloc_device<Ty>(ws_size, handle);
  sycl::event e = oneapi::mkl::lapack::potrs_batch(
      handle, &(matrix_info->uplo_info), &(matrix_info->n_info),
      &(matrix_info->nrhs_info), (Ty **)Aarray, &(matrix_info->lda_info),
      (Ty **)Barray, &(matrix_info->ldb_info), 1,
      &(matrix_info->group_size_info), ws, ws_size);
  handle.submit([&](sycl::handler &cgh) {
    cgh.depends_on(e);
    cgh.host_task([=] { std::free(matrix_info); });
  });
  handle.memset(info, 0, batchSize * sizeof(int));
#endif
}

} // namespace lapack
} // namespace dpct
