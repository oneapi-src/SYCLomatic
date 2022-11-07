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
  queue.submit([&](sycl::handler &cgh) {
    auto info_acc = info_buf.get_access<sycl::access_mode::write>(cgh);
    cgh.single_task<dpct_kernel_name<class sygvd_init_info, T>>(
        [=]() { info_acc[0] = 0; });
  });
  auto a_buffer = get_buffer<T>(a);
  auto b_buffer = get_buffer<T>(b);
  auto w_buffer = get_buffer<T>(w);
  auto scratchpad_buffer = get_buffer<T>(scratchpad);
  try {
    oneapi::mkl::lapack::sygvd(queue, itype, jobz, uplo, n, a_buffer, lda,
                               b_buffer, ldb, w_buffer, scratchpad_buffer,
                               scratchpad_size);
  } catch (oneapi::mkl::lapack::exception const& e) {
    std::cerr << "Caught synchronous SYCL exception:" << std::endl
              << "reason: " << e.what() << std::endl
              << "info: " << e.info() << std::endl;
    int info_val = static_cast<int>(e.info());
    
    queue.submit([&](sycl::handler &cgh) {
      auto info_acc = info_buf.get_access<sycl::access_mode::write>(cgh);
      cgh.single_task<dpct_kernel_name<class sygvd_set_err_info, T>>(
          [=]() { info_acc[0] = info_val; });
    }).wait();
    return 1;
  } catch (sycl::exception const& e) {
    std::cerr << "Caught synchronous SYCL exception:" << std::endl
              << "reason: " << e.what() << std::endl;
    queue.submit([&](sycl::handler &cgh) {
      auto info_acc = info_buf.get_access<sycl::access_mode::write>(cgh);
      cgh.single_task<dpct_kernel_name<class sygvd_set_err_info_dft_val, T>>(
          [=]() { info_acc[0] = -1; });
    }).wait();
    return 1;
  }
  return 0;
#else
  queue.memset(info, 0, sizeof(int));
  try {
    oneapi::mkl::lapack::sygvd(queue, itype, jobz, uplo, n, a, lda, b, ldb, w,
                               scratchpad, scratchpad_size);
  } catch (oneapi::mkl::lapack::exception const& e) {
    std::cerr << "Caught synchronous SYCL exception:" << std::endl
              << "reason: " << e.what() << std::endl
              << "info: " << e.info() << std::endl;
    int info_val = static_cast<int>(e.info());
    queue.memcpy(info, &info_val, sizeof(int)).wait();
    return 1;
  } catch (sycl::exception const& e) {
    std::cerr << "Caught synchronous SYCL exception:" << std::endl
              << "reason: " << e.what() << std::endl;
    int info_val = -1;
    queue.memcpy(info, &info_val, sizeof(int)).wait();
    return 1;
  }
  return 0;
#endif
}
// DPCT_LABEL_END
} // namespace lapack
} // namespace dpct

#endif // __DPCT_LAPACK_UTILS_HPP__
