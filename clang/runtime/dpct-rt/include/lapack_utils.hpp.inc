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
#include "lib_common_utils.hpp"

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
// DPCT_LABEL_END

namespace detail {
// DPCT_LABEL_BEGIN|lapack_shim|dpct::lapack::detail
// DPCT_DEPENDENCY_BEGIN
// Memory|dpct_memcpy_detail
// Memory|dpct_memset_detail
// Memory|dpct_free
// LibCommonUtils|library_data_t
// DPCT_DEPENDENCY_END
// DPCT_CODE
template <typename functor_t, typename... args_t>
inline int lapack_shim(sycl::queue &q, library_data_t a_type, int *info,
                       const std::string &lapack_api_name, functor_t functor,
                       args_t... args) {
  try {
    switch (a_type) {
    case library_data_t::real_float: {
      functor.template operator()<float>(args...);
      break;
    }
    case library_data_t::real_double: {
      functor.template operator()<double>(args...);
      break;
    }
    case library_data_t::complex_float: {
      functor.template operator()<std::complex<float>>(args...);
      break;
    }
    case library_data_t::complex_double: {
      functor.template operator()<std::complex<double>>(args...);
      break;
    }
    default:
      throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                            "the data type is unsupported");
    }
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
    return 1;
  } catch (sycl::exception const &e) {
    std::cerr << "Caught synchronous SYCL exception:" << std::endl
              << "reason: " << e.what() << std::endl;
    if (info)
      dpct::detail::dpct_memset(q, info, 0, sizeof(int)).wait();
    return 1;
  }
  return 0;
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|getrf_scratchpad_size_impl|dpct::lapack::detail
// DPCT_DEPENDENCY_BEGIN
// LibCommonUtils|library_data_t
// DPCT_DEPENDENCY_END
// DPCT_CODE
struct getrf_scratchpad_size_impl {
  template <typename T> 
  void operator()(sycl::queue &q, std::int64_t m, std::int64_t n,
                  library_data_t a_type, std::int64_t lda,
                  size_t *device_ws_size) {
    *device_ws_size =
        oneapi::mkl::lapack::getrf_scratchpad_size<T>(q, m, n, lda) * sizeof(T);
  }
};
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|getrf_impl|dpct::lapack::detail
// DPCT_DEPENDENCY_BEGIN
// LibCommonUtils|library_data_t
// LibCommonUtils|get_memory
// Memory|dpct_memset_detail
// DPCT_DEPENDENCY_END
// DPCT_CODE
struct getrf_impl {
  template <typename T> 
  void operator()(sycl::queue &q, std::int64_t m, std::int64_t n,
                  library_data_t a_type, void *a, std::int64_t lda,
                  std::int64_t *ipiv, void *device_ws, size_t device_ws_size,
                  int *info) {
    auto ipiv_data = dpct::detail::get_memory(ipiv);
    auto a_data = dpct::detail::get_memory(reinterpret_cast<T *>(a));
    auto device_ws_data =
        dpct::detail::get_memory(reinterpret_cast<T *>(device_ws));
    oneapi::mkl::lapack::getrf(q, m, n, a_data, lda, ipiv_data, device_ws_data,
                               device_ws_size / sizeof(T));
    dpct::detail::dpct_memset(q, info, 0, sizeof(int));
  }
};
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|getrs_impl|dpct::lapack::detail
// DPCT_DEPENDENCY_BEGIN
// LibCommonUtils|library_data_t
// LibCommonUtils|get_memory
// Memory|dpct_malloc_detail
// Memory|dpct_memset_detail
// Memory|mem_mgr_mem_free|UsmNone
// Memory|async_dpct_free|UsmRestricted
// DPCT_DEPENDENCY_END
// DPCT_CODE
struct getrs_impl {
  template <typename T> 
  void operator()(sycl::queue &q, oneapi::mkl::transpose trans, std::int64_t n,
                  std::int64_t nrhs, library_data_t a_type, void *a,
                  std::int64_t lda, std::int64_t *ipiv, library_data_t b_type,
                  void *b, std::int64_t ldb, int *info) {
    class working_memory {
    public:
      working_memory(size_t size, sycl::queue q) : _q(q) {
        _ptr = dpct::detail::dpct_malloc(size, _q);
      }
      void *get_ptr() { return _ptr; }
      void set_event(sycl::event e) { _e = e; }
      ~working_memory() {
        if (_ptr) {
#ifdef DPCT_USM_LEVEL_NONE
          dpct::detail::mem_mgr::instance().mem_free(_ptr);
#else
          dpct::async_dpct_free({_ptr}, {_e}, _q);
#endif
        }
      }

    private:
      void *_ptr = nullptr;
      sycl::event _e;
      sycl::queue _q;
    };


    auto ipiv_data = dpct::detail::get_memory(ipiv);
    std::int64_t device_ws_size = oneapi::mkl::lapack::getrs_scratchpad_size<T>(
        q, trans, n, nrhs, lda, ldb);
    working_memory device_ws(device_ws_size * sizeof(T), q);
    auto device_ws_data =
        dpct::detail::get_memory(reinterpret_cast<T *>(device_ws.get_ptr()));
    auto a_data = dpct::detail::get_memory(reinterpret_cast<T *>(a));
    auto b_data = dpct::detail::get_memory(reinterpret_cast<T *>(b));
    oneapi::mkl::lapack::getrs(q, trans, n, nrhs, a_data, lda, ipiv_data,
                               b_data, ldb, device_ws_data, device_ws_size);
    sycl::event e = dpct::detail::dpct_memset(q, info, 0, sizeof(int));
    device_ws.set_event(e);
  }
};
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|geqrf_scratchpad_size_impl|dpct::lapack::detail
// DPCT_DEPENDENCY_BEGIN
// LibCommonUtils|library_data_t
// DPCT_DEPENDENCY_END
// DPCT_CODE
struct geqrf_scratchpad_size_impl {
  template <typename T> 
  void operator()(sycl::queue &q, std::int64_t m, std::int64_t n,
                  library_data_t a_type, std::int64_t lda,
                  size_t *device_ws_size) {
    *device_ws_size =
        oneapi::mkl::lapack::geqrf_scratchpad_size<T>(q, m, n, lda) * sizeof(T);
  }
};
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|geqrf_impl|dpct::lapack::detail
// DPCT_DEPENDENCY_BEGIN
// LibCommonUtils|get_memory
// LibCommonUtils|library_data_t
// Memory|dpct_memset_detail
// DPCT_DEPENDENCY_END
// DPCT_CODE
struct geqrf_impl {
  template <typename T> 
  void operator()(sycl::queue &q, std::int64_t m, std::int64_t n,
                  library_data_t a_type, void *a, std::int64_t lda,
                  library_data_t tau_type, void *tau, void *device_ws,
                  size_t device_ws_size, int *info) {
    auto a_data = dpct::detail::get_memory(reinterpret_cast<T *>(a));
    auto tau_data = dpct::detail::get_memory(reinterpret_cast<T *>(tau));
    auto device_ws_data =
        dpct::detail::get_memory(reinterpret_cast<T *>(device_ws));
    oneapi::mkl::lapack::geqrf(q, m, n, a_data, lda, tau_data, device_ws_data,
                               device_ws_size / sizeof(T));
    dpct::detail::dpct_memset(q, info, 0, sizeof(int));
  }
};
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|getrfnp_impl|dpct::lapack::detail
// DPCT_DEPENDENCY_BEGIN
// LibCommonUtils|get_memory
// LibCommonUtils|library_data_t
// Memory|dpct_memcpy_detail
// Memory|dpct_memset_detail
// DPCT_DEPENDENCY_END
// DPCT_CODE
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
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|char2jobsvd|dpct::lapack::detail
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
inline oneapi::mkl::jobsvd char2jobsvd(char job) {
  switch (job) {
  case 'A':
    return oneapi::mkl::jobsvd::vectors;
  case 'S':
    return oneapi::mkl::jobsvd::somevec;
  case 'O':
    return oneapi::mkl::jobsvd::vectorsina;
  case 'N':
    return oneapi::mkl::jobsvd::novec;
  default:
    throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                          "the job type is unsupported");
  }
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|gesvd_scratchpad_size_impl|dpct::lapack::detail
// DPCT_DEPENDENCY_BEGIN
// LibCommonUtils|library_data_t
// LapackUtils|char2jobsvd
// DPCT_DEPENDENCY_END
// DPCT_CODE
struct gesvd_scratchpad_size_impl {
  template <typename T>
  void operator()(sycl::queue &q, signed char jobu, signed char jobvt,
                  std::int64_t m, std::int64_t n, library_data_t a_type,
                  std::int64_t lda, library_data_t u_type, std::int64_t ldu,
                  library_data_t vt_type, std::int64_t ldvt,
                  size_t *device_ws_size) {
    *device_ws_size =
        oneapi::mkl::lapack::gesvd_scratchpad_size<T>(
            q, char2jobsvd(jobu), char2jobsvd(jobvt), m, n, lda, ldu, ldvt) *
        sizeof(T);
  }
};
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|gesvd_impl|dpct::lapack::detail
// DPCT_DEPENDENCY_BEGIN
// LibCommonUtils|library_data_t
// LapackUtils|char2jobsvd
// LibCommonUtils|get_memory
// Memory|dpct_memset_detail
// DPCT_DEPENDENCY_END
// DPCT_CODE
struct gesvd_impl {
  template <typename T> struct ElementType {
    using T2 = T;
  };
  template <typename T> struct ElementType<std::complex<T>> {
    using T2 = T;
  };
  template <typename T>
  void operator()(sycl::queue &q, signed char jobu, signed char jobvt,
                  std::int64_t m, std::int64_t n, library_data_t a_type,
                  void *a, std::int64_t lda, library_data_t s_type, void *s,
                  library_data_t u_type, void *u, std::int64_t ldu,
                  library_data_t vt_type, void *vt, std::int64_t ldvt,
                  void *device_ws, size_t device_ws_size, int *info) {
    auto a_data = dpct::detail::get_memory(reinterpret_cast<T *>(a));
    auto s_data = dpct::detail::get_memory(
        reinterpret_cast<typename ElementType<T>::T2 *>(s));
    auto u_data = dpct::detail::get_memory(reinterpret_cast<T *>(u));
    auto vt_data = dpct::detail::get_memory(reinterpret_cast<T *>(vt));
    auto device_ws_data =
        dpct::detail::get_memory(reinterpret_cast<T *>(device_ws));
    oneapi::mkl::lapack::gesvd(q, char2jobsvd(jobu), char2jobsvd(jobvt), m, n,
                               a_data, lda, s_data, u_data, ldu, vt_data, ldvt,
                               device_ws_data, device_ws_size / sizeof(T));
    dpct::detail::dpct_memset(q, info, 0, sizeof(int));
  }
};
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|potrf_scratchpad_size_impl|dpct::lapack::detail
// DPCT_DEPENDENCY_BEGIN
// LibCommonUtils|library_data_t
// DPCT_DEPENDENCY_END
// DPCT_CODE
struct potrf_scratchpad_size_impl {
  template <typename T>
  void operator()(sycl::queue &q, oneapi::mkl::uplo uplo, std::int64_t n,
                  library_data_t a_type, std::int64_t lda,
                  size_t *device_ws_size) {
    *device_ws_size =
        oneapi::mkl::lapack::potrf_scratchpad_size<T>(q, uplo, n, lda) *
        sizeof(T);
  }
};
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|potrf_impl|dpct::lapack::detail
// DPCT_DEPENDENCY_BEGIN
// LibCommonUtils|library_data_t
// LibCommonUtils|get_memory
// Memory|dpct_memset_detail
// DPCT_DEPENDENCY_END
// DPCT_CODE
struct potrf_impl {
  template <typename T>
  void operator()(sycl::queue &q, oneapi::mkl::uplo uplo, std::int64_t n,
                  library_data_t a_type, void *a, std::int64_t lda,
                  void *device_ws, size_t device_ws_size, int *info) {
    auto a_data = dpct::detail::get_memory(reinterpret_cast<T *>(a));
    auto device_ws_data =
        dpct::detail::get_memory(reinterpret_cast<T *>(device_ws));
    oneapi::mkl::lapack::potrf(q, uplo, n, a_data, lda, device_ws_data,
                               device_ws_size / sizeof(T));
    dpct::detail::dpct_memset(q, info, 0, sizeof(int));
  }
};
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|potrs_impl|dpct::lapack::detail
// DPCT_DEPENDENCY_BEGIN
// LibCommonUtils|library_data_t
// LibCommonUtils|get_memory
// Memory|dpct_malloc_detail
// Memory|dpct_memset_detail
// Memory|mem_mgr_mem_free|UsmNone
// Memory|async_dpct_free|UsmRestricted
// DPCT_DEPENDENCY_END
// DPCT_CODE
struct potrs_impl {
  template <typename T>
  void operator()(sycl::queue &q, oneapi::mkl::uplo uplo, std::int64_t n,
                  std::int64_t nrhs, library_data_t a_type, void *a,
                  std::int64_t lda, library_data_t b_type, void *b,
                  std::int64_t ldb, int *info) {
    class working_memory {
    public:
      working_memory(size_t size, sycl::queue q) : _q(q) {
        _ptr = dpct::detail::dpct_malloc(size, _q);
      }
      void *get_ptr() { return _ptr; }
      void set_event(sycl::event e) { _e = e; }
      ~working_memory() {
        if (_ptr) {
#ifdef DPCT_USM_LEVEL_NONE
          dpct::detail::mem_mgr::instance().mem_free(_ptr);
#else
          dpct::async_dpct_free({_ptr}, {_e}, _q);
#endif
        }
      }

    private:
      void *_ptr = nullptr;
      sycl::event _e;
      sycl::queue _q;
    };

    std::int64_t device_ws_size = oneapi::mkl::lapack::potrs_scratchpad_size<T>(
        q, uplo, n, nrhs, lda, ldb);
    working_memory device_ws(device_ws_size * sizeof(T), q);
    auto device_ws_data =
        dpct::detail::get_memory(reinterpret_cast<T *>(device_ws.get_ptr()));
    auto a_data = dpct::detail::get_memory(reinterpret_cast<T *>(a));
    auto b_data = dpct::detail::get_memory(reinterpret_cast<T *>(b));
    oneapi::mkl::lapack::potrs(q, uplo, n, nrhs, a_data, lda, b_data, ldb,
                               device_ws_data, device_ws_size);
    sycl::event e = dpct::detail::dpct_memset(q, info, 0, sizeof(int));
    device_ws.set_event(e);
  }
};
// DPCT_LABEL_END
} // namespace detail

// DPCT_LABEL_BEGIN|getrf_scratchpad_size|dpct::lapack
// DPCT_DEPENDENCY_BEGIN
// LibCommonUtils|library_data_t
// LapackUtils|lapack_shim
// LapackUtils|getrf_scratchpad_size_impl
// DPCT_DEPENDENCY_END
// DPCT_CODE
/// Computes the size of workspace memory of getrf function.
/// \return Returns 0 if no synchronous exception, otherwise returns 1.
/// \param [in] q Device queue where computation will be performed.
/// \param [in] m The number of rows in the matrix A.
/// \param [in] n The number of columns in the matrix A.
/// \param [in] a_type The data type of the matrix A.
/// \param [in] lda The leading dimension of the matrix A.
/// \param [out] device_ws_size The workspace size in bytes.
/// \param [out] host_ws_size The host workspace size in bytes. Currently the
/// value is always zero.
inline int getrf_scratchpad_size(sycl::queue &q, std::int64_t m, std::int64_t n,
                                 library_data_t a_type, std::int64_t lda,
                                 size_t *device_ws_size,
                                 size_t *host_ws_size = nullptr) {
  if (host_ws_size)
    *host_ws_size = 0;
  return detail::lapack_shim(q, a_type, nullptr, "getrf_scratchpad_size",
                             detail::getrf_scratchpad_size_impl(), q, m, n,
                             a_type, lda, device_ws_size);
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|getrf|dpct::lapack
// DPCT_DEPENDENCY_BEGIN
// LibCommonUtils|library_data_t
// LapackUtils|lapack_shim
// LapackUtils|getrf_impl
// LapackUtils|getrfnp_impl
// DPCT_DEPENDENCY_END
// DPCT_CODE
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
  return detail::lapack_shim(q, a_type, info, "getrf", detail::getrf_impl(), q,
                             m, n, a_type, a, lda, ipiv, device_ws,
                             device_ws_size, info);
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|getrs|dpct::lapack
// DPCT_DEPENDENCY_BEGIN
// LibCommonUtils|library_data_t
// LapackUtils|lapack_shim
// LapackUtils|getrs_impl
// DPCT_DEPENDENCY_END
// DPCT_CODE
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
  return detail::lapack_shim(q, a_type, info, "getrs_scratchpad_size/getrs",
                             detail::getrs_impl(), q, trans, n, nrhs, a_type, a,
                             lda, ipiv, b_type, b, ldb, info);
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|geqrf_scratchpad_size|dpct::lapack
// DPCT_DEPENDENCY_BEGIN
// LibCommonUtils|library_data_t
// LapackUtils|lapack_shim
// LapackUtils|geqrf_scratchpad_size_impl
// DPCT_DEPENDENCY_END
// DPCT_CODE
/// Computes the size of workspace memory of geqrf function.
/// \return Returns 0 if no synchronous exception, otherwise returns 1.
/// \param [in] q Device queue where computation will be performed.
/// \param [in] m The number of rows in the matrix A.
/// \param [in] n The number of columns in the matrix A.
/// \param [in] a_type The data type of the matrix A.
/// \param [in] lda The leading dimension of the matrix A.
/// \param [out] device_ws_size The device workspace size in bytes.
/// \param [out] host_ws_size The host workspace size in bytes. Currently the
/// value is always zero.
inline int geqrf_scratchpad_size(sycl::queue &q, std::int64_t m, std::int64_t n,
                                 library_data_t a_type, std::int64_t lda,
                                 size_t *device_ws_size,
                                 size_t *host_ws_size = nullptr) {
  if (host_ws_size)
    *host_ws_size = 0;
  return detail::lapack_shim(q, a_type, nullptr, "geqrf_scratchpad_size",
                             detail::geqrf_scratchpad_size_impl(), q, m, n,
                             a_type, lda, device_ws_size);
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|geqrf|dpct::lapack
// DPCT_DEPENDENCY_BEGIN
// LibCommonUtils|library_data_t
// LapackUtils|lapack_shim
// LapackUtils|geqrf_impl
// DPCT_DEPENDENCY_END
// DPCT_CODE
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
  return detail::lapack_shim(q, a_type, info, "geqrf", detail::geqrf_impl(), q,
                             m, n, a_type, a, lda, tau_type, tau, device_ws,
                             device_ws_size, info);
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|gesvd_scratchpad_size|dpct::lapack
// DPCT_DEPENDENCY_BEGIN
// LibCommonUtils|library_data_t
// LapackUtils|lapack_shim
// LapackUtils|gesvd_scratchpad_size_impl
// DPCT_DEPENDENCY_END
// DPCT_CODE
inline int gesvd_scratchpad_size(sycl::queue &q, signed char jobu,
                                 signed char jobvt, std::int64_t m,
                                 std::int64_t n, library_data_t a_type,
                                 std::int64_t lda, library_data_t u_type,
                                 std::int64_t ldu, library_data_t vt_type,
                                 std::int64_t ldvt, size_t *device_ws_size,
                                 size_t *host_ws_size = nullptr) {
  if (host_ws_size)
    *host_ws_size = 0;
  return detail::lapack_shim(q, a_type, nullptr, "gesvd_scratchpad_size",
                             detail::gesvd_scratchpad_size_impl(), q, jobu,
                             jobvt, m, n, a_type, lda, u_type, ldu, vt_type,
                             ldvt, device_ws_size);
}

inline int gesvd_scratchpad_size(sycl::queue &q, oneapi::mkl::job jobz,
                                 std::int64_t econ, std::int64_t m,
                                 std::int64_t n, library_data_t a_type,
                                 std::int64_t lda, library_data_t u_type,
                                 std::int64_t ldu, library_data_t vt_type,
                                 std::int64_t ldvt, int *device_ws_size,
                                 size_t *host_ws_size = nullptr) {
  if (host_ws_size)
    *host_ws_size = 0;
  signed char jobu;
  signed char jobvt;
  if (jobz == oneapi::mkl::job::vec) {
    if (econ) {
      jobu = 'S';
      jobvt = 'S';
    } else {
      jobu = 'A';
      jobvt = 'A';
    }
  } else if (jobz == oneapi::mkl::job::novec) {
    jobu = 'N';
    jobvt = 'N';
  } else {
    throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                          "the job type is unsupported");
  }
  std::size_t device_ws_size_64;
  int ret = detail::lapack_shim(q, a_type, nullptr, "gesvd_scratchpad_size",
                                detail::gesvd_scratchpad_size_impl(), q, jobu,
                                jobvt, m, n, a_type, lda, u_type, ldu, vt_type,
                                ldvt, &device_ws_size_64);

#define CASE(TYPE_NAME, TYPE)                                                  \
  case TYPE_NAME: {                                                            \
    *device_ws_size = device_ws_size_64 / sizeof(TYPE);                        \
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
  return ret;
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|gesvd|dpct::lapack
// DPCT_DEPENDENCY_BEGIN
// LibCommonUtils|library_data_t
// LapackUtils|lapack_shim
// LapackUtils|gesvd_impl
// LibCommonUtils|get_memory
// DPCT_DEPENDENCY_END
// DPCT_CODE
inline int gesvd(sycl::queue &q, signed char jobu, signed char jobvt,
                 std::int64_t m, std::int64_t n, library_data_t a_type, void *a,
                 std::int64_t lda, library_data_t s_type, void *s,
                 library_data_t u_type, void *u, std::int64_t ldu,
                 library_data_t vt_type, void *vt, std::int64_t ldvt,
                 void *device_ws, size_t device_ws_size, int *info) {
  return detail::lapack_shim(q, a_type, info, "gesvd", detail::gesvd_impl(), q,
                             jobu, jobvt, m, n, a_type, a, lda, s_type, s,
                             u_type, u, ldu, vt_type, vt, ldvt, device_ws,
                             device_ws_size, info);
}

inline int gesvd(sycl::queue &q, oneapi::mkl::job jobz, std::int64_t econ,
                 std::int64_t m, std::int64_t n, library_data_t a_type, void *a,
                 std::int64_t lda, library_data_t s_type, void *s,
                 library_data_t u_type, void *u, std::int64_t ldu,
                 library_data_t vt_type, void *vt, std::int64_t ldvt,
                 void *device_ws, size_t device_ws_size, int *info) {
  signed char jobu;
  signed char jobvt;
  if (jobz == oneapi::mkl::job::vec) {
    if (econ) {
      jobu = 'S';
      jobvt = 'S';
    } else {
      jobu = 'A';
      jobvt = 'A';
    }
  } else if (jobz == oneapi::mkl::job::novec) {
    jobu = 'N';
    jobvt = 'N';
  } else {
    throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                          "the job type is unsupported");
  }

#define CASE(TYPE_NAME, TYPE)                                                  \
  case TYPE_NAME: {                                                            \
    device_ws_size = device_ws_size * sizeof(TYPE);                            \
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

  int ret = detail::lapack_shim(q, a_type, info, "gesvd", detail::gesvd_impl(),
                                q, jobu, jobvt, m, n, a_type, a, lda, s_type, s,
                                u_type, u, ldu, vt_type, vt, ldvt, device_ws,
                                device_ws_size, info);
  if (ret)
    return ret;

#define CASE(TYPE_NAME, TYPE)                                                  \
  case TYPE_NAME: {                                                            \
    auto vt_data = dpct::detail::get_memory(reinterpret_cast<TYPE *>(vt));     \
    oneapi::mkl::blas::row_major::imatcopy(                                    \
        q, oneapi::mkl::transpose::conjtrans, n, n, TYPE(1.0f), vt_data, ldvt, \
        ldvt);                                                                 \
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
  return 0;
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|potrf_scratchpad_size|dpct::lapack
// DPCT_DEPENDENCY_BEGIN
// LibCommonUtils|library_data_t
// LapackUtils|lapack_shim
// LapackUtils|potrf_scratchpad_size_impl
// DPCT_DEPENDENCY_END
// DPCT_CODE
inline int potrf_scratchpad_size(sycl::queue &q, oneapi::mkl::uplo uplo,
                                 std::int64_t n, library_data_t a_type,
                                 std::int64_t lda, size_t *device_ws_size,
                                 size_t *host_ws_size = nullptr) {
  if (host_ws_size)
    *host_ws_size = 0;
  return detail::lapack_shim(q, a_type, nullptr, "potrf_scratchpad_size",
                             detail::potrf_scratchpad_size_impl(), q, uplo, n,
                             a_type, lda, device_ws_size);
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|potrf|dpct::lapack
// DPCT_DEPENDENCY_BEGIN
// LibCommonUtils|library_data_t
// LapackUtils|lapack_shim
// LapackUtils|potrf_impl
// DPCT_DEPENDENCY_END
// DPCT_CODE
inline int potrf(sycl::queue &q, oneapi::mkl::uplo uplo, std::int64_t n,
                 library_data_t a_type, void *a, std::int64_t lda,
                 void *device_ws, size_t device_ws_size, int *info) {
  return detail::lapack_shim(q, a_type, info, "potrf", detail::potrf_impl(), q,
                             uplo, n, a_type, a, lda, device_ws, device_ws_size,
                             info);
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|potrs|dpct::lapack
// DPCT_DEPENDENCY_BEGIN
// LibCommonUtils|library_data_t
// LapackUtils|lapack_shim
// LapackUtils|potrs_impl
// DPCT_DEPENDENCY_END
// DPCT_CODE
inline int potrs(sycl::queue &q, oneapi::mkl::uplo uplo, std::int64_t n,
                 std::int64_t nrhs, library_data_t a_type, void *a,
                 std::int64_t lda, library_data_t b_type, void *b,
                 std::int64_t ldb, int *info) {
  return detail::lapack_shim(q, a_type, info, "potrs_scratchpad_size/potrs",
                             detail::potrs_impl(), q, uplo, n, nrhs, a_type, a,
                             lda, b_type, b, ldb, info);
}
// DPCT_LABEL_END
} // namespace lapack
} // namespace dpct

#endif // __DPCT_LAPACK_UTILS_HPP__
