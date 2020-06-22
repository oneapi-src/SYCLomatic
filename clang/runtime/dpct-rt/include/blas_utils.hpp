/******************************************************************************
*
* Copyright 2018 - 2020 Intel Corporation.
*
* This software and the related documents are Intel copyrighted materials,
* and your use of them is governed by the express license under which they
* were provided to you ("License"). Unless the License provides otherwise,
* you may not use, modify, copy, publish, distribute, disclose or transmit
* this software or the related documents without Intel's prior written
* permission.

* This software and the related documents are provided as is, with no express
* or implied warranties, other than those that are expressly stated in the
* License.
*****************************************************************************/

//===--- blas_utils.hpp ------------------------------*- C++ -*---===//

#ifndef __DPCT_BLAS_HPP__
#define __DPCT_BLAS_HPP__

#include "memory.hpp"
#include "util.hpp"
#include <CL/sycl.hpp>
#include <mkl_lapack_sycl.hpp>
#include <utility>
#include <vector>

namespace dpct {

inline mkl::transpose get_transpose(int t) {
  if (t == 0) {
    return mkl::transpose::nontrans;
  } else if (t == 1) {
    return mkl::transpose::trans;
  } else {
    return mkl::transpose::conjtrans;
  }
}

/// Get the value of \param s.
/// Copy the data to host synchronously, then return the data.
/// \param [in] p The pointer points the data.
/// \param [in] q The queue where the memory copy should be executed.
template <typename T>
inline typename DataType<T>::T2 get_value(const T *s, cl::sycl::queue &q){
  using Ty = typename DataType<T>::T2;
  Ty s_h;
  detail::dpct_memcpy(q, (void *)&s_h, (void *)s, sizeof(T), automatic).wait();
  return s_h;
}

/// Cast \param vec data to int, then assign to \param ptr array.
/// \param [out] ptr A data pointer.
/// \param [in] vec Input vector with int64_t type elements.
inline void copy_back(int *ptr, const std::vector<int64_t> &vec) {
  auto allocation_ptr = detail::mem_mgr::instance().translate_ptr(ptr);
  auto buffer_ptr = allocation_ptr.buffer.template reinterpret<int, 1>(
      cl::sycl::range<1>(allocation_ptr.size / sizeof(int)));
  auto acc_ptr_write =
      buffer_ptr.template get_access<cl::sycl::access::mode::write>();
  int vec_size = vec.size();
  for (int i = 0; i < vec_size; ++i) {
    acc_ptr_write[i] = static_cast<int>(vec[i]);
  }
}

/// Computes the LU factorizations of a batch of general matrices.
/// \param [in] exec_queue The queue where the routine should be executed.
/// \param [in] n The order of the matrices.
/// \param [in, out] a Array of pointers to matrices. These matrices will be overwritten
/// by lower triangulars with unit diagonal elements and upper triangulars.
/// \param [in] lda The leading dimension of the matrices.
/// \param [out] ipiv An array stores the pivot indices.
/// \param [out] info An array stores the error information.
/// \param [in] batch_size The size of the batch.
template <typename T>
inline void getrf_batch_wrapper(cl::sycl::queue &exec_queue, int n, T *a[],
                                int lda, int *ipiv, int *info, int batch_size) {
  using Ty = typename DataType<T>::T2;
  // Need to construct std::vector to store m, n and lda
  std::vector<int64_t> mn_vec = std::vector<int64_t>(batch_size, n);
  std::vector<int64_t> lda_vec = std::vector<int64_t>(batch_size, lda);
  std::vector<int64_t> info_vec(batch_size, 0);
  std::vector<int64_t> ipiv_vec(batch_size * n, 0);

  {
    std::vector<cl::sycl::buffer<Ty, 1>> a_buf_vec;
    std::vector<cl::sycl::buffer<int64_t, 1>> info_buf_vec;
    std::vector<cl::sycl::buffer<int64_t, 1>> ipiv_buf_vec;
    for (int64_t i = 0; i < batch_size; ++i) {
      // assumes data is in column-major order
      auto allocation_a = detail::mem_mgr::instance().translate_ptr(a[i]);
      a_buf_vec.emplace_back(allocation_a.buffer.template reinterpret<Ty, 1>(
          cl::sycl::range<1>(allocation_a.size / sizeof(Ty))));
      info_buf_vec.emplace_back(&info_vec[i], cl::sycl::range<1>(1));
      ipiv_buf_vec.emplace_back(&ipiv_vec[i * n], cl::sycl::range<1>(n));
    }
    mkl::lapack::getrf_batch(exec_queue, mn_vec, mn_vec, a_buf_vec, lda_vec,
                             ipiv_buf_vec, info_buf_vec);
  }

  // Copy back to the original buffers while casting variables from int64_t to
  // int
  copy_back(ipiv, ipiv_vec);
  copy_back(info, info_vec);
}

/// Solves a system of linear equations with a batch of LU-factored square
/// coefficient matrices, with multiple right-hand sides.
/// \param [in] exec_queue The queue where the routine should be executed.
/// \param [in] trans Indicates the form of the linear equations.
/// \param [in] n The order of the matrices.
/// \param [in] nrhs The number of right hand sides.
/// \param [in, out] a Array of pointers to matrices.
/// \param [in] lda The leading dimension of the matrices in \param a.
/// \param [out] ipiv An array stores the pivots.
/// \param [out] b Array of pointers to matrices, whose columns are
/// the right-hand sides for the systems of equations.
/// \param [in] ldb The leading dimension of the matrices in \param b.
/// \param [out] info An array stores the error information.
/// \param [in] batch_size The size of the batch.
template <typename T>
inline void getrs_batch_wrapper(cl::sycl::queue &exec_queue,
                                mkl::transpose trans, int n, int nrhs,
                                const T *a[], int lda, int *ipiv, T *b[],
                                int ldb, int *info, int batch_size) {
  using Ty = typename DataType<T>::T2;
  // Need to construct std::vector to store trans, n, nrhs, lda and ldb
  std::vector<mkl::transpose> trans_vec =
      std::vector<mkl::transpose>(batch_size, trans);
  std::vector<int64_t> n_vec = std::vector<int64_t>(batch_size, n);
  std::vector<int64_t> nrhs_vec = std::vector<int64_t>(batch_size, nrhs);
  std::vector<int64_t> lda_vec = std::vector<int64_t>(batch_size, lda);
  std::vector<int64_t> ldb_vec = std::vector<int64_t>(batch_size, ldb);
  std::vector<int64_t> info_vec(batch_size, 0);
  std::vector<int64_t> ipiv_vec(batch_size * n, 0);

  std::vector<cl::sycl::buffer<Ty, 1>> a_buf_vec;
  std::vector<cl::sycl::buffer<Ty, 1>> b_buf_vec;
  for (int64_t i = 0; i < batch_size; i++) {
    // assumes data is in column-major order
    auto allocation_a = detail::mem_mgr::instance().translate_ptr(a[i]);
    a_buf_vec.emplace_back(allocation_a.buffer.template reinterpret<Ty, 1>(
        cl::sycl::range<1>(allocation_a.size / sizeof(Ty))));
    auto allocation_b = detail::mem_mgr::instance().translate_ptr(b[i]);
    b_buf_vec.emplace_back(allocation_b.buffer.template reinterpret<Ty, 1>(
        cl::sycl::range<1>(allocation_b.size / sizeof(Ty))));
  }

  {
    auto allocation_ipiv = detail::mem_mgr::instance().translate_ptr(ipiv);
    auto buffer_ipiv = allocation_ipiv.buffer.template reinterpret<int, 1>(
        cl::sycl::range<1>(allocation_ipiv.size / sizeof(int)));
    auto acc_ipiv =
        buffer_ipiv.template get_access<cl::sycl::access::mode::read>();
    for (int64_t i = 0; i < batch_size * n; i++) {
      ipiv_vec.emplace_back(static_cast<int64_t>(acc_ipiv[i]));
    }

    std::vector<cl::sycl::buffer<int64_t, 1>> info_buf_vec;
    std::vector<cl::sycl::buffer<int64_t, 1>> ipiv_buf_vec;
    for (int64_t i = 0; i < batch_size; i++) {
      info_buf_vec.emplace_back(&info_vec[i], cl::sycl::range<1>(1));
      ipiv_buf_vec.emplace_back(&ipiv_vec[i * n], cl::sycl::range<1>(n));
    }
    mkl::lapack::getrs_batch(exec_queue, trans_vec, n_vec, nrhs_vec, a_buf_vec,
                             lda_vec, ipiv_buf_vec, b_buf_vec, ldb_vec,
                             info_buf_vec);
  }

  // Copy back to the original buffers while casting variables from int64_t to
  // int
  copy_back(info, info_vec);
}

/// Computes the inverses of a batch of LU-factored matrices.
/// \param [in] exec_queue The queue where the routine should be executed.
/// \param [in] n The order of the matrices.
/// \param [in, out] a Array of pointers to matrices.
/// \param [in] lda The leading dimension of the matrices in \param a.
/// \param [out] ipiv An array stores the pivots.
/// \param [out] b b Array of pointers to inverse matrices.
/// \param [in] ldb The leading dimension of the matrices in \param b.
/// \param [out] info An array stores the error information.
/// \param [in] batch_size The size of the batch.
template <typename T>
inline void getri_batch_wrapper(cl::sycl::queue &exec_queue, int n,
                                const T *a[], int lda, int *ipiv, T *b[],
                                int ldb, int *info, int batch_size) {
  using Ty = typename DataType<T>::T2;
  // Need to construct std::vector to store n and lda
  std::vector<int64_t> n_vec = std::vector<int64_t>(batch_size, n);
  std::vector<int64_t> ldb_vec = std::vector<int64_t>(batch_size, ldb);
  std::vector<int64_t> info_vec(batch_size, 0);
  std::vector<int64_t> ipiv_vec(batch_size * n, 0);

  {
    std::vector<cl::sycl::buffer<Ty, 1>> b_buf_vec;
    std::vector<cl::sycl::buffer<int64_t, 1>> ipiv_buf_vec;
    std::vector<cl::sycl::buffer<int64_t, 1>> info_buf_vec;
    for (int64_t i = 0; i < batch_size; i++) {
      // Need to create a copy of input matrices A to keep them unchanged.
      // B (copy of A) will be used as input and output parameter in MKL API
      // call.
      matrix_mem_copy(b[i], a[i], ldb, lda, n, n, dpct::device_to_device,
                      exec_queue);
      // assumes data is in column-major order
      auto allocation_b = detail::mem_mgr::instance().translate_ptr(b[i]);
      b_buf_vec.emplace_back(allocation_b.buffer.template reinterpret<Ty, 1>(
          cl::sycl::range<1>(allocation_b.size / sizeof(Ty))));
      info_buf_vec.emplace_back(&info_vec[i], cl::sycl::range<1>(1));
      ipiv_buf_vec.emplace_back(&ipiv_vec[i * n], cl::sycl::range<1>(n));
    }
    mkl::lapack::getri_batch(exec_queue, n_vec, b_buf_vec, ldb_vec,
                             ipiv_buf_vec, info_buf_vec);
  }

  // Copy back to the original buffers while casting variables from int64_t to
  // int
  copy_back(ipiv, ipiv_vec);
  copy_back(info, info_vec);
}

/// Computes the QR factorizations of a batch of general matrices.
/// \param [in] exec_queue The queue where the routine should be executed.
/// \param [in] m The number of rows in the matrices.
/// \param [in] n The number of columns in the matrices.
/// \param [in, out] a Array of pointers to matrices. These
/// matrices will be overwritten by the factorization data.
/// \param [in] lda The leading dimension of the matrices in \param a.
/// \param [out] tau An array stores the scalars.
/// \param [out] info An array stores the error information.
/// \param [in] batch_size The size of the batch.
template <typename T>
inline void geqrf_batch_wrapper(cl::sycl::queue exec_queue, int m, int n,
                                T *a[], int lda, T *tau[], int *info,
                                int batchSize) {
  using Ty = typename DataType<T>::T2;
  // Need to construct std::vector to store m, n and lda
  std::vector<int64_t> m_vec = std::vector<int64_t>(batchSize, m);
  std::vector<int64_t> n_vec = std::vector<int64_t>(batchSize, n);
  std::vector<int64_t> lda_vec = std::vector<int64_t>(batchSize, lda);
  std::vector<int64_t> info_vec(batchSize, 0);

  {
    std::vector<cl::sycl::buffer<Ty, 1>> a_buf_vec;
    std::vector<cl::sycl::buffer<Ty, 1>> tau_buf_vec;
    std::vector<cl::sycl::buffer<int64_t, 1>> info_buf_vec;
    for (int64_t i = 0; i < batchSize; i++) {
      // assumes data is in column-major order
      auto allocation_a =
          detail::mem_mgr::instance().translate_ptr(a[i]);
      a_buf_vec.emplace_back(allocation_a.buffer.template reinterpret<Ty, 1>(
          cl::sycl::range<1>(allocation_a.size / sizeof(Ty))));
      auto allocation_tau =
          detail::mem_mgr::instance().translate_ptr(tau[i]);
      tau_buf_vec.emplace_back(
          allocation_tau.buffer.template reinterpret<Ty, 1>(
              cl::sycl::range<1>(allocation_tau.size / sizeof(Ty))));
      info_buf_vec.emplace_back(&info_vec[i], cl::sycl::range<1>(1));
    }
    mkl::lapack::geqrf_batch(exec_queue, m_vec, n_vec, a_buf_vec, lda_vec,
                             tau_buf_vec, info_buf_vec);
  }

  // Copy back to the original buffers while casting variables from int64_t to
  // int
  copy_back(info, info_vec);
}

} // namespace dpct
#endif // __DPCT_BLAS_HPP__
