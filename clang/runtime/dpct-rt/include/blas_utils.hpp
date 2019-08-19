/******************************************************************************
* INTEL CONFIDENTIAL
*
* Copyright 2018 - 2019 Intel Corporation.
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

#ifndef __DPCT_BLAS_H__
#define __DPCT_BLAS_H__

#include "memory.hpp"
#include "util.hpp"
#include <CL/sycl.hpp>
#include <mkl_lapack_sycl.hpp>
#include <utility>
#include <vector>

namespace dpct {

/// Cast \param vec data to int, then assign to \param ptr array.
/// \param [out] ptr A data pointer.
/// \param [in] vec Input vector with int64_t type elements.
void copy_back(int *ptr, const std::vector<int64_t> &vec) {
  auto allocation_ptr = dpct::memory_manager::get_instance().translate_ptr(ptr);
  auto buffer_ptr = allocation_ptr.buffer.template reinterpret<int, 1>(
      cl::sycl::range<1>(allocation_ptr.size / sizeof(int)));
  auto acc_ptr_write =
      buffer_ptr.template get_access<cl::sycl::access::mode::write>();
  for (int i = 0; i < vec.size(); ++i) {
    acc_ptr_write[i] = static_cast<int>(vec[i]);
  }
}

/// Computes the LU factorizations of a batch of general matrices.
/// \param [in] exec_queue The queue where the routine should be executed.
/// \param [in] n The order of the batch matrices.
/// \param [in, out] a A pointer array, each pointer points a matrix. These
/// matrices will be overwirtten by lower triangulars with unit diagonal
/// elements and upper triangulars.
/// \param [in] lda The leading dimension of the batch matrices.
/// \param [out] ipiv An array stores the pivot indices.
/// \param [out] info An array stores the error information.
/// \param [in] batch_size The size of the batch.
template <typename T>
inline void getrf_batch_wrapper(cl::sycl::queue &exec_queue, int n, T *a[],
                                int lda, int *ipiv, int *info, int batch_size) {
  using Ty = typename DataType<T>::T2;
  // Need construct std::vector to store m, n and lda
  std::vector<int64_t> mn_vec = std::vector<int64_t>(batch_size, n);
  std::vector<int64_t> lda_vec = std::vector<int64_t>(batch_size, lda);
  std::vector<int64_t> info_vec(batch_size, 0);
  std::vector<int64_t> ipiv_vec(batch_size * n, 0);

  // geqrf buffer block
  {
    std::vector<cl::sycl::buffer<Ty, 1>> a_buf_vec;
    std::vector<cl::sycl::buffer<int64_t, 1>> info_buf_vec;
    std::vector<cl::sycl::buffer<int64_t, 1>> ipiv_buf_vec;
    for (int64_t i = 0; i < batch_size; ++i) {
      // assumes data is in column-major order
      auto allocation_a =
          dpct::memory_manager::get_instance().translate_ptr(a[i]);
      a_buf_vec.emplace_back(allocation_a.buffer.template reinterpret<Ty, 1>(
          cl::sycl::range<1>(allocation_a.size / sizeof(Ty))));
      info_buf_vec.emplace_back(&info_vec[i], cl::sycl::range<1>(1));
      ipiv_buf_vec.emplace_back(&ipiv_vec[i * n], cl::sycl::range<1>(n));
    }
    mkl::getrf_batch(exec_queue, mn_vec, mn_vec, a_buf_vec, lda_vec,
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
/// \param [in] n The order of the batch matrices.
/// \param [in] nrhs The number of right hand sides.
/// \param [in] a A pointer array, each pointer points a matrix.
/// \param [in] lda The leading dimension of the batch matrices in \param a.
/// \param [out] ipiv An array stores the pivots.
/// \param [out] b A pointer array, each pointer points a matrix whose columns
/// are the right-hand sides for the systems of equations.
/// \param [in] ldb The leading dimension of the batch matrices in \param b.
/// \param [out] info An array stores the error information.
/// \param [in] batch_size The size of the batch.
template <typename T>
inline void getrs_batch_wrapper(cl::sycl::queue &exec_queue,
                                mkl::transpose trans, int n, int nrhs,
                                const T *a[], int lda, int *ipiv, T *b[],
                                int ldb, int *info, int batch_size) {
  using Ty = typename DataType<T>::T2;
  // Need construct std::vector to store trans, n, nrhs, lda and ldb
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
    auto allocation_a =
        dpct::memory_manager::get_instance().translate_ptr(a[i]);
    a_buf_vec.emplace_back(allocation_a.buffer.template reinterpret<Ty, 1>(
        cl::sycl::range<1>(allocation_a.size / sizeof(Ty))));
    auto allocation_b =
        dpct::memory_manager::get_instance().translate_ptr(b[i]);
    b_buf_vec.emplace_back(allocation_b.buffer.template reinterpret<Ty, 1>(
        cl::sycl::range<1>(allocation_b.size / sizeof(Ty))));
  }

  // geqrs buffer block
  {
    auto allocation_ipiv =
        dpct::memory_manager::get_instance().translate_ptr(ipiv);
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
    mkl::getrs_batch(exec_queue, trans_vec, n_vec, nrhs_vec, a_buf_vec, lda_vec,
                     ipiv_buf_vec, b_buf_vec, ldb_vec, info_buf_vec);
  }

  // Copy back to the original buffers while casting variables from int64_t to
  // int
  copy_back(info, info_vec);
}

/// Computes the inverses of a batch of LU-factored matrices.
/// \param [in] exec_queue The queue where the routine should be executed.
/// \param [in] n The order of the batch matrices.
/// \param [in] a A pointer array, each pointer points a matrix.
/// \param [in] lda The leading dimension of the batch matrices in \param a.
/// \param [out] ipiv An array stores the pivots.
/// \param [out] b A pointer array, each pointer points an inverse matrix.
/// \param [in] ldb The leading dimension of the batch matrices in \param b.
/// \param [out] info An array stores the error information.
/// \param [in] batch_size The size of the batch.
template <typename T>
inline void getri_batch_wrapper(cl::sycl::queue &exec_queue, int n,
                                const T *a[], int lda, int *ipiv, T *b[],
                                int ldb, int *info, int batch_size) {
  using Ty = typename DataType<T>::T2;
  // Need construct std::vector to store n and lda
  std::vector<int64_t> n_vec = std::vector<int64_t>(batch_size, n);
  std::vector<int64_t> ldb_vec = std::vector<int64_t>(batch_size, ldb);
  std::vector<int64_t> info_vec(batch_size, 0);
  std::vector<int64_t> ipiv_vec(batch_size * n, 0);
  std::vector<int64_t> lwork_vec(batch_size, 0);

  std::vector<cl::sycl::buffer<Ty, 1>> b_buf_vec;
  for (int64_t i = 0; i < batch_size; i++) {
    // Original code is input A and output B while MKL API is input A and output
    // A So copy A to B and use B as the parameter of MKL API.
    matrix_mem_copy(b[i], a[i], ldb, lda, n, n, dpct::device_to_device,
                    exec_queue, false);
    // assumes data is in column-major order
    auto allocation_b =
        dpct::memory_manager::get_instance().translate_ptr(b[i]);
    b_buf_vec.emplace_back(allocation_b.buffer.template reinterpret<Ty, 1>(
        cl::sycl::range<1>(allocation_b.size / sizeof(Ty))));
  }

  // getri_get_lwork buffer block
  {
    std::vector<cl::sycl::buffer<int64_t>> lwork_buf_vec;
    std::vector<cl::sycl::buffer<int64_t, 1>> ipiv_buf_vec;
    for (int64_t i = 0; i < batch_size; ++i) {
      lwork_buf_vec.emplace_back(&lwork_vec[i], cl::sycl::range<1>(1));
      ipiv_buf_vec.emplace_back(&ipiv_vec[i * n], cl::sycl::range<1>(n));
    }
    mkl::getri_get_lwork_batch(exec_queue, n_vec, b_buf_vec, ldb_vec,
                               ipiv_buf_vec, lwork_buf_vec);
  }

  // getri buffer block
  {
    std::vector<cl::sycl::buffer<int64_t, 1>> info_buf_vec;
    std::vector<cl::sycl::buffer<int64_t, 1>> ipiv_buf_vec;
    std::vector<cl::sycl::buffer<Ty, 1>> work_buf_vec;
    for (int64_t i = 0; i < batch_size; ++i) {
      info_buf_vec.emplace_back(&info_vec[i], cl::sycl::range<1>(1));
      ipiv_buf_vec.emplace_back(&ipiv_vec[i * n], cl::sycl::range<1>(n));
      work_buf_vec.emplace_back(cl::sycl::range<1>(lwork_vec[i]));
    }
    mkl::getri_batch(exec_queue, n_vec, b_buf_vec, ldb_vec, ipiv_buf_vec,
                     work_buf_vec, lwork_vec, info_buf_vec);
  }

  // Copy back to the original buffers while casting variables from int64_t to
  // int
  copy_back(ipiv, ipiv_vec);
  copy_back(info, info_vec);
}

/// Computes the QR factorizations of a batch of general matrices.
/// \param [in] exec_queue The queue where the routine should be executed.
/// \param [in] m The number of rows of the batch matrices.
/// \param [in] n The number of columns of the batch matrices.
/// \param [in, out] a A pointer array, each pointer points a matrix. These
/// matrices will be overwritten by the factorization data.
/// \param [in] lda The leading dimension of the batch matrices in \param a.
/// \param [out] tau An array stores the scalars.
/// \param [out] info An array stores the error information.
/// \param [in] batch_size The size of the batch.
template <typename T>
inline void geqrf_batch_wrapper(cl::sycl::queue exec_queue, int m, int n,
                                T *a[], int lda, T *tau[], int *info,
                                int batchSize) {
  using Ty = typename DataType<T>::T2;
  // Need construct std::vector to store m, n and lda
  std::vector<int64_t> m_vec = std::vector<int64_t>(batchSize, m);
  std::vector<int64_t> n_vec = std::vector<int64_t>(batchSize, n);
  std::vector<int64_t> lda_vec = std::vector<int64_t>(batchSize, lda);
  std::vector<int64_t> info_vec(batchSize, 0);
  std::vector<int64_t> lwork_vec(batchSize, 0);

  std::vector<cl::sycl::buffer<Ty, 1>> a_buf_vec;
  std::vector<cl::sycl::buffer<Ty, 1>> tau_buf_vec;
  for (int64_t i = 0; i < batchSize; i++) {
    // assumes data is in column-major order
    auto allocation_a =
        dpct::memory_manager::get_instance().translate_ptr(a[i]);
    a_buf_vec.emplace_back(allocation_a.buffer.template reinterpret<Ty, 1>(
        cl::sycl::range<1>(allocation_a.size / sizeof(Ty))));
    auto allocation_tau =
        dpct::memory_manager::get_instance().translate_ptr(tau[i]);
    tau_buf_vec.emplace_back(allocation_tau.buffer.template reinterpret<Ty, 1>(
        cl::sycl::range<1>(allocation_tau.size / sizeof(Ty))));
  }

  // geqrf_get_lwork buffer block
  {
    std::vector<cl::sycl::buffer<int64_t>> lwork_buf_vec;
    for (int64_t i = 0; i < batchSize; i++) {
      lwork_buf_vec.emplace_back(&lwork_vec[i], cl::sycl::range<1>(1));
    }
    mkl::geqrf_get_lwork_batch(exec_queue, m_vec, n_vec, a_buf_vec, lda_vec,
                               tau_buf_vec, lwork_buf_vec);
  }

  // geqrf buffer block
  {
    std::vector<cl::sycl::buffer<Ty, 1>> work_buf_vec;
    std::vector<cl::sycl::buffer<int64_t, 1>> info_buf_vec;
    for (int64_t i = 0; i < batchSize; i++) {
      work_buf_vec.emplace_back(cl::sycl::range<1>(lwork_vec[i]));
      info_buf_vec.emplace_back(&info_vec[i], cl::sycl::range<1>(1));
    }
    mkl::geqrf_batch(exec_queue, m_vec, n_vec, a_buf_vec, lda_vec, tau_buf_vec,
                     work_buf_vec, lwork_vec, info_buf_vec);
  }

  // Copy back to the original buffers while casting variables from int64_t to
  // int
  copy_back(info, info_vec);
}

} // namespace dpct
#endif // __DPCT_BLAS_H__
