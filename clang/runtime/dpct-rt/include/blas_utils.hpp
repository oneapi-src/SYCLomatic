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
#include <thread>

namespace dpct {

namespace detail {
inline void mem_free(cl::sycl::queue *exec_queue,
                     std::vector<void *> pointers_array, cl::sycl::event e) {
  e.wait();
  for (auto p : pointers_array)
    cl::sycl::free(p, *exec_queue);
}

inline int stride_for(int num_elems, int mem_align_in_elems) {
  return ((num_elems - 1) / mem_align_in_elems + 1) * mem_align_in_elems;
}
}

inline oneapi::mkl::transpose get_transpose(int t) {
  if (t == 0) {
    return oneapi::mkl::transpose::nontrans;
  } else if (t == 1) {
    return oneapi::mkl::transpose::trans;
  } else {
    return oneapi::mkl::transpose::conjtrans;
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

/// Computes the LU factorizations of a batch of general matrices.
/// \param [in] exec_queue The queue where the routine should be executed.
/// \param [in] n The order of the matrices.
/// \param [in, out] a Array of pointers to matrices. These matrices will be
/// overwritten by lower triangulars with unit diagonal elements and upper
/// triangulars.
/// \param [in] lda The leading dimension of the matrices.
/// \param [out] ipiv An array stores the pivot indices.
/// \param [out] info An array stores the error information.
/// \param [in] batch_size The size of the batch.
template <typename T>
inline void getrf_batch_wrapper(cl::sycl::queue &exec_queue, int n, T *a[],
                                int lda, int *ipiv, int *info, int batch_size) {
  using Ty = typename DataType<T>::T2;
  // Set the info array value to 0
  detail::dpct_memset(exec_queue, info, 0, sizeof(int) * batch_size);
#ifdef DPCT_USM_LEVEL_NONE
  int mem_base_addr_align =
      exec_queue.get_device()
          .get_info<cl::sycl::info::device::mem_base_addr_align>();
  std::int64_t stride_a =
      detail::stride_for(n * lda, mem_base_addr_align / sizeof(T));
  std::int64_t stride_ipiv =
      detail::stride_for(n, mem_base_addr_align / sizeof(T));
  std::int64_t scratchpad_size = oneapi::mkl::lapack::getrf_batch_scratchpad_size<Ty>(
      exec_queue, n, n, lda, stride_a, stride_ipiv, batch_size);

  T *a_buffer_ptr;
  dpct_malloc((void **)&a_buffer_ptr, stride_a * batch_size * sizeof(T));

  T **host_a = (T **)malloc(batch_size * sizeof(T *));
  dpct_memcpy(host_a, a, batch_size * sizeof(T *));
  for (int64_t i = 0; i < batch_size; ++i)
    dpct_memcpy(a_buffer_ptr + i * stride_a, host_a[i], n * lda * sizeof(T));

  {
    cl::sycl::buffer<int64_t, 1> ipiv_buf(
        cl::sycl::range<1>(batch_size * stride_ipiv));
    cl::sycl::buffer<Ty, 1> scratchpad{cl::sycl::range<1>(scratchpad_size)};
    auto a_buffer = get_buffer<Ty>(a_buffer_ptr);
    oneapi::mkl::lapack::getrf_batch(exec_queue, n, n, a_buffer, lda, stride_a,
                             ipiv_buf, stride_ipiv, batch_size, scratchpad,
                             scratchpad_size);

    auto to_buffer = get_buffer<int>(ipiv);
    exec_queue.submit([&](cl::sycl::handler &cgh) {
      auto from_acc = ipiv_buf.get_access<cl::sycl::access::mode::read>(cgh);
      auto to_acc = to_buffer.get_access<cl::sycl::access::mode::write>(cgh);
      cgh.parallel_for<class device_int64_to_int>(
          cl::sycl::range<2>(batch_size, n), [=](cl::sycl::id<2> id) {
            to_acc[id.get(0) * n + id.get(1)] =
                static_cast<int>(from_acc[id.get(0) * stride_ipiv + id.get(1)]);
          });
    });
  }

  // Copy back to the original buffers
  cl::sycl::event e;
  for (int64_t i = 0; i < batch_size; ++i)
    e = detail::dpct_memcpy(exec_queue, host_a[i], a_buffer_ptr + i * stride_a,
                            n * lda * sizeof(T), automatic);

  std::vector<void *> ptrs{host_a};
  std::thread mem_free_thread(
      [=](std::vector<void *> pointers_array, cl::sycl::event e) {
        e.wait();
        for (auto p : pointers_array)
          free(p);
      },
      ptrs, e);
  mem_free_thread.detach();
#else
  std::int64_t m_int64 = n;
  std::int64_t n_int64 = n;
  std::int64_t lda_int64 = lda;
  std::int64_t group_sizes = batch_size;
  std::int64_t scratchpad_size = oneapi::mkl::lapack::getrf_batch_scratchpad_size<Ty>(
      exec_queue, &m_int64, &n_int64, &lda_int64, 1, &group_sizes);

  Ty *scratchpad = cl::sycl::malloc_device<Ty>(scratchpad_size, exec_queue);
  std::int64_t *ipiv_int64 =
      cl::sycl::malloc_device<std::int64_t>(batch_size * n, exec_queue);
  std::int64_t **ipiv_int64_ptr =
      cl::sycl::malloc_shared<std::int64_t *>(1, exec_queue);
  ipiv_int64_ptr[0] = ipiv_int64;

  oneapi::mkl::lapack::getrf_batch(exec_queue, &m_int64, &n_int64, (Ty **)a, &lda_int64,
                           ipiv_int64_ptr, 1, &group_sizes, scratchpad,
                           scratchpad_size);

  cl::sycl::event e = exec_queue.submit([&](cl::sycl::handler &cgh) {
    cgh.parallel_for<class device_int64_to_int>(
        cl::sycl::range<1>(batch_size * n), [=](cl::sycl::id<1> idx) {
          ipiv[idx] = static_cast<int>(ipiv_int64[idx]);
        });
  });

  std::vector<void *> ptrs{scratchpad, ipiv_int64, ipiv_int64_ptr};
  std::thread mem_free_thread(detail::mem_free, &exec_queue, ptrs, e);
  mem_free_thread.detach();
#endif
}

/// Solves a system of linear equations with a batch of LU-factored square
/// coefficient matrices, with multiple right-hand sides.
/// \param [in] exec_queue The queue where the routine should be executed.
/// \param [in] trans Indicates the form of the linear equations.
/// \param [in] n The order of the matrices.
/// \param [in] nrhs The number of right hand sides.
/// \param [in] a Array of pointers to matrices.
/// \param [in] lda The leading dimension of the matrices in \param a.
/// \param [in] ipiv An array stores the pivots.
/// \param [in, out] b Array of pointers to matrices, whose columns are
/// the right-hand sides for the systems of equations.
/// \param [in] ldb The leading dimension of the matrices in \param b.
/// \param [out] info A value stores the error information.
/// \param [in] batch_size The size of the batch.
template <typename T>
inline void getrs_batch_wrapper(cl::sycl::queue &exec_queue,
                                oneapi::mkl::transpose trans, int n, int nrhs,
                                const T *a[], int lda, int *ipiv, T *b[],
                                int ldb, int *info, int batch_size) {
  using Ty = typename DataType<T>::T2;
  // Set the info value to 0
  *info = 0;
#ifdef DPCT_USM_LEVEL_NONE
  int mem_base_addr_align =
      exec_queue.get_device()
          .get_info<cl::sycl::info::device::mem_base_addr_align>();
  std::int64_t stride_a =
      detail::stride_for(n * lda, mem_base_addr_align / sizeof(T));
  std::int64_t stride_b =
      detail::stride_for(nrhs * ldb, mem_base_addr_align / sizeof(T));
  std::int64_t stride_ipiv =
      detail::stride_for(n, mem_base_addr_align / sizeof(T));
  std::int64_t scratchpad_size = oneapi::mkl::lapack::getrs_batch_scratchpad_size<Ty>(
      exec_queue, trans, n, nrhs, lda, stride_a, stride_ipiv, ldb, stride_b,
      batch_size);

  T *a_buffer_ptr, *b_buffer_ptr;
  dpct_malloc((void **)&a_buffer_ptr, stride_a * batch_size * sizeof(T));
  dpct_malloc((void **)&b_buffer_ptr, stride_b * batch_size * sizeof(T));

  T **host_a = (T **)malloc(batch_size * sizeof(T *));
  T **host_b = (T **)malloc(batch_size * sizeof(T *));
  dpct_memcpy(host_a, a, batch_size * sizeof(T *));
  dpct_memcpy(host_b, b, batch_size * sizeof(T *));
  for (int64_t i = 0; i < batch_size; ++i) {
    dpct_memcpy(a_buffer_ptr + i * stride_a, host_a[i], n * lda * sizeof(T));
    dpct_memcpy(b_buffer_ptr + i * stride_b, host_b[i], nrhs * ldb * sizeof(T));
  }

  {
    auto a_buffer = get_buffer<Ty>(a_buffer_ptr);
    auto b_buffer = get_buffer<Ty>(b_buffer_ptr);
    cl::sycl::buffer<Ty, 1> scratchpad{cl::sycl::range<1>(scratchpad_size)};
    cl::sycl::buffer<int64_t, 1> ipiv_buf(
        cl::sycl::range<1>(batch_size * stride_ipiv));
    auto from_buf = get_buffer<int>(ipiv);
    exec_queue.submit([&](cl::sycl::handler &cgh) {
      auto from_acc = from_buf.get_access<cl::sycl::access::mode::read>(cgh);
      auto to_acc = ipiv_buf.get_access<cl::sycl::access::mode::write>(cgh);
      cgh.parallel_for<class device_int_to_int64>(
          cl::sycl::range<2>(batch_size, n), [=](cl::sycl::id<2> id) {
            to_acc[id.get(0) * stride_ipiv + id.get(1)] =
                static_cast<std::int64_t>(from_acc[id.get(0) * n + id.get(1)]);
          });
    });

    oneapi::mkl::lapack::getrs_batch(exec_queue, trans, n, nrhs, a_buffer, lda,
                             stride_a, ipiv_buf, stride_ipiv, b_buffer, ldb,
                             stride_b, batch_size, scratchpad, scratchpad_size);
  }

  // Copy back to the original buffers
  cl::sycl::event e;
  for (int64_t i = 0; i < batch_size; ++i)
    e = detail::dpct_memcpy(exec_queue, host_b[i], b_buffer_ptr + i * stride_b,
                            nrhs * ldb * sizeof(T), automatic);
  std::vector<void *> ptrs{host_a, host_b};
  std::thread mem_free_thread(
      [=](std::vector<void *> pointers_array, cl::sycl::event e) {
        e.wait();
        for (auto p : pointers_array)
          free(p);
      },
      ptrs, e);
  mem_free_thread.detach();
#else
  std::int64_t n_int64 = n;
  std::int64_t nrhs_int64 = nrhs;
  std::int64_t lda_int64 = lda;
  std::int64_t ldb_int64 = ldb;
  std::int64_t group_sizes = batch_size;
  std::int64_t scratchpad_size = oneapi::mkl::lapack::getrs_batch_scratchpad_size<Ty>(
      exec_queue, &trans, &n_int64, &nrhs_int64, &lda_int64, &ldb_int64, 1,
      &group_sizes);

  Ty *scratchpad = cl::sycl::malloc_device<Ty>(scratchpad_size, exec_queue);
  std::int64_t *ipiv_int64 =
      cl::sycl::malloc_device<std::int64_t>(batch_size * n, exec_queue);
  std::int64_t **ipiv_int64_ptr =
      cl::sycl::malloc_shared<std::int64_t *>(1, exec_queue);
  ipiv_int64_ptr[0] = ipiv_int64;

  exec_queue.submit([&](cl::sycl::handler &cgh) {
    cgh.parallel_for<class device_int_to_int64>(
        cl::sycl::range<1>(batch_size * n), [=](cl::sycl::id<1> idx) {
          ipiv_int64[idx] = static_cast<std::int64_t>(ipiv[idx]);
        });
  });

  cl::sycl::event e = oneapi::mkl::lapack::getrs_batch(
      exec_queue, &trans, &n_int64, &nrhs_int64, (Ty **)a, &lda_int64,
      ipiv_int64_ptr, (Ty **)b, &ldb_int64, 1, &group_sizes, scratchpad,
      scratchpad_size);

  std::vector<void *> ptrs{scratchpad, ipiv_int64_ptr, ipiv_int64};
  std::thread mem_free_thread(detail::mem_free, &exec_queue, ptrs, e);
  mem_free_thread.detach();
#endif
}

/// Computes the inverses of a batch of LU-factored matrices.
/// \param [in] exec_queue The queue where the routine should be executed.
/// \param [in] n The order of the matrices.
/// \param [in] a Array of pointers to matrices.
/// \param [in] lda The leading dimension of the matrices in \param a.
/// \param [in] ipiv An array stores the pivots.
/// \param [out] b Array of pointers to inverse matrices.
/// \param [in] ldb The leading dimension of the matrices in \param b.
/// \param [out] info An array stores the error information.
/// \param [in] batch_size The size of the batch.
template <typename T>
inline void getri_batch_wrapper(cl::sycl::queue &exec_queue, int n,
                                const T *a[], int lda, int *ipiv, T *b[],
                                int ldb, int *info, int batch_size) {
  using Ty = typename DataType<T>::T2;
  // Set the info array value to 0
  detail::dpct_memset(exec_queue, info, 0, sizeof(int) * batch_size);
#ifdef DPCT_USM_LEVEL_NONE
  int mem_base_addr_align =
      exec_queue.get_device()
          .get_info<cl::sycl::info::device::mem_base_addr_align>();
  std::int64_t stride_b =
      detail::stride_for(n * ldb, mem_base_addr_align / sizeof(T));
  std::int64_t stride_ipiv =
      detail::stride_for(n, mem_base_addr_align / sizeof(T));
  std::int64_t scratchpad_size = oneapi::mkl::lapack::getri_batch_scratchpad_size<Ty>(
      exec_queue, n, ldb, stride_b, stride_ipiv, batch_size);

  T *b_buffer_ptr;
  dpct_malloc((void **)&b_buffer_ptr, stride_b * batch_size * sizeof(T));

  T **host_a = (T **)malloc(batch_size * sizeof(T *));
  T **host_b = (T **)malloc(batch_size * sizeof(T *));
  dpct_memcpy(host_a, a, batch_size * sizeof(T *));
  dpct_memcpy(host_b, b, batch_size * sizeof(T *));

  for (int64_t i = 0; i < batch_size; ++i) {
    // Need to create a copy of input matrices A to keep them unchanged.
    // B (copy of A) will be used as input and output parameter in MKL API
    // call.
    matrix_mem_copy(b_buffer_ptr + i * stride_b, host_a[i], ldb, lda, n, n,
                    dpct::device_to_device, exec_queue);
  }

  {
    auto b_buffer = get_buffer<Ty>(b_buffer_ptr);
    cl::sycl::buffer<Ty, 1> scratchpad{cl::sycl::range<1>(scratchpad_size)};
    cl::sycl::buffer<int64_t, 1> ipiv_buf(
        cl::sycl::range<1>(batch_size * stride_ipiv));
    auto from_buf = get_buffer<int>(ipiv);
    exec_queue.submit([&](cl::sycl::handler &cgh) {
      auto from_acc = from_buf.get_access<cl::sycl::access::mode::read>(cgh);
      auto to_acc = ipiv_buf.get_access<cl::sycl::access::mode::write>(cgh);
      cgh.parallel_for<class device_int_to_int64>(
          cl::sycl::range<2>(batch_size, n), [=](cl::sycl::id<2> id) {
            to_acc[id.get(0) * stride_ipiv + id.get(1)] =
                static_cast<std::int64_t>(from_acc[id.get(0) * n + id.get(1)]);
          });
    });

    oneapi::mkl::lapack::getri_batch(exec_queue, n, b_buffer, ldb, stride_b, ipiv_buf,
                             stride_ipiv, batch_size, scratchpad,
                             scratchpad_size);
  }

  // Copy back to the original buffers
  cl::sycl::event e;
  for (int64_t i = 0; i < batch_size; ++i)
    e = detail::dpct_memcpy(exec_queue, host_b[i], b_buffer_ptr + i * stride_b,
                            n * ldb * sizeof(T), automatic);
  std::vector<void *> ptrs{host_a, host_b};
  std::thread mem_free_thread(
      [=](std::vector<void *> pointers_array, cl::sycl::event e) {
        e.wait();
        for (auto p : pointers_array)
          free(p);
      },
      ptrs, e);
  mem_free_thread.detach();
#else
  std::int64_t n_int64 = n;
  std::int64_t ldb_int64 = ldb;
  std::int64_t group_sizes = batch_size;
  std::int64_t scratchpad_size = oneapi::mkl::lapack::getri_batch_scratchpad_size<Ty>(
      exec_queue, &n_int64, &ldb_int64, 1, &group_sizes);

  Ty *scratchpad = cl::sycl::malloc_device<Ty>(scratchpad_size, exec_queue);
  std::int64_t *ipiv_int64 =
      cl::sycl::malloc_device<std::int64_t>(batch_size * n, exec_queue);
  std::int64_t **ipiv_int64_ptr =
      cl::sycl::malloc_shared<std::int64_t *>(1, exec_queue);
  ipiv_int64_ptr[0] = ipiv_int64;

  exec_queue.submit([&](cl::sycl::handler &cgh) {
    cgh.parallel_for<class device_int_to_int64>(
        cl::sycl::range<1>(batch_size * n), [=](cl::sycl::id<1> idx) {
          ipiv_int64[idx] = static_cast<std::int64_t>(ipiv[idx]);
        });
  });

  for (int64_t i = 0; i < batch_size; ++i) {
    // Need to create a copy of input matrices A to keep them unchanged.
    // B (copy of A) will be used as input and output parameter in MKL API
    // call.
    matrix_mem_copy(b[i], a[i], ldb, lda, n, n, dpct::device_to_device,
                    exec_queue);
  }

  cl::sycl::event e = oneapi::mkl::lapack::getri_batch(
      exec_queue, &n_int64, (Ty **)b, &ldb_int64, ipiv_int64_ptr, 1,
      &group_sizes, scratchpad, scratchpad_size);

  std::vector<void *> ptrs{scratchpad, ipiv_int64_ptr, ipiv_int64};
  std::thread mem_free_thread(detail::mem_free, &exec_queue, ptrs, e);
  mem_free_thread.detach();
#endif
}

/// Computes the QR factorizations of a batch of general matrices.
/// \param [in] exec_queue The queue where the routine should be executed.
/// \param [in] m The number of rows in the matrices.
/// \param [in] n The number of columns in the matrices.
/// \param [in, out] a Array of pointers to matrices. These
/// matrices will be overwritten by the factorization data.
/// \param [in] lda The leading dimension of the matrices in \param a.
/// \param [out] tau An array stores the scalars.
/// \param [out] info A value stores the error information.
/// \param [in] batch_size The size of the batch.
template <typename T>
inline void geqrf_batch_wrapper(cl::sycl::queue exec_queue, int m, int n,
                                T *a[], int lda, T *tau[], int *info,
                                int batch_size) {
  using Ty = typename DataType<T>::T2;
  // Set the info value to 0
  *info = 0;
#ifdef DPCT_USM_LEVEL_NONE
  int mem_base_addr_align =
      exec_queue.get_device()
          .get_info<cl::sycl::info::device::mem_base_addr_align>();
  std::int64_t stride_a =
      detail::stride_for(n * lda, mem_base_addr_align / sizeof(T));
  std::int64_t stride_tau = detail::stride_for(std::max(1, std::min(m, n)),
                                               mem_base_addr_align / sizeof(T));
  std::int64_t scratchpad_size = oneapi::mkl::lapack::geqrf_batch_scratchpad_size<Ty>(
      exec_queue, m, n, lda, stride_a, stride_tau, batch_size);

  T *a_buffer_ptr, *tau_buffer_ptr;
  dpct_malloc((void **)&a_buffer_ptr, stride_a * batch_size * sizeof(T));
  dpct_malloc((void **)&tau_buffer_ptr, stride_tau * batch_size * sizeof(T));

  T **host_a = (T **)malloc(batch_size * sizeof(T *));
  T **host_tau = (T **)malloc(batch_size * sizeof(T *));
  dpct_memcpy(host_a, a, batch_size * sizeof(T *));
  dpct_memcpy(host_tau, tau, batch_size * sizeof(T *));

  for (int64_t i = 0; i < batch_size; ++i)
    dpct_memcpy(a_buffer_ptr + i * stride_a, host_a[i], n * lda * sizeof(T));
  {
    auto a_buffer = get_buffer<Ty>(a_buffer_ptr);
    auto tau_buffer = get_buffer<Ty>(tau_buffer_ptr);
    cl::sycl::buffer<Ty, 1> scratchpad{cl::sycl::range<1>(scratchpad_size)};
    oneapi::mkl::lapack::geqrf_batch(exec_queue, m, n, a_buffer, lda, stride_a,
                             tau_buffer, stride_tau, batch_size, scratchpad,
                             scratchpad_size);
  }

  // Copy back to the original buffers
  cl::sycl::event e_a;
  cl::sycl::event e_tau;
  for (int64_t i = 0; i < batch_size; ++i) {
    e_a =
        detail::dpct_memcpy(exec_queue, host_a[i], a_buffer_ptr + i * stride_a,
                            n * lda * sizeof(T), automatic);
    e_tau = detail::dpct_memcpy(
        exec_queue, host_tau[i], tau_buffer_ptr + i * stride_tau,
        std::max(1, std::min(m, n)) * sizeof(T), automatic);
  }
  std::vector<void *> ptr_a{host_a};
  std::vector<void *> ptr_tau{host_tau};
  std::thread mem_free_thread_a(
      [=](std::vector<void *> pointers_array, cl::sycl::event e) {
        e.wait();
        for (auto p : pointers_array)
          free(p);
      },
      ptr_a, e_a);
  std::thread mem_free_thread_tau(
      [=](std::vector<void *> pointers_array, cl::sycl::event e) {
        e.wait();
        for (auto p : pointers_array)
          free(p);
      },
      ptr_tau, e_tau);
  mem_free_thread_a.detach();
  mem_free_thread_tau.detach();
#else
  std::int64_t m_int64 = n;
  std::int64_t n_int64 = n;
  std::int64_t lda_int64 = lda;
  std::int64_t group_sizes = batch_size;
  std::int64_t scratchpad_size = oneapi::mkl::lapack::geqrf_batch_scratchpad_size<Ty>(
      exec_queue, &m_int64, &n_int64, &lda_int64, 1, &group_sizes);

  Ty *scratchpad = cl::sycl::malloc_device<Ty>(scratchpad_size, exec_queue);

  cl::sycl::event e = oneapi::mkl::lapack::geqrf_batch(
      exec_queue, &m_int64, &n_int64, (Ty **)a, &lda_int64, (Ty **)tau, 1,
      &group_sizes, scratchpad, scratchpad_size);

  std::vector<void *> ptrs{scratchpad};
  std::thread mem_free_thread(detail::mem_free, &exec_queue, ptrs, e);
  mem_free_thread.detach();
#endif
}

} // namespace dpct
#endif // __DPCT_BLAS_HPP__
