// DPCT_LABEL_BEGIN|License|
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
//==---- blas_utils.hpp----------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
// DPCT_LABEL_END

#ifndef __DPCT_BLAS_HPP__
#define __DPCT_BLAS_HPP__
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
// DPCT_COMMENT   non_local_include_dependency: other headler files

// DPCT_LABEL_BEGIN|local_include_dependency|
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
// DPCT_LABEL_END
#include "memory.hpp"
#include "util.hpp"
// DPCT_LABEL_BEGIN|non_local_include_dependency|
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
#include <CL/sycl.hpp>
#include <oneapi/mkl.hpp>
#include <utility>
#include <vector>
#include <thread>
// DPCT_LABEL_END

namespace dpct {

namespace detail {
// DPCT_LABEL_BEGIN|mem_free|dpct::detail
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
inline void mem_free(cl::sycl::queue *exec_queue,
                     std::vector<void *> pointers_array, cl::sycl::event e) {
  e.wait();
  for (auto p : pointers_array)
    cl::sycl::free(p, *exec_queue);
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|stride_for|dpct::detail
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
inline int stride_for(int num_elems, int mem_align_in_elems) {
  return ((num_elems - 1) / mem_align_in_elems + 1) * mem_align_in_elems;
}
// DPCT_LABEL_END
}

// DPCT_LABEL_BEGIN|get_transpose|dpct
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
inline oneapi::mkl::transpose get_transpose(int t) {
  if (t == 0) {
    return oneapi::mkl::transpose::nontrans;
  } else if (t == 1) {
    return oneapi::mkl::transpose::trans;
  } else {
    return oneapi::mkl::transpose::conjtrans;
  }
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|get_value|dpct
// DPCT_DEPENDENCY_BEGIN
// Memory|dpct_memcpy_detail
// Util|DataType
// DPCT_DEPENDENCY_END
// DPCT_CODE
/// Get the value of \p s.
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
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|getrf_batch_wrapper|dpct
// DPCT_DEPENDENCY_BEGIN
// Memory|dpct_malloc|UsmNone
// Memory|dpct_memcpy|UsmNone
// Memory|dpct_memcpy_detail|UsmNone
// Memory|async_dpct_free|UsmRestricted
// Util|DataType
// Memory|dpct_memset_detail
// Memory|get_buffer_T|UsmNone
// Dpct|dpct_named_lambda
// DPCT_DEPENDENCY_END
// DPCT_CODE
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
  std::int64_t stride_a = n * lda;
  std::int64_t stride_ipiv = n;
  std::int64_t scratchpad_size = oneapi::mkl::lapack::getrf_batch_scratchpad_size<Ty>(
      exec_queue, n, n, lda, stride_a, stride_ipiv, batch_size);

  T *a_buffer_ptr;
  a_buffer_ptr = (T *)dpct_malloc(stride_a * batch_size * sizeof(T));

  T **host_a = (T **)malloc(batch_size * sizeof(T *));
  dpct_memcpy(host_a, a, batch_size * sizeof(T *));
  for (std::int64_t i = 0; i < batch_size; ++i)
    dpct_memcpy(a_buffer_ptr + i * stride_a, host_a[i], n * lda * sizeof(T));

  {
    cl::sycl::buffer<std::int64_t, 1> ipiv_buf(
        cl::sycl::range<1>(batch_size * stride_ipiv));
    cl::sycl::buffer<Ty, 1> scratchpad{cl::sycl::range<1>(scratchpad_size)};
    auto a_buffer = get_buffer<Ty>(a_buffer_ptr);
    oneapi::mkl::lapack::getrf_batch(exec_queue, n, n, a_buffer, lda, stride_a,
                             ipiv_buf, stride_ipiv, batch_size, scratchpad,
                             scratchpad_size);

    auto to_buffer = get_buffer<int>(ipiv);
    exec_queue.submit([&](cl::sycl::handler &cgh) {
      auto from_acc = ipiv_buf.get_access<cl::sycl::access_mode::read>(cgh);
      auto to_acc = to_buffer.get_access<cl::sycl::access_mode::write>(cgh);
      cgh.parallel_for<dpct_kernel_name<class getrf_device_int64_to_int, T>>(
          cl::sycl::range<2>(batch_size, n), [=](cl::sycl::id<2> id) {
            to_acc[id.get(0) * n + id.get(1)] =
                static_cast<int>(from_acc[id.get(0) * stride_ipiv + id.get(1)]);
          });
    });
  }

  // Copy back to the original buffers
  std::vector<cl::sycl::event> events;
  for (std::int64_t i = 0; i < batch_size; ++i)
    events.push_back(detail::dpct_memcpy(exec_queue, host_a[i],
                                         a_buffer_ptr + i * stride_a,
                                         n * lda * sizeof(T), automatic));

  std::vector<void *> ptrs{host_a};
  std::thread mem_free_thread(
      [=](std::vector<void *> pointers_array,
          std::vector<cl::sycl::event> events_array) {
        cl::sycl::event::wait(events_array);
        for (auto p : pointers_array)
          free(p);
      },
      ptrs, events);
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
      cl::sycl::malloc_shared<std::int64_t *>(batch_size, exec_queue);
  T **a_shared = cl::sycl::malloc_shared<T *>(batch_size, exec_queue);
  exec_queue.memcpy(a_shared, a, batch_size * sizeof(T *)).wait();
  for (std::int64_t i = 0; i < batch_size; ++i)
    ipiv_int64_ptr[i] = ipiv_int64 + n * i;

  oneapi::mkl::lapack::getrf_batch(exec_queue, &m_int64, &n_int64, (Ty **)a_shared, &lda_int64,
                           ipiv_int64_ptr, 1, &group_sizes, scratchpad,
                           scratchpad_size);

  cl::sycl::event e = exec_queue.submit([&](cl::sycl::handler &cgh) {
    cgh.parallel_for<dpct_kernel_name<class getrf_device_int64_to_int, T>>(
        cl::sycl::range<1>(batch_size * n), [=](cl::sycl::id<1> idx) {
          ipiv[idx] = static_cast<int>(ipiv_int64[idx]);
        });
  });

  std::vector<void *> ptrs{scratchpad, ipiv_int64, ipiv_int64_ptr, a_shared};
  async_dpct_free(ptrs, {e}, exec_queue);
#endif
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|getrs_batch_wrapper|dpct
// DPCT_DEPENDENCY_BEGIN
// Memory|dpct_malloc|UsmNone
// Memory|dpct_memcpy|UsmNone
// Memory|dpct_memcpy_detail|UsmNone
// Memory|async_dpct_free|UsmRestricted
// Util|DataType
// Memory|get_buffer_T|UsmNone
// Dpct|dpct_named_lambda
// DPCT_DEPENDENCY_END
// DPCT_CODE
/// Solves a system of linear equations with a batch of LU-factored square
/// coefficient matrices, with multiple right-hand sides.
/// \param [in] exec_queue The queue where the routine should be executed.
/// \param [in] trans Indicates the form of the linear equations.
/// \param [in] n The order of the matrices.
/// \param [in] nrhs The number of right hand sides.
/// \param [in] a Array of pointers to matrices.
/// \param [in] lda The leading dimension of the matrices in \p a.
/// \param [in] ipiv An array stores the pivots.
/// \param [in, out] b Array of pointers to matrices, whose columns are
/// the right-hand sides for the systems of equations.
/// \param [in] ldb The leading dimension of the matrices in \p b.
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
  std::int64_t stride_a = n * lda;
  std::int64_t stride_b = nrhs * ldb;
  std::int64_t stride_ipiv = n;
  std::int64_t scratchpad_size = oneapi::mkl::lapack::getrs_batch_scratchpad_size<Ty>(
      exec_queue, trans, n, nrhs, lda, stride_a, stride_ipiv, ldb, stride_b,
      batch_size);

  T *a_buffer_ptr, *b_buffer_ptr;
  a_buffer_ptr = (T *)dpct_malloc(stride_a * batch_size * sizeof(T));
  b_buffer_ptr = (T *)dpct_malloc(stride_b * batch_size * sizeof(T));

  T **host_a = (T **)malloc(batch_size * sizeof(T *));
  T **host_b = (T **)malloc(batch_size * sizeof(T *));
  dpct_memcpy(host_a, a, batch_size * sizeof(T *));
  dpct_memcpy(host_b, b, batch_size * sizeof(T *));
  for (std::int64_t i = 0; i < batch_size; ++i) {
    dpct_memcpy(a_buffer_ptr + i * stride_a, host_a[i], n * lda * sizeof(T));
    dpct_memcpy(b_buffer_ptr + i * stride_b, host_b[i], nrhs * ldb * sizeof(T));
  }

  {
    auto a_buffer = get_buffer<Ty>(a_buffer_ptr);
    auto b_buffer = get_buffer<Ty>(b_buffer_ptr);
    cl::sycl::buffer<Ty, 1> scratchpad{cl::sycl::range<1>(scratchpad_size)};
    cl::sycl::buffer<std::int64_t, 1> ipiv_buf(
        cl::sycl::range<1>(batch_size * stride_ipiv));
    auto from_buf = get_buffer<int>(ipiv);
    exec_queue.submit([&](cl::sycl::handler &cgh) {
      auto from_acc = from_buf.get_access<cl::sycl::access_mode::read>(cgh);
      auto to_acc = ipiv_buf.get_access<cl::sycl::access_mode::write>(cgh);
      cgh.parallel_for<dpct_kernel_name<class getrs_device_int64_to_int, T>>(
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
  std::vector<cl::sycl::event> events;
  for (std::int64_t i = 0; i < batch_size; ++i)
    events.push_back(detail::dpct_memcpy(exec_queue, host_b[i],
                                         b_buffer_ptr + i * stride_b,
                                         nrhs * ldb * sizeof(T), automatic));
  std::vector<void *> ptrs{host_a, host_b};
  std::thread mem_free_thread(
      [=](std::vector<void *> pointers_array,
          std::vector<cl::sycl::event> events_array) {
        cl::sycl::event::wait(events_array);
        for (auto p : pointers_array)
          free(p);
      },
      ptrs, events);
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
      cl::sycl::malloc_shared<std::int64_t *>(batch_size, exec_queue);
  T **a_shared = cl::sycl::malloc_shared<T *>(batch_size, exec_queue);
  T **b_shared = cl::sycl::malloc_shared<T *>(batch_size, exec_queue);
  exec_queue.memcpy(a_shared, a, batch_size * sizeof(T *));
  exec_queue.memcpy(b_shared, b, batch_size * sizeof(T *));

  exec_queue.submit([&](cl::sycl::handler &cgh) {
    cgh.parallel_for<dpct_kernel_name<class getrs_device_int64_to_int, T>>(
        cl::sycl::range<1>(batch_size * n), [=](cl::sycl::id<1> idx) {
          ipiv_int64[idx] = static_cast<std::int64_t>(ipiv[idx]);
        });
  }).wait();

  for (std::int64_t i = 0; i < batch_size; ++i)
    ipiv_int64_ptr[i] = ipiv_int64 + n * i;

  cl::sycl::event e = oneapi::mkl::lapack::getrs_batch(
      exec_queue, &trans, &n_int64, &nrhs_int64, (Ty **)a_shared, &lda_int64,
      ipiv_int64_ptr, (Ty **)b_shared, &ldb_int64, 1, &group_sizes, scratchpad,
      scratchpad_size);

  std::vector<void *> ptrs{scratchpad, ipiv_int64_ptr, ipiv_int64, a_shared, b_shared};
  async_dpct_free(ptrs, {e}, exec_queue);
#endif
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|getri_batch_wrapper|dpct
// DPCT_DEPENDENCY_BEGIN
// Memory|dpct_malloc|UsmNone
// Memory|dpct_memcpy|UsmNone
// Memory|dpct_memcpy_detail|UsmNone
// Memory|async_dpct_free|UsmRestricted
// Util|DataType
// Util|matrix_mem_copy_T
// Memory|dpct_memset_detail
// Memory|get_buffer_T|UsmNone
// Dpct|dpct_named_lambda
// DPCT_DEPENDENCY_END
// DPCT_CODE
/// Computes the inverses of a batch of LU-factored matrices.
/// \param [in] exec_queue The queue where the routine should be executed.
/// \param [in] n The order of the matrices.
/// \param [in] a Array of pointers to matrices.
/// \param [in] lda The leading dimension of the matrices in \p a.
/// \param [in] ipiv An array stores the pivots.
/// \param [out] b Array of pointers to inverse matrices.
/// \param [in] ldb The leading dimension of the matrices in \p b.
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
  std::int64_t stride_b = n * ldb;
  std::int64_t stride_ipiv = n;
  std::int64_t scratchpad_size = oneapi::mkl::lapack::getri_batch_scratchpad_size<Ty>(
      exec_queue, n, ldb, stride_b, stride_ipiv, batch_size);

  T *b_buffer_ptr;
  b_buffer_ptr = (T *)dpct_malloc(stride_b * batch_size * sizeof(T));

  T **host_a = (T **)malloc(batch_size * sizeof(T *));
  T **host_b = (T **)malloc(batch_size * sizeof(T *));
  dpct_memcpy(host_a, a, batch_size * sizeof(T *));
  dpct_memcpy(host_b, b, batch_size * sizeof(T *));

  for (std::int64_t i = 0; i < batch_size; ++i) {
    // Need to create a copy of input matrices "a" to keep them unchanged.
    // Matrices "b" (copy of matrices "a") will be used as input and output
    // parameter in oneapi::mkl::lapack::getri_batch call.
    matrix_mem_copy(b_buffer_ptr + i * stride_b, host_a[i], ldb, lda, n, n,
                    dpct::device_to_device, exec_queue);
  }

  {
    auto b_buffer = get_buffer<Ty>(b_buffer_ptr);
    cl::sycl::buffer<Ty, 1> scratchpad{cl::sycl::range<1>(scratchpad_size)};
    cl::sycl::buffer<std::int64_t, 1> ipiv_buf(
        cl::sycl::range<1>(batch_size * stride_ipiv));
    auto from_buf = get_buffer<int>(ipiv);
    exec_queue.submit([&](cl::sycl::handler &cgh) {
      auto from_acc = from_buf.get_access<cl::sycl::access_mode::read>(cgh);
      auto to_acc = ipiv_buf.get_access<cl::sycl::access_mode::write>(cgh);
      cgh.parallel_for<dpct_kernel_name<class getri_device_int64_to_int, T>>(
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
  std::vector<cl::sycl::event> events;
  for (std::int64_t i = 0; i < batch_size; ++i)
    events.push_back(detail::dpct_memcpy(exec_queue, host_b[i],
                                         b_buffer_ptr + i * stride_b,
                                         n * ldb * sizeof(T), automatic));
  std::vector<void *> ptrs{host_a, host_b};
  std::thread mem_free_thread(
      [=](std::vector<void *> pointers_array,
          std::vector<cl::sycl::event> events_array) {
        cl::sycl::event::wait(events_array);
        for (auto p : pointers_array)
          free(p);
      },
      ptrs, events);
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
      cl::sycl::malloc_shared<std::int64_t *>(batch_size, exec_queue);

  exec_queue.submit([&](cl::sycl::handler &cgh) {
    cgh.parallel_for<dpct_kernel_name<class getri_device_int64_to_int, T>>(
        cl::sycl::range<1>(batch_size * n), [=](cl::sycl::id<1> idx) {
          ipiv_int64[idx] = static_cast<std::int64_t>(ipiv[idx]);
        });
  });

  T **a_shared = cl::sycl::malloc_shared<T *>(batch_size, exec_queue);
  T **b_shared = cl::sycl::malloc_shared<T *>(batch_size, exec_queue);
  exec_queue.memcpy(a_shared, a, batch_size * sizeof(T *));
  exec_queue.memcpy(b_shared, b, batch_size * sizeof(T *)).wait();
  for (std::int64_t i = 0; i < batch_size; ++i) {
    ipiv_int64_ptr[i] = ipiv_int64 + n * i;
    // Need to create a copy of input matrices "a" to keep them unchanged.
    // Matrices "b" (copy of matrices "a") will be used as input and output
    // parameter in oneapi::mkl::lapack::getri_batch call.
    matrix_mem_copy(b_shared[i], a_shared[i], ldb, lda, n, n, dpct::device_to_device,
                    exec_queue);
  }

  cl::sycl::event e = oneapi::mkl::lapack::getri_batch(
      exec_queue, &n_int64, (Ty **)b_shared, &ldb_int64, ipiv_int64_ptr, 1,
      &group_sizes, scratchpad, scratchpad_size);

  std::vector<void *> ptrs{scratchpad, ipiv_int64_ptr, ipiv_int64, a_shared, b_shared};
  async_dpct_free(ptrs, {e}, exec_queue);
#endif
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|geqrf_batch_wrapper|dpct
// DPCT_DEPENDENCY_BEGIN
// Memory|dpct_malloc|UsmNone
// Memory|dpct_memcpy|UsmNone
// Memory|dpct_memcpy_detail|UsmNone
// Memory|async_dpct_free|UsmRestricted
// Util|DataType
// Memory|get_buffer_T|UsmNone
// DPCT_DEPENDENCY_END
// DPCT_CODE
/// Computes the QR factorizations of a batch of general matrices.
/// \param [in] exec_queue The queue where the routine should be executed.
/// \param [in] m The number of rows in the matrices.
/// \param [in] n The number of columns in the matrices.
/// \param [in, out] a Array of pointers to matrices. These
/// matrices will be overwritten by the factorization data.
/// \param [in] lda The leading dimension of the matrices in \p a.
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
  std::int64_t stride_a = n * lda;
  std::int64_t stride_tau = std::max(1, std::min(m, n));
  std::int64_t scratchpad_size = oneapi::mkl::lapack::geqrf_batch_scratchpad_size<Ty>(
      exec_queue, m, n, lda, stride_a, stride_tau, batch_size);

  T *a_buffer_ptr, *tau_buffer_ptr;
  a_buffer_ptr = (T *)dpct_malloc(stride_a * batch_size * sizeof(T));
  tau_buffer_ptr = (T *)dpct_malloc(stride_tau * batch_size * sizeof(T));

  T **host_a = (T **)malloc(batch_size * sizeof(T *));
  T **host_tau = (T **)malloc(batch_size * sizeof(T *));
  dpct_memcpy(host_a, a, batch_size * sizeof(T *));
  dpct_memcpy(host_tau, tau, batch_size * sizeof(T *));

  for (std::int64_t i = 0; i < batch_size; ++i)
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
  std::vector<cl::sycl::event> events_a;
  std::vector<cl::sycl::event> events_tau;
  for (std::int64_t i = 0; i < batch_size; ++i) {
    events_a.push_back(detail::dpct_memcpy(exec_queue, host_a[i],
                                           a_buffer_ptr + i * stride_a,
                                           n * lda * sizeof(T), automatic));
    events_tau.push_back(detail::dpct_memcpy(
        exec_queue, host_tau[i], tau_buffer_ptr + i * stride_tau,
        std::max(1, std::min(m, n)) * sizeof(T), automatic));
  }
  std::vector<void *> ptr_a{host_a};
  std::vector<void *> ptr_tau{host_tau};
  std::thread mem_free_thread_a(
      [=](std::vector<void *> pointers_array,
          std::vector<cl::sycl::event> events_array) {
        cl::sycl::event::wait(events_array);
        for (auto p : pointers_array)
          free(p);
      },
      ptr_a, events_a);
  std::thread mem_free_thread_tau(
      [=](std::vector<void *> pointers_array,
          std::vector<cl::sycl::event> events_array) {
        cl::sycl::event::wait(events_array);
        for (auto p : pointers_array)
          free(p);
      },
      ptr_tau, events_tau);
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
  T **a_shared = cl::sycl::malloc_shared<T *>(batch_size, exec_queue);
  T **tau_shared = cl::sycl::malloc_shared<T *>(batch_size, exec_queue);
  exec_queue.memcpy(a_shared, a, batch_size * sizeof(T *));
  exec_queue.memcpy(tau_shared, tau, batch_size * sizeof(T *)).wait();

  cl::sycl::event e = oneapi::mkl::lapack::geqrf_batch(
      exec_queue, &m_int64, &n_int64, (Ty **)a_shared, &lda_int64, (Ty **)tau_shared, 1,
      &group_sizes, scratchpad, scratchpad_size);

  std::vector<void *> ptrs{scratchpad, a_shared, tau_shared};
  async_dpct_free(ptrs, {e}, exec_queue);
#endif
}
// DPCT_LABEL_END

} // namespace dpct
#endif // __DPCT_BLAS_HPP__
