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

//===--- dpct_blas.hpp ------------------------------*- C++ -*---===//

#ifndef __DPCT_BLAS_H__
#define __DPCT_BLAS_H__

#include "memory.hpp"
#include "util.hpp"
#include <CL/sycl.hpp>
#include <mkl_lapack_batch_sycl.hpp>
#include <utility>
#include <vector>

namespace dpct {

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

template <typename T>
inline void getrf_batch_wrapper(cl::sycl::queue &queue, int n, T *a_array[],
                                int lda, int *pivot, int *info, int batchSize) {
  using Ty = typename DataType<T>::T2;
  // Need construct std::vector to store m, n and lda
  std::vector<int64_t> mn_vec = std::vector<int64_t>(batchSize, n);
  std::vector<int64_t> lda_vec = std::vector<int64_t>(batchSize, lda);
  std::vector<int64_t> info_vec(batchSize, 0);
  std::vector<int64_t> pivot_vec(batchSize * n, 0);

  // geqrf buffer block
  {
    std::vector<cl::sycl::buffer<Ty, 1>> a_buf_vec;
    std::vector<cl::sycl::buffer<int64_t, 1>> info_buf_vec;
    std::vector<cl::sycl::buffer<int64_t, 1>> pivot_buf_vec;
    for (int64_t i = 0; i < batchSize; ++i) {
      // assumes data is in column-major order
      auto allocation_Aarray =
          dpct::memory_manager::get_instance().translate_ptr(a_array[i]);
      a_buf_vec.emplace_back(
          allocation_Aarray.buffer.template reinterpret<Ty, 1>(
              cl::sycl::range<1>(allocation_Aarray.size / sizeof(Ty))));
      info_buf_vec.emplace_back(&info_vec[i], cl::sycl::range<1>(1));
      pivot_buf_vec.emplace_back(&pivot_vec[i * n], cl::sycl::range<1>(n));
    }
    mkl::getrf_batch(queue, mn_vec, mn_vec, a_buf_vec, lda_vec, pivot_buf_vec,
                     info_buf_vec);
  }

  // Copy back to the original buffers while casting variables from int64_t to
  // int
  copy_back(pivot, pivot_vec);
  copy_back(info, info_vec);
}

template <typename T>
inline void getri_batch_wrapper(cl::sycl::queue &queue, int n,
                                const T *a_array[], int lda, int *pivot,
                                T *c_array[], int ldc, int *info,
                                int batchSize) {
  using Ty = typename DataType<T>::T2;
  // Need construct std::vector to store n and lda
  std::vector<int64_t> n_vec = std::vector<int64_t>(batchSize, n);
  std::vector<int64_t> ldc_vec = std::vector<int64_t>(batchSize, ldc);
  std::vector<int64_t> info_vec(batchSize, 0);
  std::vector<int64_t> pivot_vec(batchSize * n, 0);
  std::vector<int64_t> lwork_vec(batchSize, 0);

  std::vector<cl::sycl::buffer<Ty, 1>> c_buf_vec;
  for (int64_t i = 0; i < batchSize; i++) {
    // Original code is input A and output C while MKL API is input A and output
    // A
    // So copy A to C and use C as the parameter of MKL API.
    matrix_mem_copy(c_array[i], a_array[i], ldc, lda, n, n,
                    dpct::device_to_device);
    // assumes data is in column-major order
    auto allocation_c_array =
        dpct::memory_manager::get_instance().translate_ptr(c_array[i]);
    c_buf_vec.emplace_back(
        allocation_c_array.buffer.template reinterpret<Ty, 1>(
            cl::sycl::range<1>(allocation_c_array.size / sizeof(Ty))));
  }

  // getri_get_lwork buffer block
  {
    std::vector<cl::sycl::buffer<int64_t>> lwork_buf_vec;
    std::vector<cl::sycl::buffer<int64_t, 1>> pivot_buf_vec;
    for (int64_t i = 0; i < batchSize; ++i) {
      lwork_buf_vec.emplace_back(&lwork_vec[i], cl::sycl::range<1>(1));
      pivot_buf_vec.emplace_back(&pivot_vec[i * n], cl::sycl::range<1>(n));
    }
    mkl::getri_get_lwork_batch(queue, n_vec, c_buf_vec, ldc_vec, pivot_buf_vec,
                               lwork_buf_vec);
  }

  // getri buffer block
  {
    std::vector<cl::sycl::buffer<int64_t, 1>> info_buf_vec;
    std::vector<cl::sycl::buffer<int64_t, 1>> pivot_buf_vec;
    std::vector<cl::sycl::buffer<int64_t>> lwork_buf_vec;
    std::vector<cl::sycl::buffer<Ty, 1>> work_buf_vec;
    for (int64_t i = 0; i < batchSize; ++i) {
      info_buf_vec.emplace_back(&info_vec[i], cl::sycl::range<1>(1));
      pivot_buf_vec.emplace_back(&pivot_vec[i * n], cl::sycl::range<1>(n));
      lwork_buf_vec.emplace_back(&lwork_vec[i], cl::sycl::range<1>(1));
      work_buf_vec.emplace_back(cl::sycl::range<1>(lwork_vec[i]));
    }
    mkl::getri_batch(queue, n_vec, c_buf_vec, ldc_vec, pivot_buf_vec,
                     work_buf_vec, lwork_buf_vec, info_buf_vec);
  }

  // Copy back to the original buffers while casting variables from int64_t to
  // int
  copy_back(pivot, pivot_vec);
  copy_back(info, info_vec);
}

template <typename T>
inline void geqrf_batch_wrapper(cl::sycl::queue queue, int m, int n,
                                T *a_array[], int lda, T *tau_array[],
                                int *info, int batchSize) {
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
    auto allocation_a_array =
        dpct::memory_manager::get_instance().translate_ptr(a_array[i]);
    a_buf_vec.emplace_back(
        allocation_a_array.buffer.template reinterpret<Ty, 1>(
            cl::sycl::range<1>(allocation_a_array.size / sizeof(Ty))));
    auto allocation_tau_array =
        dpct::memory_manager::get_instance().translate_ptr(tau_array[i]);
    tau_buf_vec.emplace_back(
        allocation_tau_array.buffer.template reinterpret<Ty, 1>(
            cl::sycl::range<1>(allocation_tau_array.size / sizeof(Ty))));
  }

  // geqrf_get_lwork buffer block
  {
    std::vector<cl::sycl::buffer<int64_t>> lwork_buf_vec;
    for (int64_t i = 0; i < batchSize; i++) {
      lwork_buf_vec.emplace_back(&lwork_vec[i], cl::sycl::range<1>(1));
    }
    mkl::geqrf_get_lwork_batch(queue, m_vec, n_vec, a_buf_vec, lda_vec,
                               tau_buf_vec, lwork_buf_vec);
  }

  // geqrf buffer block
  {
    std::vector<cl::sycl::buffer<int64_t>> lwork_buf_vec;
    std::vector<cl::sycl::buffer<Ty, 1>> work_buf_vec;
    std::vector<cl::sycl::buffer<int64_t, 1>> info_buf_vec;
    for (int64_t i = 0; i < batchSize; i++) {
      lwork_buf_vec.emplace_back(&lwork_vec[i], cl::sycl::range<1>(1));
      work_buf_vec.emplace_back(cl::sycl::range<1>(lwork_vec[i]));
      info_buf_vec.emplace_back(&info_vec[i], cl::sycl::range<1>(1));
    }
    mkl::geqrf_batch(queue, m_vec, n_vec, a_buf_vec, lda_vec, tau_buf_vec,
                     work_buf_vec, lwork_buf_vec, info_buf_vec);
  }

  // Copy back to the original buffers while casting variables from int64_t to
  // int
  copy_back(info, info_vec);
}

} // namespace dpct
#endif // __DPCT_BLAS_H__
