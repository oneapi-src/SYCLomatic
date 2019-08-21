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

//===--- atomic.hpp -------------------------------*- C++ -*---===//

#ifndef DPCT_ATOMIC_H
#define DPCT_ATOMIC_H

#include <CL/sycl.hpp>

namespace dpct {

/// Atomically add operand to *addr, Int version.
/// \param [in, out] addr Point to a data.
/// \param operand Be added to data pointed by \p addr.
/// \param memoryOrder The memory ordering used.
/// \return The old value addr point to.
template <cl::sycl::access::address_space addressSpace =
              cl::sycl::access::address_space::global_space,
          typename T>
inline T atomic_fetch_add(
    T *addr, T operand,
    cl::sycl::memory_order memoryOrder = cl::sycl::memory_order::relaxed) {
  cl::sycl::atomic<T, addressSpace> obj(
      (cl::sycl::multi_ptr<T, addressSpace>(addr)));
  return cl::sycl::atomic_fetch_add(obj, operand, memoryOrder);
}

/// Atomically add operand to *addr, Float version.
/// \param [in, out] addr Point to a data.
/// \param operand Be added to data pointed by \p addr.
/// \param memoryOrder The memory ordering used.
/// \return The old value addr point to.
template <cl::sycl::access::address_space addressSpace =
              cl::sycl::access::address_space::global_space>
inline float atomic_fetch_add(
    float *addr, float operand,
    cl::sycl::memory_order memoryOrder = cl::sycl::memory_order::relaxed) {
  static_assert(sizeof(float) == sizeof(int), "Mismatched type size");

  cl::sycl::atomic<int, addressSpace> obj(
      (cl::sycl::multi_ptr<int, addressSpace>(reinterpret_cast<int *>(addr))));

  const int old_value = obj.load(memoryOrder);
  const float old_float_value = *reinterpret_cast<const float *>(&old_value);
  const float new_float_value = old_float_value + operand;
  const int new_value = *reinterpret_cast<const int *>(&new_float_value);

  while (true) {
    int expected = old_value;
    if (obj.compare_exchange_strong(expected, new_value, memoryOrder))
      break;
  }

  return old_float_value;
}

/// Atomically add operand to *addr, Double version.
/// \param [in, out] addr Point to a data.
/// \param operand Be added to data pointed by \p addr.
/// \param memoryOrder The memory ordering used.
/// \return The old value addr point to.
template <cl::sycl::access::address_space addressSpace =
              cl::sycl::access::address_space::global_space>
inline double atomic_fetch_add(
    double *addr, double operand,
    cl::sycl::memory_order memoryOrder = cl::sycl::memory_order::relaxed) {
  static_assert(sizeof(double) == sizeof(unsigned long long int),
                "Mismatched type size");

  cl::sycl::atomic<unsigned long long int, addressSpace> obj(
      (cl::sycl::multi_ptr<unsigned long long int, addressSpace>(
          reinterpret_cast<unsigned long long int *>(addr))));

  const unsigned long long int old_value = obj.load(memoryOrder);
  const double old_double_value = *reinterpret_cast<const double *>(&old_value);
  const double new_double_value = old_double_value + operand;
  const unsigned long long int new_value =
      *reinterpret_cast<const unsigned long long int *>(&new_double_value);

  while (true) {
    unsigned long long int expected = old_value;
    if (obj.compare_exchange_strong(expected, new_value, memoryOrder))
      break;
  }

  return old_double_value;
}

/// Atomically subtract operand from *addr.
/// \param [in, out] addr Point to a data.
/// \param operand Be substracted from data pointed by \p addr.
/// \param memoryOrder The memory ordering used.
/// \return The old value addr point to.
template <cl::sycl::access::address_space addressSpace =
              cl::sycl::access::address_space::global_space,
          typename T>
inline T atomic_fetch_sub(
    T *addr, T operand,
    cl::sycl::memory_order memoryOrder = cl::sycl::memory_order::relaxed) {
  cl::sycl::atomic<T, addressSpace> obj(
      (cl::sycl::multi_ptr<T, addressSpace>(addr)));
  return cl::sycl::atomic_fetch_sub(obj, operand, memoryOrder);
}

/// Atomically and *addr with operand.
/// \param [in, out] addr Point to a data.
/// \param operand Be anded with data pointed by \p addr.
/// \param memoryOrder The memory ordering used.
/// \return The old value addr point to.
template <cl::sycl::access::address_space addressSpace =
              cl::sycl::access::address_space::global_space,
          typename T>
inline T atomic_fetch_and(
    T *addr, T operand,
    cl::sycl::memory_order memoryOrder = cl::sycl::memory_order::relaxed) {
  cl::sycl::atomic<T, addressSpace> obj(
      (cl::sycl::multi_ptr<T, addressSpace>(addr)));
  return cl::sycl::atomic_fetch_and(obj, operand, memoryOrder);
}

/// Atomically or *addr with operand.
/// \param [in, out] addr Point to a data.
/// \param operand Be or-ed with data pointed by \p addr.
/// \param memoryOrder The memory ordering used.
/// \return The old value addr point to.
template <cl::sycl::access::address_space addressSpace =
              cl::sycl::access::address_space::global_space,
          typename T>
inline T atomic_fetch_or(
    T *addr, T operor,
    cl::sycl::memory_order memoryOrder = cl::sycl::memory_order::relaxed) {
  cl::sycl::atomic<T, addressSpace> obj(
      (cl::sycl::multi_ptr<T, addressSpace>(addr)));
  return cl::sycl::atomic_fetch_or(obj, operor, memoryOrder);
}

/// Atomically xor *addr with operxor.
/// \param [in, out] addr Point to a data.
/// \param operxor Be xor-ed with data pointed by \p addr.
/// \param memoryOrder The memory ordering used.
/// \return The old value addr point to.
template <cl::sycl::access::address_space addressSpace =
              cl::sycl::access::address_space::global_space,
          typename T>
inline T atomic_fetch_xor(
    T *addr, T operxor,
    cl::sycl::memory_order memoryOrder = cl::sycl::memory_order::relaxed) {
  cl::sycl::atomic<T, addressSpace> obj(
      (cl::sycl::multi_ptr<T, addressSpace>(addr)));
  return cl::sycl::atomic_fetch_xor(obj, operxor, memoryOrder);
}

/// Atomically do *addr=min(*addr, opermin).
/// \param [in, out] addr Point to a data.
/// \param opermin.
/// \param memoryOrder The memory ordering used.
/// \return The old value addr point to.
template <cl::sycl::access::address_space addressSpace =
              cl::sycl::access::address_space::global_space,
          typename T>
inline T atomic_fetch_min(
    T *addr, T opermin,
    cl::sycl::memory_order memoryOrder = cl::sycl::memory_order::relaxed) {
  cl::sycl::atomic<T, addressSpace> obj(
      (cl::sycl::multi_ptr<T, addressSpace>(addr)));
  return cl::sycl::atomic_fetch_min(obj, opermin, memoryOrder);
}

/// Atomically do *addr=max(*addr, opermax).
/// \param [in, out] addr Point to a data.
/// \param opermax.
/// \param memoryOrder The memory ordering used.
/// \return The old value addr point to.
template <cl::sycl::access::address_space addressSpace =
              cl::sycl::access::address_space::global_space,
          typename T>
inline T atomic_fetch_max(
    T *addr, T opermax,
    cl::sycl::memory_order memoryOrder = cl::sycl::memory_order::relaxed) {
  cl::sycl::atomic<T, addressSpace> obj(
      (cl::sycl::multi_ptr<T, addressSpace>(addr)));
  return cl::sycl::atomic_fetch_max(obj, opermax, memoryOrder);
}

/// Atomically exchange *addr with operand.
/// \param [in, out] addr Point to a data.
/// \param operand Be exchanged with data pointed by \p addr.
/// \param memoryOrder The memory ordering used.
/// \return The old value addr point to.
template <cl::sycl::access::address_space addressSpace =
              cl::sycl::access::address_space::global_space,
          typename T>
inline T atomic_exchange(
    T *addr, T operand,
    cl::sycl::memory_order memoryOrder = cl::sycl::memory_order::relaxed) {
  cl::sycl::atomic<T, addressSpace> obj(
      (cl::sycl::multi_ptr<T, addressSpace>(addr)));
  return cl::sycl::atomic_exchange(obj, operand, memoryOrder);
}

/// Atomically compare and exchange.
/// if(*addr==expected) *addr=desired, return expected
/// if(*addr!=expected)  return *addr
/// \param [in, out] addr Multi_ptr.
/// \param expected The value compare against *addr.
/// \param desired The value store to *addr on success.
/// \param success The memory ordering used when comparison succeeds.
/// \param fail The memory ordering used when comparison fails.
/// \return old data of *addr.
template <cl::sycl::access::address_space addressSpace =
              cl::sycl::access::address_space::global_space,
          typename T>
T atomic_compare_exchange_strong(
    cl::sycl::multi_ptr<T, cl::sycl::access::address_space::global_space> addr,
    T expected, T desired,
    cl::sycl::memory_order success = cl::sycl::memory_order::relaxed,
    cl::sycl::memory_order fail = cl::sycl::memory_order::relaxed) {
  cl::sycl::atomic<T, addressSpace> obj(addr);
  obj.compare_exchange_strong(expected, desired, success, fail);
  return expected;
}

/// Atomically compare and exchange.
/// if(*addr==expected) *addr=desired, return true
/// if(*addr!=expected)  return false
/// \param [in, out] addr Point to a data.
/// \param expected The value compare against *addr.
/// \param desired The value store to *addr on success.
/// \param success The memory ordering used when comparison succeeds.
/// \param fail The memory ordering used when comparison fails.
/// \return (*addr==expected)
template <cl::sycl::access::address_space addressSpace =
              cl::sycl::access::address_space::global_space,
          typename T>
T atomic_compare_exchange_strong(
    T *addr, T expected, T desired,
    cl::sycl::memory_order success = cl::sycl::memory_order::relaxed,
    cl::sycl::memory_order fail = cl::sycl::memory_order::relaxed) {
  return atomic_compare_exchange_strong(
      cl::sycl::multi_ptr<T, addressSpace>(addr), expected, desired, success,
      fail);
}

} // namespace dpct

#endif // DPCT_ATOMIC_H
