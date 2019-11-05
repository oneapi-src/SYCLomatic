/******************************************************************************
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

#ifndef __DPCT_ATOMIC_HPP__
#define __DPCT_ATOMIC_HPP__

#include <CL/sycl.hpp>

namespace dpct {

/// Atomically add the value operand to the value at the addr and assign the
/// result to the value at addr, Int version.
/// \param [in, out] addr The pointer to the data.
/// \param operand The value to add to the value at \p addr.
/// \param memoryOrder The memory ordering used.
/// \return The value at the \p addr before the call.
template <typename T, cl::sycl::access::address_space addressSpace =
              cl::sycl::access::address_space::global_space>
inline T atomic_fetch_add(
    T *addr, T operand,
    cl::sycl::memory_order memoryOrder = cl::sycl::memory_order::relaxed) {
  cl::sycl::atomic<T, addressSpace> obj(
      (cl::sycl::multi_ptr<T, addressSpace>(addr)));
  return cl::sycl::atomic_fetch_add(obj, operand, memoryOrder);
}

/// Atomically add the value operand to the value at the addr and assign the
/// result to the value at addr, Float version.
/// \param [in, out] addr The pointer to the data.
/// \param operand The value to add to the value at \p addr.
/// \param memoryOrder The memory ordering used.
/// \return The value at the \p addr before the call.
template <cl::sycl::access::address_space addressSpace =
              cl::sycl::access::address_space::global_space>
inline float atomic_fetch_add(
    float *addr, float operand,
    cl::sycl::memory_order memoryOrder = cl::sycl::memory_order::relaxed) {
  static_assert(sizeof(float) == sizeof(int), "Mismatched type size");

  cl::sycl::atomic<int, addressSpace> obj(
      (cl::sycl::multi_ptr<int, addressSpace>(reinterpret_cast<int *>(addr))));

  int old_value = obj.load(memoryOrder);

  float old_float_value;
  do {
    old_float_value = *reinterpret_cast<const float *>(&old_value);
    const float new_float_value = old_float_value + operand;
    const int new_value = *reinterpret_cast<const int *>(&new_float_value);
    if (obj.compare_exchange_strong(old_value, new_value, memoryOrder))
      break;
  } while (true);

  return old_float_value;
}

/// Atomically add the value operand to the value at the addr and assign the
/// result to the value at addr, Double version.
/// \param [in, out] addr The pointer to the data.
/// \param operand The value to add to the value at \p addr
/// \param memoryOrder The memory ordering used.
/// \return The value at the \p addr before the call.
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

  unsigned long long int old_value = obj.load(memoryOrder);

  double old_double_value;
  do {
    old_double_value = *reinterpret_cast<const double *>(&old_value);
    const double new_double_value = old_double_value + operand;
    const unsigned long long int new_value =
      *reinterpret_cast<const unsigned long long int *>(&new_double_value);

    if (obj.compare_exchange_strong(old_value, new_value, memoryOrder))
      break;
  } while (true);

  return old_double_value;
}

/// Atomically subtract the value operand from the value at the addr and assign
/// the result to the value at addr.
/// \param [in, out] addr The pointer to the data.
/// \param operand The value to substract from the value at \p addr
/// \param memoryOrder The memory ordering used.
/// \return The value at the \p addr before the call.
template <typename T, cl::sycl::access::address_space addressSpace =
              cl::sycl::access::address_space::global_space>
inline T atomic_fetch_sub(
    T *addr, T operand,
    cl::sycl::memory_order memoryOrder = cl::sycl::memory_order::relaxed) {
  cl::sycl::atomic<T, addressSpace> obj(
      (cl::sycl::multi_ptr<T, addressSpace>(addr)));
  return cl::sycl::atomic_fetch_sub(obj, operand, memoryOrder);
}

/// Atomically perform a bitwise AND between the value operand and the value at the addr
/// and assign the result to the value at addr.
/// \param [in, out] addr The pointer to the data.
/// \param operand The value to use in bitwise AND operation with the value at the \p addr.
/// \param memoryOrder The memory ordering used.
/// \return The value at the \p addr before the call.
template <typename T, cl::sycl::access::address_space addressSpace =
              cl::sycl::access::address_space::global_space>
inline T atomic_fetch_and(
    T *addr, T operand,
    cl::sycl::memory_order memoryOrder = cl::sycl::memory_order::relaxed) {
  cl::sycl::atomic<T, addressSpace> obj(
      (cl::sycl::multi_ptr<T, addressSpace>(addr)));
  return cl::sycl::atomic_fetch_and(obj, operand, memoryOrder);
}

/// Atomically or the value at the addr with the value operand, and assign
/// the result to the value at addr.
/// \param [in, out] addr The pointer to the data.
/// \param operand The value to use in bitwise OR operation with the value at the \p addr.
/// \param memoryOrder The memory ordering used.
/// \return The value at the \p addr before the call.
template <typename T, cl::sycl::access::address_space addressSpace =
              cl::sycl::access::address_space::global_space>
inline T atomic_fetch_or(
    T *addr, T operand,
    cl::sycl::memory_order memoryOrder = cl::sycl::memory_order::relaxed) {
  cl::sycl::atomic<T, addressSpace> obj(
      (cl::sycl::multi_ptr<T, addressSpace>(addr)));
  return cl::sycl::atomic_fetch_or(obj, operand, memoryOrder);
}

/// Atomically xor the value at the addr with the value operand, and assign
/// the result to the value at addr.
/// \param [in, out] addr The pointer to the data.
/// \param operand The value to use in bitwise XOR operation with the value at the \p addr.
/// \param memoryOrder The memory ordering used.
/// \return The value at the \p addr before the call.
template <typename T, cl::sycl::access::address_space addressSpace =
              cl::sycl::access::address_space::global_space>
inline T atomic_fetch_xor(
    T *addr, T operand,
    cl::sycl::memory_order memoryOrder = cl::sycl::memory_order::relaxed) {
  cl::sycl::atomic<T, addressSpace> obj(
      (cl::sycl::multi_ptr<T, addressSpace>(addr)));
  return cl::sycl::atomic_fetch_xor(obj, operand, memoryOrder);
}

/// Atomically calculate the minimum of the value at addr and the value operand
/// and assign the result to the value at addr.
/// \param [in, out] addr The pointer to the data.
/// \param operand.
/// \param memoryOrder The memory ordering used.
/// \return The value at the \p addr before the call.
template <typename T, cl::sycl::access::address_space addressSpace =
              cl::sycl::access::address_space::global_space>
inline T atomic_fetch_min(
    T *addr, T operand,
    cl::sycl::memory_order memoryOrder = cl::sycl::memory_order::relaxed) {
  cl::sycl::atomic<T, addressSpace> obj(
      (cl::sycl::multi_ptr<T, addressSpace>(addr)));
  return cl::sycl::atomic_fetch_min(obj, operand, memoryOrder);
}

/// Atomically calculate the maximum of the value at addr and the value operand
/// and assign the result to the value at addr.
/// \param [in, out] addr The pointer to the data.
/// \param operand.
/// \param memoryOrder The memory ordering used.
/// \return The value at the \p addr before the call.
template <typename T, cl::sycl::access::address_space addressSpace =
              cl::sycl::access::address_space::global_space>
inline T atomic_fetch_max(
    T *addr, T operand,
    cl::sycl::memory_order memoryOrder = cl::sycl::memory_order::relaxed) {
  cl::sycl::atomic<T, addressSpace> obj(
      (cl::sycl::multi_ptr<T, addressSpace>(addr)));
  return cl::sycl::atomic_fetch_max(obj, operand, memoryOrder);
}

/// Atomically exchange the value at the address addr with the value operand.
/// \param [in, out] addr The pointer to the data.
/// \param operand The value to be exchanged with the value pointed by \p addr.
/// \param memoryOrder The memory ordering used.
/// \return The value at the \p addr before the call.
template <typename T, cl::sycl::access::address_space addressSpace =
              cl::sycl::access::address_space::global_space>
inline T atomic_exchange(
    T *addr, T operand,
    cl::sycl::memory_order memoryOrder = cl::sycl::memory_order::relaxed) {
  cl::sycl::atomic<T, addressSpace> obj(
      (cl::sycl::multi_ptr<T, addressSpace>(addr)));
  return cl::sycl::atomic_exchange(obj, operand, memoryOrder);
}

/// Atomically compare the value at \p addr to the value expected and exchange
/// with the value desired if the value at \p addr is equal to the value expected.
/// Returns the value at the \p addr before the call.
/// \param [in, out] addr Multi_ptr.
/// \param expected The value to compare against the value at \p addr.
/// \param desired The value to assign to \p addr if the value at \p addr is expected.
/// \param success The memory ordering used when comparison succeeds.
/// \param fail The memory ordering used when comparison fails.
/// \return The value at the \p addr before the call.
template <typename T, cl::sycl::access::address_space addressSpace =
              cl::sycl::access::address_space::global_space>
T atomic_compare_exchange_strong(
    cl::sycl::multi_ptr<T, cl::sycl::access::address_space::global_space> addr,
    T expected, T desired,
    cl::sycl::memory_order success = cl::sycl::memory_order::relaxed,
    cl::sycl::memory_order fail = cl::sycl::memory_order::relaxed) {
  cl::sycl::atomic<T, addressSpace> obj(addr);
  obj.compare_exchange_strong(expected, desired, success, fail);
  return expected;
}

/// Atomically compare the value at \p addr to the value expected and exchange
/// with the value desired if the value at \p addr is equal to the value expected.
/// Returns the value at the \p addr before the call.
/// \param [in] addr The pointer to the data.
/// \param expected The value to compare against the value at \p addr.
/// \param desired The value to assign to \p addr if the value at \p addr is expected.
/// \param success The memory ordering used when comparison succeeds.
/// \param fail The memory ordering used when comparison fails.
/// \return The value at the \p addr before the call.
template <typename T, cl::sycl::access::address_space addressSpace =
              cl::sycl::access::address_space::global_space>
T atomic_compare_exchange_strong(
    T *addr, T expected, T desired,
    cl::sycl::memory_order success = cl::sycl::memory_order::relaxed,
    cl::sycl::memory_order fail = cl::sycl::memory_order::relaxed) {
  return atomic_compare_exchange_strong(
      cl::sycl::multi_ptr<T, addressSpace>(addr), expected, desired, success,
      fail);
}

} // namespace dpct
#endif // __DPCT_ATOMIC_HPP__
