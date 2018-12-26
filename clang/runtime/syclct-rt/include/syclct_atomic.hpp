//===--- syclct_atomic.hpp -------------------------------*- C++ -*---===//
//
// Copyright (C) 2019 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

#ifndef SYCLCT_ATOMIC_H
#define SYCLCT_ATOMIC_H

#include <CL/sycl.hpp>

namespace syclct {

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

template <cl::sycl::access::address_space addressSpace =
              cl::sycl::access::address_space::global_space,
          typename T>
inline cl::sycl::cl_bool atomic_compare_exchange_strong(
    T *addr, T expected, T desired,
    cl::sycl::memory_order success = cl::sycl::memory_order::relaxed,
    cl::sycl::memory_order fail = cl::sycl::memory_order::relaxed) {
  cl::sycl::atomic<T, addressSpace> obj(
      (cl::sycl::multi_ptr<T, addressSpace>(addr)));
  return obj.compare_exchange_strong(expected, desired, success, fail);
}

} // namespace syclct

#endif // SYCLCT_ATOMIC_H
