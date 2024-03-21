//==---- atomic.hpp -------------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief Atomic functions
 * 
 * @copyright Copyright (C) Intel Corporation
 * 
 */

#ifndef __DPCT_ATOMIC_HPP__
#define __DPCT_ATOMIC_HPP__

#include <sycl/sycl.hpp>

namespace dpct {
/**
 * @brief Add the operand to the value at \p addr and assigns the
 * result to the value at addr.
 * @tparam T The type of the data and the operand.
 * @tparam addressSpace The address space of \p addr.
 * @tparam memoryOrder The memory ordering to be used.
 * @tparam memoryScope The memory scope used.
 * @param [in, out] addr The pointer to the data.
 * @param [in] operand The value to add to the value at \p addr.
 * @returns The value stored at \p addr before the call.
 */
template <typename T,
          sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space,
          sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
          sycl::memory_scope memoryScope = sycl::memory_scope::device>
inline T atomic_fetch_add(T *addr, T operand) {
  auto atm =
      sycl::atomic_ref<T, memoryOrder, memoryScope, addressSpace>(addr[0]);
  return atm.fetch_add(operand);
}
/**
 * @brief Add the operand to the value at \p addr and assigns the
 * result to the value at addr.
 * @tparam addressSpace The address space of \p addr.
 * @tparam memoryOrder The memory ordering to be used.
 * @tparam memoryScope The memory scope used.
 * @tparam T1 The type of the value in \p addr.
 * @tparam T2 The type of the value to add.
 * @param [in, out] addr The pointer to the data.
 * @param [in] operand The value to add to the value at \p addr.
 * @returns The value stored at \p addr before the call.
 */
template <sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space,
          sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
          sycl::memory_scope memoryScope = sycl::memory_scope::device,
          typename T1, typename T2>
inline T1 atomic_fetch_add(T1 *addr, T2 operand) {
  auto atm =
      sycl::atomic_ref<T1, memoryOrder, memoryScope, addressSpace>(addr[0]);
  return atm.fetch_add(operand);
}

/**
 * @brief Add the operand to the value at \p addr and assigns the
 * result to the value at addr.
 * @tparam T The type of the data and the operand.
 * @tparam addressSpace The address space of \p addr.
 * @param [in, out] addr The pointer to the data.
 * @param [in] operand The value to add to the value at \p addr.
 * @param [in] memoryOrder The memory ordering of \p addr.
 * @returns The value stored at \p addr before the call.
 */
template <typename T, sycl::access::address_space addressSpace =
                          sycl::access::address_space::global_space>
inline T atomic_fetch_add(T *addr, T operand,
                          sycl::memory_order memoryOrder) {
  switch (memoryOrder) {
  case sycl::memory_order::relaxed:
    return atomic_fetch_add<T, addressSpace, sycl::memory_order::relaxed,
                            sycl::memory_scope::device>(addr, operand);
  case sycl::memory_order::acq_rel:
    return atomic_fetch_add<T, addressSpace, sycl::memory_order::acq_rel,
                            sycl::memory_scope::device>(addr, operand);
  case sycl::memory_order::seq_cst:
    return atomic_fetch_add<T, addressSpace, sycl::memory_order::seq_cst,
                            sycl::memory_scope::device>(addr, operand);
  default:
    assert(false && "Invalid memory_order for atomics. Valid memory_order for "
                    "atomics are: sycl::memory_order::relaxed, "
                    "sycl::memory_order::acq_rel, sycl::memory_order::seq_cst!");
  }
}
/**
 * @brief Add the operand to the value at \p addr and assigns the
 * result to the value at addr.
 * @tparam addressSpace The address space of \p addr.
 * @tparam T1 The type of the value in \p addr.
 * @tparam T2 The type of the value to add.
 * @param [in, out] addr The pointer to the data.
 * @param [in] operand The value to add to the value at \p addr.
 * @param [in] memoryOrder The memory ordering of \p addr.
 * @returns The value stored at \p addr before the call.
 */
template <sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space,
          typename T1, typename T2>
inline T1 atomic_fetch_add(T1 *addr, T2 operand,
                           sycl::memory_order memoryOrder) {
  atomic_fetch_add<T1, addressSpace>(addr, operand, memoryOrder);
}

/**
 * @brief Sub the operand to the value at \p addr and assigns the
 * result to the value at addr.
 * @tparam T The type of the data and the operand.
 * @tparam addressSpace The address space of \p addr.
 * @tparam memoryOrder The memory ordering to be used.
 * @tparam memoryScope The memory scope used.
 * @param [in, out] addr The pointer to the data.
 * @param [in] operand The value to sub to the value at \p addr.
 * @returns The value stored at \p addr before the call.
 */
template <typename T,
          sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space,
          sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
          sycl::memory_scope memoryScope = sycl::memory_scope::device>
inline T atomic_fetch_sub(T *addr, T operand) {
  auto atm =
      sycl::atomic_ref<T, memoryOrder, memoryScope, addressSpace>(addr[0]);
  return atm.fetch_sub(operand);
}
/**
 * @brief Sub the operand to the value at \p addr and assigns the
 * result to the value at addr.
 * @tparam addressSpace The address space of \p addr.
 * @tparam memoryOrder The memory ordering to be used.
 * @tparam memoryScope The memory scope used.
 * @tparam T1 The type of the value in \p addr.
 * @tparam T2 The type of the value to sub.
 * @param [in, out] addr The pointer to the data.
 * @param [in] operand The value to sub to the value at \p addr.
 * @returns The value stored at \p addr before the call.
 */
template <sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space,
          sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
          sycl::memory_scope memoryScope = sycl::memory_scope::device,
          typename T1, typename T2>
inline T1 atomic_fetch_sub(T1 *addr, T2 operand) {
  auto atm =
      sycl::atomic_ref<T1, memoryOrder, memoryScope, addressSpace>(addr[0]);
  return atm.fetch_sub(operand);
}

/**
 * @brief Sub the operand to the value at \p addr and assigns the
 * result to the value at addr.
 * @tparam T The type of the data and the operand.
 * @tparam addressSpace The address space of \p addr.
 * @param [in, out] addr The pointer to the data.
 * @param [in] operand The value to sub to the value at \p addr.
 * @param [in] memoryOrder The memory ordering of \p addr.
 * @returns The value stored at \p addr before the call.
 */
template <typename T, sycl::access::address_space addressSpace =
                          sycl::access::address_space::global_space>
inline T atomic_fetch_sub(T *addr, T operand,
                          sycl::memory_order memoryOrder) {
  switch (memoryOrder) {
  case sycl::memory_order::relaxed:
    return atomic_fetch_sub<T, addressSpace, sycl::memory_order::relaxed,
                            sycl::memory_scope::device>(addr, operand);
  case sycl::memory_order::acq_rel:
    return atomic_fetch_sub<T, addressSpace, sycl::memory_order::acq_rel,
                            sycl::memory_scope::device>(addr, operand);
  case sycl::memory_order::seq_cst:
    return atomic_fetch_sub<T, addressSpace, sycl::memory_order::seq_cst,
                            sycl::memory_scope::device>(addr, operand);
  default:
    assert(false && "Invalid memory_order for atomics. Valid memory_order for "
                    "atomics are: sycl::memory_order::relaxed, "
                    "sycl::memory_order::acq_rel, sycl::memory_order::seq_cst!");
  }
}
/**
 * @brief Sub the operand to the value at \p addr and assigns the
 * result to the value at addr.
 * @tparam addressSpace The address space of \p addr.
 * @tparam T1 The type of the value in \p addr.
 * @tparam T2 The type of the value to sub.
 * @param [in, out] addr The pointer to the data.
 * @param [in] operand The value to sub to the value at \p addr.
 * @param [in] memoryOrder The memory ordering of \p addr.
 * @returns The value stored at \p addr before the call.
 */
template <sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space,
          typename T1, typename T2>
inline T1 atomic_fetch_sub(T1 *addr, T2 operand,
                           sycl::memory_order memoryOrder) {
  atomic_fetch_sub<T1, addressSpace>(addr, operand, memoryOrder);
}

/**
 * @brief Performs a bitwise AND between the operand and the value at \p addr
 * and assigns the result to the value at \p addr.
 * @tparam T The type of the data and the operand.
 * @tparam addressSpace The address space of \p addr.
 * @tparam memoryOrder The memory ordering to be used.
 * @tparam memoryScope The memory scope used.
 * @param [in, out] addr The pointer to the data.
 * @param [in] operand The value to AND with the value at \p addr.
 * @returns The value stored at \p addr before the call.
 */
template <typename T,
          sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space,
          sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
          sycl::memory_scope memoryScope = sycl::memory_scope::device>
inline T atomic_fetch_and(T *addr, T operand) {
  auto atm =
      sycl::atomic_ref<T, memoryOrder, memoryScope, addressSpace>(addr[0]);
  return atm.fetch_and(operand);
}

/**
 * @brief Performs a bitwise AND between the operand and the value at \p addr
 * and assigns the result to the value at \p addr.
 * @tparam addressSpace The address space of \p addr.
 * @tparam memoryOrder The memory ordering to be used.
 * @tparam memoryScope The memory scope used.
 * @tparam T1 The type of the value in \p addr.
 * @tparam T2 The type of the value to AND with.
 * @param [in, out] addr The pointer to the data.
 * @param [in] operand The value to AND with the value at \p addr.
 * @returns The value stored at \p addr before the call.
 */
template <sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space,
          sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
          sycl::memory_scope memoryScope = sycl::memory_scope::device,
          typename T1, typename T2>
inline T1 atomic_fetch_and(T1 *addr, T2 operand) {
  auto atm =
      sycl::atomic_ref<T1, memoryOrder, memoryScope, addressSpace>(addr[0]);
  return atm.fetch_and(operand);
}

/**
 * @brief Performs a bitwise AND between the operand and the value at \p addr
 * and assigns the result to the value at \p addr.
 * @tparam T The type of the data and the operand.
 * @tparam addressSpace The address space of \p addr.
 * @param [in, out] addr The pointer to the data.
 * @param [in] operand The value to AND with the value at \p addr.
 * @param [in] memoryOrder The memory ordering of \p addr.
 * @returns The value stored at \p addr before the call.
 */
template <typename T, sycl::access::address_space addressSpace =
                          sycl::access::address_space::global_space>
inline T atomic_fetch_and(T *addr, T operand,
                          sycl::memory_order memoryOrder) {
  switch (memoryOrder) {
  case sycl::memory_order::relaxed:
    return atomic_fetch_and<T, addressSpace, sycl::memory_order::relaxed,
                            sycl::memory_scope::device>(addr, operand);
  case sycl::memory_order::acq_rel:
    return atomic_fetch_and<T, addressSpace, sycl::memory_order::acq_rel,
                            sycl::memory_scope::device>(addr, operand);
  case sycl::memory_order::seq_cst:
    return atomic_fetch_and<T, addressSpace, sycl::memory_order::seq_cst,
                            sycl::memory_scope::device>(addr, operand);
  default:
    assert(false && "Invalid memory_order for atomics. Valid memory_order for "
                    "atomics are: sycl::memory_order::relaxed, "
                    "sycl::memory_order::acq_rel, sycl::memory_order::seq_cst!");
  }
}
/**
 * @brief Performs a bitwise AND between the operand and the value at \p addr
 * and assigns the result to the value at \p addr.
 * @tparam addressSpace The address space of \p addr.
 * @tparam T1 The type of the value in \p addr.
 * @tparam T2 The type of the value to AND with.
 * @param [in, out] addr The pointer to the data.
 * @param [in] operand The value to AND with the value at \p addr.
 * @param [in] memoryOrder The memory ordering of \p addr.
 * @returns The value stored at \p addr before the call.
 */
template <sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space,
          typename T1, typename T2>
inline T1 atomic_fetch_and(T1 *addr, T2 operand,
                           sycl::memory_order memoryOrder) {
  atomic_fetch_and<T1, addressSpace>(addr, operand, memoryOrder);
}

/**
 * @brief Performs a bitwise OR between the operand and the value at \p addr
 * and assigns the result to the value at \p addr.
 * @tparam T The type of the data and the operand.
 * @tparam addressSpace The address space of \p addr.
 * @tparam memoryOrder The memory ordering to be used.
 * @tparam memoryScope The memory scope used.
 * @param [in, out] addr The pointer to the data.
 * @param [in] operand The value to OR with the value at \p addr.
 * @returns The value stored at \p addr before the call.
 */
template <typename T,
          sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space,
          sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
          sycl::memory_scope memoryScope = sycl::memory_scope::device>
inline T atomic_fetch_or(T *addr, T operand) {
  auto atm =
      sycl::atomic_ref<T, memoryOrder, memoryScope, addressSpace>(addr[0]);
  return atm.fetch_or(operand);
}
/**
 * @brief Performs a bitwise OR between the operand and the value at \p addr
 * and assigns the result to the value at \p addr.
 * @tparam addressSpace The address space of \p addr.
 * @tparam memoryOrder The memory ordering to be used.
 * @tparam memoryScope The memory scope used.
 * @tparam T1 The type of the value in \p addr.
 * @tparam T2 The type of the value to OR with.
 * @param [in, out] addr The pointer to the data.
 * @param [in] operand The value to OR with the value at \p addr.
 * @returns The value stored at \p addr before the call.
 */
template <sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space,
          sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
          sycl::memory_scope memoryScope = sycl::memory_scope::device,
          typename T1, typename T2>
inline T1 atomic_fetch_or(T1 *addr, T2 operand) {
  auto atm =
      sycl::atomic_ref<T1, memoryOrder, memoryScope, addressSpace>(addr[0]);
  return atm.fetch_or(operand);
}

/**
 * @brief Performs a bitwise OR between the operand and the value at \p addr
 * and assigns the result to the value at \p addr.
 * @tparam T The type of the data and the operand.
 * @tparam addressSpace The address space of \p addr.
 * @param [in, out] addr The pointer to the data.
 * @param [in] operand The value to OR with the value at \p addr.
 * @param [in] memoryOrder The memory ordering of \p addr.
 * @returns The value stored at \p addr before the call.
 */
template <typename T, sycl::access::address_space addressSpace =
                          sycl::access::address_space::global_space>
inline T atomic_fetch_or(T *addr, T operand,
                         sycl::memory_order memoryOrder) {
  switch (memoryOrder) {
  case sycl::memory_order::relaxed:
    return atomic_fetch_or<T, addressSpace, sycl::memory_order::relaxed,
                           sycl::memory_scope::device>(addr, operand);
  case sycl::memory_order::acq_rel:
    return atomic_fetch_or<T, addressSpace, sycl::memory_order::acq_rel,
                           sycl::memory_scope::device>(addr, operand);
  case sycl::memory_order::seq_cst:
    return atomic_fetch_or<T, addressSpace, sycl::memory_order::seq_cst,
                           sycl::memory_scope::device>(addr, operand);
  default:
    assert(false && "Invalid memory_order for atomics. Valid memory_order for "
                    "atomics are: sycl::memory_order::relaxed, "
                    "sycl::memory_order::acq_rel, sycl::memory_order::seq_cst!");
  }
}
/**
 * @brief Performs a bitwise OR between the operand and the value at \p addr
 * and assigns the result to the value at \p addr.
 * @tparam addressSpace The address space of \p addr.
 * @tparam T1 The type of the value in \p addr.
 * @tparam T2 The type of the value to OR with.
 * @param [in, out] addr The pointer to the data.
 * @param [in] operand The value to OR with the value at \p addr.
 * @param [in] memoryOrder The memory ordering of \p addr.
 * @returns The value stored at \p addr before the call.
 */
template <sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space,
          typename T1, typename T2>
inline T1 atomic_fetch_or(T1 *addr, T2 operand,
                           sycl::memory_order memoryOrder) {
  atomic_fetch_or<T1, addressSpace>(addr, operand, memoryOrder);
}

/**
 * @brief Performs a bitwise XOR between the operand and the value at \p addr
 * and assigns the result to the value at \p addr.
 * @tparam T The type of the data and the operand.
 * @tparam addressSpace The address space of \p addr.
 * @tparam memoryOrder The memory ordering to be used.
 * @tparam memoryScope The memory scope used.
 * @param [in, out] addr The pointer to the data.
 * @param [in] operand The value to XOR with the value at \p addr.
 * @returns The value stored at \p addr before the call.
 */
template <typename T,
          sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space,
          sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
          sycl::memory_scope memoryScope = sycl::memory_scope::device>
inline T atomic_fetch_xor(T *addr, T operand) {
  auto atm =
      sycl::atomic_ref<T, memoryOrder, memoryScope, addressSpace>(addr[0]);
  return atm.fetch_xor(operand);
}
/**
 * @brief Performs a bitwise XOR between the operand and the value at \p addr
 * and assigns the result to the value at \p addr.
 * @tparam addressSpace The address space of \p addr.
 * @tparam memoryOrder The memory ordering to be used.
 * @tparam memoryScope The memory scope used.
 * @tparam T1 The type of the value in \p addr.
 * @tparam T2 The type of the value to XOR with.
 * @param [in, out] addr The pointer to the data.
 * @param [in] operand The value to XOR with the value at \p addr.
 * @returns The value stored at \p addr before the call.
 */
template <sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space,
          sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
          sycl::memory_scope memoryScope = sycl::memory_scope::device,
          typename T1, typename T2>
inline T1 atomic_fetch_xor(T1 *addr, T2 operand) {
  auto atm =
      sycl::atomic_ref<T1, memoryOrder, memoryScope, addressSpace>(addr[0]);
  return atm.fetch_xor(operand);
}

/**
 * @brief Performs a bitwise XOR between the operand and the value at \p addr
 * and assigns the result to the value at \p addr.
 * @tparam T The type of the data and the operand.
 * @tparam addressSpace The address space of \p addr.
 * @param [in, out] addr The pointer to the data.
 * @param [in] operand The value to XOR with the value at \p addr.
 * @param [in] memoryOrder The memory ordering of \p addr.
 * @returns The value stored at \p addr before the call.
 */
template <typename T, sycl::access::address_space addressSpace =
                          sycl::access::address_space::global_space>
inline T atomic_fetch_xor(T *addr, T operand,
                          sycl::memory_order memoryOrder) {
  switch (memoryOrder) {
  case sycl::memory_order::relaxed:
    return atomic_fetch_xor<T, addressSpace, sycl::memory_order::relaxed,
                            sycl::memory_scope::device>(addr, operand);
  case sycl::memory_order::acq_rel:
    return atomic_fetch_xor<T, addressSpace, sycl::memory_order::acq_rel,
                            sycl::memory_scope::device>(addr, operand);
  case sycl::memory_order::seq_cst:
    return atomic_fetch_xor<T, addressSpace, sycl::memory_order::seq_cst,
                            sycl::memory_scope::device>(addr, operand);
  default:
    assert(false && "Invalid memory_order for atomics. Valid memory_order for "
                    "atomics are: sycl::memory_order::relaxed, "
                    "sycl::memory_order::acq_rel, sycl::memory_order::seq_cst!");
  }
}
/**
 * @brief Performs a bitwise XOR between the operand and the value at \p addr
 * and assigns the result to the value at \p addr.
 * @tparam addressSpace The address space of \p addr.
 * @tparam T1 The type of the value in \p addr.
 * @tparam T2 The type of the value to XOR with.
 * @param [in, out] addr The pointer to the data.
 * @param [in] operand The value to XOR with the value at \p addr.
 * @param [in] memoryOrder The memory ordering of \p addr.
 * @returns The value stored at \p addr before the call.
 */
template <sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space,
          typename T1, typename T2>
inline T1 atomic_fetch_xor(T1 *addr, T2 operand,
                           sycl::memory_order memoryOrder) {
  atomic_fetch_xor<T1, addressSpace>(addr, operand, memoryOrder);
}

/**
 * @brief Calculates the minimum of the value at \p addr and the operand
 * and assigns the result to the value at addr.
 * @tparam T The type of the data and the operand.
 * @tparam addressSpace The address space of \p addr.
 * @tparam memoryOrder The memory ordering to be used.
 * @tparam memoryScope The memory scope used.
 * @param [in, out] addr The pointer to the data.
 * @param [in] operand The operand.
 * @returns The value stored at \p addr before the call.
 */
template <typename T,
          sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space,
          sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
          sycl::memory_scope memoryScope = sycl::memory_scope::device>
inline T atomic_fetch_min(T *addr, T operand) {
  auto atm =
      sycl::atomic_ref<T, memoryOrder, memoryScope, addressSpace>(addr[0]);
  return atm.fetch_min(operand);
}
/**
 * @brief Calculates the minimum of the value at \p addr and the operand
 * and assigns the result to the value at addr.
 * @tparam addressSpace The address space of \p addr.
 * @tparam memoryOrder The memory ordering to be used.
 * @tparam memoryScope The memory scope used.
 * @tparam T1 The type of the data.
 * @tparam T2 The type of the operands.
 * @param [in, out] addr The pointer to the data.
 * @param [in] operand The operand.
 * @returns The value stored at \p addr before the call.
 */
template <sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space,
          sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
          sycl::memory_scope memoryScope = sycl::memory_scope::device,
          typename T1, typename T2>
inline T1 atomic_fetch_min(T1 *addr, T2 operand) {
  auto atm =
      sycl::atomic_ref<T1, memoryOrder, memoryScope, addressSpace>(addr[0]);
  return atm.fetch_min(operand);
}

/**
 * @brief Calculates the minimum of the value at \p addr and the operand
 * and assigns the result to the value at addr.
 * @tparam T The type of the data and the operand.
 * @tparam addressSpace The address space of \p addr.
 * @param [in, out] addr The pointer to the data.
 * @param [in] operand The operand.
 * @param [in] memoryOrder The memory ordering of \p addr.
 * @returns The value stored at \p addr before the call.
 */
template <typename T, sycl::access::address_space addressSpace =
                          sycl::access::address_space::global_space>
inline T atomic_fetch_min(T *addr, T operand,
                          sycl::memory_order memoryOrder) {
  switch (memoryOrder) {
  case sycl::memory_order::relaxed:
    return atomic_fetch_min<T, addressSpace, sycl::memory_order::relaxed,
                            sycl::memory_scope::device>(addr, operand);
  case sycl::memory_order::acq_rel:
    return atomic_fetch_min<T, addressSpace, sycl::memory_order::acq_rel,
                            sycl::memory_scope::device>(addr, operand);
  case sycl::memory_order::seq_cst:
    return atomic_fetch_min<T, addressSpace, sycl::memory_order::seq_cst,
                            sycl::memory_scope::device>(addr, operand);
  default:
    assert(false && "Invalid memory_order for atomics. Valid memory_order for "
                    "atomics are: sycl::memory_order::relaxed, "
                    "sycl::memory_order::acq_rel, sycl::memory_order::seq_cst!");
  }
}
/**
 * @brief Calculates the minimum of the value at \p addr and the operand
 * and assigns the result to the value at addr.
 * @tparam addressSpace The address space of \p addr.
 * @tparam T1 The type of the data.
 * @tparam T2 The type of the operands.
 * @param [in, out] addr The pointer to the data.
 * @param [in] operand The operand.
 * @param [in] memoryOrder The memory ordering of \p addr.
 * @returns The value stored at \p addr before the call.
 */
template <sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space,
          typename T1, typename T2>
inline T1 atomic_fetch_min(T1 *addr, T2 operand,
                           sycl::memory_order memoryOrder) {
  atomic_fetch_min<T1, addressSpace>(addr, operand, memoryOrder);
}
/**
 * @brief Calculates the maximum of the value at \p addr and the operand
 * and assigns the result to the value at addr.
 * @tparam T The type of the data and the operand.
 * @tparam addressSpace The address space of \p addr.
 * @tparam memoryOrder The memory ordering to be used.
 * @tparam memoryScope The memory scope used.
 * @param [in, out] addr The pointer to the data.
 * @param [in] operand The operand.
 * @returns The value stored at \p addr before the call.
 */
template <typename T,
          sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space,
          sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
          sycl::memory_scope memoryScope = sycl::memory_scope::device>
inline T atomic_fetch_max(T *addr, T operand) {
  auto atm =
      sycl::atomic_ref<T, memoryOrder, memoryScope, addressSpace>(addr[0]);
  return atm.fetch_max(operand);
}
/**
 * @brief Calculates the maximum of the value at \p addr and the operand
 * and assigns the result to the value at addr.
 * @tparam addressSpace The address space of \p addr.
 * @tparam memoryOrder The memory ordering to be used.
 * @tparam memoryScope The memory scope used.
 * @tparam T1 The type of the data.
 * @tparam T2 The type of the operands.
 * @param [in, out] addr The pointer to the data.
 * @param [in] operand The operand.
 * @returns The value stored at \p addr before the call.
 */
template <sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space,
          sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
          sycl::memory_scope memoryScope = sycl::memory_scope::device,
          typename T1, typename T2>
inline T1 atomic_fetch_max(T1 *addr, T2 operand) {
  auto atm =
      sycl::atomic_ref<T1, memoryOrder, memoryScope, addressSpace>(addr[0]);
  return atm.fetch_max(operand);
}

/**
 * @brief Calculates the maximum of the value at \p addr and the operand
 * and assigns the result to the value at addr.
 * @tparam T The type of the data and the operand.
 * @tparam addressSpace The address space of \p addr.
 * @param [in, out] addr The pointer to the data.
 * @param [in] operand The operand.
 * @param [in] memoryOrder The memory ordering of \p addr.
 * @returns The value stored at \p addr before the call.
 */
template <typename T, sycl::access::address_space addressSpace =
                          sycl::access::address_space::global_space>
inline T atomic_fetch_max(T *addr, T operand,
                          sycl::memory_order memoryOrder) {
  switch (memoryOrder) {
  case sycl::memory_order::relaxed:
    return atomic_fetch_max<T, addressSpace, sycl::memory_order::relaxed,
                            sycl::memory_scope::device>(addr, operand);
  case sycl::memory_order::acq_rel:
    return atomic_fetch_max<T, addressSpace, sycl::memory_order::acq_rel,
                            sycl::memory_scope::device>(addr, operand);
  case sycl::memory_order::seq_cst:
    return atomic_fetch_max<T, addressSpace, sycl::memory_order::seq_cst,
                            sycl::memory_scope::device>(addr, operand);
  default:
    assert(false && "Invalid memory_order for atomics. Valid memory_order for "
                    "atomics are: sycl::memory_order::relaxed, "
                    "sycl::memory_order::acq_rel, sycl::memory_order::seq_cst!");
  }
}
/**
 * @brief Calculates the maximum of the value at \p addr and the operand
 * and assigns the result to the value at addr.
 * @tparam addressSpace The address space of \p addr.
 * @tparam T1 The type of the data.
 * @tparam T2 The type of the operands.
 * @param [in, out] addr The pointer to the data.
 * @param [in] operand The operand.
 * @param [in] memoryOrder The memory ordering of \p addr.
 * @returns The value stored at \p addr before the call.
 */
template <sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space,
          typename T1, typename T2>
inline T1 atomic_fetch_max(T1 *addr, T2 operand,
                           sycl::memory_order memoryOrder) {
  atomic_fetch_max<T1, addressSpace>(addr, operand, memoryOrder);
}

/**
 * @brief If the original value stored in \p addr is equal to zero or greater
 * than \p operand, set \p operand to the value stored in \p addr. Otherwise,
 * decrease the value stored in \p addr.
 * @tparam addressSpace The address space of \p addr.
 * @tparam memoryOrder The memory ordering to be used.
 * @tparam memoryScope The memory scope used.
 * @param [in, out] addr The pointer to the data.
 * @param [in] operand The operand.
 * @returns The value stored at \p addr before the call.
 */
template <sycl::access::address_space addressSpace = sycl::access::address_space::global_space,
          sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
          sycl::memory_scope memoryScope = sycl::memory_scope::device>
inline unsigned int atomic_fetch_compare_dec(unsigned int *addr,
                                             unsigned int operand) {
  auto atm = sycl::atomic_ref<unsigned int, memoryOrder, memoryScope,
                                  addressSpace>(addr[0]);
  unsigned int old;

  while (true) {
    old = atm.load();
    if (old == 0 || old > operand) {
      if (atm.compare_exchange_strong(old, operand))
        break;
    } else if (atm.compare_exchange_strong(old, old - 1))
      break;
  }

  return old;
}

/**
 * @brief If the original value stored in \p addr is equal to zero or greater
 * than \p operand, set \p operand to the value stored in \p addr. Otherwise,
 * decrease the value stored in \p addr.
 * @tparam addressSpace The address space of \p addr.
 * @param [in, out] addr The pointer to the data.
 * @param [in] operand The operand.
 * @param [in] memoryOrder The memory ordering of \p addr.
 * @returns The value stored at \p addr before the call.
 */
template <sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space>
inline unsigned int
atomic_fetch_compare_dec(unsigned int *addr, unsigned int operand,
                         sycl::memory_order memoryOrder) {
  switch (memoryOrder) {
  case sycl::memory_order::relaxed:
    return atomic_fetch_compare_dec<addressSpace, sycl::memory_order::relaxed,
                                    sycl::memory_scope::device>(addr,
                                                                   operand);
  case sycl::memory_order::acq_rel:
    return atomic_fetch_compare_dec<addressSpace, sycl::memory_order::acq_rel,
                                    sycl::memory_scope::device>(addr,
                                                                   operand);
  case sycl::memory_order::seq_cst:
    return atomic_fetch_compare_dec<addressSpace, sycl::memory_order::seq_cst,
                                    sycl::memory_scope::device>(addr,
                                                                   operand);
  default:
    assert(false && "Invalid memory_order for atomics. Valid memory_order for "
                    "atomics are: sycl::memory_order::relaxed, "
                    "sycl::memory_order::acq_rel, sycl::memory_order::seq_cst!");
  }
}

/**
 * @brief If the original value stored in \p addr is equal to zero or greater
 * than \p operand, set \p operand to the value stored in \p addr. Otherwise,
 * increase the value stored in \p addr.
 * @tparam addressSpace The address space of \p addr.
 * @tparam memoryOrder The memory ordering to be used.
 * @tparam memoryScope The memory scope used.
 * @param [in, out] addr The pointer to the data.
 * @param [in] operand The operand.
 * @returns The value stored at \p addr before the call.
 */
template <sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space,
          sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
          sycl::memory_scope memoryScope = sycl::memory_scope::device>
inline unsigned int atomic_fetch_compare_inc(unsigned int *addr,
                                             unsigned int operand) {
  auto atm = sycl::atomic_ref<unsigned int, memoryOrder, memoryScope,
                                  addressSpace>(addr[0]);
  unsigned int old;
  while (true) {
    old = atm.load();
    if (old >= operand) {
      if (atm.compare_exchange_strong(old, 0))
        break;
    } else if (atm.compare_exchange_strong(old, old + 1))
      break;
  }
  return old;
}

/**
 * @brief If the original value stored in \p addr is equal to zero or greater
 * than \p operand, set \p operand to the value stored in \p addr. Otherwise,
 * increase the value stored in \p addr.
 * @tparam addressSpace The address space of \p addr.
 * @param [in, out] addr The pointer to the data.
 * @param [in] operand The operand.
 * @param [in] memoryOrder The memory ordering of \p addr.
 * @returns The value stored at \p addr before the call.
 */
template <sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space>
inline unsigned int
atomic_fetch_compare_inc(unsigned int *addr, unsigned int operand,
                         sycl::memory_order memoryOrder) {
  switch (memoryOrder) {
  case sycl::memory_order::relaxed:
    return atomic_fetch_compare_inc<addressSpace, sycl::memory_order::relaxed,
                                    sycl::memory_scope::device>(addr,
                                                                   operand);
  case sycl::memory_order::acq_rel:
    return atomic_fetch_compare_inc<addressSpace, sycl::memory_order::acq_rel,
                                    sycl::memory_scope::device>(addr,
                                                                   operand);
  case sycl::memory_order::seq_cst:
    return atomic_fetch_compare_inc<addressSpace, sycl::memory_order::seq_cst,
                                    sycl::memory_scope::device>(addr,
                                                                   operand);
  default:
    assert(false && "Invalid memory_order for atomics. Valid memory_order for "
                    "atomics are: sycl::memory_order::relaxed, "
                    "sycl::memory_order::acq_rel, sycl::memory_order::seq_cst!");
  }
}

/**
 * @brief Exchanges the value at \p addr with the operand.
 * @tparam T The type of the data and the operand.
 * @tparam addressSpace The address space of \p addr.
 * @tparam memoryOrder The memory ordering to be used.
 * @tparam memoryScope The memory scope used.
 * @param [in, out] addr The pointer to the data.
 * @param [in] operand The operand.
 * @returns The value stored at \p addr before the call.
 */
template <typename T,
          sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space,
          sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
          sycl::memory_scope memoryScope = sycl::memory_scope::device>
inline T atomic_exchange(T *addr, T operand) {
  auto atm =
      sycl::atomic_ref<T, memoryOrder, memoryScope, addressSpace>(addr[0]);
  return atm.exchange(operand);
}
/**
 * @brief Exchanges the value at \p addr with the operand.
 * @tparam addressSpace The address space of \p addr.
 * @tparam memoryOrder The memory ordering to be used.
 * @tparam memoryScope The memory scope used.
 * @tparam T1 The type of the data.
 * @tparam T2 The type of the operand.
 * @param [in, out] addr The pointer to the data.
 * @param [in] operand The operand.
 * @returns The value stored at \p addr before the call.
 */
template <sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space,
          sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
          sycl::memory_scope memoryScope = sycl::memory_scope::device,
          typename T1, typename T2>
inline T1 atomic_exchange(T1 *addr, T2 operand) {
  auto atm =
      sycl::atomic_ref<T1, memoryOrder, memoryScope, addressSpace>(addr[0]);
  return atm.exchange(operand);
}

/**
 * @brief Exchanges the value at \p addr with the operand.
 * @tparam T The type of the data and the operand.
 * @tparam addressSpace The address space of \p addr.
 * @param [in, out] addr The pointer to the data.
 * @param [in] operand The operand.
 * @param [in] memoryOrder The memory ordering of \p addr.
 * @returns The value stored at \p addr before the call.
 */
template <typename T, sycl::access::address_space addressSpace =
                          sycl::access::address_space::global_space>
inline T atomic_exchange(T *addr, T operand,
                         sycl::memory_order memoryOrder) {
  switch (memoryOrder) {
  case sycl::memory_order::relaxed:
    return atomic_exchange<T, addressSpace, sycl::memory_order::relaxed,
                           sycl::memory_scope::device>(addr, operand);
  case sycl::memory_order::acq_rel:
    return atomic_exchange<T, addressSpace, sycl::memory_order::acq_rel,
                           sycl::memory_scope::device>(addr, operand);
  case sycl::memory_order::seq_cst:
    return atomic_exchange<T, addressSpace, sycl::memory_order::seq_cst,
                           sycl::memory_scope::device>(addr, operand);
  default:
    assert(false && "Invalid memory_order for atomics. Valid memory_order for "
                    "atomics are: sycl::memory_order::relaxed, "
                    "sycl::memory_order::acq_rel, sycl::memory_order::seq_cst!");
  }
}
/**
 * @brief Exchanges the value at \p addr with the operand.
 * @tparam addressSpace The address space of \p addr.
 * @tparam T1 The type of the data.
 * @tparam T2 The type of the operand.
 * @param [in, out] addr The pointer to the data.
 * @param [in] operand The operand.
 * @param [in] memoryOrder The memory ordering of \p addr.
 * @returns The value stored at \p addr before the call.
 */
template <sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space,
          typename T1, typename T2>
inline T1 atomic_exchange(T1 *addr, T2 operand,
                           sycl::memory_order memoryOrder) {
  atomic_exchange<T1, addressSpace>(addr, operand, memoryOrder);
}
/**
 * @brief Compares the value at \p addr to \p expected and exchange the value at
 * \p addr with \p desired if the value at \p addr is equal to \p expected.
 * @tparam T The type of the data.
 * @tparam addressSpace The address space of \p addr.
 * @tparam memoryOrder The memory ordering to be used.
 * @tparam memoryScope The memory scope used.
 * @param [in, out] addr multi_ptr to the data.
 * @param [in] expected The value to compare against the value at \p addr.
 * @param [in] desired The value to assign to \p addr if the value at \p addr is
 * expected.
 * @param [in] success The memory ordering used when comparison succeeds.
 * @param [in] fail The memory ordering used when comparison fails.
 * @returns The value at the \p addr before the call.
 */
template <typename T,
          sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space,
          sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
          sycl::memory_scope memoryScope = sycl::memory_scope::device>
T atomic_compare_exchange_strong(
    sycl::multi_ptr<T, addressSpace> addr, T expected, T desired,
    sycl::memory_order success = sycl::memory_order::relaxed,
    sycl::memory_order fail = sycl::memory_order::relaxed) {
  auto atm = sycl::atomic_ref<T, memoryOrder, memoryScope, addressSpace>(*addr);

  atm.compare_exchange_strong(expected, desired, success, fail);
  return expected;
}

/**
 * @brief Compares the value at \p addr to \p expected and exchange the value at
 * \p addr with \p desired if the value at \p addr is equal to \p expected.
 * @tparam addressSpace The address space of \p addr.
 * @tparam memoryOrder The memory ordering to be used.
 * @tparam memoryScope The memory scope used.
 * @tparam T1 The type of the data.
 * @tparam T2 The type of \p expected.
 * @tparam T3 The type of \p desired.
 * @param [in, out] addr multi_ptr to the data.
 * @param [in] expected The value to compare against the value at \p addr.
 * @param [in] desired The value to assign to \p addr if the value at \p addr is
 * expected.
 * @param [in] success The memory ordering used when comparison succeeds.
 * @param [in] fail The memory ordering used when comparison fails.
 * @returns The value at the \p addr before the call.
 */
template <sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space,
          sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
          sycl::memory_scope memoryScope = sycl::memory_scope::device,
          typename T1, typename T2, typename T3>
T1 atomic_compare_exchange_strong(
    sycl::multi_ptr<T1, addressSpace> addr, T2 expected, T3 desired,
    sycl::memory_order success = sycl::memory_order::relaxed,
    sycl::memory_order fail = sycl::memory_order::relaxed) {
  auto atm =
      sycl::atomic_ref<T1, memoryOrder, memoryScope, addressSpace>(*addr);
  T1 expected_value = expected;
  atm.compare_exchange_strong(expected_value, desired, success, fail);
  return expected_value;
}

/**
 * @brief Compares the value at \p addr to \p expected and exchange the value at
 * \p addr with \p desired if the value at \p addr is equal to \p expected.
 * @tparam T The type of the data.
 * @tparam addressSpace The address space of \p addr.
 * @tparam memoryOrder The memory ordering to be used.
 * @tparam memoryScope The memory scope to be used.
 * @param [in, out] addr pointer to the data.
 * @param [in] expected The value to compare against the value at \p addr.
 * @param [in] desired The value to assign to \p addr if the value at \p addr is
 * expected.
 * @param [in] success The memory ordering used when comparison succeeds.
 * @param [in] fail The memory ordering used when comparison fails.
 * @returns The value at the \p addr before the call.
 */
template <typename T,
          sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space,
          sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
          sycl::memory_scope memoryScope = sycl::memory_scope::device>
T atomic_compare_exchange_strong(
    T *addr, T expected, T desired,
    sycl::memory_order success = sycl::memory_order::relaxed,
    sycl::memory_order fail = sycl::memory_order::relaxed) {
  auto atm =
      sycl::atomic_ref<T, memoryOrder, memoryScope, addressSpace>(addr[0]);
  atm.compare_exchange_strong(expected, desired, success, fail);
  return expected;
}
/**
 * @brief Compares the value at \p addr to \p expected and exchange the value at
 * \p addr with \p desired if the value at \p addr is equal to \p expected.
 * @tparam addressSpace The address space of \p addr.
 * @tparam memoryOrder The memory ordering to be used.
 * @tparam memoryScope The memory scope used.
 * @tparam T1 The type of the data.
 * @tparam T2 The type of \p expected.
 * @tparam T3 The type of \p desired.
 * @param [in, out] addr pointer to the data.
 * @param expected The value to compare against the value at \p addr.
 * @param desired The value to assign to \p addr if the value at \p addr is
 * expected.
 * @param success The memory ordering used when comparison succeeds.
 * @param fail The memory ordering used when comparison fails.
 * @returns The value at the \p addr before the call.
 */
template <sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space,
          sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
          sycl::memory_scope memoryScope = sycl::memory_scope::device,
          typename T1, typename T2, typename T3>
T1 atomic_compare_exchange_strong(
    T1 *addr, T2 expected, T3 desired,
    sycl::memory_order success = sycl::memory_order::relaxed,
    sycl::memory_order fail = sycl::memory_order::relaxed) {
  T1 expected_value = expected;
  auto atm =
      sycl::atomic_ref<T1, memoryOrder, memoryScope, addressSpace>(addr[0]);
  atm.compare_exchange_strong(expected_value, desired, success, fail);
  return expected_value;
}


namespace detail{
/**
 * @class IsValidAtomicType
 * @brief Utility class to check if the template type is valid for atomic
 * functions
 */
template <typename T> struct IsValidAtomicType {
  static constexpr bool value =
      (std::is_same<T, int>::value || std::is_same<T, unsigned int>::value ||
       std::is_same<T, long>::value || std::is_same<T, unsigned long>::value ||
       std::is_same<T, long long>::value ||
       std::is_same<T, unsigned long long>::value ||
       std::is_same<T, float>::value || std::is_same<T, double>::value ||
       std::is_pointer<T>::value);
};
} // namespace detail


/**
 * @class atomic
 * @brief Emulates atomic object behavior with sycl::atomic_ref.
 */
template <typename T,
          sycl::memory_scope DefaultScope = sycl::memory_scope::system,
          sycl::memory_order DefaultOrder = sycl::memory_order::seq_cst,
          sycl::access::address_space Space =
              sycl::access::address_space::generic_space>
class atomic{
  static_assert(
    detail::IsValidAtomicType<T>::value,
    "Invalid atomic type.  Valid types are int, unsigned int, long, "
      "unsigned long, long long, unsigned long long, float, double "
      "and pointer types");
  T __d;

public:
  /**
   * @brief Alias of sycl::atomic_ref<T, DefaultOrder, DefaultScope, Space>::default_read_order
   */
  static constexpr sycl::memory_order default_read_order =
      sycl::atomic_ref<T, DefaultOrder, DefaultScope,
                       Space>::default_read_order;
  /** 
   * @brief Alias of sycl::atomic_ref<T, DefaultOrder, DefaultScope, Space>::default_write_order
   */
  static constexpr sycl::memory_order default_write_order =
      sycl::atomic_ref<T, DefaultOrder, DefaultScope,
                       Space>::default_write_order;
  /** 
   * @brief Alias of DefaultScope
   */
  static constexpr sycl::memory_scope default_scope =
    DefaultScope;
  /** 
   * @brief Alias of DefaultOrder
   */
  static constexpr sycl::memory_order default_read_modify_write_order =
      DefaultOrder;
  /** 
   * @brief The default constructor.
   */
  constexpr atomic() noexcept = default;
  /** 
   * @brief The constructor with initial value.
   */
  constexpr atomic(T d) noexcept : __d(d){};

  /**
   * @brief Replaces the referenced object value with the operand
   * @param [in] operand The value to replace the pointed value.
   * @param [in] memoryOrder The memory ordering to be used.
   * @param [in] memoryScope The memory scope to be used.
   */
  void store(T operand, sycl::memory_order memoryOrder = default_write_order,
             sycl::memory_scope memoryScope = default_scope) noexcept {
    sycl::atomic_ref<T, DefaultOrder, DefaultScope, Space> atm(__d);
    atm.store(operand, memoryOrder, memoryScope);
  }

  /**
   * @brief Obtains the value of the referenced object
   * @param [in] memoryOrder The memory ordering to be used.
   * @param [in] memoryScope The memory scope to be used.
   * @return The value of the referenced object
   */
  T load(sycl::memory_order memoryOrder = default_read_order,
         sycl::memory_scope memoryScope = default_scope) const noexcept {
    sycl::atomic_ref<T, DefaultOrder, DefaultScope, Space> atm(
      const_cast<T &>(__d));
    return atm.load(memoryOrder, memoryScope);
  }

  /**
   * @brief Obtains the original value of the referenced object and then replaces
   * the value to \p operand
   * @param [in] operand The value to replace the pointed value.
   * @param [in] memoryOrder The memory ordering to be used.
   * @param [in] memoryScope The memory scope to be used.
   * @return The value of the referenced object before the call.
   */
  T exchange(T operand,
             sycl::memory_order memoryOrder = default_read_modify_write_order,
             sycl::memory_scope memoryScope = default_scope) noexcept {

    sycl::atomic_ref<T, DefaultOrder, DefaultScope, Space> atm(__d);
    return atm.exchange(operand, memoryOrder, memoryScope);
  }

  /**
   * @brief Compares the value of the referenced object with \p expected and
   * attemps to performs atomic exchange if equal or atomic load if not.
   * @param [in] expected The expected value of the referenced object.
   * @param [in] desired  The value to store in the referenced object if it is
   * expected.
   * @param [in] success The memory models for the read-modify-write.
   * @param [in] failure The memory models for load operations.
   * @param [in] memoryScope The memory scope to be used.
   * @return true if the referenced object is successfully changed, false
   * otherwise.
   */
  bool compare_exchange_weak(
      T &expected, T desired,
      sycl::memory_order success, sycl::memory_order failure,
      sycl::memory_scope memoryScope = default_scope) noexcept {
    sycl::atomic_ref<T, DefaultOrder, DefaultScope, Space> atm(__d);
    return atm.compare_exchange_weak(expected, desired, success, failure, memoryScope);
  }
  /**
   * @brief Compares the value of the referenced object with \p expected and
   * attemps to perform atomic exchange if equal or atomic load if not.
   * @param [in] expected The expected value of the referenced object.
   * @param [in] desired  The value to store in the referenced object if it is
   * expected.
   * @param [in] memoryOrder The memory synchronization ordering for operations.
   * @param [in] memoryScope The memory scope to be used.
   * @return true if the referenced object is successfully changed, false
   * otherwise.
   */
  bool compare_exchange_weak(T &expected, T desired,
                  sycl::memory_order memoryOrder = default_read_modify_write_order,
                  sycl::memory_scope memoryScope = default_scope) noexcept {
    sycl::atomic_ref<T, DefaultOrder, DefaultScope, Space> atm(__d);
    return atm.compare_exchange_weak(expected, desired, memoryOrder, memoryScope);
  }

  /**
   * @brief Compares the value of the referenced object with \p expected and
   * performs atomic exchange if equal or atomic load if not.
   * @param [in] expected The expected value of the referenced object.
   * @param [in] desired  The value to store in the referenced object if it is
   * expected.
   * @param [in] success The memory models for the read-modify-write.
   * @param [in] failure The memory models for load operations.
   * @param [in] memoryScope The memory scope to be used.
   * @return true if the referenced object is successfully changed, false
   * otherwise.
   */
  bool compare_exchange_strong(
      T &expected, T desired,
      sycl::memory_order success, sycl::memory_order failure,
      sycl::memory_scope memoryScope = default_scope) noexcept {

    sycl::atomic_ref<T, DefaultOrder, DefaultScope, Space> atm(__d);
    return atm.compare_exchange_strong(expected, desired, success, failure, memoryScope);
  }
  /**
   * @brief Compares the value of the referenced object with \p expected and
   * performs atomic exchange if equal or atomic load if not.
   * @param [in] expected The expected value of the referenced object.
   * @param [in] desired  The value to store in the referenced object if it is
   * expected.
   * @param [in] memoryOrder The memory synchronization ordering for operations.
   * @param [in] memoryScope The memory scope to be used.
   * @return true if the referenced object is successfully changed, false
   * otherwise.
   */
  bool compare_exchange_strong(T &expected, T desired,
                    sycl::memory_order memoryOrder = default_read_modify_write_order,
                    sycl::memory_scope memoryScope = default_scope) noexcept {
    sycl::atomic_ref<T, DefaultOrder, DefaultScope, Space> atm(__d);
    return atm.compare_exchange_strong(expected, desired, memoryOrder, memoryScope);
  }

  /**
   * @brief Obtains the original value of the referenced object and adds the \p operand to the value stored.
   * @param [in] operand The other argument of arithmetic addition
   * @param [in] memoryOrder The memory ordering to be used.
   * @param [in] memoryScope The memory scope to be used.
   * @return The value of the referenced object before the call.
   */
  T fetch_add(T operand,
              sycl::memory_order memoryOrder = default_read_modify_write_order,
              sycl::memory_scope  memoryScope = default_scope) noexcept {

    sycl::atomic_ref<T, DefaultOrder, DefaultScope, Space> atm(__d);
    return atm.fetch_add(operand, memoryOrder,  memoryScope);
  }

  /**
   * @brief Obtains the original value of the referenced object and subs the \p operand to the value stored.
   * @param [in] operand The other argument of arithmetic subtraction
   * @param [in] memoryOrder The memory ordering to be used.
   * @param [in] memoryScope The memory scope to be used.
   * @return The value of the referenced object before the call.
   */
  T fetch_sub(T operand,
              sycl::memory_order memoryOrder = default_read_modify_write_order,
              sycl::memory_scope memoryScope = default_scope) noexcept {

    sycl::atomic_ref<T, DefaultOrder, DefaultScope, Space> atm(__d);
    return atm.fetch_sub(operand, memoryOrder, memoryScope);
  }
};

} // namespace dpct
#endif // __DPCT_ATOMIC_HPP__
