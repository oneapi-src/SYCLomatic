// DPCT_LABEL_BEGIN|License|
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
//==---- atomic.hpp -------------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
// DPCT_LABEL_END

#ifndef __DPCT_ATOMIC_HPP__
#define __DPCT_ATOMIC_HPP__
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

// DPCT_LABEL_BEGIN|non_local_include_dependency|
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
#include <CL/sycl.hpp>
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|local_include_dependency|
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
// DPCT_LABEL_END

namespace dpct {

// DPCT_LABEL_BEGIN|atomic_fetch_add|dpct
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
/// Atomically add the value operand to the value at the addr and assign the
/// result to the value at addr.
/// \param [in, out] addr The pointer to the data.
/// \param operand The value to add to the value at \p addr.
/// \param memoryOrder The memory ordering used.
/// \returns The value at the \p addr before the call.
template <typename T,
          cl::sycl::access::address_space addressSpace =
              cl::sycl::access::address_space::global_space,
          cl::sycl::memory_order memoryOrder = cl::sycl::memory_order::relaxed,
          cl::sycl::memory_scope memoryScope = cl::sycl::memory_scope::device>
inline T atomic_fetch_add(T *addr, T operand) {
  auto atm =
      cl::sycl::atomic_ref<T, memoryOrder, memoryScope, addressSpace>(addr[0]);
  return atm.fetch_add(operand);
}

/// Atomically add the value operand to the value at the addr and assign the
/// result to the value at addr.
/// \param [in, out] addr The pointer to the data.
/// \param operand The value to add to the value at \p addr.
/// \param memoryOrder The memory ordering used.
/// \returns The value at the \p addr before the call.
template <typename T, cl::sycl::access::address_space addressSpace =
                          cl::sycl::access::address_space::global_space>
inline T atomic_fetch_add(T *addr, T operand,
                          cl::sycl::memory_order memoryOrder) {
  switch (memoryOrder) {
  case cl::sycl::memory_order::relaxed:
    return atomic_fetch_add<T, addressSpace, cl::sycl::memory_order::relaxed,
                            cl::sycl::memory_scope::device>(addr, operand);
  case cl::sycl::memory_order::acq_rel:
    return atomic_fetch_add<T, addressSpace, cl::sycl::memory_order::acq_rel,
                            cl::sycl::memory_scope::device>(addr, operand);
  case cl::sycl::memory_order::seq_cst:
    return atomic_fetch_add<T, addressSpace, cl::sycl::memory_order::seq_cst,
                            cl::sycl::memory_scope::device>(addr, operand);
  default:
    assert(false && "Invalid memory_order for atomics. Valid memory_order for "
                    "atomics are: cl::sycl::memory_order::relaxed, "
                    "cl::sycl::memory_order::acq_rel, cl::sycl::memory_order::seq_cst!");
  }
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|atomic_fetch_sub|dpct
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
/// Atomically subtract the value operand from the value at the addr and assign
/// the result to the value at addr.
/// \param [in, out] addr The pointer to the data.
/// \param operand The value to substract from the value at \p addr
/// \param memoryOrder The memory ordering used.
/// \returns The value at the \p addr before the call.
template <typename T,
          cl::sycl::access::address_space addressSpace =
              cl::sycl::access::address_space::global_space,
          cl::sycl::memory_order memoryOrder = cl::sycl::memory_order::relaxed,
          cl::sycl::memory_scope memoryScope = cl::sycl::memory_scope::device>
inline T atomic_fetch_sub(T *addr, T operand) {
  auto atm =
      cl::sycl::atomic_ref<T, memoryOrder, memoryScope, addressSpace>(addr[0]);
  return atm.fetch_sub(operand);
}

/// Atomically subtract the value operand from the value at the addr and assign
/// the result to the value at addr.
/// \param [in, out] addr The pointer to the data.
/// \param operand The value to substract from the value at \p addr
/// \param memoryOrder The memory ordering used.
/// \returns The value at the \p addr before the call.
template <typename T, cl::sycl::access::address_space addressSpace =
                          cl::sycl::access::address_space::global_space>
inline T atomic_fetch_sub(T *addr, T operand,
                          cl::sycl::memory_order memoryOrder) {
  switch (memoryOrder) {
  case cl::sycl::memory_order::relaxed:
    return atomic_fetch_sub<T, addressSpace, cl::sycl::memory_order::relaxed,
                            cl::sycl::memory_scope::device>(addr, operand);
  case cl::sycl::memory_order::acq_rel:
    return atomic_fetch_sub<T, addressSpace, cl::sycl::memory_order::acq_rel,
                            cl::sycl::memory_scope::device>(addr, operand);
  case cl::sycl::memory_order::seq_cst:
    return atomic_fetch_sub<T, addressSpace, cl::sycl::memory_order::seq_cst,
                            cl::sycl::memory_scope::device>(addr, operand);
  default:
    assert(false && "Invalid memory_order for atomics. Valid memory_order for "
                    "atomics are: cl::sycl::memory_order::relaxed, "
                    "cl::sycl::memory_order::acq_rel, cl::sycl::memory_order::seq_cst!");
  }
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|atomic_fetch_and|dpct
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
/// Atomically perform a bitwise AND between the value operand and the value at the addr
/// and assign the result to the value at addr.
/// \param [in, out] addr The pointer to the data.
/// \param operand The value to use in bitwise AND operation with the value at the \p addr.
/// \param memoryOrder The memory ordering used.
/// \returns The value at the \p addr before the call.
template <typename T,
          cl::sycl::access::address_space addressSpace =
              cl::sycl::access::address_space::global_space,
          cl::sycl::memory_order memoryOrder = cl::sycl::memory_order::relaxed,
          cl::sycl::memory_scope memoryScope = cl::sycl::memory_scope::device>
inline T atomic_fetch_and(T *addr, T operand) {
  auto atm =
      cl::sycl::atomic_ref<T, memoryOrder, memoryScope, addressSpace>(addr[0]);
  return atm.fetch_and(operand);
}

/// Atomically perform a bitwise AND between the value operand and the value at the addr
/// and assign the result to the value at addr.
/// \param [in, out] addr The pointer to the data.
/// \param operand The value to use in bitwise AND operation with the value at the \p addr.
/// \param memoryOrder The memory ordering used.
/// \returns The value at the \p addr before the call.
template <typename T, cl::sycl::access::address_space addressSpace =
                          cl::sycl::access::address_space::global_space>
inline T atomic_fetch_and(T *addr, T operand,
                          cl::sycl::memory_order memoryOrder) {
  switch (memoryOrder) {
  case cl::sycl::memory_order::relaxed:
    return atomic_fetch_and<T, addressSpace, cl::sycl::memory_order::relaxed,
                            cl::sycl::memory_scope::device>(addr, operand);
  case cl::sycl::memory_order::acq_rel:
    return atomic_fetch_and<T, addressSpace, cl::sycl::memory_order::acq_rel,
                            cl::sycl::memory_scope::device>(addr, operand);
  case cl::sycl::memory_order::seq_cst:
    return atomic_fetch_and<T, addressSpace, cl::sycl::memory_order::seq_cst,
                            cl::sycl::memory_scope::device>(addr, operand);
  default:
    assert(false && "Invalid memory_order for atomics. Valid memory_order for "
                    "atomics are: cl::sycl::memory_order::relaxed, "
                    "cl::sycl::memory_order::acq_rel, cl::sycl::memory_order::seq_cst!");
  }
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|atomic_fetch_or|dpct
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
/// Atomically or the value at the addr with the value operand, and assign
/// the result to the value at addr.
/// \param [in, out] addr The pointer to the data.
/// \param operand The value to use in bitwise OR operation with the value at the \p addr.
/// \param memoryOrder The memory ordering used.
/// \returns The value at the \p addr before the call.
template <typename T,
          cl::sycl::access::address_space addressSpace =
              cl::sycl::access::address_space::global_space,
          cl::sycl::memory_order memoryOrder = cl::sycl::memory_order::relaxed,
          cl::sycl::memory_scope memoryScope = cl::sycl::memory_scope::device>
inline T atomic_fetch_or(T *addr, T operand) {
  auto atm =
      cl::sycl::atomic_ref<T, memoryOrder, memoryScope, addressSpace>(addr[0]);
  return atm.fetch_or(operand);
}

/// Atomically or the value at the addr with the value operand, and assign
/// the result to the value at addr.
/// \param [in, out] addr The pointer to the data.
/// \param operand The value to use in bitwise OR operation with the value at the \p addr.
/// \param memoryOrder The memory ordering used.
/// \returns The value at the \p addr before the call.
template <typename T, cl::sycl::access::address_space addressSpace =
                          cl::sycl::access::address_space::global_space>
inline T atomic_fetch_or(T *addr, T operand,
                         cl::sycl::memory_order memoryOrder) {
  switch (memoryOrder) {
  case cl::sycl::memory_order::relaxed:
    return atomic_fetch_or<T, addressSpace, cl::sycl::memory_order::relaxed,
                           cl::sycl::memory_scope::device>(addr, operand);
  case cl::sycl::memory_order::acq_rel:
    return atomic_fetch_or<T, addressSpace, cl::sycl::memory_order::acq_rel,
                           cl::sycl::memory_scope::device>(addr, operand);
  case cl::sycl::memory_order::seq_cst:
    return atomic_fetch_or<T, addressSpace, cl::sycl::memory_order::seq_cst,
                           cl::sycl::memory_scope::device>(addr, operand);
  default:
    assert(false && "Invalid memory_order for atomics. Valid memory_order for "
                    "atomics are: cl::sycl::memory_order::relaxed, "
                    "cl::sycl::memory_order::acq_rel, cl::sycl::memory_order::seq_cst!");
  }
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|atomic_fetch_xor|dpct
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
/// Atomically xor the value at the addr with the value operand, and assign
/// the result to the value at addr.
/// \param [in, out] addr The pointer to the data.
/// \param operand The value to use in bitwise XOR operation with the value at the \p addr.
/// \param memoryOrder The memory ordering used.
/// \returns The value at the \p addr before the call.
template <typename T,
          cl::sycl::access::address_space addressSpace =
              cl::sycl::access::address_space::global_space,
          cl::sycl::memory_order memoryOrder = cl::sycl::memory_order::relaxed,
          cl::sycl::memory_scope memoryScope = cl::sycl::memory_scope::device>
inline T atomic_fetch_xor(T *addr, T operand) {
  auto atm =
      cl::sycl::atomic_ref<T, memoryOrder, memoryScope, addressSpace>(addr[0]);
  return atm.fetch_xor(operand);
}

/// Atomically xor the value at the addr with the value operand, and assign
/// the result to the value at addr.
/// \param [in, out] addr The pointer to the data.
/// \param operand The value to use in bitwise XOR operation with the value at the \p addr.
/// \param memoryOrder The memory ordering used.
/// \returns The value at the \p addr before the call.
template <typename T, cl::sycl::access::address_space addressSpace =
                          cl::sycl::access::address_space::global_space>
inline T atomic_fetch_xor(T *addr, T operand,
                          cl::sycl::memory_order memoryOrder) {
  switch (memoryOrder) {
  case cl::sycl::memory_order::relaxed:
    return atomic_fetch_xor<T, addressSpace, cl::sycl::memory_order::relaxed,
                            cl::sycl::memory_scope::device>(addr, operand);
  case cl::sycl::memory_order::acq_rel:
    return atomic_fetch_xor<T, addressSpace, cl::sycl::memory_order::acq_rel,
                            cl::sycl::memory_scope::device>(addr, operand);
  case cl::sycl::memory_order::seq_cst:
    return atomic_fetch_xor<T, addressSpace, cl::sycl::memory_order::seq_cst,
                            cl::sycl::memory_scope::device>(addr, operand);
  default:
    assert(false && "Invalid memory_order for atomics. Valid memory_order for "
                    "atomics are: cl::sycl::memory_order::relaxed, "
                    "cl::sycl::memory_order::acq_rel, cl::sycl::memory_order::seq_cst!");
  }
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|atomic_fetch_min|dpct
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
/// Atomically calculate the minimum of the value at addr and the value operand
/// and assign the result to the value at addr.
/// \param [in, out] addr The pointer to the data.
/// \param operand.
/// \param memoryOrder The memory ordering used.
/// \returns The value at the \p addr before the call.
template <typename T,
          cl::sycl::access::address_space addressSpace =
              cl::sycl::access::address_space::global_space,
          cl::sycl::memory_order memoryOrder = cl::sycl::memory_order::relaxed,
          cl::sycl::memory_scope memoryScope = cl::sycl::memory_scope::device>
inline T atomic_fetch_min(T *addr, T operand) {
  auto atm =
      cl::sycl::atomic_ref<T, memoryOrder, memoryScope, addressSpace>(addr[0]);
  return atm.fetch_min(operand);
}

/// Atomically calculate the minimum of the value at addr and the value operand
/// and assign the result to the value at addr.
/// \param [in, out] addr The pointer to the data.
/// \param operand.
/// \param memoryOrder The memory ordering used.
/// \returns The value at the \p addr before the call.
template <typename T, cl::sycl::access::address_space addressSpace =
                          cl::sycl::access::address_space::global_space>
inline T atomic_fetch_min(T *addr, T operand,
                          cl::sycl::memory_order memoryOrder) {
  switch (memoryOrder) {
  case cl::sycl::memory_order::relaxed:
    return atomic_fetch_min<T, addressSpace, cl::sycl::memory_order::relaxed,
                            cl::sycl::memory_scope::device>(addr, operand);
  case cl::sycl::memory_order::acq_rel:
    return atomic_fetch_min<T, addressSpace, cl::sycl::memory_order::acq_rel,
                            cl::sycl::memory_scope::device>(addr, operand);
  case cl::sycl::memory_order::seq_cst:
    return atomic_fetch_min<T, addressSpace, cl::sycl::memory_order::seq_cst,
                            cl::sycl::memory_scope::device>(addr, operand);
  default:
    assert(false && "Invalid memory_order for atomics. Valid memory_order for "
                    "atomics are: cl::sycl::memory_order::relaxed, "
                    "cl::sycl::memory_order::acq_rel, cl::sycl::memory_order::seq_cst!");
  }
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|atomic_fetch_max|dpct
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
/// Atomically calculate the maximum of the value at addr and the value operand
/// and assign the result to the value at addr.
/// \param [in, out] addr The pointer to the data.
/// \param operand.
/// \param memoryOrder The memory ordering used.
/// \returns The value at the \p addr before the call.
template <typename T,
          cl::sycl::access::address_space addressSpace =
              cl::sycl::access::address_space::global_space,
          cl::sycl::memory_order memoryOrder = cl::sycl::memory_order::relaxed,
          cl::sycl::memory_scope memoryScope = cl::sycl::memory_scope::device>
inline T atomic_fetch_max(T *addr, T operand) {
  auto atm =
      cl::sycl::atomic_ref<T, memoryOrder, memoryScope, addressSpace>(addr[0]);
  return atm.fetch_max(operand);
}

/// Atomically calculate the maximum of the value at addr and the value operand
/// and assign the result to the value at addr.
/// \param [in, out] addr The pointer to the data.
/// \param operand.
/// \param memoryOrder The memory ordering used.
/// \returns The value at the \p addr before the call.
template <typename T, cl::sycl::access::address_space addressSpace =
                          cl::sycl::access::address_space::global_space>
inline T atomic_fetch_max(T *addr, T operand,
                          cl::sycl::memory_order memoryOrder) {
  switch (memoryOrder) {
  case cl::sycl::memory_order::relaxed:
    return atomic_fetch_max<T, addressSpace, cl::sycl::memory_order::relaxed,
                            cl::sycl::memory_scope::device>(addr, operand);
  case cl::sycl::memory_order::acq_rel:
    return atomic_fetch_max<T, addressSpace, cl::sycl::memory_order::acq_rel,
                            cl::sycl::memory_scope::device>(addr, operand);
  case cl::sycl::memory_order::seq_cst:
    return atomic_fetch_max<T, addressSpace, cl::sycl::memory_order::seq_cst,
                            cl::sycl::memory_scope::device>(addr, operand);
  default:
    assert(false && "Invalid memory_order for atomics. Valid memory_order for "
                    "atomics are: cl::sycl::memory_order::relaxed, "
                    "cl::sycl::memory_order::acq_rel, cl::sycl::memory_order::seq_cst!");
  }
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|atomic_fetch_compare_inc|dpct
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
/// Atomically increment the value stored in \p addr if old value stored in \p
/// addr is less than \p operand, else set 0 to the value stored in \p addr.
/// \param [in, out] addr The pointer to the data.
/// \param operand The threshold value.
/// \param memoryOrder The memory ordering used.
/// \returns The old value stored in \p addr.
template <cl::sycl::access::address_space addressSpace =
              cl::sycl::access::address_space::global_space,
          cl::sycl::memory_order memoryOrder = cl::sycl::memory_order::relaxed,
          cl::sycl::memory_scope memoryScope = cl::sycl::memory_scope::device>
inline unsigned int atomic_fetch_compare_inc(unsigned int *addr,
                                             unsigned int operand) {
  auto atm = cl::sycl::atomic_ref<unsigned int, memoryOrder, memoryScope,
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

/// Atomically increment the value stored in \p addr if old value stored in \p
/// addr is less than \p operand, else set 0 to the value stored in \p addr.
/// \param [in, out] addr The pointer to the data.
/// \param operand The threshold value.
/// \param memoryOrder The memory ordering used.
/// \returns The old value stored in \p addr.
template <cl::sycl::access::address_space addressSpace =
              cl::sycl::access::address_space::global_space>
inline unsigned int
atomic_fetch_compare_inc(unsigned int *addr, unsigned int operand,
                         cl::sycl::memory_order memoryOrder) {
  switch (memoryOrder) {
  case cl::sycl::memory_order::relaxed:
    return atomic_fetch_compare_inc<addressSpace, cl::sycl::memory_order::relaxed,
                                    cl::sycl::memory_scope::device>(addr,
                                                                   operand);
  case cl::sycl::memory_order::acq_rel:
    return atomic_fetch_compare_inc<addressSpace, cl::sycl::memory_order::acq_rel,
                                    cl::sycl::memory_scope::device>(addr,
                                                                   operand);
  case cl::sycl::memory_order::seq_cst:
    return atomic_fetch_compare_inc<addressSpace, cl::sycl::memory_order::seq_cst,
                                    cl::sycl::memory_scope::device>(addr,
                                                                   operand);
  default:
    assert(false && "Invalid memory_order for atomics. Valid memory_order for "
                    "atomics are: cl::sycl::memory_order::relaxed, "
                    "cl::sycl::memory_order::acq_rel, cl::sycl::memory_order::seq_cst!");
  }
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|atomic_exchange|dpct
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
/// Atomically exchange the value at the address addr with the value operand.
/// \param [in, out] addr The pointer to the data.
/// \param operand The value to be exchanged with the value pointed by \p addr.
/// \param memoryOrder The memory ordering used.
/// \returns The value at the \p addr before the call.
template <typename T,
          cl::sycl::access::address_space addressSpace =
              cl::sycl::access::address_space::global_space,
          cl::sycl::memory_order memoryOrder = cl::sycl::memory_order::relaxed,
          cl::sycl::memory_scope memoryScope = cl::sycl::memory_scope::device>
inline T atomic_exchange(T *addr, T operand) {
  auto atm =
      cl::sycl::atomic_ref<T, memoryOrder, memoryScope, addressSpace>(addr[0]);
  return atm.exchange(operand);
}

/// Atomically exchange the value at the address addr with the value operand.
/// \param [in, out] addr The pointer to the data.
/// \param operand The value to be exchanged with the value pointed by \p addr.
/// \param memoryOrder The memory ordering used.
/// \returns The value at the \p addr before the call.
template <typename T, cl::sycl::access::address_space addressSpace =
                          cl::sycl::access::address_space::global_space>
inline T atomic_exchange(T *addr, T operand,
                         cl::sycl::memory_order memoryOrder) {
  switch (memoryOrder) {
  case cl::sycl::memory_order::relaxed:
    return atomic_exchange<T, addressSpace, cl::sycl::memory_order::relaxed,
                           cl::sycl::memory_scope::device>(addr, operand);
  case cl::sycl::memory_order::acq_rel:
    return atomic_exchange<T, addressSpace, cl::sycl::memory_order::acq_rel,
                           cl::sycl::memory_scope::device>(addr, operand);
  case cl::sycl::memory_order::seq_cst:
    return atomic_exchange<T, addressSpace, cl::sycl::memory_order::seq_cst,
                           cl::sycl::memory_scope::device>(addr, operand);
  default:
    assert(false && "Invalid memory_order for atomics. Valid memory_order for "
                    "atomics are: cl::sycl::memory_order::relaxed, "
                    "cl::sycl::memory_order::acq_rel, cl::sycl::memory_order::seq_cst!");
  }
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|atomic_compare_exchange_strong|dpct
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
/// Atomically compare the value at \p addr to the value expected and exchange
/// with the value desired if the value at \p addr is equal to the value expected.
/// Returns the value at the \p addr before the call.
/// \param [in, out] addr Multi_ptr.
/// \param expected The value to compare against the value at \p addr.
/// \param desired The value to assign to \p addr if the value at \p addr is expected.
/// \param success The memory ordering used when comparison succeeds.
/// \param fail The memory ordering used when comparison fails.
/// \returns The value at the \p addr before the call.
template <typename T,
          cl::sycl::access::address_space addressSpace =
              cl::sycl::access::address_space::global_space,
          cl::sycl::memory_order memoryOrder = cl::sycl::memory_order::relaxed,
          cl::sycl::memory_scope memoryScope = cl::sycl::memory_scope::device>
T atomic_compare_exchange_strong(
    cl::sycl::multi_ptr<T, cl::sycl::access::address_space::global_space> addr,
    T expected, T desired,
    cl::sycl::memory_order success = cl::sycl::memory_order::relaxed,
    cl::sycl::memory_order fail = cl::sycl::memory_order::relaxed) {
  auto atm =
      cl::sycl::atomic_ref<T, memoryOrder, memoryScope, addressSpace>(*addr);

  atm.compare_exchange_strong(expected, desired, success, fail);
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
/// \returns The value at the \p addr before the call.
template <typename T,
          cl::sycl::access::address_space addressSpace =
              cl::sycl::access::address_space::global_space,
          cl::sycl::memory_order memoryOrder = cl::sycl::memory_order::relaxed,
          cl::sycl::memory_scope memoryScope = cl::sycl::memory_scope::device>
T atomic_compare_exchange_strong(
    T *addr, T expected, T desired,
    cl::sycl::memory_order success = cl::sycl::memory_order::relaxed,
    cl::sycl::memory_order fail = cl::sycl::memory_order::relaxed) {
  auto atm =
      cl::sycl::atomic_ref<T, memoryOrder, memoryScope, addressSpace>(addr[0]);
  atm.compare_exchange_strong(expected, desired, success, fail);
  return expected;
}
// DPCT_LABEL_END

} // namespace dpct
#endif // __DPCT_ATOMIC_HPP__
