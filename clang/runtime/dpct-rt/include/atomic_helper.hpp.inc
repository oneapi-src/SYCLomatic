// DPCT_LABEL_BEGIN|License|
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
//==---- atomic_helper.hpp -------------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
// DPCT_LABEL_END

#ifndef __DPCT_ATOMIC_HELPER_HPP__
#define __DPCT_ATOMIC_HELPER_HPP__

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
// DPCT_COMMENT   non_local_include_dependency: other header files

// DPCT_LABEL_BEGIN|non_local_include_dependency|
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
#include <CL/sycl.hpp>
#include <cassert>
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|local_include_dependency|
// DPCT_DEPENDENCY_BEGIN
// AtomicHelper|atomic_helper
// DPCT_DEPENDENCY_END
// DPCT_CODE
// DPCT_LABEL_END

namespace dpct {

// DPCT_LABEL_BEGIN|atomic_helper|dpct
// DPCT_DEPENDENCY_BEGIN
// AtomicHelper|atomic_helper
// DPCT_DEPENDENCY_END
// DPCT_CODE
/// Atomic extension to implement standard APIs in std::atomic
template <typename T,
          cl::sycl::memory_order DefaultOrder = cl::sycl::memory_order::seq_cst,
          cl::sycl::memory_scope DefaultScope = cl::sycl::memory_scope::system,
          cl::sycl::access::address_space Space =
              cl::sycl::access::address_space::generic_space>
class atomic{
  T __d;

public:
  /// default memory synchronization order
  static constexpr cl::sycl::memory_order default_read_order =
      cl::sycl::detail::memory_order_traits<DefaultOrder>::read_order;
  static constexpr cl::sycl::memory_order default_write_order =
      cl::sycl::detail::memory_order_traits<DefaultOrder>::write_order;
  static constexpr cl::sycl::memory_scope default_scope = DefaultScope;
  static constexpr cl::sycl::memory_order default_read_modify_write_order =
      DefaultOrder;
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|atomic_helper_1|dpct
// DPCT_DEPENDENCY_BEGIN
// AtomicHelper|atomic_helper
// DPCT_DEPENDENCY_END
// DPCT_CODE
  /// Default constructor.
  constexpr atomic() noexcept = default;
  /// Constructor with initialize value.
  constexpr atomic(T d) noexcept : __d(d){};
// DPCT_LABEL_END


// DPCT_LABEL_BEGIN|atomic_helper_store|dpct
// DPCT_DEPENDENCY_BEGIN
// AtomicHelper|atomic_helper
// DPCT_DEPENDENCY_END
// DPCT_CODE
  /// atomically replaces the value of the referenced object with a non-atomic argument
  /// \param [in]  replaces the value of the referenced object
  /// \param operand The value to replace the pointed value.
  /// \param memoryOrder The memory ordering used.
  /// \param memoryScope The memory scope used.
  void store(T operand, cl::sycl::memory_order memoryOrder = default_write_order,
             cl::sycl::memory_scope memoryScope = default_scope) noexcept {
    cl::sycl::atomic_ref<T, DefaultOrder, DefaultScope, Space> atm(
        const_cast<T &>(__d));
    atm.store(operand, memoryOrder, memoryScope);
  }
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|atomic_helper_load|dpct
// DPCT_DEPENDENCY_BEGIN
// AtomicHelper|atomic_helper
// DPCT_DEPENDENCY_END
// DPCT_CODE
  /// atomically obtains the value of the referenced object
  /// \param [in, out]  replaces the value of the referenced object
  /// \param memoryOrder The memory ordering used.
  /// \param memoryScope The memory scope used.
  /// \returns The value of the referenced object
  T load(cl::sycl::memory_order memoryOrder = default_read_order,
         cl::sycl::memory_scope memoryScope = default_scope) const noexcept {
    cl::sycl::atomic_ref<T, DefaultOrder, DefaultScope, Space> atm(
        const_cast<T &>(__d));
    return atm.load(memoryOrder, memoryScope);
  }
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|atomic_helper_exchange|dpct
// DPCT_DEPENDENCY_BEGIN
// AtomicHelper|atomic_helper
// DPCT_DEPENDENCY_END
// DPCT_CODE
  /// atomically replaces the value of the referenced object and obtains the value held previously
  /// \param [in, out]  replaces the value of the referenced object
  /// \param operand The value to replace the pointed value.
  /// \param memoryOrder The memory ordering used.
  /// \param memoryScope The memory scope used.
  /// \returns The value of the referenced object before the call.
  T exchange(T operand,
             cl::sycl::memory_order memoryOrder = default_read_modify_write_order,
             cl::sycl::memory_scope memoryScope = default_scope) noexcept {

    cl::sycl::atomic_ref<T, DefaultOrder, DefaultScope, Space> atm(__d);
    return atm.exchange(operand, memoryOrder, memoryScope);
  }
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|atomic_helper_compare_exchange_weak|dpct
// DPCT_DEPENDENCY_BEGIN
// AtomicHelper|atomic_helper
// DPCT_DEPENDENCY_END
// DPCT_CODE
  /// atomically compares the value of the referenced object with non-atomic argument 
  /// and performs atomic exchange if equal or atomic load if not
  /// \param [in, out]  replaces the value of the referenced object
  /// \param except The value expected to be found in the object referenced by the atomic_ref object
  /// \param desired  The value to store in the referenced object if it is as expected
  /// \param success The memory models for the read-modify-write
  /// \param failure The memory models for load operations
  /// \param memoryScope The memory scope used.
  /// \returns true if the referenced object was successfully changed, false otherwise.
  bool compare_exchange_weak(
      T &expected, T desired,
      cl::sycl::memory_order success = cl::sycl::memory_order::relaxed,
      cl::sycl::memory_order failure = cl::sycl::memory_order::relaxed,
      cl::sycl::memory_scope scope = default_scope) noexcept {
    cl::sycl::atomic_ref<T, DefaultOrder, DefaultScope, Space> atm(__d);
    return atm.compare_exchange_weak(expected, desired, failure, scope);
  }
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|atomic_helper_compare_exchange_strong|dpct
// DPCT_DEPENDENCY_BEGIN
// AtomicHelper|atomic_helper
// DPCT_DEPENDENCY_END
// DPCT_CODE
  /// atomically compares the value of the referenced object with non-atomic argument 
  /// and performs atomic exchange if equal or atomic load if not
  /// \param [in, out]  replaces the value of the referenced object
  /// \param except The value expected to be found in the object referenced by the atomic_ref object
  /// \param desired  The value to store in the referenced object if it is as expected
  /// \param success The memory models for the read-modify-write
  /// \param failure The memory models for load operations
  /// \param memoryScope The memory scope used.
  /// \returns true if the referenced object was successfully changed, false otherwise.
  bool compare_exchange_strong(
      T &expected, T desired,
      cl::sycl::memory_order success = cl::sycl::memory_order::relaxed,
      cl::sycl::memory_order failure = cl::sycl::memory_order::relaxed,
      cl::sycl::memory_scope scope = default_scope) noexcept {

    cl::sycl::atomic_ref<T, DefaultOrder, DefaultScope, Space> atm(__d);
    return atm.compare_exchange_strong(expected, desired, failure, scope);
  }
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|atomic_helper_fetch_add|dpct
// DPCT_DEPENDENCY_BEGIN
// AtomicHelper|atomic_helper
// DPCT_DEPENDENCY_END
// DPCT_CODE
  /// atomically replaces the value of the referenced object and obtains the value held previously
  /// \param [in, out]  replaces the value of the referenced object
  /// \param operand 	The other argument of arithmetic addition
  /// \param memoryOrder The memory ordering used.
  /// \param memoryScope The memory scope used.
  /// \returns The value of the referenced object before the call.
  T fetch_add(T operand,
              cl::sycl::memory_order memoryOrder = default_read_modify_write_order,
              cl::sycl::memory_scope  memoryScope = default_scope) noexcept {

    cl::sycl::atomic_ref<T, DefaultOrder, DefaultScope, Space> atm(__d);
    return atm.fetch_add(operand, memoryOrder,  memoryScope);
  }
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|atomic_helper_fetch_sub|dpct
// DPCT_DEPENDENCY_BEGIN
// AtomicHelper|atomic_helper
// DPCT_DEPENDENCY_END
// DPCT_CODE
  /// atomically replaces the value of the referenced object and obtains the value held previously
  /// \param [in, out]  replaces the value of the referenced object
  /// \param operand 	The other argument of arithmetic subtraction
  /// \param memoryOrder The memory ordering used.
  /// \param memoryScope The memory scope used.
  /// \returns The value of the referenced object before the call.
  T fetch_sub(T operand,
              cl::sycl::memory_order memoryOrder = default_read_modify_write_order,
              cl::sycl::memory_scope memoryScope = default_scope) noexcept {

    cl::sycl::atomic_ref<T, DefaultOrder, DefaultScope, Space> atm(__d);
    return atm.fetch_sub(operand, memoryOrder, memoryScope);
  }
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|atomic_helper_fetch_and|dpct
// DPCT_DEPENDENCY_BEGIN
// AtomicHelper|atomic_helper
// DPCT_DEPENDENCY_END
// DPCT_CODE
  /// atomically replaces the value of the referenced object and obtains the value held previously
  /// \param [in, out]  replaces the value of the referenced object
  /// \param operand 	The other argument of bitwise AND
  /// \param memoryOrder The memory ordering used.
  /// \param memoryScope The memory scope used.
  /// \returns The value of the referenced object before the call.
  T fetch_and(T operand,
              cl::sycl::memory_order memoryOrder = default_read_modify_write_order,
              cl::sycl::memory_scope memoryScope = default_scope) noexcept {
    cl::sycl::atomic_ref<T, DefaultOrder, DefaultScope, Space> atm(__d);
    return atm.fetch_and(operand, memoryOrder, memoryScope);
  }
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|atomic_helper_fetch_or|dpct
// DPCT_DEPENDENCY_BEGIN
// AtomicHelper|atomic_helper
// DPCT_DEPENDENCY_END
// DPCT_CODE
  /// atomically replaces the value of the referenced object and obtains the value held previously
  /// \param [in, out]  replaces the value of the referenced object
  /// \param operand 	The other argument of bitwise OR
  /// \param memoryOrder The memory ordering used.
  /// \param memoryScope The memory scope used.
  /// \returns The value of the referenced object before the call.
  T fetch_or(T operand,
             cl::sycl::memory_order memoryOrder = default_read_modify_write_order,
             cl::sycl::memory_scope memoryScope = default_scope) noexcept {
    cl::sycl::atomic_ref<T, DefaultOrder, DefaultScope, Space> atm(__d);
    return atm.fetch_or(operand, memoryOrder, memoryScope);
  }
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|atomic_helper_fetch_xor|dpct
// DPCT_DEPENDENCY_BEGIN
// AtomicHelper|atomic_helper
// DPCT_DEPENDENCY_END
// DPCT_CODE
  /// atomically replaces the value of the referenced object and obtains the value held previously
  /// \param [in, out]  replaces the value of the referenced object
  /// \param operand 	The other argument of bitwise XOR
  /// \param memoryOrder The memory ordering used.
  /// \param memoryScope The memory scope used.
  /// \returns The value of the referenced object before the call.
  T fetch_xor(T operand,
              cl::sycl::memory_order memoryOrder = default_read_modify_write_order,
              cl::sycl::memory_scope memoryScope = default_scope) noexcept {
    cl::sycl::atomic_ref<T, DefaultOrder, DefaultScope, Space> atm(__d);
    return atm.fetch_xor(operand, memoryOrder, memoryScope);
  }

};
// DPCT_LABEL_END


} // namespace dpct

#endif // __DPCT_ATOMIC_HELPER_HPP__