//==---- group_utils.h ------------------*- C++ -*---------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------===//

#ifndef __DPCT_DPCPP_EXTENSIONS_H__
#define __DPCT_DPCPP_EXTENSIONS_H__

#include <stdexcept>
#include <sycl/sycl.hpp>

#ifdef SYCL_EXT_ONEAPI_USER_DEFINED_REDUCTIONS
#include <sycl/ext/oneapi/experimental/user_defined_reductions.hpp>
#endif

#include "dpct.hpp"
#include "dpl_extras/dpcpp_extensions.h"
#include "functional.h"

namespace dpct {
namespace group {

/// Implements scatter to blocked exchange pattern used in radix sort algorithm.
///
/// \tparam T type of the data elements exchanges
/// \tparam VALUES_PER_THREAD number of data elements assigned to a thread
/// Implements blocked to striped exchange pattern
template <typename T, int VALUES_PER_THREAD> class exchange {
public:
  static size_t get_local_memory_size(size_t group_threads) {
    size_t padding_values =
        (INSERT_PADDING)
            ? ((group_threads * VALUES_PER_THREAD) >> LOG_LOCAL_MEMORY_BANKS)
            : 0;
    return (group_threads * VALUES_PER_THREAD + padding_values) * sizeof(T);
  }

  exchange(uint8_t *local_memory) : _local_memory(local_memory) {}

  // TODO: Investigate if padding is required for performance,
  // and if specializations are required for specific target hardware.
  static size_t adjust_by_padding(size_t offset) {

    if constexpr (INSERT_PADDING) {
      offset = detail::shr_add(offset, LOG_LOCAL_MEMORY_BANKS, offset);
    }
    return offset;
  }

  struct blocked_offset {
    template <typename Item> size_t operator()(Item item, size_t i) {
      size_t offset = item.get_local_linear_id() * VALUES_PER_THREAD + i;
      return adjust_by_padding(offset);
    }
  };

  struct striped_offset {
    template <typename Item> size_t operator()(Item item, size_t i) {
      size_t offset = i * item.get_local_range(2) * item.get_local_range(1) *
                          item.get_local_range(0) +
                      item.get_local_linear_id();
      return adjust_by_padding(offset);
    }
  };

  template <typename Iterator> struct scatter_offset {
    Iterator begin;
    scatter_offset(const int (&ranks)[VALUES_PER_THREAD]) {
      begin = std::begin(ranks);
    }
    template <typename Item> size_t operator()(Item item, size_t i) const {
      // iterator i is expected to be within bounds [0,VALUES_PER_THREAD)
      return adjust_by_padding(begin[i]);
    }
  };

  template <typename Item, typename offsetFunctorTypeFW,
            typename offsetFunctorTypeRV>
  __dpct_inline__ void helper_exchange(Item item, T (&keys)[VALUES_PER_THREAD],
                                       offsetFunctorTypeFW &offset_functor_fw,
                                       offsetFunctorTypeRV &offset_functor_rv) {

    T *buffer = reinterpret_cast<T *>(_local_memory);

#pragma unroll
    for (size_t i = 0; i < VALUES_PER_THREAD; i++) {
      size_t offset = offset_functor_fw(item, i);
      buffer[offset] = keys[i];
    }

    item.barrier(sycl::access::fence_space::local_space);

#pragma unroll
    for (size_t i = 0; i < VALUES_PER_THREAD; i++) {
      size_t offset = offset_functor_rv(item, i);
      keys[i] = buffer[offset];
    }
  }

  /// Rearrange elements from blocked order to striped order
  template <typename Item>
  __dpct_inline__ void blocked_to_striped(Item item,
                                          T (&keys)[VALUES_PER_THREAD]) {

    striped_offset get_striped_offset;
    blocked_offset get_blocked_offset;
    helper_exchange(item, keys, get_blocked_offset, get_striped_offset);
  }

  /// Rearrange elements from striped order to blocked order
  template <typename Item>
  __dpct_inline__ void striped_to_blocked(Item item,
                                          T (&keys)[VALUES_PER_THREAD]) {

    blocked_offset get_blocked_offset;
    striped_offset get_striped_offset;
    helper_exchange(item, keys, get_striped_offset, get_blocked_offset);
  }

  /// Rearrange elements from rank order to blocked order
  template <typename Item>
  __dpct_inline__ void scatter_to_blocked(Item item,
                                          T (&keys)[VALUES_PER_THREAD],
                                          int (&ranks)[VALUES_PER_THREAD]) {

    scatter_offset<const int *> get_scatter_offset(ranks);
    blocked_offset get_blocked_offset;
    helper_exchange(item, keys, get_scatter_offset, get_blocked_offset);
  }

  /// Rearrange elements from scatter order to striped order
  template <typename Item>
  __dpct_inline__ void scatter_to_striped(Item item,
                                          T (&keys)[VALUES_PER_THREAD],
                                          int (&ranks)[VALUES_PER_THREAD]) {

    scatter_offset<const int *> get_scatter_offset(ranks);
    striped_offset get_striped_offset;
    helper_exchange(item, keys, get_scatter_offset, get_striped_offset);
  }

private:
  static constexpr int LOG_LOCAL_MEMORY_BANKS = 4;
  static constexpr bool INSERT_PADDING =
      (VALUES_PER_THREAD > 4) &&
      (dpct::group::detail::power_of_two<VALUES_PER_THREAD>::VALUE);

  uint8_t *_local_memory;
};

/// Implements radix sort to sort integer data elements assigned to all threads
/// in the group.
///
/// \tparam T type of the data elements exchanges
/// \tparam VALUES_PER_THREAD number of data elements assigned to a thread
/// \tparam DECENDING boolean value indicating if data elements are sorted in
/// decending order.
template <typename T, int VALUES_PER_THREAD, bool DESCENDING = false>
class radix_sort {
public:
  static size_t get_local_memory_size(size_t group_threads) {
    size_t ranks_size =
        dpct::group::detail::radix_rank<RADIX_BITS>::get_local_memory_size(
            group_threads);
    size_t exchange_size =
        exchange<T, VALUES_PER_THREAD>::get_local_memory_size(group_threads);
    return sycl::max(ranks_size, exchange_size);
  }

  radix_sort(uint8_t *local_memory) : _local_memory(local_memory) {}

  template <typename Item>
  __dpct_inline__ void
  helper_sort(const Item &item, T (&keys)[VALUES_PER_THREAD], int begin_bit = 0,
              int end_bit = 8 * sizeof(T), bool is_striped = false) {

    uint32_t(&unsigned_keys)[VALUES_PER_THREAD] =
        reinterpret_cast<uint32_t(&)[VALUES_PER_THREAD]>(keys);

#pragma unroll
    for (int i = 0; i < VALUES_PER_THREAD; ++i) {
      unsigned_keys[i] =
          dpct::group::detail::traits<T>::twiddle_in(unsigned_keys[i]);
    }

    for (int i = begin_bit; i < end_bit; i += RADIX_BITS) {
      int pass_bits = sycl::min(RADIX_BITS, end_bit - begin_bit);

      int ranks[VALUES_PER_THREAD];
      dpct::group::detail::radix_rank<RADIX_BITS, DESCENDING>(_local_memory)
          .template rank_keys(item, unsigned_keys, ranks, i, pass_bits);

      item.barrier(sycl::access::fence_space::local_space);

      bool last_iter = i + RADIX_BITS > end_bit;
      if (last_iter && is_striped) {
        exchange<T, VALUES_PER_THREAD>(_local_memory)
            .scatter_to_striped(item, keys, ranks);

      } else {
        exchange<T, VALUES_PER_THREAD>(_local_memory)
            .scatter_to_blocked(item, keys, ranks);
      }

      item.barrier(sycl::access::fence_space::local_space);
    }

#pragma unroll
    for (int i = 0; i < VALUES_PER_THREAD; ++i) {
      unsigned_keys[i] =
          dpct::group::detail::traits<T>::twiddle_out(unsigned_keys[i]);
    }
  }

  template <typename Item>
  __dpct_inline__ void
  sort_blocked(const Item &item, T (&keys)[VALUES_PER_THREAD],
               int begin_bit = 0, int end_bit = 8 * sizeof(T)) {
    helper_sort(item, keys, begin_bit, end_bit, false);
  }

  template <typename Item>
  __dpct_inline__ void
  sort_blocked_to_striped(const Item &item, T (&keys)[VALUES_PER_THREAD],
                          int begin_bit = 0, int end_bit = 8 * sizeof(T)) {
    helper_sort(item, keys, begin_bit, end_bit, true);
  }

private:
  static constexpr int RADIX_BITS = 4;

  uint8_t *_local_memory;
};

} // namespace group
} // namespace dpct

#endif
