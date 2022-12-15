// DPCT_LABEL_BEGIN|License|
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
//==---- dpcpp_extensions.h ------------------*- C++ -*---------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------===//
// DPCT_LABEL_END

#ifndef __DPCT_DPCPP_EXTENSIONS_H__
#define __DPCT_DPCPP_EXTENSIONS_H__

// DPCT_LABEL_BEGIN|non_local_include_dependency|
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
#include <sycl/sycl.hpp>
#include <stdexcept>
// DPCT_LABEL_END

namespace dpct {
namespace group {

namespace detail {

// DPCT_LABEL_BEGIN|group_reduce|dpct::group::detail
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
template <typename... _Args>
constexpr auto __reduce_over_group(_Args... __args) {
  return sycl::reduce_over_group(__args...);
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|group_broadcast|dpct::group::detail
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
template <typename... _Args> constexpr auto __group_broadcast(_Args... __args) {
  return sycl::group_broadcast(__args...);
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|group_scan|dpct::group::detail
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
template <typename... _Args>
constexpr auto __exclusive_scan_over_group(_Args... __args) {
  return sycl::exclusive_scan_over_group(__args...);
}

template <typename... _Args>
constexpr auto __inclusive_scan_over_group(_Args... __args) {
  return sycl::inclusive_scan_over_group(__args...);
}
// DPCT_LABEL_END

} // end namespace detail

// DPCT_LABEL_BEGIN|exclusive_scan|dpct::group
// DPCT_DEPENDENCY_BEGIN
// DplExtrasDpcppExtensions|group_broadcast
// DplExtrasDpcppExtensions|group_scan
// DPCT_DEPENDENCY_END
// DPCT_CODE
/// Perform an exclusive scan over the values of inputs from all work-items in
/// the group using the operator binary_op, which must be one of the SYCL 2020
/// group algorithms library function objects.
///
/// \param item A work-item in a group.
/// \param inputs Pointer to the input data for the scan operation.
/// \param outputs Pointer to the location where scan results will be stored.
/// \param init initial value of the scan result.
/// \param binary_op functor that implements the binary operation used to
/// perform the scan.
template <typename Item, typename T, class BinaryOperation,
          int VALUES_PER_THREAD>
__attribute__((always_inline)) void
exclusive_scan(const Item &item, T (&inputs)[VALUES_PER_THREAD],
               T (&outputs)[VALUES_PER_THREAD], T init,
               BinaryOperation binary_op) {
  T result = inputs[0];

#pragma unroll
  for (int i = 1; i < VALUES_PER_THREAD; ++i) {
    result = binary_op(result, inputs[i]);
  }

  T exclusive_result =
      detail::__exclusive_scan_over_group(item.get_group(), result, binary_op);

  T input = inputs[0];
  if (item.get_local_linear_id() == 0) {
    outputs[0] = init;
  } else {
    outputs[0] = exclusive_result;
  }

#pragma unroll
  for (int i = 1; i < VALUES_PER_THREAD; ++i) {
    T output = binary_op(input, outputs[i - 1]);
    input = inputs[i];
    outputs[i] = output;
  }
}

/// Perform an exclusive scan over the values of input from all work-items in
/// the group using the operator binary_op, which must be one of the SYCL 2020
/// group algorithms library function objects.
///
/// \param item A work-item in a group.
/// \param input Input data for the scan operation.
/// \param init initial value of the scan result.
/// \param binary_op functor that implements the binary operation used to
/// perform the scan. \param group_aggregate group-wide aggregate of all inputs
/// in the work-items of the group. \returns exclusive scan of the first i
/// work-items where item is the i-th work item.
template <typename Item, typename T, class BinaryOperation>
__attribute__((always_inline)) T
exclusive_scan(const Item &item, T input, T init, BinaryOperation binary_op,
               T &group_aggregate) {
  T output = detail::__exclusive_scan_over_group(item.get_group(), input, init,
                                                 binary_op);
  if (item.get_local_linear_id() == item.get_local_range().size() - 1) {
    group_aggregate = binary_op(output, input);
  }

  group_aggregate = detail::__group_broadcast(
      item.get_group(), group_aggregate, item.get_local_range().size() - 1);
  return output;
}

/// Perform an exclusive scan over the values of input from all work-items in
/// the group using the operator binary_op, which must be one of the SYCL 2020
/// group algorithms library function objects.
///
/// \param item A work-item in a group.
/// \param input Input data for the scan operation.
/// \param binary_op functor that implements the binary operation used to
/// perform the scan. \param prefix_callback_op functor invoked by the first
/// work-item in the group that returns the
///        initial value in the resulting scan of the work-items in the group.
/// \returns exclusive scan of the input elements assigned to work-items in the
/// group.
template <typename Item, typename T, class BinaryOperation,
          class GroupPrefixCallbackOperation>
__attribute__((always_inline)) T
exclusive_scan(const Item &item, T input, BinaryOperation binary_op,
               GroupPrefixCallbackOperation &prefix_callback_op) {
  T group_aggregate;

  T output =
      detail::__exclusive_scan_over_group(item.get_group(), input, binary_op);
  if (item.get_local_linear_id() == item.get_local_range().size() - 1) {
    group_aggregate = binary_op(output, input);
  }

  group_aggregate = detail::__group_broadcast(
      item.get_group(), group_aggregate, item.get_local_range().size() - 1);

  T group_prefix = prefix_callback_op(group_aggregate);
  if (item.get_local_linear_id() == 0) {
    output = group_prefix;
  } else {
    output = binary_op(group_prefix, output);
  }

  return output;
}
// DPCT_LABEL_END

namespace detail {

// DPCT_LABEL_BEGIN|radix_rank|dpct::group::detail
// DPCT_DEPENDENCY_BEGIN
// DplExtrasDpcppExtensions|exclusive_scan
// DPCT_DEPENDENCY_END
// DPCT_CODE
typedef uint16_t digit_counter_type;
typedef uint32_t packed_counter_type;

template <int N, int CURRENT_VAL = N, int COUNT = 0> struct log2 {
  enum { VALUE = log2<N, (CURRENT_VAL >> 1), COUNT + 1>::VALUE };
};

template <int N, int COUNT> struct log2<N, 0, COUNT> {
  enum { VALUE = (1 << (COUNT - 1) < N) ? COUNT : COUNT - 1 };
};

inline uint32_t bfe(uint32_t source, uint32_t bit_start,
                    uint32_t num_bits) {
  const uint32_t MASK = (1 << num_bits) - 1;
  return (source >> bit_start) & MASK;
}

template <int RADIX_BITS, bool DESCENDING = false> class radix_rank {
public:
  static size_t get_local_memory_size(size_t group_threads) {
    return group_threads * PADDED_COUNTER_LANES * sizeof(packed_counter_type);
  }

  radix_rank(uint8_t *local_memory) : _local_memory(local_memory) {}

  template <typename Item, int VALUES_PER_THREAD>
  __attribute__((always_inline)) void
  rank_keys(const Item &item, uint32_t (&keys)[VALUES_PER_THREAD],
            int (&ranks)[VALUES_PER_THREAD], int current_bit, int num_bits) {

    digit_counter_type thread_prefixes[VALUES_PER_THREAD];
    digit_counter_type *digit_counters[VALUES_PER_THREAD];
    digit_counter_type *buffer =
        reinterpret_cast<digit_counter_type *>(_local_memory);

    reset_local_memory(item);

    item.barrier(sycl::access::fence_space::local_space);

#pragma unroll
    for (int i = 0; i < VALUES_PER_THREAD; ++i) {
      uint32_t digit = bfe(keys[i], current_bit, num_bits);
      uint32_t sub_counter = digit >> LOG_COUNTER_LANES;
      uint32_t counter_lane = digit & (COUNTER_LANES - 1);

      if (DESCENDING) {
        sub_counter = PACKING_RATIO - 1 - sub_counter;
        counter_lane = COUNTER_LANES - 1 - counter_lane;
      }

      digit_counters[i] =
          &buffer[counter_lane * item.get_local_range().size() * PACKING_RATIO +
                  item.get_local_linear_id() * PACKING_RATIO + sub_counter];
      thread_prefixes[i] = *digit_counters[i];
      *digit_counters[i] = thread_prefixes[i] + 1;
    }

    item.barrier(sycl::access::fence_space::local_space);

    scan_counters(item);

    item.barrier(sycl::access::fence_space::local_space);

    for (int i = 0; i < VALUES_PER_THREAD; ++i) {
      ranks[i] = thread_prefixes[i] + *digit_counters[i];
    }
  }

private:
  template <typename Item>
  __attribute__((always_inline)) void reset_local_memory(const Item &item) {
    packed_counter_type *ptr =
        reinterpret_cast<packed_counter_type *>(_local_memory);

#pragma unroll
    for (int i = 0; i < PADDED_COUNTER_LANES; ++i) {
      ptr[i * item.get_local_range().size() + item.get_local_linear_id()] = 0;
    }
  }

  template <typename Item>
  __attribute__((always_inline)) packed_counter_type upsweep(const Item &item) {
    packed_counter_type sum = 0;
    packed_counter_type *ptr =
        reinterpret_cast<packed_counter_type *>(_local_memory);

#pragma unroll
    for (int i = 0; i < PADDED_COUNTER_LANES; i++) {
      cached_segment[i] =
          ptr[item.get_local_linear_id() * PADDED_COUNTER_LANES + i];
    }

#pragma unroll
    for (int i = 0; i < PADDED_COUNTER_LANES; ++i) {
      sum += cached_segment[i];
    }

    return sum;
  }

  template <typename Item>
  __attribute__((always_inline)) void
  exclusive_downsweep(const Item &item, packed_counter_type raking_partial) {
    packed_counter_type *ptr =
        reinterpret_cast<packed_counter_type *>(_local_memory);
    packed_counter_type sum = raking_partial;

#pragma unroll
    for (int i = 0; i < PADDED_COUNTER_LANES; ++i) {
      packed_counter_type value = cached_segment[i];
      cached_segment[i] = sum;
      sum += value;
    }

#pragma unroll
    for (int i = 0; i < PADDED_COUNTER_LANES; ++i) {
      ptr[item.get_local_linear_id() * PADDED_COUNTER_LANES + i] =
          cached_segment[i];
    }
  }

  struct prefix_callback {
    __attribute__((always_inline)) packed_counter_type
    operator()(packed_counter_type block_aggregate) {
      packed_counter_type block_prefix = 0;

#pragma unroll
      for (int packed = 1; packed < PACKING_RATIO; packed++) {
        block_prefix += block_aggregate
                        << (sizeof(digit_counter_type) * 8 * packed);
      }

      return block_prefix;
    }
  };

  template <typename Item>
  __attribute__((always_inline)) void scan_counters(const Item &item) {
    packed_counter_type raking_partial = upsweep(item);

    prefix_callback callback;
    packed_counter_type exclusive_partial = exclusive_scan(
        item, raking_partial, sycl::ext::oneapi::plus<packed_counter_type>(),
        callback);

    exclusive_downsweep(item, exclusive_partial);
  }

private:
  static constexpr int PACKING_RATIO =
      sizeof(packed_counter_type) / sizeof(digit_counter_type);
  static constexpr int LOG_PACKING_RATIO = log2<PACKING_RATIO>::VALUE;
  static constexpr int LOG_COUNTER_LANES = RADIX_BITS - LOG_PACKING_RATIO;
  static constexpr int COUNTER_LANES = 1 << LOG_COUNTER_LANES;
  static constexpr int PADDED_COUNTER_LANES = COUNTER_LANES + 1;

  packed_counter_type cached_segment[PADDED_COUNTER_LANES];
  uint8_t *_local_memory;
};
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|traits|dpct::group::detail
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
template <typename T, typename U> struct base_traits {

  static __attribute__((always_inline)) U twiddle_in(U key) {
    throw std::runtime_error("Not implemented");
  }
  static __attribute__((always_inline)) U twiddle_out(U key) {
    throw std::runtime_error("Not implemented");
  }
};

template <typename U> struct base_traits<uint32_t, U> {
  static __attribute__((always_inline)) U twiddle_in(U key) { return key; }
  static __attribute__((always_inline)) U twiddle_out(U key) { return key; }
};

template <typename U> struct base_traits<int, U> {
  static constexpr U HIGH_BIT = U(1) << ((sizeof(U) * 8) - 1);
  static __attribute__((always_inline)) U twiddle_in(U key) {
    return key ^ HIGH_BIT;
  }
  static __attribute__((always_inline)) U twiddle_out(U key) {
    return key ^ HIGH_BIT;
  }
};

template <typename U> struct base_traits<float, U> {
  static constexpr U HIGH_BIT = U(1) << ((sizeof(U) * 8) - 1);
  static __attribute__((always_inline)) U twiddle_in(U key) {
    U mask = (key & HIGH_BIT) ? U(-1) : HIGH_BIT;
    return key ^ mask;
  }
  static __attribute__((always_inline)) U twiddle_out(U key) {
    U mask = (key & HIGH_BIT) ? HIGH_BIT : U(-1);
    return key ^ mask;
  }
};

template <typename T> struct traits : base_traits<T, T> {};
template <> struct traits<uint32_t> : base_traits<uint32_t, uint32_t> {};
template <> struct traits<int> : base_traits<int, uint32_t> {};
template <> struct traits<float> : base_traits<float, uint32_t> {};
// DPCT_LABEL_END

} // namespace detail

// DPCT_LABEL_BEGIN|exchange|dpct::group
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
namespace detail {

template <int N> struct power_of_two {
  enum { VALUE = ((N & (N - 1)) == 0) };
};

inline uint32_t shr_add(uint32_t x, uint32_t shift,
                        uint32_t addend) {
  return (x >> shift) + addend;
}

} // namespace detail

/// Implements scatter to blocked exchange pattern used in radix sort algorithm.
///
/// \tparam T type of the data elements exchanges
/// \tparam VALUES_PER_THREAD number of data elements assigned to a thread
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

  /// Rearrange elements from rank order to blocked order
  template <typename Item>
  __attribute__((always_inline)) void
  scatter_to_blocked(Item item, T (&keys)[VALUES_PER_THREAD],
                     int (&ranks)[VALUES_PER_THREAD]) {
    T *buffer = reinterpret_cast<T *>(_local_memory);

#pragma unroll
    for (int i = 0; i < VALUES_PER_THREAD; i++) {
      int offset = ranks[i];
      if (INSERT_PADDING)
        offset = detail::shr_add(offset, LOG_LOCAL_MEMORY_BANKS, offset);
      buffer[offset] = keys[i];
    }

    item.barrier(sycl::access::fence_space::local_space);

#pragma unroll
    for (int i = 0; i < VALUES_PER_THREAD; i++) {
      int offset = (item.get_local_id(0) * VALUES_PER_THREAD) + i;
      if (INSERT_PADDING)
        offset = detail::shr_add(offset, LOG_LOCAL_MEMORY_BANKS, offset);
      keys[i] = buffer[offset];
    }
  }

private:
  static constexpr int LOG_LOCAL_MEMORY_BANKS = 5;
  static constexpr bool INSERT_PADDING =
      (VALUES_PER_THREAD > 4) &&
      (detail::power_of_two<VALUES_PER_THREAD>::VALUE);

  uint8_t *_local_memory;
};
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|radix_sort|dpct::group
// DPCT_DEPENDENCY_BEGIN
// DplExtrasDpcppExtensions|traits
// DPCT_DEPENDENCY_END
// DPCT_CODE
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
        detail::radix_rank<RADIX_BITS>::get_local_memory_size(group_threads);
    size_t exchange_size =
        exchange<T, VALUES_PER_THREAD>::get_local_memory_size(group_threads);
    return sycl::max(ranks_size, exchange_size);
  }

  radix_sort(uint8_t *local_memory) : _local_memory(local_memory) {}

  template <typename Item>
  __attribute__((always_inline)) void
  sort(const Item &item, T (&keys)[VALUES_PER_THREAD], int begin_bit = 0,
       int end_bit = 8 * sizeof(T)) {

    uint32_t(&unsigned_keys)[VALUES_PER_THREAD] =
        reinterpret_cast<uint32_t(&)[VALUES_PER_THREAD]>(keys);

#pragma unroll
    for (int i = 0; i < VALUES_PER_THREAD; ++i) {
      unsigned_keys[i] = detail::traits<T>::twiddle_in(unsigned_keys[i]);
    }

    while (true) {
      int pass_bits = sycl::min(RADIX_BITS, end_bit - begin_bit);

      int ranks[VALUES_PER_THREAD];
      detail::radix_rank<RADIX_BITS, DESCENDING>(_local_memory)
          .template rank_keys(item, unsigned_keys, ranks, begin_bit, pass_bits);
      begin_bit += RADIX_BITS;

      item.barrier(sycl::access::fence_space::local_space);

      exchange<T, VALUES_PER_THREAD>(_local_memory)
          .scatter_to_blocked(item, keys, ranks);

      item.barrier(sycl::access::fence_space::local_space);

      if (begin_bit >= end_bit)
        break;
    }

#pragma unroll
    for (int i = 0; i < VALUES_PER_THREAD; ++i) {
      unsigned_keys[i] = detail::traits<T>::twiddle_out(unsigned_keys[i]);
    }
  }

private:
  static constexpr int RADIX_BITS = 4;

  uint8_t *_local_memory;
};
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|reduce|dpct::group
// DPCT_DEPENDENCY_BEGIN
// DplExtrasDpcppExtensions|group_reduce
// DPCT_DEPENDENCY_END
// DPCT_CODE
/// Perform a reduction of the data elements assigned to all threads in the
/// group.
///
/// \param item A work-item in a group.
/// \param inputs Pointer to the input data for the reduce operation.
/// \param binary_op functor that implements the binary operation used to
/// perform the scan. \returns value of the reduction using binary_op
template <typename Item, typename T, class BinaryOperation,
          int VALUES_PER_THREAD>
__attribute__((always_inline)) T
reduce(Item item, T (&inputs)[VALUES_PER_THREAD], BinaryOperation binary_op) {
  T result = inputs[0];

#pragma unroll
  for (int i = 1; i < VALUES_PER_THREAD; i++) {
    result = binary_op(result, inputs[i]);
  }
  return detail::__reduce_over_group(item.get_group(), result, binary_op);
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|inclusive_scan|dpct::group
// DPCT_DEPENDENCY_BEGIN
// DplExtrasDpcppExtensions|group_broadcast
// DplExtrasDpcppExtensions|group_scan
// DPCT_DEPENDENCY_END
// DPCT_CODE
/// Perform an inclusive scan over the values of inputs from all work-items in
/// the group using the operator binary_op, which must be one of the SYCL 2020
/// group algorithms library function objects.
///
/// \param item A work-item in a group.
/// \param inputs Pointer to the input data for the scan operation.
/// \param outputs Pointer to the location where scan results will be stored.
/// \param binary_op functor that implements the binary operation used to
/// perform the scan. \returns inclusive scan of the input elements assigned to
/// work-items in the group.
template <typename Item, typename T, class BinaryOperation,
          int VALUES_PER_THREAD>
__attribute__((always_inline)) void
inclusive_scan(const Item &item, T (&inputs)[VALUES_PER_THREAD],
               T (&outputs)[VALUES_PER_THREAD], BinaryOperation binary_op) {
  T result = inputs[0];

#pragma unroll
  for (int i = 1; i < VALUES_PER_THREAD; ++i) {
    result = binary_op(result, inputs[i]);
  }

  T exclusive_result =
      detail::__exclusive_scan_over_group(item.get_group(), result, binary_op);

  if (item.get_local_linear_id() == 0) {
    outputs[0] = inputs[0];
  } else {
    outputs[0] = binary_op(inputs[0], exclusive_result);
  }

#pragma unroll
  for (int i = 1; i < VALUES_PER_THREAD; ++i) {
    outputs[i] = binary_op(inputs[i], outputs[i - 1]);
  }
}

/// Perform an inclusive scan over the values of inputs from all work-items in
/// the group using the operator binary_op, which must be one of the SYCL 2020
/// group algorithms library function objects.
///
/// \param item A work-item in a group.
/// \param input Pointer to the input data for the scan operation.
/// \param binary_op functor that implements the binary operation used to
/// perform the scan. \param group_aggregate group-wide aggregate of all inputs
/// in the work-items of the group. \returns inclusive scan of the input
/// elements assigned to work-items in the group.
template <typename Item, typename T, class BinaryOperation>
__attribute__((always_inline)) T inclusive_scan(const Item &item, T input,
                                                BinaryOperation binary_op,
                                                T &group_aggregate) {
  T output =
      detail::__inclusive_scan_over_group(item.get_group(), input, binary_op);
  if (item.get_local_linear_id() == item.get_local_range().size() - 1) {
    group_aggregate = output;
  }

  group_aggregate = detail::__group_broadcast(
      item.get_group(), group_aggregate, item.get_local_range().size() - 1);
  return output;
}

/// Perform an inclusive scan over the values of input from all work-items in
/// the group using the operator binary_op, which must be one of the SYCL 2020
/// group algorithms library function objects.
///
/// \param item A work-item in a group.
/// \param input Input data for the scan operation.
/// \param binary_op functor that implements the binary operation used to
/// perform the scan. \param prefix_callback_op functor invoked by the first
/// work-item in the group that returns the
///        initial value in the resulting scan of the work-items in the group.
/// \returns inclusive scan of the input elements assigned to work-items in the
/// group.
template <typename Item, typename T, class BinaryOperation,
          class GroupPrefixCallbackOperation>
__attribute__((always_inline)) T
inclusive_scan(const Item &item, T input, BinaryOperation binary_op,
               GroupPrefixCallbackOperation &prefix_callback_op) {
  T group_aggregate;

  T output = inclusive_scan(item, input, binary_op, group_aggregate);
  T group_prefix = prefix_callback_op(group_aggregate);

  return binary_op(group_prefix, output);
}
// DPCT_LABEL_END

} // namespace group

namespace device {

namespace detail {

// DPCT_LABEL_BEGIN|joint_reduce|dpct::device::detail
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
template <typename... _Args> constexpr auto __joint_reduce(_Args... __args) {
  return sycl::joint_reduce(__args...);
}
// DPCT_LABEL_END

} // namespace detail

// DPCT_LABEL_BEGIN|segmented_reduce|dpct::device
// DPCT_DEPENDENCY_BEGIN
// DplExtrasDpcppExtensions|joint_reduce
// DPCT_DEPENDENCY_END
// DPCT_CODE
/// Perform a reduce on each of the segments specified within data stored on
/// the device.
///
/// \param queue Command queue used to access device used for reduction
/// \param inputs Pointer to the data elements on the device to be reduced
/// \param outputs Pointer to the storage where the reduced value for each
/// segment will be stored \param segment_count number of segments to be reduced
/// \param begin_offsets Pointer to the set of indices that are the first
/// element in each segment \param end_offsets Pointer to the set of indices
/// that are one past the last element in each segment \param binary_op functor
/// that implements the binary operation used to perform the scan. \param init
/// initial value of the reduction for each segment.
template <int GROUP_SIZE, typename T, typename OffsetT, class BinaryOperation>
void segmented_reduce(sycl::queue queue, T *inputs, T *outputs,
                      size_t segment_count, OffsetT *begin_offsets,
                      OffsetT *end_offsets, BinaryOperation binary_op, T init) {

  sycl::range<1> global_size(segment_count * GROUP_SIZE);
  sycl::range<1> local_size(GROUP_SIZE);

  queue.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>(global_size, local_size), [=](sycl::nd_item<1> item) {
          OffsetT segment_begin = begin_offsets[item.get_group_linear_id()];
          OffsetT segment_end = end_offsets[item.get_group_linear_id()];
          if (segment_begin == segment_end) {
            if (item.get_local_linear_id() == 0) {
              outputs[item.get_group_linear_id()] = init;
            }
            return;
          }

          sycl::multi_ptr<T, sycl::access::address_space::global_space>
              input_ptr = inputs;
          T group_aggregate = detail::__joint_reduce(
              item.get_group(), input_ptr + segment_begin,
              input_ptr + segment_end, init, binary_op);

          if (item.get_local_linear_id() == 0) {
            outputs[item.get_group_linear_id()] = group_aggregate;
          }
        });
  });
}
// DPCT_LABEL_END

} // namespace device
} // namespace dpct

#endif
