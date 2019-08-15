/******************************************************************************
* INTEL CONFIDENTIAL
*
* Copyright 2019 Intel Corporation.
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

#ifndef __DPCT_ALGORITHM_H
#define __DPCT_ALGORITHM_H

#include "functional.h"

namespace dpct {

template <class Policy, class InputIt1, class InputIt2, class OutputIt1,
          class OutputIt2, class UnaryPredicate>
std::pair<OutputIt1, OutputIt2>
stable_partition_copy(Policy &&policy, InputIt1 first, InputIt1 last,
                      InputIt2 stencil, OutputIt1 out_true, OutputIt2 out_false,
                      UnaryPredicate p) {
  using OutRef1 = typename std::iterator_traits<OutputIt1>::reference;
  using OutRef2 = typename std::iterator_traits<OutputIt2>::reference;
  auto ret_val = std::partition_copy(
      std::forward<Policy>(policy), dpstd::make_zip_iterator(first, stencil),
      dpstd::make_zip_iterator(last, stencil + std::distance(first, last)),
      dpstd::make_transform_iterator(out_true,
                                     internal::discard_fun<OutRef1>()),
      dpstd::make_transform_iterator(out_false,
                                     internal::discard_fun<OutRef2>()),
      internal::predicate_key_fun<UnaryPredicate>(p));
  return std::make_pair(ret_val.first.base(), ret_val.second.base());
}

template <class Policy, class InputIt1, class OutputIt1, class OutputIt2,
          class UnaryPredicate>
std::pair<OutputIt1, OutputIt2>
stable_partition_copy(Policy &&policy, InputIt1 first, InputIt1 last,
                      OutputIt1 out_true, OutputIt2 out_false,
                      UnaryPredicate p) {
  return std::partition_copy(std::forward<Policy>(policy), first, last,
                             out_true, out_false, p);
}

template <typename Policy, typename InputIt1, typename InputIt2,
          typename OutputIt, typename Predicate>
OutputIt copy_if(Policy &&policy, InputIt1 first, InputIt1 last,
                 InputIt2 stencil, OutputIt result, Predicate pred) {
  using Ref3 = typename std::iterator_traits<OutputIt>::reference;
  auto ret_val = std::copy_if(
      std::forward<Policy>(policy), dpstd::make_zip_iterator(first, stencil),
      dpstd::make_zip_iterator(last, stencil + std::distance(first, last)),
      dpstd::make_transform_iterator(result, internal::discard_fun<Ref3>()),
      internal::predicate_key_fun<Predicate>(pred));
  return ret_val.base();
}

template <typename Policy, typename InputIt1, typename InputIt2,
          typename InputIt3, typename InputIt4, typename OutputIt1,
          typename OutputIt2>
std::pair<OutputIt1, OutputIt2>
merge_by_key(Policy &&policy, InputIt1 keys_first1, InputIt1 keys_last1,
             InputIt2 keys_first2, InputIt2 keys_last2, InputIt3 values_first1,
             InputIt4 values_first2, OutputIt1 keys_result,
             OutputIt2 values_result) {
  auto n1 = std::distance(keys_first1, keys_last1);
  auto n2 = std::distance(keys_first2, keys_last2);
  std::merge(std::forward<Policy>(policy),
             dpstd::make_zip_iterator(keys_first1, values_first1),
             dpstd::make_zip_iterator(keys_last1, values_first1 + n1),
             dpstd::make_zip_iterator(keys_first2, values_first2),
             dpstd::make_zip_iterator(keys_last2, values_first2 + n2),
             dpstd::make_zip_iterator(keys_result, values_result),
             internal::compare_key_fun<>());
  return std::make_pair(keys_result + n1 + n2, values_result + n1 + n2);
}

template <typename Policy, typename InputIt1, typename InputIt2,
          typename InputIt3, typename InputIt4, typename OutputIt1,
          typename OutputIt2, typename Compare>
std::pair<OutputIt1, OutputIt2>
merge_by_key(Policy &&policy, InputIt1 keys_first1, InputIt1 keys_last1,
             InputIt2 keys_first2, InputIt2 keys_last2, InputIt3 values_first1,
             InputIt4 values_first2, OutputIt1 keys_result,
             OutputIt2 values_result, Compare comp) {
  auto n1 = std::distance(keys_first1, keys_last1);
  auto n2 = std::distance(keys_first2, keys_last2);
  std::merge(std::forward<Policy>(policy),
             dpstd::make_zip_iterator(keys_first1, values_first1),
             dpstd::make_zip_iterator(keys_last1, values_first1 + n1),
             dpstd::make_zip_iterator(keys_first2, values_first2),
             dpstd::make_zip_iterator(keys_last2, values_first2 + n2),
             dpstd::make_zip_iterator(keys_result, values_result),
             internal::compare_key_fun<Compare>(comp));
  return std::make_pair(keys_result + n1 + n2, values_result + n1 + n2);
}

template <typename Policy, typename ForwardIt1, typename ForwardIt2,
          typename UnaryPredicate>
ForwardIt1 partition(Policy &&policy, ForwardIt1 first, ForwardIt1 last,
                     ForwardIt2 stencil, UnaryPredicate p) {
  return stable_partition(std::forward<Policy>(policy), first, last, stencil,
                          p);
}

template <class Policy, class InputIter, class T>
void sequence(Policy &&policy, InputIter first, InputIter last, T init,
              T step) {
  using DiffSize = typename std::iterator_traits<InputIter>::difference_type;
  std::transform(std::forward<Policy>(policy),
                 dpstd::counting_iterator<DiffSize>(0),
                 dpstd::counting_iterator<DiffSize>(std::distance(first, last)),
                 first, internal::sequence_fun<T>(init, step));
}

template <class Policy, class InputIter, class T>
void sequence(Policy &&policy, InputIter first, InputIter last, T init) {
  using DiffSize = typename std::iterator_traits<InputIter>::difference_type;
  sequence(std::forward<Policy>(policy), first, last, init, T(1));
}

template <class Policy, class InputIter>
void sequence(Policy &&policy, InputIter first, InputIter last) {
  using DiffSize = typename std::iterator_traits<InputIter>::difference_type;
  sequence(std::forward<Policy>(policy), first, last, DiffSize(0), DiffSize(1));
}

template <class Policy, class InputIter1, class InputIter2, class Compare>
void sort_by_key(Policy &&policy, InputIter1 keys_first, InputIter1 keys_last,
                 InputIter2 values_first, Compare comp) {
  auto first = dpstd::make_zip_iterator(keys_first, values_first);
  auto last = first + std::distance(keys_first, keys_last);
  std::sort(std::forward<Policy>(policy), first, last,
            internal::compare_key_fun<Compare>(comp));
}

template <class Policy, class InputIter1, class InputIter2>
void sort_by_key(Policy &&policy, InputIter1 keys_first, InputIter1 keys_last,
                 InputIter2 values_first) {
  sort_by_key(std::forward<Policy>(policy), keys_first, keys_last, values_first,
              dpstd::__internal::__pstl_less());
}

template <class Policy, class InputIter1, class InputIter2, class Compare>
void stable_sort_by_key(Policy &&policy, InputIter1 keys_first,
                        InputIter1 keys_last, InputIter2 values_first,
                        Compare comp) {
  std::stable_sort(
      std::forward<Policy>(policy),
      dpstd::make_zip_iterator(keys_first, values_first),
      dpstd::make_zip_iterator(
          keys_last, values_first + std::distance(keys_first, keys_last)),
      internal::compare_key_fun<Compare>(comp));
}

template <class Policy, class RandomAccessIter1, class RandomAccessIter2>
void stable_sort_by_key(Policy &&policy, RandomAccessIter1 keys_first,
                        RandomAccessIter1 keys_last,
                        RandomAccessIter2 values_first) {
  stable_sort_by_key(std::forward<Policy>(policy), keys_first, keys_last,
                     values_first, dpstd::__internal::__pstl_less());
}

template <class Policy, class InputIter, class Operator>
void tabulate(Policy &&policy, InputIter first, InputIter last,
              Operator unary_op) {
  using DiffSize = typename std::iterator_traits<InputIter>::difference_type;
  std::transform(std::forward<Policy>(policy),
                 dpstd::counting_iterator<DiffSize>(0),
                 dpstd::counting_iterator<DiffSize>(std::distance(first, last)),
                 first, unary_op);
}

} // end namespace dpct

#endif
