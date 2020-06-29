/******************************************************************************
*
* Copyright 2019 - 2020 Intel Corporation.
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

#ifndef __DPCT_ALGORITHM_H__
#define __DPCT_ALGORITHM_H__

#include <dpstd/execution>
#include <dpstd/algorithm>
#include <dpstd/numeric>

#include "functional.h"
#include "iterators.h"

namespace dpct {

template <typename Policy, typename InputIter1, typename InputIter2,
          typename Predicate, typename T>
void replace_if(Policy &&policy, InputIter1 first, InputIter1 last,
                InputIter2 stencil, Predicate p, const T &new_value) {
  std::transform(std::forward<Policy>(policy), first, last, stencil, first,
                 internal::replace_if_fun<T, Predicate>(p, new_value));
}

template <typename Policy, typename InputIter1, typename InputIter2,
          typename OutputIter, typename Predicate, typename T>
OutputIter replace_copy_if(Policy &&policy, InputIter1 first, InputIter1 last,
                           InputIter2 stencil, OutputIter result, Predicate p,
                           const T &new_value) {
  return std::transform(std::forward<Policy>(policy), first, last, stencil,
                        result,
                        internal::replace_if_fun<T, Predicate>(p, new_value));
}

template <typename Policy, typename ForwardIt, typename InputIt,
          typename Predicate>
ForwardIt remove_if(Policy &&policy, ForwardIt first, ForwardIt last,
                    InputIt stencil, Predicate p) {
  using dpstd::make_zip_iterator;
  using policy_type = typename std::decay<Policy>::type;
  using internal::__buffer;
  using ValueType = typename std::iterator_traits<ForwardIt>::value_type;

  __buffer<ValueType> _tmp(std::distance(first, last));

  typename internal::rebind_policy<policy_type, class RemoveIf1>::type policy1(
      policy);
  auto end = std::copy_if(
      policy1, make_zip_iterator(first, stencil),
      make_zip_iterator(last, stencil + std::distance(first, last)),
      make_zip_iterator(_tmp.get(), discard_iterator()),
      internal::negate_predicate_key_fun<Predicate>(p));
  typename internal::rebind_policy<policy_type, class RemoveIf2>::type policy2(
      policy);
  return std::copy(policy2, _tmp.get(), std::get<0>(end.base()), first);
}

template <typename Policy, typename InputIt1, typename InputIt2,
          typename OutputIt, typename Predicate>
OutputIt remove_copy_if(Policy &&policy, InputIt1 first, InputIt1 last,
                        InputIt2 stencil, OutputIt result, Predicate p) {
  using dpstd::make_zip_iterator;
  auto ret_val = std::remove_copy_if(
      std::forward<Policy>(policy), make_zip_iterator(first, stencil),
      make_zip_iterator(last, stencil + std::distance(first, last)),
      make_zip_iterator(result, discard_iterator()),
      internal::predicate_key_fun<Predicate>(p));
  return std::get<0>(ret_val.base());
}

template <class Policy, class ForwardIter1, class ForwardIter2,
          class BinaryPredicate>
std::pair<ForwardIter1, ForwardIter2>
unique_by_key(Policy &&policy, ForwardIter1 keys_first, ForwardIter1 keys_last,
              ForwardIter2 values_first, BinaryPredicate binary_pred) {
  auto ret_val = std::unique(
      std::forward<Policy>(policy),
      dpstd::make_zip_iterator(keys_first, values_first),
      dpstd::make_zip_iterator(
          keys_last, values_first + std::distance(keys_first, keys_last)),
      internal::compare_key_fun<BinaryPredicate>(binary_pred));
  auto n1 = std::distance(dpstd::make_zip_iterator(keys_first, values_first),
                          ret_val);
  return std::make_pair(keys_first + n1, values_first + n1);
}

template <class Policy, class ForwardIter1, class ForwardIter2>
std::pair<ForwardIter1, ForwardIter2>
unique_by_key(Policy &&policy, ForwardIter1 keys_first, ForwardIter1 keys_last,
              ForwardIter2 values_first) {
  using T = typename std::iterator_traits<ForwardIter1>::value_type;
  return unique_by_key(std::forward<Policy>(policy), keys_first, keys_last,
                       values_first, std::equal_to<T>());
}

template <class Policy, class ForwardIter1, class ForwardIter2,
          class OutputIter1, class OutputIter2, class BinaryPredicate>
std::pair<ForwardIter1, ForwardIter2>
unique_by_key_copy(Policy &&policy, ForwardIter1 keys_first,
                   ForwardIter1 keys_last, ForwardIter2 values_first,
                   OutputIter1 keys_result, OutputIter2 values_result,
                   BinaryPredicate binary_pred) {
  auto ret_val = std::unique_copy(
      std::forward<Policy>(policy),
      dpstd::make_zip_iterator(keys_first, values_first),
      dpstd::make_zip_iterator(
          keys_last, values_first + std::distance(keys_first, keys_last)),
      dpstd::make_zip_iterator(keys_result, values_result),
      internal::unique_by_key_fun<BinaryPredicate>(binary_pred));
  auto n1 = std::distance(dpstd::make_zip_iterator(keys_result, values_result),
                          ret_val);
  return std::make_pair(keys_result + n1, values_result + n1);
}

template <class Policy, class ForwardIter1, class ForwardIter2,
          class OutputIter1, class OutputIter2>
std::pair<ForwardIter1, ForwardIter2>
unique_by_key_copy(Policy &&policy, ForwardIter1 keys_first,
                   ForwardIter1 keys_last, ForwardIter2 values_first,
                   OutputIter1 keys_result, OutputIter2 values_result) {
  using T = typename std::iterator_traits<ForwardIter1>::value_type;
  auto comp = std::equal_to<T>();
  return unique_by_key_copy(std::forward<Policy>(policy), keys_first, keys_last,
                            values_first, keys_result, values_result, comp);
}

template <typename Policy, typename InputIt, typename Predicate>
InputIt partition_point(Policy &&policy, InputIt first, InputIt last,
                        Predicate p) {
  if (std::is_partitioned(std::forward<Policy>(policy), first, last, p))
    return std::find_if_not(std::forward<Policy>(policy), first, last, p);
  else
    return first;
}

template <typename Policy, typename InputIt1, typename InputIt2,
          typename OutputIt, typename Predicate>
OutputIt copy_if(Policy &&policy, InputIt1 first, InputIt1 last,
                 InputIt2 stencil, OutputIt result, Predicate pred) {
  auto ret_val = std::copy_if(
      std::forward<Policy>(policy), dpstd::make_zip_iterator(first, stencil),
      dpstd::make_zip_iterator(last, stencil + std::distance(first, last)),
      dpstd::make_zip_iterator(result, discard_iterator()),
      internal::predicate_key_fun<Predicate>(pred));
  return std::get<0>(ret_val.base());
}

template <class Policy, class InputIt, class OutputIt, class UnaryOperation>
OutputIt transform(Policy &&policy, InputIt first, InputIt last,
                   OutputIt result, UnaryOperation unary_op) {
  return std::transform(std::forward<Policy>(policy), first, last, result,
                        unary_op);
}

template <class Policy, class InputIt1, class InputIt2, class OutputIt,
          class UnaryOperation>
OutputIt transform(Policy &&policy, InputIt1 first1, InputIt1 last1,
                   InputIt2 first2, OutputIt result, UnaryOperation unary_op) {
  return std::transform(std::forward<Policy>(policy), first1, last1, first2,
                        result, unary_op);
}

template <class Policy, class ForwardIt1, class ForwardIt2,
          class UnaryOperation, class Predicate>
ForwardIt2 transform_if(Policy &&policy, ForwardIt1 first, ForwardIt1 last,
                        ForwardIt2 result, UnaryOperation unary_op,
                        Predicate pred) {
  using T = typename std::iterator_traits<ForwardIt1>::value_type;
  return std::transform(
      std::forward<Policy>(policy), first, last, result,
      internal::transform_if_fun<T, Predicate, UnaryOperation>(pred, unary_op));
}

template <class Policy, class ForwardIt1, class ForwardIt2, class ForwardIt3,
          class UnaryOperation, class Predicate>
ForwardIt3 transform_if(Policy &&policy, ForwardIt1 first, ForwardIt1 last,
                        ForwardIt2 stencil, ForwardIt3 result,
                        UnaryOperation unary_op, Predicate pred) {
  using Ref1 = typename std::iterator_traits<ForwardIt1>::reference;
  using Ref2 = typename std::iterator_traits<ForwardIt2>::reference;
  return std::transform(
      std::forward<Policy>(policy), first, last, stencil, result,
      [pred, unary_op](Ref1 a, Ref2 s) { return pred(s) ? unary_op(a) : a; });
}

template <class Policy, class ForwardIt1, class ForwardIt2, class ForwardIt3,
          class ForwardIt4, class BinaryOperation, class Predicate>
ForwardIt4 transform_if(Policy &&policy, ForwardIt1 first1, ForwardIt1 last1,
                        ForwardIt2 first2, ForwardIt3 stencil,
                        ForwardIt4 result, BinaryOperation binary_op,
                        Predicate pred) {
  const auto n = std::distance(first1, last1);
  using ZipIterator = typename dpstd::zip_iterator<ForwardIt1, ForwardIt2,
                                                   ForwardIt3, ForwardIt4>;
  using T = typename std::iterator_traits<ZipIterator>::value_type;
  std::for_each(
      std::forward<Policy>(policy),
      dpstd::make_zip_iterator(first1, first2, stencil, result),
      dpstd::make_zip_iterator(last1, first2 + n, stencil + n, result + n),
      internal::transform_if_zip_stencil_fun<T, Predicate, BinaryOperation>(
          pred, binary_op));
  return result + n;
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
              internal::__less());
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
                     values_first, internal::__less());
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

template <class Policy, class InputIter1, class InputIter2, class InputIter3,
          class OutputIter1, class OutputIter2>
std::pair<OutputIter1, OutputIter2>
set_intersection_by_key(Policy &&policy, InputIter1 keys_first1,
                        InputIter1 keys_last1, InputIter2 keys_first2,
                        InputIter2 keys_last2, InputIter3 values_first1,
                        OutputIter1 keys_result, OutputIter2 values_result) {
  auto ret_val = std::set_intersection(
      std::forward<Policy>(policy),
      dpstd::make_zip_iterator(keys_first1, values_first1),
      dpstd::make_zip_iterator(
          keys_last1, values_first1 + std::distance(keys_first1, keys_last1)),
      dpstd::make_zip_iterator(keys_first2, discard_iterator()),
      dpstd::make_zip_iterator(keys_last2, discard_iterator()),
      dpstd::make_zip_iterator(keys_result, values_result),
      internal::compare_key_fun<>());
  auto n1 = std::distance(dpstd::make_zip_iterator(keys_result, values_result),
                          ret_val);
  return std::make_pair(keys_result + n1, values_result + n1);
}

template <class Policy, class InputIter1, class InputIter2, class InputIter3,
          class OutputIter1, class OutputIter2, class Compare>
std::pair<OutputIter1, OutputIter2> set_intersection_by_key(
    Policy &&policy, InputIter1 keys_first1, InputIter1 keys_last1,
    InputIter2 keys_first2, InputIter2 keys_last2, InputIter3 values_first1,
    OutputIter1 keys_result, OutputIter2 values_result, Compare comp) {
  auto ret_val = std::set_intersection(
      std::forward<Policy>(policy),
      dpstd::make_zip_iterator(keys_first1, values_first1),
      dpstd::make_zip_iterator(
          keys_last1, values_first1 + std::distance(keys_first1, keys_last1)),
      dpstd::make_zip_iterator(keys_first2, discard_iterator()),
      dpstd::make_zip_iterator(keys_last2, discard_iterator()),
      dpstd::make_zip_iterator(keys_result, values_result),
      internal::compare_key_fun<Compare>(comp));
  auto n1 = std::distance(dpstd::make_zip_iterator(keys_result, values_result),
                          ret_val);
  return std::make_pair(keys_result + n1, values_result + n1);
}

template <class Policy, class InputIter1, class InputIter2, class InputIter3,
          class InputIter4, class OutputIter1, class OutputIter2>
std::pair<OutputIter1, OutputIter2> set_symmetric_difference_by_key(
    Policy &&policy, InputIter1 keys_first1, InputIter1 keys_last1,
    InputIter2 keys_first2, InputIter2 keys_last2, InputIter3 values_first1,
    InputIter4 values_first2, OutputIter1 keys_result,
    OutputIter2 values_result) {
  auto ret_val = std::set_symmetric_difference(
      std::forward<Policy>(policy),
      dpstd::make_zip_iterator(keys_first1, values_first1),
      dpstd::make_zip_iterator(
          keys_last1, values_first1 + std::distance(keys_first1, keys_last1)),
      dpstd::make_zip_iterator(keys_first2, values_first2),
      dpstd::make_zip_iterator(
          keys_last2, values_first2 + std::distance(keys_first2, keys_last2)),
      dpstd::make_zip_iterator(keys_result, values_result),
      internal::compare_key_fun<>());
  auto n1 = std::distance(dpstd::make_zip_iterator(keys_result, values_result),
                          ret_val);
  return std::make_pair(keys_result + n1, values_result + n1);
}

template <class Policy, class InputIter1, class InputIter2, class InputIter3,
          class InputIter4, class OutputIter1, class OutputIter2, class Compare>
std::pair<OutputIter1, OutputIter2> set_symmetric_difference_by_key(
    Policy &&policy, InputIter1 keys_first1, InputIter1 keys_last1,
    InputIter2 keys_first2, InputIter2 keys_last2, InputIter3 values_first1,
    InputIter4 values_first2, OutputIter1 keys_result,
    OutputIter2 values_result, Compare comp) {
  auto ret_val = std::set_symmetric_difference(
      std::forward<Policy>(policy),
      dpstd::make_zip_iterator(keys_first1, values_first1),
      dpstd::make_zip_iterator(
          keys_last1, values_first1 + std::distance(keys_first1, keys_last1)),
      dpstd::make_zip_iterator(keys_first2, values_first2),
      dpstd::make_zip_iterator(
          keys_last2, values_first2 + std::distance(keys_first2, keys_last2)),
      dpstd::make_zip_iterator(keys_result, values_result),
      internal::compare_key_fun<Compare>(comp));
  auto n1 = std::distance(dpstd::make_zip_iterator(keys_result, values_result),
                          ret_val);
  return std::make_pair(keys_result + n1, values_result + n1);
}

template <class Policy, class InputIter1, class InputIter2, class InputIter3,
          class InputIter4, class OutputIter1, class OutputIter2>
std::pair<OutputIter1, OutputIter2>
set_difference_by_key(Policy &&policy, InputIter1 keys_first1,
                      InputIter1 keys_last1, InputIter2 keys_first2,
                      InputIter2 keys_last2, InputIter3 values_first1,
                      InputIter4 values_first2, OutputIter1 keys_result,
                      OutputIter2 values_result) {
  auto ret_val = std::set_difference(
      std::forward<Policy>(policy),
      dpstd::make_zip_iterator(keys_first1, values_first1),
      dpstd::make_zip_iterator(
          keys_last1, values_first1 + std::distance(keys_first1, keys_last1)),
      dpstd::make_zip_iterator(keys_first2, values_first2),
      dpstd::make_zip_iterator(
          keys_last2, values_first2 + std::distance(keys_first2, keys_last2)),
      dpstd::make_zip_iterator(keys_result, values_result),
      internal::compare_key_fun<>());
  auto n1 = std::distance(dpstd::make_zip_iterator(keys_result, values_result),
                          ret_val);
  return std::make_pair(keys_result + n1, values_result + n1);
}

template <class Policy, class InputIter1, class InputIter2, class InputIter3,
          class InputIter4, class OutputIter1, class OutputIter2, class Compare>
std::pair<OutputIter1, OutputIter2>
set_difference_by_key(Policy &&policy, InputIter1 keys_first1,
                      InputIter1 keys_last1, InputIter2 keys_first2,
                      InputIter2 keys_last2, InputIter3 values_first1,
                      InputIter4 values_first2, OutputIter1 keys_result,
                      OutputIter2 values_result, Compare comp) {
  auto ret_val = std::set_difference(
      std::forward<Policy>(policy),
      dpstd::make_zip_iterator(keys_first1, values_first1),
      dpstd::make_zip_iterator(
          keys_last1, values_first1 + std::distance(keys_first1, keys_last1)),
      dpstd::make_zip_iterator(keys_first2, values_first2),
      dpstd::make_zip_iterator(
          keys_last2, values_first2 + std::distance(keys_first2, keys_last2)),
      dpstd::make_zip_iterator(keys_result, values_result),
      internal::compare_key_fun<Compare>(comp));
  auto n1 = std::distance(dpstd::make_zip_iterator(keys_result, values_result),
                          ret_val);
  return std::make_pair(keys_result + n1, values_result + n1);
}

template <class Policy, class InputIter1, class InputIter2, class InputIter3,
          class InputIter4, class OutputIter1, class OutputIter2>
internal::enable_if_execution_policy<Policy,
                                     std::pair<OutputIter1, OutputIter2>>
set_union_by_key(Policy &&policy, InputIter1 keys_first1, InputIter1 keys_last1,
                 InputIter2 keys_first2, InputIter2 keys_last2,
                 InputIter3 values_first1, InputIter4 values_first2,
                 OutputIter1 keys_result, OutputIter2 values_result) {
  auto ret_val = std::set_union(
      std::forward<Policy>(policy),
      dpstd::make_zip_iterator(keys_first1, values_first1),
      dpstd::make_zip_iterator(
          keys_last1, values_first1 + std::distance(keys_first1, keys_last1)),
      dpstd::make_zip_iterator(keys_first2, values_first2),
      dpstd::make_zip_iterator(
          keys_last2, values_first2 + std::distance(keys_first2, keys_last2)),
      dpstd::make_zip_iterator(keys_result, values_result),
      internal::compare_key_fun<>());
  auto n1 = std::distance(dpstd::make_zip_iterator(keys_result, values_result),
                          ret_val);
  return std::make_pair(keys_result + n1, values_result + n1);
}

template <class Policy, class InputIter1, class InputIter2, class InputIter3,
          class InputIter4, class OutputIter1, class OutputIter2, class Compare>
internal::enable_if_execution_policy<Policy,
                                     std::pair<OutputIter1, OutputIter2>>
set_union_by_key(Policy &&policy, InputIter1 keys_first1, InputIter1 keys_last1,
                 InputIter2 keys_first2, InputIter2 keys_last2,
                 InputIter3 values_first1, InputIter4 values_first2,
                 OutputIter1 keys_result, OutputIter2 values_result,
                 Compare comp) {
  auto ret_val = std::set_union(
      std::forward<Policy>(policy),
      dpstd::make_zip_iterator(keys_first1, values_first1),
      dpstd::make_zip_iterator(
          keys_last1, values_first1 + std::distance(keys_first1, keys_last1)),
      dpstd::make_zip_iterator(keys_first2, values_first2),
      dpstd::make_zip_iterator(
          keys_last2, values_first2 + std::distance(keys_first2, keys_last2)),
      dpstd::make_zip_iterator(keys_result, values_result),
      internal::compare_key_fun<Compare>(comp));
  auto n1 = std::distance(dpstd::make_zip_iterator(keys_result, values_result),
                          ret_val);
  return std::make_pair(keys_result + n1, values_result + n1);
}

template <typename Policy, typename InputIt1, typename InputIt2,
          typename OutputIt1, typename OutputIt2, typename Predicate>
internal::enable_if_execution_policy<Policy, std::pair<OutputIt1, OutputIt2>>
stable_partition_copy(Policy &&policy, InputIt1 first, InputIt1 last,
                      InputIt2 stencil, OutputIt1 out_true, OutputIt2 out_false,
                      Predicate p) {
  auto ret_val = std::partition_copy(
      std::forward<Policy>(policy), dpstd::make_zip_iterator(first, stencil),
      dpstd::make_zip_iterator(last, stencil + std::distance(first, last)),
      dpstd::make_zip_iterator(out_true, discard_iterator()),
      dpstd::make_zip_iterator(out_false, discard_iterator()),
      internal::predicate_key_fun<Predicate>(p));
  return std::make_pair(std::get<0>(ret_val.first.base()),
                        std::get<0>(ret_val.second.base()));
}

template <typename Policy, typename InputIt1, typename OutputIt1,
          typename OutputIt2, typename Predicate>
internal::enable_if_execution_policy<Policy, std::pair<OutputIt1, OutputIt2>>
stable_partition_copy(Policy &&policy, InputIt1 first, InputIt1 last,
                      OutputIt1 out_true, OutputIt2 out_false, Predicate p) {
  return std::partition_copy(std::forward<Policy>(policy), first, last,
                             out_true, out_false, p);
}

template <typename Policy, typename InputIt1, typename InputIt2,
          typename OutputIt1, typename OutputIt2, typename Predicate>
internal::enable_if_execution_policy<Policy, std::pair<OutputIt1, OutputIt2>>
partition_copy(Policy &&policy, InputIt1 first, InputIt1 last, InputIt2 stencil,
               OutputIt1 out_true, OutputIt2 out_false, Predicate p) {
  return stable_partition_copy(std::forward<Policy>(policy), first, last,
                               stencil, out_true, out_false, p);
}

template <typename Policy, typename ForwardIt, typename InputIt,
          typename Predicate>
internal::enable_if_execution_policy<Policy, ForwardIt>
stable_partition(Policy &&policy, ForwardIt first, ForwardIt last,
                 InputIt stencil, Predicate p) {
  typedef typename std::decay<Policy>::type policy_type;
  internal::__buffer<typename std::iterator_traits<ForwardIt>::value_type> _tmp(
      std::distance(first, last));

  std::copy(std::forward<Policy>(policy), stencil,
            stencil + std::distance(first, last), _tmp.get());

  auto ret_val = std::stable_partition(
      std::forward<Policy>(policy), dpstd::make_zip_iterator(first, _tmp.get()),
      dpstd::make_zip_iterator(last, _tmp.get() + std::distance(first, last)),
      internal::predicate_key_fun<Predicate>(p));
  return std::get<0>(ret_val.base());
}

template <typename Policy, typename ForwardIt, typename InputIt,
          typename Predicate>
internal::enable_if_execution_policy<Policy, ForwardIt>
partition(Policy &&policy, ForwardIt first, ForwardIt last, InputIt stencil,
          Predicate p) {
  return stable_partition(std::forward<Policy>(policy), first, last, stencil,
                          p);
}

} // end namespace dpct

#endif
