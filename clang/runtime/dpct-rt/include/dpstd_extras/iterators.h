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

#ifndef __DPCT_ITERATORS_H__
#define __DPCT_ITERATORS_H__

#include <dpstd/iterator>

#include "functional.h"

namespace dpct {

using std::advance;

using std::distance;

template <typename T>
dpstd::counting_iterator<T> make_counting_iterator(const T &input) {
  return dpstd::counting_iterator<T>(input);
}

template <typename _Tp> class constant_iterator {
public:
  typedef std::false_type is_hetero;
  typedef std::true_type is_passed_directly;
  typedef std::ptrdiff_t difference_type;
  typedef _Tp value_type;
  typedef _Tp *pointer;
  typedef _Tp &reference;
  typedef const _Tp &const_reference;
  typedef std::random_access_iterator_tag iterator_category;

  explicit constant_iterator(_Tp __init)
      : __my_value_(__init), __my_counter_(0) {}

private:
  // used to construct iterator instances with different counter values required
  // by arithmetic operators
  constant_iterator(const _Tp &__value, const difference_type &__offset)
      : __my_value_(__value), __my_counter_(__offset) {}

public:
  // non-const variants of access operators are not provided so unintended
  // writes are caught at compile time.
  const_reference operator*() const { return __my_value_; }
  const_reference operator[](difference_type) const { return __my_value_; }

  difference_type operator-(const constant_iterator &__it) const {
    return __my_counter_ - __it.__my_counter_;
  }

  constant_iterator &operator+=(difference_type __forward) {
    __my_counter_ += __forward;
    return *this;
  }
  constant_iterator &operator-=(difference_type __backward) {
    return *this += -__backward;
  }
  constant_iterator &operator++() { return *this += 1; }
  constant_iterator &operator--() { return *this -= 1; }

  constant_iterator operator++(int) {
    constant_iterator __it(*this);
    ++(*this);
    return __it;
  }
  constant_iterator operator--(int) {
    constant_iterator __it(*this);
    --(*this);
    return __it;
  }

  constant_iterator operator-(difference_type __backward) const {
    return constant_iterator(__my_value_, __my_counter_ - __backward);
  }
  constant_iterator operator+(difference_type __forward) const {
    return constant_iterator(__my_value_, __my_counter_ + __forward);
  }
  friend constant_iterator operator+(difference_type __forward,
                                     const constant_iterator __it) {
    return __it + __forward;
  }

  bool operator==(const constant_iterator &__it) const {
    return __my_value_ == __it.__my_value_ &&
           this->__my_counter_ == __it.__my_counter_;
  }
  bool operator!=(const constant_iterator &__it) const {
    return !(*this == __it);
  }
  bool operator<(const constant_iterator &__it) const {
    return *this - __it < 0;
  }
  bool operator>(const constant_iterator &__it) const { return __it < *this; }
  bool operator<=(const constant_iterator &__it) const {
    return !(*this > __it);
  }
  bool operator>=(const constant_iterator &__it) const {
    return !(*this < __it);
  }

private:
  const _Tp __my_value_;
  uint64_t __my_counter_;
};

template <typename _Tp>
constant_iterator<_Tp> make_constant_iterator(_Tp __value) {
  return constant_iterator<_Tp>(__value);
}

template <typename _Ip> class discard_iterator {
public:
  typedef std::ptrdiff_t difference_type;
  typedef _Ip value_type;
  typedef _Ip *pointer;
  typedef _Ip &reference;
  typedef const _Ip &const_reference;
  typedef std::random_access_iterator_tag iterator_category;

  // const variants of access operators are not provided so unintended reads are
  // caught at compile time.
  reference operator*() { return __my_placeholder_; }
  reference operator[](difference_type) { return __my_placeholder_; }

  constexpr bool operator==(const discard_iterator &) const { return true; }
  constexpr bool operator!=(const discard_iterator &) const { return false; }

  difference_type operator-(const discard_iterator &__it) const {
    return difference_type{};
  }

  discard_iterator &operator++() { return *this; }
  discard_iterator &operator--() { return *this; }
  discard_iterator operator++(int) { return *this; }
  discard_iterator operator--(int) { return *this; }
  discard_iterator &operator+=(difference_type) { return *this; }
  discard_iterator &operator-=(difference_type) { return *this; }

  discard_iterator operator+(difference_type) const { return *this; }
  discard_iterator operator-(difference_type) const { return *this; }

private:
  _Ip __my_placeholder_;
};

template <typename Iter1, typename Iter2> class permutation_iterator {
public:
  typedef typename std::iterator_traits<Iter1>::difference_type difference_type;
  typedef typename std::iterator_traits<Iter1>::value_type value_type;
  typedef typename std::iterator_traits<Iter1>::pointer pointer;
  typedef typename std::iterator_traits<Iter1>::reference reference;
  typedef std::random_access_iterator_tag iterator_category;

  permutation_iterator(const Iter1 &input1, const Iter2 &input2)
      : my_source(input1), my_map(input2) {}

  reference operator*() const {
    return *(my_source + difference_type(*my_map));
  }

  reference operator[](difference_type i) const { return *(*this + i); }

  permutation_iterator &operator++() {
    ++my_map;
    return *this;
  }

  permutation_iterator operator++(int) {
    permutation_iterator it(*this);
    ++(*this);
    return it;
  }

  permutation_iterator &operator--() {
    --my_map;
    return *this;
  }

  permutation_iterator operator--(int) {
    permutation_iterator it(*this);
    --(*this);
    return it;
  }

  permutation_iterator operator+(difference_type forward) const {
    return permutation_iterator(my_source, my_map + forward);
  }

  permutation_iterator operator-(difference_type backward) {
    return permutation_iterator(my_source, my_map - backward);
  }

  permutation_iterator &operator+=(difference_type forward) {
    my_map += forward;
    return *this;
  }

  permutation_iterator &operator-=(difference_type forward) {
    my_map -= forward;
    return *this;
  }

  difference_type operator-(const permutation_iterator &it) const {
    return my_map - it.my_map;
  }

  bool operator==(const permutation_iterator &it) const {
    return *this - it == 0;
  }
  bool operator!=(const permutation_iterator &it) const {
    return !(*this == it);
  }
  bool operator<(const permutation_iterator &it) const {
    return *this - it < 0;
  }
  bool operator>(const permutation_iterator &it) const { return it < *this; }
  bool operator<=(const permutation_iterator &it) const {
    return !(*this > it);
  }
  bool operator>=(const permutation_iterator &it) const {
    return !(*this < it);
  }

private:
  Iter1 my_source;
  Iter2 my_map;
};

template <typename Iter1, typename Iter2>
permutation_iterator<Iter1, Iter2> make_permutation_iterator(Iter1 source,
                                                             Iter2 map) {
  return permutation_iterator<Iter1, Iter2>(source, map);
}

template <typename Iterator> struct iterator_difference {
  using type = typename std::iterator_traits<Iterator>::difference_type;
};
} // end namespace dpct

#endif
