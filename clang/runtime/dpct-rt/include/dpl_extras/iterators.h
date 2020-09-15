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
  typedef const _Tp &reference;
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

class discard_iterator {
public:
  typedef std::ptrdiff_t difference_type;
  typedef decltype(std::ignore) value_type;
  typedef void *pointer;
  typedef value_type reference;
  typedef std::random_access_iterator_tag iterator_category;
  using is_passed_directly = std::true_type;

  discard_iterator() : __my_position_() {}
  explicit discard_iterator(difference_type __init) : __my_position_(__init) {}

  auto operator*() const -> decltype(std::ignore) { return std::ignore; }
  auto operator[](difference_type) const -> decltype(std::ignore) {
    return std::ignore;
  }

  constexpr bool operator==(const discard_iterator &__it) const {
    return __my_position_ - __it.__my_position_ == 0;
  }
  constexpr bool operator!=(const discard_iterator &__it) const {
    return !(*this == __it);
  }

  bool operator<(const discard_iterator &__it) const {
    return __my_position_ - __it.__my_position_ < 0;
  }
  bool operator>(const discard_iterator &__it) const {
    return __my_position_ - __it.__my_position_ > 0;
  }

  difference_type operator-(const discard_iterator &__it) const {
    return __my_position_ - __it.__my_position_;
  }

  discard_iterator &operator++() {
    ++__my_position_;
    return *this;
  }
  discard_iterator &operator--() {
    --__my_position_;
    return *this;
  }
  discard_iterator operator++(int) {
    discard_iterator __it(__my_position_);
    ++__my_position_;
    return __it;
  }
  discard_iterator operator--(int) {
    discard_iterator __it(__my_position_);
    --__my_position_;
    return __it;
  }
  discard_iterator &operator+=(difference_type __forward) {
    __my_position_ += __forward;
    return *this;
  }
  discard_iterator &operator-=(difference_type __backward) {
    __my_position_ += __backward;
    return *this;
  }

  discard_iterator operator+(difference_type __forward) const {
    return discard_iterator(__my_position_ + __forward);
  }
  discard_iterator operator-(difference_type __backward) const {
    return discard_iterator(__my_position_ - __backward);
  }

private:
  difference_type __my_position_;
};
} // end namespace dpct

#endif
