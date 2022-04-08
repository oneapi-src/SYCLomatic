// C2S_LABEL_BEGIN|License|
// C2S_DEPENDENCY_EMPTY
// C2S_CODE
//==---- iterators.h ------------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
// C2S_LABEL_END

#ifndef __C2S_DPL_EXTRAS_ITERATORS_H__
#define __C2S_DPL_EXTRAS_ITERATORS_H__
// C2S_COMMENT
// C2S_COMMENT Example1:
// C2S_COMMENT // C2S_LABEL_BEGIN|FeatureNameDef|[Namespace]
// C2S_COMMENT // C2S_DEPENDENCY_EMPTY
// C2S_COMMENT // C2S_CODE
// C2S_COMMENT some code
// C2S_COMMENT // C2S_LABEL_END
// C2S_COMMENT
// C2S_COMMENT Example2:
// C2S_COMMENT // C2S_LABEL_BEGIN|FeatureNameDef|[Namespace]
// C2S_COMMENT // C2S_DEPENDENCY_BEGIN
// C2S_COMMENT // FileID|FeatureNameRef
// C2S_COMMENT [// FileID|FeatureNameRef]
// C2S_COMMENT ...
// C2S_COMMENT // C2S_DEPENDENCY_END
// C2S_COMMENT // C2S_CODE
// C2S_COMMENT some code
// C2S_COMMENT // C2S_LABEL_END
// C2S_COMMENT
// C2S_COMMENT For header file including dependency, please use predefined feature name:
// C2S_COMMENT   local_include_dependency: c2s helper files
// C2S_COMMENT   non_local_include_dependency: other headler files

// C2S_LABEL_BEGIN|non_local_include_dependency|
// C2S_DEPENDENCY_EMPTY
// C2S_CODE
#include <oneapi/dpl/iterator>
// C2S_LABEL_END

// C2S_LABEL_BEGIN|local_include_dependency|
// C2S_DEPENDENCY_EMPTY
// C2S_CODE
// C2S_LABEL_END
#include "functional.h"

namespace c2s {

// C2S_LABEL_BEGIN|using_std_advance|c2s
// C2S_DEPENDENCY_EMPTY
// C2S_CODE
using std::advance;
// C2S_LABEL_END

// C2S_LABEL_BEGIN|using_std_distance|c2s
// C2S_DEPENDENCY_EMPTY
// C2S_CODE
using std::distance;
// C2S_LABEL_END

// C2S_LABEL_BEGIN|make_counting_iterator|c2s
// C2S_DEPENDENCY_EMPTY
// C2S_CODE
template <typename T>
oneapi::dpl::counting_iterator<T> make_counting_iterator(const T &input) {
  return oneapi::dpl::counting_iterator<T>(input);
}
// C2S_LABEL_END

// C2S_LABEL_BEGIN|constant_iterator|c2s
// C2S_DEPENDENCY_EMPTY
// C2S_CODE
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
// C2S_LABEL_END

// C2S_LABEL_BEGIN|make_constant_iterator|c2s
// C2S_DEPENDENCY_BEGIN
// DplExtrasIterators|constant_iterator
// C2S_DEPENDENCY_END
// C2S_CODE
template <typename _Tp>
constant_iterator<_Tp> make_constant_iterator(_Tp __value) {
  return constant_iterator<_Tp>(__value);
}
// C2S_LABEL_END

} // end namespace c2s

#endif
