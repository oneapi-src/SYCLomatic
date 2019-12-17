/******************************************************************************
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

#ifndef __DPCT_ITERATORS_H__
#define __DPCT_ITERATORS_H__

#include <dpstd/iterators.h>

#include "functional.h"

namespace dpct {

using std::advance;

using std::distance;

template <typename T> using counting_iterator = dpstd::counting_iterator<T>;

template <typename T>
counting_iterator<T> make_counting_iterator(const T &input) {
  return counting_iterator<T>(input);
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

using dpstd::make_transform_iterator;

template <typename UnaryFunc, typename Iter>
using transform_iterator = dpstd::transform_iterator<UnaryFunc, Iter>;

template <typename... Types>
auto zipit_decl(std::tuple<Types...>) -> dpstd::zip_iterator<Types...>;
template <typename U> using zip_iterator = decltype(zipit_decl(U()));

namespace detail {
template <typename T, size_t... Is>
constexpr auto make_zip_iterator(T tuple, internal::index_sequence<Is...>)
    -> decltype(dpstd::make_zip_iterator(std::get<Is>(tuple)...)) {
  return dpstd::make_zip_iterator(std::get<Is>(tuple)...);
}
} // namespace detail

template <typename... T>
constexpr dpstd::zip_iterator<T...>
make_zip_iterator(const std::tuple<T...> &arg) {
  return detail::make_zip_iterator(
      arg, internal::make_index_sequence<sizeof...(T)>{});
}

} // end namespace dpct

#endif
