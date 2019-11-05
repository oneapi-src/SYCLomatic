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

#ifndef __DPCT_ITERATORS_H
#define __DPCT_ITERATORS_H

#include <dpstd/iterators.h>

#include "functional.h"

#include <dpstd/pstl/utils.h>

namespace dpct {

using std::advance;

using std::distance;

template <typename T> using counting_iterator = dpstd::counting_iterator<T>;

template <typename T>
counting_iterator<T> make_counting_iterator(const T &input) {
  return counting_iterator<T>(input);
}

using dpstd::make_transform_iterator;

template <typename UnaryFunc, typename Iter>
using transform_iterator = dpstd::transform_iterator<UnaryFunc, Iter>;

template <typename... Types>
auto zipit_decl(std::tuple<Types...>) -> dpstd::zip_iterator<Types...>;
template <typename U> using zip_iterator = decltype(zipit_decl(U()));

namespace detail {
template <typename T, size_t... Is>
constexpr auto make_zip_iterator(T tuple,
                                 dpstd::__internal::__index_sequence<Is...>)
    -> decltype(dpstd::make_zip_iterator(std::get<Is>(tuple)...)) {
  return dpstd::make_zip_iterator(std::get<Is>(tuple)...);
}
}

template <typename... T>
constexpr dpstd::zip_iterator<T...>
make_zip_iterator(const std::tuple<T...> &arg) {
  return detail::make_zip_iterator(
      arg, dpstd::__internal::__make_index_sequence<sizeof...(T)>{});
}

template <typename Iter1, typename Iter2>
class permutation_iterator
    : public dpstd::transform_iterator<internal::perm_fun<Iter1, Iter2>,
                                       Iter2> {
public:
  permutation_iterator(Iter1 source, Iter2 map)
      : dpstd::transform_iterator<internal::perm_fun<Iter1, Iter2>, Iter2>(
            map, internal::perm_fun<Iter1, Iter2>(source)) {}
};

template <typename Iter1, typename Iter2>
dpstd::transform_iterator<internal::perm_fun<Iter1, Iter2>, Iter2>
make_permutation_iterator(Iter1 source, Iter2 map) {
  return make_transform_iterator(map, internal::perm_fun<Iter1, Iter2>(source));
};

} // end namespace dpct

#endif //__DPCT_ITERATORS_H