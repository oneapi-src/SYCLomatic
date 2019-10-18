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

#ifndef __DPCT_FUNCTIONAL_H
#define __DPCT_FUNCTIONAL_H

#include <functional>
#include <dpstd/internal/function.h>
#include <dpstd/iterators.h>

#ifdef __PSTL_BACKEND_SYCL
#include <dpstd/pstl/parallel_backend_sycl_utils.h>
#endif

#include <tuple>
#include <utility>

namespace dpstd {
namespace internal {
using std::get;
#ifdef __PSTL_BACKEND_SYCL
using dpstd::__par_backend_hetero::__internal::get;
#endif
}
}

namespace dpct {

namespace internal {

template <typename Policy, typename NewName> struct rebind_policy {
  using type = Policy;
};

template <typename DevicePolicy, typename KernelName, typename NewName>
struct rebind_policy<
    dpstd::execution::v1::sycl_policy<DevicePolicy, KernelName>, NewName> {
  using type = dpstd::execution::v1::sycl_policy<DevicePolicy, NewName>;
};

template <typename T1, typename T2,
          typename R1 = typename std::iterator_traits<T1>::reference,
          typename R2 = typename std::iterator_traits<T2>::reference>
struct perm_fun {
  typedef R2 result_of;
  perm_fun(T1 input) : source(input) {}

  R2 operator()(R1 x) const { return *(source + x); }

private:
  T1 source;
};

} // end namespace internal

struct identity {
  template <typename _T> constexpr _T &&operator()(_T &&x) const noexcept {
    return std::forward<_T>(x);
  }
};

template <typename T> struct maximum {
  typedef T result_type;
  typedef T first_argument_type;
  typedef T second_argument_type;

  T operator()(const T &lhs, const T &rhs) const {
    return lhs < rhs ? rhs : lhs;
  }
};

template <typename T> struct minimum {
  typedef T result_type;
  typedef T first_argument_type;
  typedef T second_argument_type;

  T operator()(const T &lhs, const T &rhs) const {
    return lhs < rhs ? lhs : rhs;
  }
};

namespace internal {

/// Functor replacing a zip & discard iterator combination; useful for stencil
/// algorithm
/// Used by: copy_if, remove_copy_if, stable_partition_copy
/// Lambda: [](OutRef1 x) { return std::tie(x, std::ignore); }
template <typename T> struct discard_fun {

#ifdef __PSTL_BACKEND_SYCL
  template <typename _T>
  auto operator()(_T &&x) const
      -> decltype(dpstd::__par_backend_hetero::__internal::make_tuplewrapper(
          x, std::ignore)) {
    return dpstd::__par_backend_hetero::__internal::make_tuplewrapper(
        x, std::ignore);
  }
#else
  template <typename _T>
  auto operator()(_T &&x) const -> decltype(std::tie(x, std::ignore)) {
    return std::tie(x, std::ignore);
  }
#endif
};

/// Functor compares first element (key) from tied sequence.
template <typename Compare = class dpstd::__internal::__pstl_less>
struct compare_key_fun {
  typedef bool result_of;
  compare_key_fun(Compare _comp = dpstd::__internal::__pstl_less())
      : comp(_comp) {}

  template <typename _T1, typename _T2>
  result_of operator()(_T1 &&a, _T2 &&b) const {
    using std::get;
    return comp(dpstd::internal::get<0>(a), dpstd::internal::get<0>(b));
  }

private:
  Compare comp;
};

/// Functor evaluates second element of tied sequence with predicate.
/// Used by: copy_if, remove_copy_if, stable_partition_copy
/// Lambda:
template <typename Predicate> struct predicate_key_fun {
  typedef bool result_of;
  predicate_key_fun(Predicate _pred) : pred(_pred) {}

  template <typename _T1> result_of operator()(_T1 &&a) const {
    using std::get;
    return pred(dpstd::internal::get<1>(a));
  }

private:
  Predicate pred;
};

template <typename Predicate>
struct negate_predicate_key_fun {
  typedef bool result_of;
  negate_predicate_key_fun(Predicate _pred) : pred(_pred) {}

  template <typename _T1> result_of operator()(_T1 &&a) const {
    using std::get;
    return !pred(dpstd::internal::get<1>(a));
  }

private:
  Predicate pred;
};

template <typename T> struct sequence_fun {
  using result_type = T;
  sequence_fun(T _init, T _step) : init(_init), step(_step) {}

  template <typename _T> result_type operator()(_T &&i) const {
    return static_cast<T>(init + step * i);
  }

private:
  const T init;
  const T step;
};

/// [binary_pred](Ref a, Ref b){ return(binary_pred(get<0>(a),get<0>(b)));
template <typename Predicate> struct unique_by_key_fun {
  typedef bool result_of;
  unique_by_key_fun(Predicate _pred) : pred(_pred) {}
  template <typename _T> result_of operator()(_T &&a, _T &&b) const {
    using std::get;
    return pred(dpstd::internal::get<0>(a), dpstd::internal::get<0>(b));
  }

private:
  Predicate pred;
};

/// Lambda: [pred, &new_value](Ref1 a, Ref2 s) {return pred(s) ? new_value : a;
/// });
template <typename T, typename Predicate>
struct replace_if_fun {
public:
  typedef T result_of;
  replace_if_fun(Predicate _pred, T _new_value)
      : pred(_pred), new_value(_new_value) {}

  template <typename _T1, typename _T2> T operator()(_T1 &&a, _T2 &&s) const {
    return pred(s) ? new_value : a;
  }

private:
  Predicate pred;
  const T new_value;
};

/// [pred,op](Ref a){return pred(a) ? op(a) : a; }
template <typename T, typename Predicate, typename Operator>
struct transform_if_fun {
  typedef T result_of;
  transform_if_fun(Predicate _pred, Operator _op) : pred(_pred), op(_op) {}
  result_of operator()(const T &input) const {
    return pred(input) ? op(input) : input;
  }

private:
  Predicate pred;
  Operator op;
};

template <typename T, typename Predicate, typename BinaryOperation>
class transform_if_zip_stencil_fun {
public:
  transform_if_zip_stencil_fun(Predicate _pred = identity(),
                               BinaryOperation _op = identity())
      : pred(_pred), op(_op) {}
  template <typename _T> void operator()(_T &&t) {
    if (pred(dpstd::internal::get<2>(t)))
      dpstd::internal::get<3>(t) =
          op(dpstd::internal::get<0>(t), dpstd::internal::get<1>(t));
  }

private:
  Predicate pred;
  BinaryOperation op;
};
} // end namespace internal

} // end namespace dpct

#endif //__DPCT_FUNCTIONAL_H