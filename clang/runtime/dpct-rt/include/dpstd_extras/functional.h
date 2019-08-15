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

#ifndef __DPCT_FUNCTIONAL_H
#define __DPCT_FUNCTIONAL_H

#include <dpstd/iterators.h>

#ifdef __PSTL_BACKEND_SYCL
#include <dpstd/pstl/parallel_backend_sycl_utils.h>
#endif

#include <tuple>

namespace dpct {

namespace internal {

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

template <typename T> struct identity {
  T operator()(const T &x) const { return x; }
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

#ifdef __PSTL_BACKEND_SYCL
using dpstd::__par_backend::__internal::make_tuplewrapper;
#endif

// Functor replacing a zip & discard iterator combination; useful for stencil
// algorithm Used by: copy_if, remove_copy_if, stable_partition_copy Lambda:
// [](OutRef1 x) { return std::tie(x, std::ignore); }
template <typename T> struct discard_fun {

#ifdef __PSTL_BACKEND_SYCL
  template <typename _T>
  auto operator()(_T &&x) const -> decltype(make_tuplewrapper(x, std::ignore)) {
    return make_tuplewrapper(x, std::ignore);
  }
#else
  template <typename _T>
  auto operator()(_T &&x) const -> decltype(std::tie(x, std::ignore)) {
    return std::tie(x, std::ignore);
  }
#endif
};

// Functor compares first element (key) from tied sequence.
template <typename Compare = class dpstd::__internal::__pstl_less>
struct compare_key_fun {
  typedef bool result_of;
  compare_key_fun(Compare _comp = dpstd::__internal::__pstl_less())
      : comp(_comp) {}

  template <typename _T1, typename _T2>
  result_of operator()(_T1 &&a, _T2 &&b) const {
    using std::get;
    return comp(get<0>(a), get<0>(b));
  }

private:
  Compare comp;
};

// Functor evaluates second element of tied sequence with predicate.
// Used by: copy_if, remove_copy_if, stable_partition_copy
// Lambda:
template <typename Predicate> struct predicate_key_fun {
  typedef bool result_of;
  predicate_key_fun(Predicate _pred) : pred(_pred) {}

  template <typename _T1> result_of operator()(_T1 &&a) const {
    using std::get;
    return pred(get<1>(a));
  }

private:
  Predicate pred;
};

// Used by: remove_if
template <typename Predicate>
struct negate_predicate_key_fun {
  typedef bool result_of;
  negate_predicate_key_fun(Predicate _pred) : pred(_pred) {}

  template <typename _T1> result_of operator()(_T1 &&a) const {
    using std::get;
    return !pred(get<1>(a));
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

//[binary_pred](Ref a, Ref b){
//return(binary_pred(std::get<0>(a),std::get<0>(b)));
template <typename Predicate> struct unique_by_key_fun {
  typedef bool result_of;
  unique_by_key_fun(Predicate _pred) : pred(_pred) {}
  template <typename _T> result_of operator()(_T &&a, _T &&b) const {
    using std::get;
    return pred(get<0>(a), get<0>(b));
  }

private:
  Predicate pred;
};

// Functor applies function if predicate
//[pred,op](Ref a){return pred(a) ? op(a) : a; }
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

// called by: transform_if(6 args)
template <typename T, /*typename T2, */ typename Predicate = class identity<T>,
          typename UnaryOperation = class identity<T>>
class transform_if_stencil_fun1 {
public:
  typedef typename std::tuple_element<2, T>::type result_of;
  transform_if_stencil_fun1(Predicate _pred = identity<T>(),
                            UnaryOperation _op = identity<T>())
      : pred(_pred), op(_op) {}
  // template<typename _T> result_of operator() (const T& a, _T&& s) const {
  // return pred(s) ? op(a) : a; }
  template <typename _T> result_of operator()(_T &&t) {
    using std::get;
    //#ifdef __PSTL_BACKEND_SYCL
    //            using dpstd::par_backend::internal::get;
    //#endif
    if (pred(get<1>(t)))
      return op(get<0>(t));
    else
      return get<0>(t);
  }

private:
  Predicate pred;
  UnaryOperation op;
};

// called by: transform_if(7 args)
//[pred,binary_op](Ref1 a, Ref2 s){return pred(s)
//binary_op(std::get<0>(a),std::get<1>(a)) : std::get<0>(a); }
template <typename T, /*typename T2, */ typename Predicate,
          typename BinaryOperation>
class transform_if_zip_stencil_fun {
public:
  typedef typename std::tuple_element<3, T>::type result_of;
  transform_if_zip_stencil_fun(Predicate _pred = identity<T>(),
                               BinaryOperation _op = identity<T>())
      : pred(_pred), op(_op) {}
  // template<typename _T> result_of operator() (const T& a, _T&& s) const {
  // return pred(s) ? op(a) : a; }
  template <typename _T> result_of operator()(_T t) {
    using std::get;
    //#ifdef __PSTL_BACKEND_SYCL
    //            using dpstd::par_backend::internal::get;
    //#endif
    if (pred(get<2>(t)))
      return op(get<0>(t), get<1>(t));
    else
      return get<0>(t);
  }

private:
  Predicate pred;
  BinaryOperation op;
};

} // end namespace internal

} // end namespace dpct

#endif
