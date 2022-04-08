//==---- functional.h -----------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __C2S_DPL_EXTRAS_FUNCTIONAL_H__
#define __C2S_DPL_EXTRAS_FUNCTIONAL_H__

#include <functional>
#include <oneapi/dpl/functional>
#include <oneapi/dpl/iterator>

#if ONEDPL_USE_DPCPP_BACKEND
#include <oneapi/dpl/pstl/hetero/dpcpp/parallel_backend_sycl_utils.h>
#endif

#include <tuple>
#include <utility>

namespace c2s {

namespace internal {

template <class _ExecPolicy, class _T>
using enable_if_execution_policy =
    typename std::enable_if<oneapi::dpl::execution::is_execution_policy<
                                typename std::decay<_ExecPolicy>::type>::value,
                            _T>::type;

template <typename _T> struct is_hetero_execution_policy : ::std::false_type {};

template <typename... PolicyParams>
struct is_hetero_execution_policy<
    oneapi::dpl::execution::device_policy<PolicyParams...>> : ::std::true_type {
};

template <typename _T> struct is_fpga_execution_policy : ::std::false_type {};

#if _ONEDPL_FPGA_DEVICE
template <unsigned int unroll_factor, typename... PolicyParams>
struct is_hetero_execution_policy<
    execution::fpga_policy<unroll_factor, PolicyParams...>> : ::std::true_type {
};
#endif

template <class _ExecPolicy, class _T>
using enable_if_hetero_execution_policy = typename std::enable_if<
    is_hetero_execution_policy<typename std::decay<_ExecPolicy>::type>::value,
    _T>::type;

#if _ONEDPL_CPP14_INTEGER_SEQUENCE_PRESENT

template <std::size_t... _Sp>
using index_sequence = std::index_sequence<_Sp...>;
template <std::size_t _Np>
using make_index_sequence = std::make_index_sequence<_Np>;

#else

template <std::size_t... _Sp> class index_sequence {};

template <std::size_t _Np, std::size_t... _Sp>
struct make_index_sequence_impl
    : make_index_sequence_impl<_Np - 1, _Np - 1, _Sp...> {};

template <std::size_t... _Sp> struct make_index_sequence_impl<0, _Sp...> {
  using type = index_sequence<_Sp...>;
};

template <std::size_t _Np>
using make_index_sequence = typename make_index_sequence_impl<_Np>::type;
#endif

// Minimal buffer implementations for temporary storage in mapping rules
// Some of our algorithms need to start with raw memory buffer,
// not an initialized array, because initialization/destruction
// would make the span be at least O(N).
#if ONEDPL_USE_DPCPP_BACKEND
template <typename _Tp> class __buffer {
  cl::sycl::buffer<_Tp, 1> __buf;

  __buffer(const __buffer &) = delete;

  void operator=(const __buffer &) = delete;

public:
  // Try to obtain buffer of given size to store objects of _Tp type
  __buffer(std::size_t __n) : __buf(sycl::range<1>(__n)) {}

  // Return pointer to buffer, or  NULL if buffer could not be obtained.
  auto get() -> decltype(oneapi::dpl::begin(__buf)) const {
    return oneapi::dpl::begin(__buf);
  }
};
#else
template <typename _Tp> class __buffer {
  std::unique_ptr<_Tp> _M_ptr;

  __buffer(const __buffer &) = delete;

  void operator=(const __buffer &) = delete;

public:
  // Try to obtain buffer of given size to store objects of _Tp type
  __buffer(const std::size_t __n) : _M_ptr(new _Tp[__n]) {}

  // Return pointer to buffer, or  NULL if buffer could not be obtained.
  _Tp *get() const { return _M_ptr.get(); }
};
#endif

// Implements C++14 std::less<void> specialization to allow parameter type
// deduction.
class __less {
public:
  template <typename _Xp, typename _Yp>
  bool operator()(_Xp &&__x, _Yp &&__y) const {
    return std::forward<_Xp>(__x) < std::forward<_Yp>(__y);
  }
};

template <typename Policy, typename NewName> struct rebind_policy {
  using type = Policy;
};

template <typename KernelName, typename NewName>
struct rebind_policy<oneapi::dpl::execution::device_policy<KernelName>,
                     NewName> {
  using type = oneapi::dpl::execution::device_policy<NewName>;
};

#if _ONEDPL_FPGA_DEVICE
template <unsigned int factor, typename KernelName, typename NewName>
struct rebind_policy<oneapi::dpl::execution::fpga_policy<factor, KernelName>,
                     NewName> {
  using type = oneapi::dpl::execution::fpga_policy<factor, NewName>;
};
#endif

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

// Functor compares first element (key) from tied sequence.
template <typename Compare = class internal::__less> struct compare_key_fun {
  typedef bool result_of;
  compare_key_fun(Compare _comp = internal::__less()) : comp(_comp) {}

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
template <typename Predicate> struct negate_predicate_key_fun {
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

//[binary_pred](Ref a, Ref b){ return(binary_pred(get<0>(a),get<0>(b)));
template <typename Predicate> struct unique_fun {
  typedef bool result_of;
  unique_fun(Predicate _pred) : pred(_pred) {}
  template <typename _T> result_of operator()(_T &&a, _T &&b) const {
    using std::get;
    return pred(get<0>(a), get<0>(b));
  }

private:
  Predicate pred;
};

// Lambda: [pred, &new_value](Ref1 a, Ref2 s) {return pred(s) ? new_value : a;
// });
template <typename T, typename Predicate> struct replace_if_fun {
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

//[pred,op](Ref a){return pred(a) ? op(a) : a; }
template <typename T, typename Predicate, typename Operator>
struct transform_if_fun {
  transform_if_fun(Predicate _pred, Operator _op) : pred(_pred), op(_op) {}
  template <typename _T>
  void operator()(_T&& t) const {
    using std::get;
    if (pred(get<0>(t)))
      get<1>(t) = op(get<0>(t));
  }

private:
  Predicate pred;
  Operator op;
};

//[pred, op](Ref1 a, Ref2 s) { return pred(s) ? op(a) : a; });
template <typename T, typename Predicate, typename Operator>
struct transform_if_unary_zip_mask_fun {
  transform_if_unary_zip_mask_fun(Predicate _pred, Operator _op) : pred(_pred), op(_op) {}
  template <typename _T>
  void operator()(_T&& t) const {
    using std::get;
    if (pred(get<1>(t)))
      get<2>(t) = op(get<0>(t));
  }

private:
  Predicate pred;
  Operator op;
};

template <typename T, typename Predicate, typename BinaryOperation>
class transform_if_zip_mask_fun {
public:
  transform_if_zip_mask_fun(Predicate _pred = oneapi::dpl::identity(),
                            BinaryOperation _op = oneapi::dpl::identity())
      : pred(_pred), op(_op) {}
  template <typename _T> void operator()(_T &&t) const {
    using std::get;
    if (pred(get<2>(t)))
      get<3>(t) = op(get<0>(t), get<1>(t));
  }

private:
  Predicate pred;
  BinaryOperation op;
};
} // end namespace internal

} // end namespace c2s

#endif
