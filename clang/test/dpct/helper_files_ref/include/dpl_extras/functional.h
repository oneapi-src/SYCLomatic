//==---- functional.h -----------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __DPCT_FUNCTIONAL_H__
#define __DPCT_FUNCTIONAL_H__

#include <functional>
#include <oneapi/dpl/functional>
#include <oneapi/dpl/iterator>

#if ONEDPL_USE_DPCPP_BACKEND
#include <oneapi/dpl/pstl/hetero/dpcpp/parallel_backend_sycl_utils.h>
#endif

#include <tuple>
#include <utility>

namespace dpct {

struct null_type {};

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
  sycl::buffer<_Tp, 1> __buf;

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
  mutable Compare comp;
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
  mutable Predicate pred;
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
  mutable Predicate pred;
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
  mutable Predicate pred;
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
  mutable Predicate pred;
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
  mutable Predicate pred;
  mutable Operator op;
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
  mutable Predicate pred;
  mutable Operator op;
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
  mutable Predicate pred;
  mutable BinaryOperation op;
};

#ifdef DPCT_USM_LEVEL_NONE


template <typename _ExecutionPolicyHost, typename _ExecutionPolicyDevice,
          typename _Ptr1, typename Size, typename _Ptr2, typename _Ptr3, typename _Ptr4, typename _Ptr5, typename _Ptr6, typename _Func, typename... _Args>
inline
std::enable_if_t<std::is_pointer_v<_Ptr1> && std::is_pointer_v<_Ptr2> && std::is_pointer_v<_Ptr3> && std::is_pointer_v<_Ptr4> && std::is_pointer_v<_Ptr5> && std::is_pointer_v<_Ptr6>, ::std::pair<_Ptr5, _Ptr6>>
check_device_ptr_and_launch(_ExecutionPolicyHost&& host_policy, _ExecutionPolicyDevice&& dev_policy, _Ptr1 start1,
                       Size size, _Ptr2 start2, _Ptr3 start3, _Ptr4 start4, _Ptr5 start5, _Ptr6 start6, _Func func, _Args... args)
{
    if (dpct::is_device_ptr(start1)) {
      using value_type1 = typename std::iterator_traits<_Ptr1>::value_type;
      using value_type2 = typename std::iterator_traits<_Ptr2>::value_type;
      using value_type3 = typename std::iterator_traits<_Ptr3>::value_type;
      using value_type4 = typename std::iterator_traits<_Ptr4>::value_type;
      using value_type5 = typename std::iterator_traits<_Ptr5>::value_type;
      using value_type6 = typename std::iterator_traits<_Ptr6>::value_type;
      auto dev_start1 = dpct::device_pointer<value_type1>(start1);
      auto dev_end1 = dpct::device_pointer<value_type1>(start1) + size;
      auto dev_start2 = dpct::device_pointer<value_type2>(start2);
      auto dev_end2 = dpct::device_pointer<value_type2>(start2) + size;
      auto dev_start3 = dpct::device_pointer<value_type3>(start3);
      auto dev_end3 = dpct::device_pointer<value_type3>(start3) + size;
      auto dev_start4 = dpct::device_pointer<value_type4>(start4);
      auto dev_end4 = dpct::device_pointer<value_type4>(start4) + size;
      auto dev_start5 = dpct::device_pointer<value_type5>(start5);
      auto dev_end5 = dpct::device_pointer<value_type5>(start5) + size;
      auto dev_start6 = dpct::device_pointer<value_type6>(start6);
      auto dev_end6 = dpct::device_pointer<value_type6>(start6) + size;
      auto ret = func(std::forward<_ExecutionPolicyDevice>(dev_policy), dev_start1, dev_end1, dev_start2, dev_start3, dev_start4, dev_start5, dev_start6, args...);
      return ::std::pair<_Ptr5, _Ptr6>(start5 + (ret.first - dev_start5), start6 + (ret.second - dev_start6));
    } else {
      return func(std::forward<_ExecutionPolicyHost>(host_policy), start1, start1 + size, start2, start3, start4, start5, start6, args...);
    }
}


template <typename _ExecutionPolicyHost, typename _ExecutionPolicyDevice,
          typename _Ptr1, typename Size, typename _Ptr2, typename _Ptr3, typename _Ptr4, typename _Func, typename... _Args>
inline
std::enable_if_t<std::is_pointer_v<_Ptr1> && std::is_pointer_v<_Ptr2> && std::is_pointer_v<_Ptr3> && std::is_pointer_v<_Ptr4>, ::std::pair<_Ptr3, _Ptr4>>
check_device_ptr_and_launch(_ExecutionPolicyHost&& host_policy, _ExecutionPolicyDevice&& dev_policy, _Ptr1 start1,
                       Size size, _Ptr2 start2, _Ptr3 start3, _Ptr4 start4, _Func func, _Args... args)
{
    if (dpct::is_device_ptr(start1)) {
      using value_type1 = typename std::iterator_traits<_Ptr1>::value_type;
      using value_type2 = typename std::iterator_traits<_Ptr2>::value_type;
      using value_type3 = typename std::iterator_traits<_Ptr3>::value_type;
      using value_type4 = typename std::iterator_traits<_Ptr4>::value_type;
      auto dev_start1 = dpct::device_pointer<value_type1>(start1);
      auto dev_end1 = dpct::device_pointer<value_type1>(start1) + size;
      auto dev_start2 = dpct::device_pointer<value_type2>(start2);
      auto dev_end2 = dpct::device_pointer<value_type2>(start2) + size;
      auto dev_start3 = dpct::device_pointer<value_type3>(start3);
      auto dev_end3 = dpct::device_pointer<value_type3>(start3) + size;
      auto dev_start4 = dpct::device_pointer<value_type4>(start4);
      auto dev_end4 = dpct::device_pointer<value_type4>(start4) + size;
      auto ret = func(std::forward<_ExecutionPolicyDevice>(dev_policy), dev_start1, dev_end1, dev_start2, dev_start3, dev_start4, args...);
      return ::std::pair<_Ptr3, _Ptr4>(start3 + (ret.first - dev_start3), start4 + (ret.second - dev_start4));
    } else {
      return func(std::forward<_ExecutionPolicyHost>(host_policy), start1, start1 + size, start2, start3, start4, args...);
    }
}



template <typename _ExecutionPolicyHost, typename _ExecutionPolicyDevice,
          typename _Ptr1, typename Size, typename _Ptr2, typename _Func, typename... _Args>
inline
std::enable_if_t<std::is_pointer_v<_Ptr1> && std::is_pointer_v<_Ptr2>, ::std::pair<_Ptr1, _Ptr2>>
check_device_ptr_and_launch(_ExecutionPolicyHost&& host_policy, _ExecutionPolicyDevice&& dev_policy, _Ptr1 start1,
                       Size size, _Ptr2 start2, _Func func, _Args... args)
{
    if (dpct::is_device_ptr(start1)) {
      using value_type1 = typename std::iterator_traits<_Ptr1>::value_type;
      using value_type2 = typename std::iterator_traits<_Ptr2>::value_type;
      auto dev_start1 = dpct::device_pointer<value_type1>(start1);
      auto dev_end1 = dpct::device_pointer<value_type1>(start1) + size;
      auto dev_start2 = dpct::device_pointer<value_type2>(start2);
      auto dev_end2 = dpct::device_pointer<value_type2>(start2) + size;
      auto ret = func(std::forward<_ExecutionPolicyDevice>(dev_policy), dev_start1, dev_end1, dev_start2, args...);
      return ::std::pair<_Ptr1, _Ptr2>(start1 + (ret.first - dev_start1), start2 + (ret.second - dev_start2));
    } else {
      return func(std::forward<_ExecutionPolicyHost>(host_policy), start1, start1 + size, start2, args...);
    }
}

template <typename _ExecutionPolicyHost, typename _ExecutionPolicyDevice,
          typename _PtrT, typename Size, typename _Func, typename... _Args>
inline
std::enable_if_t<std::is_pointer_v<_PtrT>, ::std::pair<_PtrT, _PtrT>>
check_device_ptr_and_launch(_ExecutionPolicyHost&& host_policy, _ExecutionPolicyDevice&& dev_policy, _PtrT start,
                       Size size, _Func func, _Args... args)
{
    if (dpct::is_device_ptr(start)) {
      using value_type = typename std::iterator_traits<_PtrT>::value_type;
      auto dev_start = dpct::device_pointer<value_type>(start);
      auto dev_end = dpct::device_pointer<value_type>(start) + size;
      auto ret = func(std::forward<_ExecutionPolicyDevice>(dev_policy), dev_start, dev_end, args...);
      return ::std::pair<_PtrT, _PtrT>(start + (ret.first - dev_start), start + (ret.second - dev_start));
    } else {
      return func(std::forward<_ExecutionPolicyHost>(host_policy), start, start + size, args...);
    }
}

#endif //DPCT_USM_LEVEL_NONE

// This following code is similar to a section of code in
// oneDPL/include/oneapi/dpl/pstl/hetero/dpcpp/parallel_backend_sycl_radix_sort.h
// It has a similar approach, and could be consolidated.
// Outside of some differences in approach, there are two significant
// differences in function.
//
// 1) This code allows the output type of the bit range translation to be fit
// into to the minimal type required to provide that many bits. The code in
// oneDPL to calculate the bucket for the radix is similar but its output is
// always std::uint32_t.  The assumption that the bit range desired will fit in
// 32 bits is not true for this code.
//
// 2) This code ensures that for floating point type, -0.0f and 0.0f map to the
// same value.  This allows the output of this translation to be used to provide
// a sort which ensures the stability of these values for floating point types.

template <int N> struct uint_byte_map {};
template <> struct uint_byte_map<1> { using type = uint8_t; };
template <> struct uint_byte_map<2> { using type = uint16_t; };
template <> struct uint_byte_map<4> { using type = uint32_t; };
template <> struct uint_byte_map<8> { using type = uint64_t; };

template <typename T> struct uint_map {
  using type = typename uint_byte_map<sizeof(T)>::type;
};

template <typename T, typename OutKeyT> class translate_key {
  using uint_type_t = typename uint_map<T>::type;

public:
  translate_key(int begin_bit, int end_bit) {
    shift = begin_bit;
    mask = ~OutKeyT(0); // all ones
    mask = mask >> (sizeof(OutKeyT) * 8 -
                    (end_bit - begin_bit));           // setup appropriate mask
    flip_sign = uint_type_t(1) << (sizeof(uint_type_t) * 8 - 1); // sign bit
    flip_key = ~uint_type_t(0);                       // 0xF...F
  }

  inline OutKeyT operator()(const T &key) const {
    uint_type_t intermediate;
    if constexpr (std::is_floating_point<T>::value) {
        // normal case (both -0.0f and 0.0f equal -0.0f)
        if (key != T(-0.0f)) {
        uint_type_t is_negative = reinterpret_cast<const uint_type_t &>(key) >>
              (sizeof(uint_type_t) * 8 - 1);
          intermediate = reinterpret_cast<const uint_type_t &>(key) ^
                         ((is_negative * flip_key) | flip_sign);
        } else // special case for -0.0f to keep stability with 0.0f
        {
          T negzero = T(-0.0f);
          intermediate = reinterpret_cast<const uint_type_t &>(negzero);
        }
    } else if constexpr (std::is_signed<T>::value) {
        intermediate = reinterpret_cast<const uint_type_t &>(key) ^ flip_sign;
    } else {
      intermediate = key;
    }

    return static_cast<OutKeyT>(intermediate >> shift) &
           mask; // shift, cast, and mask
  }

private:
  uint8_t shift;
  OutKeyT mask;
  uint_type_t flip_sign;
  uint_type_t flip_key;
};

} // end namespace internal

} // end namespace dpct

#endif
