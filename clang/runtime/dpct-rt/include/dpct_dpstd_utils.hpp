/******************************************************************************
* INTEL CONFIDENTIAL
*
* Copyright 2018-2019 Intel Corporation.
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

//===--- dpct_dpstd_utils.hpp ------------------------------*- C++ -*---===//

#ifndef __DPCT_DPSTD_HPP__
#define __DPCT_DPSTD_HPP__

// The definitions below are the ones that require non-trivial mapping to
// the DPC++ library

#define __PSTL_BACKEND_SYCL
#define __USE_SYCL
#define __USE_DPCT
#define DPCT_CUSTOM

// ---------------------
// From: dpct/memory.h
// ---------------------

#ifndef DPCPP_MEMORY_H
#define DPCPP_MEMORY_H

#ifdef __USE_SYCL
#include <CL/sycl.hpp>
#endif

// Memory management section:
// device_ptr, device_reference, swap, device_iterator, device_malloc, device_new, device_free, device_delete
namespace dpct {

  namespace sycl = cl::sycl;

  //template<typename T>
  //using device_reference = T&;
  template <typename T>
  class device_ptr;

  template<typename T>
  struct device_reference {
    using pointer = device_ptr<T>;
    using value_type = T;
    template<typename OtherT>
    device_reference(const device_reference<OtherT>& input) : value(input.value) {}
    //template<typename OtherT>
    //device_reference(device_reference<OtherT>&& input): value(std::move(input.value)) {}
    device_reference(const pointer& input) : value((*input).value) {}
    device_reference(value_type& input) : value(input) {}
    template<typename OtherT>
    device_reference& operator=(const device_reference<OtherT>& input) {
      value = input;
      return *this;
    };
    device_reference& operator=(const device_reference& input) {
      T val = input.value;
      value = val;
      return *this;
    };
    device_reference& operator=(const value_type& x) {
      value = x;
      return *this;
    };
    pointer operator&() const { return pointer(&value); };
    operator value_type() const { return T(value); }
    device_reference& operator++() { ++value; return *this; };
    device_reference& operator--() { --value; return *this; };
    device_reference operator++(int) {
      device_reference ref(*this);
      ++(*this);
      return ref;
    };
    device_reference operator--(int) {
      device_reference ref(*this);
      --(*this);
      return ref;
    };
    device_reference& operator+=(const T& input) { value += input; return *this; };
    device_reference& operator-=(const T& input) { value -= input; return *this; };
    device_reference& operator*=(const T& input) { value *= input; return *this; };
    device_reference& operator/=(const T& input) { value /= input; return *this; };
    device_reference& operator%=(const T& input) { value %= input; return *this; };
    device_reference& operator&=(const T& input) { value &= input; return *this; };
    device_reference& operator|=(const T& input) { value |= input; return *this; };
    device_reference& operator^=(const T& input) { value ^= input; return *this; };
    device_reference& operator<<=(const T& input) { value <<= input; return *this; };
    device_reference& operator>>=(const T& input) { value >>= input; return *this; };
    void swap(device_reference& input) {
      T tmp = (*this);
      *this = (input);
      input = (tmp);
    }
    T& value;
  };

  template<typename T>
  void swap(device_reference<T> &x, device_reference<T> &y) {
    x.swap(y);
  }

  template<typename T>
  void swap(T &x, T &y) {
    T tmp = x;
    x = y;
    y = tmp;
  }
#if 0
  template <typename T>
  struct device_iterator {
    using value_type = T;
    using difference_type = std::size_t;
    using pointer = T * ;
    using reference = device_reference<T>;
    using iterator_category = std::random_access_iterator_tag;

    sycl::buffer<T, 1> buffer;
    difference_type idx;

    device_iterator() : buffer(sycl::buffer<T, 1>(sycl::range<1>(1))), idx(difference_type{}) { }
    device_iterator(sycl::buffer<T, 1> vec, std::size_t index) : buffer(vec), idx(index) {}
    device_iterator(const device_iterator& in) : buffer(in.buffer), idx(in.idx) {}
    device_iterator& operator=(const device_iterator& in) {
      buffer = in.buffer;
      idx = in.idx;
      return *this;
    }
    reference operator*() const {
      auto ptr = (const_cast<device_iterator*>(this)->buffer.template get_access<sycl::access::mode::read_write>()).get_pointer();
      return device_reference<T>(*(ptr + idx));
    }
    reference operator[](difference_type i) const {
      return *(*this + i);
    }
    device_iterator& operator++() {
      ++idx;
      return *this;
    }
    device_iterator& operator--() {
      --idx;
      return *this;
    }
    device_iterator operator++(int) {
      device_iterator it(*this);
      ++(*this);
      return it;
    }
    device_iterator operator--(int) {
      device_iterator it(*this);
      --(*this);
      return it;
    }
    device_iterator operator+(difference_type forward) const {
      return { buffer, idx + forward };
    }
    device_iterator& operator+=(difference_type forward) {
      idx += forward;
      return *this;
    }
    device_iterator operator-(difference_type backward) const {
      return { buffer, idx - backward };
    }
    device_iterator& operator-=(difference_type backward) {
      idx -= backward;
      return *this;
    }
    friend device_iterator operator+(difference_type forward, const device_iterator& it) {
      return it + forward;
    }
    difference_type operator-(const device_iterator& it) const {
      return idx - it.idx;
    }
    bool operator==(const device_iterator& it) const { return *this - it == 0; }
    bool operator!=(const device_iterator& it) const { return !(*this == it); }
    bool operator<(const device_iterator& it) const { return *this - it < 0; }
    bool operator>(const device_iterator& it) const { return it < *this; }
    bool operator<=(const device_iterator& it) const { return !(*this > it); }
    bool operator>=(const device_iterator& it) const { return !(*this < it); }

    sycl::buffer<T, 1> get_buffer() { return buffer; }
    std::size_t get_idx() { return idx; }
  };
#else
  template<typename T> using device_iterator = pstl::sycl_iterator<sycl::access::mode::read_write, T>;
#endif
  template <typename T>
  class device_ptr : public device_iterator<T> {
    using Base = device_iterator<T>;
  public:
    template<typename OtherT>
    device_ptr(const device_iterator<OtherT>& in) : Base(in) { }
    template<typename OtherT>
    device_ptr(sycl::buffer<OtherT, 1> in) : Base(in, std::size_t{}) { }
#ifdef DPCT_CUSTOM
    /*
    template<typename OtherT>
    device_ptr(OtherT ptr) {
       dpct::memory_manager::allocation
    alloc=dpct::memory_manager::get_instance().translate_ptr((void*) ptr); int
    size=alloc.size; new
    (this)device_ptr(alloc.buffer.reinterpret<T>(cl::sycl::range<1>(size/sizeof(T))));
    }
    */

    // TODO: Compiled with ComputeCpp, cl::sycl::buffer inited by another buffer
    // or a raw pointer manager unaffected memory. But in CUDA, device_ptr manager
    // the same memory when it inited by a raw pointer.
    template <typename OtherT>
    device_ptr(OtherT ptr)
      : Base(
        cl::sycl::buffer<T, 1>(cl::sycl::range<1>(
          dpct::memory_manager::get_instance().translate_ptr(ptr).size /
          sizeof(T))),
        std::size_t{}) {}
#endif
    // needed for device_malloc
    device_ptr(const std::size_t n) : Base(sycl::buffer<T, 1>(sycl::range<1>(n)), std::size_t{ }) { }
    device_ptr() : Base() {}
    template<typename OtherT>
    device_ptr(const device_ptr<OtherT>& in) : Base(in) { }
    template<typename OtherT>
    device_ptr& operator=(const device_iterator<OtherT>& in) {
      *this = in;
      return *this;
    }
    typename Base::pointer get() const {
      auto res = (const_cast<device_ptr*>(this)->Base::buffer.template get_access<sycl::access::mode::read_write>()).get_pointer();
      return res + Base::idx;
    }
    device_ptr operator+(typename Base::difference_type forward) const {
      return device_iterator<T>{ Base::buffer, Base::idx + forward };
    }
    device_ptr operator-(typename Base::difference_type backward) const {
      return device_iterator<T>{ Base::buffer, Base::idx - backward };
    }
    typename Base::difference_type operator-(const device_ptr& it) const {
      return Base::idx - it.idx;
    }
#if 0
    // TODO: not implemented yet
    // BTW, any use cases couldn't be found
    template<typename OtherT>
    device_ptr(OtherT* in) : Base(sycl::buffer<T, 1>(in, sycl::range<1>(1)), std::size_t{}) {}
#endif
  };

  //template <typename T>
  //using device_ptr = sycl::global_ptr<T>;

//    device_ptr<void> device_malloc(const std::size_t n);
  template <typename T>
  device_ptr<T> device_malloc(const std::size_t n) {
    return device_ptr<T>(n / sizeof(T));
  }
  //    device_ptr<T> device_new(device_ptr<void> p, const T& exemplar, const std::size_t n = 1);
  template <typename T>
  device_ptr<T> device_new(device_ptr<T> p, const T& exemplar, const std::size_t n = 1) {
    std::vector<T> result(n, exemplar);
    p.buffer = sycl::buffer<T, 1>(result.begin(), result.end());
    return p + n;
  }
  //    device_ptr<T> device_new(device_ptr<void> p, const std::size_t n = 1);
  template <typename T>
  device_ptr<T> device_new(device_ptr<T> p, const std::size_t n = 1) {
    return device_new(p, T{}, n);
  }
  template <typename T>
  device_ptr<T> device_new(const std::size_t n = 1) {
    return device_ptr<T>(n);
  }

  template <typename T>
  void device_free(device_ptr<T> ptr) {}

  template <typename T>
  typename std::enable_if<!std::is_trivially_destructible<T>::value, void>::type
    device_delete(device_ptr<T> p, const std::size_t n = 1) {
    for (std::size_t i = 0; i < n; ++i) {
      p[i].~T();
    }
  }
  template <typename T>
  typename std::enable_if<std::is_trivially_destructible<T>::value, void>::type
    device_delete(device_ptr<T>, const std::size_t n = 1) {}

  // casts
  template<typename T>
  device_ptr<T> device_pointer_cast(T* ptr) {
    return device_ptr<T>(ptr);
  }

  template<typename T>
  device_ptr<T> device_pointer_cast(const device_ptr<T>& ptr) {
    return device_ptr<T>(ptr);
  }

  template<typename T>
  T* raw_pointer_cast(const device_ptr<T>& ptr) {
    return ptr.get();
  }

  template<typename Pointer>
  Pointer raw_pointer_cast(const Pointer& ptr) {
    return ptr;
  }

  //template<typename T>
  //T& raw_reference_cast(__global T& ref) {
  //    return (ref);
  //}

  // allocators (Experimental implementation. It can be extended if needed)
  template<typename T>
  using device_allocator = sycl::buffer_allocator;
  template<typename T>
  using device_malloc_allocator = sycl::buffer_allocator;
  template<typename T>
  using device_new_allocator = sycl::buffer_allocator;

} // namespace dpct
#endif // DPCPP_MEMORY_H

// ------------------------
// From: dpct/functional.h
// ------------------------

#ifndef DPCT_FUNCTIONAL_H
#define DPCT_FUNCTIONAL_H

#include <dpstd/functional.h>
#include <dpstd/iterators.h>
#ifdef __PSTL_BACKEND_SYCL
#include <pstl/internal/parallel_backend_sycl_utils.h>
#endif
#include <tuple>

namespace dpct {

  template<typename T>
  struct maximum {
    typedef T result_type;
    typedef T first_argument_type;
    typedef T second_argument_type;

    T operator()(const T &lhs, const T &rhs) const { return lhs < rhs ? rhs : lhs; }
  };

  template<typename T>
  struct minimum {
    typedef T result_type;
    typedef T first_argument_type;
    typedef T second_argument_type;

    T operator()(const T &lhs, const T &rhs) const { return lhs < rhs ? lhs : rhs; }
  };

  // Functor replacing a zip & discard iterator combination; useful for stencil algorithm
      // Used by: copy_if, remove_copy_if, stable_partition_copy
      // Lambda: [](OutRef1 x) { return std::tie(x, std::ignore); }
  template<typename T>
  struct discard_fun {
    template<typename _T> auto operator() (_T&& x) const -> decltype(std::tie(x, std::ignore)) {
      return std::tie(x, std::ignore);
    }
  };

  // Functor compares first element (key) from tied sequence.
  template<typename Compare = class __pstl::internal::pstl_less>
  struct compare_key_fun {
    typedef bool result_of;
    compare_key_fun(Compare _comp = __pstl::internal::pstl_less()) : comp(_comp) {}

    template<typename _T1, typename _T2> result_of operator() (_T1&& a, _T2&& b) const {
      using std::get;
      return comp(get<0>(a), get<0>(b));
    }
  private:
    Compare comp;
  };

  // Functor evaluates second element of tied sequence with predicate.
  // Used by: copy_if, remove_copy_if, stable_partition_copy
  // Lambda:
  template<typename Predicate>
  struct predicate_key_fun { // predicate fun?
    typedef bool result_of;
    predicate_key_fun(Predicate _pred) : pred(_pred) {}

    template<typename _T1> result_of operator() (_T1&& a) const {
      using std::get;
      return pred(get<1>(a));
    }
  private:
    Predicate pred;
  };


  // Used by: remove_if
  template<typename Predicate>
  struct negate_predicate_key_fun { // predicate fun?
    typedef bool result_of;
    negate_predicate_key_fun(Predicate _pred) : pred(_pred) {}

    template<typename _T1> result_of operator() (_T1&& a) const {
      using std::get;
      return !pred(get<1>(a));
    }
  private:
    Predicate pred;
  };

  template<typename T>
  struct sequence_fun {
    using result_type = T;
    sequence_fun(T _init, T _step) : init(_init), step(_step) {}

    template<typename _T> result_type operator() (_T&& i) const {
      return static_cast<T>(init + step * i);
    }

  private:
    const T init;
    const T step;
  };

  //[binary_pred](Ref a, Ref b){ return(binary_pred(std::get<0>(a),std::get<0>(b)));
  template<typename Predicate>
  struct unique_by_key_fun { // compare key?
    typedef bool result_of;
    unique_by_key_fun(Predicate _pred) : pred(_pred) {}
    template<typename _T> result_of operator() (_T&& a, _T&& b) const {
      using std::get;
      return pred(get<0>(a), get<0>(b));
    }
  private:
    Predicate pred;
  };

  // Functor applies function if predicate
  //[pred,op](Ref a){return pred(a) ? op(a) : a; }
  template<typename T, typename Predicate, typename Operator>
  struct transform_if_fun {
    typedef T result_of;
    transform_if_fun(Predicate _pred, Operator _op) : pred(_pred), op(_op) {}
    result_of operator() (const T& input) const { return pred(input) ? op(input) : input; }
  private:
    Predicate pred;
    Operator op;
  };

  //called by: transform_if(6 args)
  template<typename T, /*typename T2, */ typename Predicate = class dpstd::identity<T>, typename UnaryOperation = class dpstd::identity<T>>
  class transform_if_stencil_fun1 {
  public:
    typedef typename std::tuple_element<2, T>::type result_of;
    transform_if_stencil_fun1(Predicate _pred = dpstd::identity<T>(), UnaryOperation _op = dpstd::identity<T>()) : pred(_pred), op(_op) {}
    //template<typename _T> result_of operator() (const T& a, _T&& s) const { return pred(s) ? op(a) : a; }
    template<typename _T>
    result_of operator() (_T&& t)
    {
      using std::get;
#ifdef __PSTL_BACKEND_SYCL
      using __pstl::par_backend::internal::get;
#endif
      if (pred(get<1>(t))) return op(get<0>(t));
      else return get<0>(t);
    }
  private:
    Predicate pred;
    UnaryOperation op;
  };

  //called by: transform_if(7 args)
      //[pred,binary_op](Ref1 a, Ref2 s){return pred(s) ? binary_op(std::get<0>(a),std::get<1>(a)) : std::get<0>(a); }
  template<typename T, /*typename T2, */ typename Predicate, typename BinaryOperation>
  class transform_if_zip_stencil_fun {
  public:
    typedef typename std::tuple_element<3, T>::type result_of;
    transform_if_zip_stencil_fun(Predicate _pred = dpstd::identity<T>(), BinaryOperation _op = dpstd::identity<T>()) : pred(_pred), op(_op) {}
    //template<typename _T> result_of operator() (const T& a, _T&& s) const { return pred(s) ? op(a) : a; }
    template<typename _T>
    result_of operator() (_T t)
    {
      using std::get;
#ifdef __PSTL_BACKEND_SYCL
      using __pstl::par_backend::internal::get;
#endif
      if (pred(get<2>(t))) return op(get<0>(t), get<1>(t));
      else return get<0>(t);
    }
  private:
    Predicate pred;
    BinaryOperation op;
  };

}//end namespace dpct
#endif

// ------------------------
// From: dpct/transform.h
// ------------------------

#ifndef DPCT_TRANSFORM_H
#define DPCT_TRANSFORM_H

#include <dpstd/iterators.h>

#include <algorithm>
#include <type_traits>
#include <numeric>

#include <pstl/execution>
#include <pstl/algorithm>
#include <pstl/numeric>

namespace dpct {

  template<class InputIter, class T >
  void sequence(InputIter first, InputIter last, T init, T step) {
    sequence(pstl::execution::sycl, first, last, init);
  }

  template<class Policy, class InputIter, class T >
  void sequence(Policy&& policy, InputIter first, InputIter last, T init, T step) {
    using DiffSize = typename std::iterator_traits<InputIter>::difference_type;
    std::transform(std::forward<Policy>(policy), dpstd::counting_iterator<DiffSize>(0),
      dpstd::counting_iterator<DiffSize>(std::distance(first, last)), first, dpct::sequence_fun<T>(init, step));
  }

  template<class Policy, class InputIter, class T >
  void sequence(Policy&& policy, InputIter first, InputIter last, T init) {
    using DiffSize = typename std::iterator_traits<InputIter>::difference_type;
    sequence(std::forward<Policy>(policy), first, last, init, T(1));
  }

  template<class Policy, class InputIter>
  void sequence(Policy&& policy, InputIter first, InputIter last) {
    using DiffSize = typename std::iterator_traits<InputIter>::difference_type;
    sequence(std::forward<Policy>(policy), first, last, DiffSize(0), DiffSize(1));
  }

  template<class Policy, class InputIter, class Operator>
  void tabulate(Policy&& policy, InputIter first, InputIter last, Operator unary_op) {
    using DiffSize = typename std::iterator_traits<InputIter>::difference_type;
    std::transform(std::forward<Policy>(policy), dpstd::counting_iterator<DiffSize>(0),
      dpstd::counting_iterator<DiffSize>(std::distance(first, last)), first, unary_op);
  }

  template<class InputIter, class Operator>
  void tabulate(InputIter first, InputIter last, Operator unary_op) {
    tabulate(pstl::execution::sycl, first, last, unary_op);
  }

  template<class Policy, class ForwardIt1, class ForwardIt2, class UnaryOperation, class Predicate>
  ForwardIt2 transform_if(Policy&& policy, ForwardIt1 first, ForwardIt1 last, ForwardIt2 result, UnaryOperation unary_op, Predicate pred) {
    using T = typename std::iterator_traits<ForwardIt1>::value_type;
    return std::transform(std::forward<Policy>(policy), first, last, result,
      dpct::transform_if_fun<T, Predicate, UnaryOperation>(pred, unary_op));
  }

  template<class Policy, class ForwardIt1, class ForwardIt2, class ForwardIt3, class UnaryOperation, class Predicate>
  ForwardIt3 transform_if(Policy&& policy, ForwardIt1 first, ForwardIt1 last, ForwardIt2 stencil,
    ForwardIt3 result, UnaryOperation unary_op, Predicate pred) {
    using ZipIterator = typename dpstd::zip_iterator<ForwardIt1, ForwardIt2, ForwardIt3>;
    using T = typename std::iterator_traits<ZipIterator>::value_type;
    return std::transform(std::forward<Policy>(policy),
      dpstd::make_zip_iterator(first, stencil, result),
      dpstd::make_zip_iterator(last, stencil, result),
      result,
      dpct::transform_if_stencil_fun1<T, Predicate, UnaryOperation>(pred, unary_op));
  }

  template<class Policy, class ForwardIt1, class ForwardIt2, class ForwardIt3, class ForwardIt4,
    class BinaryOperation, class Predicate>
    ForwardIt4 transform_if(Policy&& policy, ForwardIt1 first1, ForwardIt1 last1, ForwardIt2 first2, ForwardIt3 stencil,
      ForwardIt4 result, BinaryOperation binary_op, Predicate pred) {
    const auto n = std::distance(first1, last1);
    using ZipIterator = typename dpstd::zip_iterator<ForwardIt1, ForwardIt2, ForwardIt3, ForwardIt4>;
    using T = typename std::iterator_traits<ZipIterator>::value_type;
    return std::transform(std::forward<Policy>(policy),
      dpstd::make_zip_iterator(first1, first2, stencil, result),
      dpstd::make_zip_iterator(last1, last1 + n, stencil, result),
      result,
      dpct::transform_if_zip_stencil_fun<T, Predicate, BinaryOperation>(pred, binary_op));
  }

  template<class ForwardIt1, class ForwardIt2, class ForwardIt3, class ForwardIt4,
    class BinaryOperation, class Predicate>
    ForwardIt4 transform_if(ForwardIt1 first1, ForwardIt1 last1, ForwardIt2 first2, ForwardIt3 stencil,
      ForwardIt4 result, BinaryOperation binary_op, Predicate pred) {
    return transform_if(pstl::execution::sycl, first1, last1, first2, stencil, result, binary_op, pred);
  }

  template<class Policy, class InputIt, class OutputIt, class UnaryOperation>
  OutputIt transform(Policy&& policy, InputIt first, InputIt last, OutputIt result, UnaryOperation unary_op) {
    return std::transform(std::forward<Policy>(policy), first, last, result, unary_op);
  }

  template<class InputIt, class OutputIt, class UnaryOperation>
  OutputIt transform(InputIt first, InputIt last, OutputIt result, UnaryOperation unary_op) {
    return transform(pstl::execution::sycl, first, last, result, unary_op);
  }

  template<class Policy, class InputIt1, class InputIt2, class OutputIt, class UnaryOperation>
  OutputIt transform(Policy&& policy, InputIt1 first1, InputIt1 last1, InputIt2 first2, OutputIt result, UnaryOperation unary_op) {
    return std::transform(std::forward<Policy>(policy), first1, last1, first2, result, unary_op);
  }

  template<class InputIt1, class InputIt2, class OutputIt, class UnaryOperation>
  OutputIt transform(InputIt1 first1, InputIt1 last1, InputIt2 first2, OutputIt result, UnaryOperation unary_op) {
    return transform(pstl::execution::sycl, first1, last1, first2, result, unary_op);
  }

  template<typename Policy, typename InputIt, typename UnaryOperation, typename OutputType, typename BinaryOperation >
  OutputType transform_reduce(Policy&& policy, InputIt first, InputIt last, UnaryOperation unary_op, OutputType init, BinaryOperation binary_op) {
    return std::transform_reduce(std::forward<Policy>(policy), first, last, init, binary_op, unary_op);
  }

  template<typename InputIt, typename UnaryOperation, typename OutputType, typename BinaryOperation >
  OutputType transform_reduce(InputIt first, InputIt last, UnaryOperation unary_op, OutputType init, BinaryOperation binary_op) {
    return transform_reduce(pstl::execution::sycl, first, last, init, binary_op, unary_op);
  }

} // namespace dpct
#endif

// -------------------
// From: dpct/sort.h
// -------------------

#ifndef DPCT_SORT_H
#define DPCT_SORT_H

#include <dpstd/iterators.h>

namespace dpct {

  template<class Policy, class InputIter1, class InputIter2, class Compare>
  void sort_by_key(Policy&& policy, InputIter1 keys_first, InputIter1 keys_last, InputIter2 values_first, Compare comp)
  {
    auto first = dpstd::make_zip_iterator(keys_first, values_first);
    auto last = first + std::distance(keys_first, keys_last);
    std::sort(std::forward<Policy>(policy), first, last, dpct::compare_key_fun<Compare>(comp));
  }

  template<class Policy, class InputIter1, class InputIter2>
  void sort_by_key(Policy&& policy, InputIter1 keys_first, InputIter1 keys_last, InputIter2 values_first)
  {
    sort_by_key(std::forward<Policy>(policy), keys_first, keys_last, values_first, __pstl::internal::pstl_less());
  }

  template<class Policy, class InputIter1, class InputIter2, class Compare>
  void stable_sort_by_key(Policy&& policy, InputIter1 keys_first, InputIter1 keys_last, InputIter2 values_first, Compare comp)
  {
    std::stable_sort(std::forward<Policy>(policy),
      dpstd::make_zip_iterator(keys_first, values_first),
      dpstd::make_zip_iterator(keys_last, values_first + std::distance(keys_first, keys_last)),
      dpct::compare_key_fun<Compare>(comp));
  }

  template<class Policy, class RandomAccessIter1, class RandomAccessIter2>
  void stable_sort_by_key(Policy&& policy, RandomAccessIter1 keys_first, RandomAccessIter1 keys_last, RandomAccessIter2 values_first)
  {
    stable_sort_by_key(std::forward<Policy>(policy), keys_first, keys_last, values_first, __pstl::internal::pstl_less());
  }

} //end namespace dpct
#endif

#endif
