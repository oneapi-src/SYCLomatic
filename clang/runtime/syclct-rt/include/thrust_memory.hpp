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

//===--- thrust_memory.hpp ------------------------------*- C++ -*---===//

#ifndef __THRUST_MEMORY_HPP_
#define __THRUST_MEMORY_HPP_

// file from runtime team, use SYCLCT_CUSTOM tag changed made by SYCLCT.
#define SYCLCT_CUSTOM

#ifdef SYCLCT_CUSTOM
#include "syclct_memory.hpp"
#endif
namespace thrust {
namespace sycl = cl::sycl;

template <typename T> class device_iterator {
public:
  using value_type = T;
  using difference_type = std::size_t;
  using pointer = T *;
  using reference = T &;
  using iterator_category = std::random_access_iterator_tag;

protected:
  sycl::buffer<T, 1> buffer;
  difference_type idx;

public:
  device_iterator()
      : buffer(sycl::buffer<T, 1>(sycl::range<1>(1))), idx(difference_type{}) {}
  device_iterator(sycl::buffer<T, 1> vec, std::size_t index)
      : buffer(vec), idx(index) {}
  device_iterator(const device_iterator &in) : buffer(in.buffer), idx(in.idx) {}
  device_iterator &operator=(const device_iterator &in) {
    buffer = in.buffer;
    idx = in.idx;
    return *this;
  }
  reference operator*() const {
    auto ptr =
        (const_cast<device_iterator *>(this)
             ->buffer.template get_access<sycl::access::mode::read_write>())
            .get_pointer();
    return *(ptr + idx);
  }
  reference operator[](difference_type i) const { return *(*this + i); }
  device_iterator &operator++() {
    ++idx;
    return *this;
  }
  device_iterator &operator--() {
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
    return {buffer, idx + forward};
  }
  device_iterator &operator+=(difference_type forward) {
    idx += forward;
    return *this;
  }
  device_iterator operator-(difference_type backward) const {
    return {buffer, idx - backward};
  }
  device_iterator &operator-=(difference_type backward) {
    idx -= backward;
    return *this;
  }
  friend device_iterator operator+(difference_type forward,
                                   const device_iterator &it) {
    return it + forward;
  }
  difference_type operator-(const device_iterator &it) const {
    return idx - it.idx;
  }
  bool operator==(const device_iterator &it) const { return *this - it == 0; }
  bool operator!=(const device_iterator &it) const { return !(*this == it); }
  bool operator<(const device_iterator &it) const { return *this - it < 0; }
  bool operator>(const device_iterator &it) const { return it < *this; }
  bool operator<=(const device_iterator &it) const { return !(*this > it); }
  bool operator>=(const device_iterator &it) const { return !(*this < it); }

  sycl::buffer<T, 1> get_buffer() { return buffer; }
  std::size_t get_idx() { return idx; }
};

template <typename T> class device_ptr : public device_iterator<T> {
  using Base = device_iterator<T>;

public:
  template <typename OtherT>
  device_ptr(const device_iterator<OtherT> &in) : Base(in) {}
  template <typename OtherT>
  device_ptr(sycl::buffer<OtherT, 1> in) : Base(in, std::size_t{}) {}

#ifdef SYCLCT_CUSTOM
  /*
  template<typename OtherT>
  device_ptr(OtherT ptr) {
     syclct::memory_manager::allocation
  alloc=syclct::memory_manager::get_instance().translate_ptr((void*) ptr); int
  size=alloc.size; new
  (this)device_ptr(alloc.buffer.reinterpret<T>(cl::sycl::range<1>(size/sizeof(T))));
  }
  */
  template <typename OtherT>
  device_ptr(OtherT ptr)
      : Base(syclct::memory_manager::get_instance()
                 .translate_ptr((void *)ptr)
                 .buffer.reinterpret<T>(
                     cl::sycl::range<1>(syclct::memory_manager::get_instance()
                                            .translate_ptr((void *)ptr)
                                            .size /
                                        sizeof(T))),
             std::size_t{}) {}
#endif

  // needed for device_malloc
  device_ptr(const std::size_t n)
      : Base(sycl::buffer<T, 1>(sycl::range<1>(n)), std::size_t{}) {}

  device_ptr() : Base() {}
  template <typename OtherT>
  device_ptr(const device_ptr<OtherT> &in) : Base(in) {}
  template <typename OtherT>
  device_ptr &operator=(const device_iterator<OtherT> &in) {
    *this = in;
    return *this;
  }
  typename Base::pointer get() const {
    auto res = (const_cast<device_ptr *>(this)
                    ->Base::buffer
                    .template get_access<sycl::access::mode::read_write>())
                   .get_pointer();
    return res + Base::idx;
  }
  device_ptr operator+(typename Base::difference_type forward) const {
    return device_iterator<T>{Base::buffer, Base::idx + forward};
  }
  device_ptr operator-(typename Base::difference_type backward) const {
    return device_iterator<T>{Base::buffer, Base::idx - backward};
  }
  typename Base::difference_type operator-(const device_ptr &it) const {
    return Base::idx - it.idx;
  }

  // TODO: not implemented yet
  // BTW, any use cases couldn't be found
  // template<typename OtherT>
  // device_ptr(OtherT* in){}
};

template <typename T> device_ptr<T> device_malloc(const std::size_t n) {
  return device_ptr<T>(n);
}

template <typename T> void device_free(device_ptr<T> ptr) {}

} // namespace thrust

namespace thrust {
using namespace cl::sycl;

// some helper functions
struct get_buffer {
  // for device_ptr
  template <typename T>
  buffer<T, 1> operator()(thrust::device_ptr<T> it,
                          thrust::device_ptr<T> = thrust::device_ptr<T>{}) {
    return it.get_buffer();
  }
  // for thrust::counting_iterator
  // To not multiply buffers without necessity it was decided to return
  // counting_iterator Counting_iterator already contains idx as dereferenced
  // value. So idx should be 0
  template <typename T>
  tbb::counting_iterator<T>
  operator()(tbb::counting_iterator<T> it,
             tbb::counting_iterator<T> = tbb::counting_iterator<T>(0)) {
    return it;
  }
  // for common iterator
  template <typename Iter>
  buffer<typename std::iterator_traits<Iter>::value_type, 1>
  operator()(Iter first, Iter last) {
    using T = typename std::iterator_traits<Iter>::value_type;
    auto n = last - first;
    std::shared_ptr<T> ret{new T[n], [first, n](T *p) {
                             std::copy(p, p + n, first);
                             delete[] p;
                           }};
    std::copy(first, last, ret.get());
    buffer<T, 1> result(ret, range<1>(n));
    result.set_final_data(ret);
    return result;
  }
};

struct get_idx {
  // for device_ptr
  template <typename T>
  typename std::iterator_traits<thrust::device_ptr<T>>::difference_type
  operator()(thrust::device_ptr<T> it) {
    return it.get_idx();
  }
  // for thrust::counting_iterator
  // To not multiply buffers without necessity it was decided to return
  // counting_iterator Counting_iterator already contains idx as dereferenced
  // value. So idx should be 0
  template <typename T>
  typename std::iterator_traits<tbb::counting_iterator<T>>::difference_type
  operator()(tbb::counting_iterator<T> it) {
    return T{};
  }
  // for common Iterator
  template <typename Iter>
  typename std::iterator_traits<Iter>::difference_type operator()(Iter it1) {
    return typename std::iterator_traits<Iter>::difference_type{};
  }
};
template <access::mode Mode> struct get_access {
  get_access(handler &cgh_) : cgh(cgh_) {}

  // for buffer
  template <typename T>
  accessor<T, 1, Mode, access::target::global_buffer>
  operator()(buffer<T, 1> buf) {
    return buf.template get_access<Mode>(cgh);
  }
  // for tbb::counting_iterator
  template <typename T>
  tbb::counting_iterator<T> operator()(tbb::counting_iterator<T> it) {
    return it;
  }

private:
  handler &cgh;
};

// create queue to work with default device
queue q(default_selector{});

template <typename InAcc, typename OutAcc, typename Size1, typename Size2,
          typename Op>
struct ForParallelFor {
  InAcc in_acc;
  OutAcc out_acc;
  Size1 result_idx;
  Size2 first_idx;
  Op op;
  ForParallelFor(InAcc in_acc_, OutAcc out_acc_, Size1 first_idx_,
                 Size2 result_idx_, Op op_)
      : in_acc(in_acc_), out_acc(out_acc_), result_idx(result_idx_),
        first_idx(first_idx_), op(op_) {}
  void operator()(item<1> it) {
    auto idx = it.get_linear_id();
    out_acc[result_idx + idx] = op(in_acc[first_idx + idx]);
  }
};

// transform
template <typename InputIterator, typename OutputIterator,
          typename UnaryFunction>
OutputIterator transform(InputIterator first, InputIterator last,
                         OutputIterator result, UnaryFunction op) {
  auto device_first = get_buffer()(first, last);
  auto device_result = get_buffer()(result, result + (last - first));
  auto first_idx = get_idx()(first);
  auto result_idx = get_idx()(result);

  std::cout << "Running on " << q.get_device().get_info<info::device::name>()
            << std::endl;
  try {
    q.submit([&](handler &cgh) {
      auto in_acc = get_access<access::mode::read_write>(cgh)(device_first);
      auto out_acc = get_access<access::mode::read_write>(cgh)(device_result);
      cgh.parallel_for(range<1>(last - first),
                       ForParallelFor<decltype(in_acc), decltype(out_acc),
                                      decltype(first_idx), decltype(result_idx),
                                      UnaryFunction>(in_acc, out_acc, first_idx,
                                                     result_idx, op));
    });
    q.wait();
  } catch (exception &ex) {
    std::cout << ex.what() << std::endl;
  }
  return result + (last - first);
}

template <typename T> struct ForCopy {
  T operator()(T x) const { return x; }
};

// copy
template <typename InputIterator, typename OutputIterator>
OutputIterator copy(InputIterator first, InputIterator last,
                    OutputIterator result) {
  using T = typename std::iterator_traits<InputIterator>::value_type;
  return transform(first, last, result, ForCopy<T>());
}

// sequence
template <typename ForwardIterator>
void sequence(ForwardIterator first, ForwardIterator last) {
  using DiffType =
      typename std::iterator_traits<ForwardIterator>::difference_type;
  copy(tbb::counting_iterator<DiffType>(DiffType(0)),
       tbb::counting_iterator<DiffType>(last - first), first);
}
#ifdef SYCLCT_CUSTOM
template <typename RandomAccessIterator1, typename RandomAccessIterator2>
void stable_sort_by_key(RandomAccessIterator1 keys_first,
                        RandomAccessIterator1 keys_last,
                        RandomAccessIterator2 values_first) {
  // todo
  return;
}
#endif

} // namespace thrust
#endif // THRUST_MEMORY_H
