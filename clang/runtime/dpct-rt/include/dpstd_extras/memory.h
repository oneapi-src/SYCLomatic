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

#ifndef __DPCT_MEMORY_H__
#define __DPCT_MEMORY_H__

#ifdef __PSTL_BACKEND_SYCL
#include <CL/sycl.hpp>
#endif

// Memory management section:
// device_ptr, device_reference, swap, device_iterator, device_malloc,
// device_new, device_free, device_delete
namespace dpct {

namespace sycl = cl::sycl;

#ifdef DPCT_USM_LEVEL_NONE
template <typename T, sycl::access::mode Mode = sycl::access::mode::read_write,
          typename Allocator = sycl::buffer_allocator>
class device_ptr;
#else
template <typename T> class device_ptr;
#endif

template <typename T> struct device_reference {
  using pointer = device_ptr<T>;
  using value_type = T;
  template <typename OtherT>
  device_reference(const device_reference<OtherT> &input)
      : value(input.value) {}
  device_reference(const pointer &input) : value((*input).value) {}
  device_reference(value_type &input) : value(input) {}
  template <typename OtherT>
  device_reference &operator=(const device_reference<OtherT> &input) {
    value = input;
    return *this;
  };
  device_reference &operator=(const device_reference &input) {
    T val = input.value;
    value = val;
    return *this;
  };
  device_reference &operator=(const value_type &x) {
    value = x;
    return *this;
  };
  pointer operator&() const { return pointer(&value); };
  operator value_type() const { return T(value); }
  device_reference &operator++() {
    ++value;
    return *this;
  };
  device_reference &operator--() {
    --value;
    return *this;
  };
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
  device_reference &operator+=(const T &input) {
    value += input;
    return *this;
  };
  device_reference &operator-=(const T &input) {
    value -= input;
    return *this;
  };
  device_reference &operator*=(const T &input) {
    value *= input;
    return *this;
  };
  device_reference &operator/=(const T &input) {
    value /= input;
    return *this;
  };
  device_reference &operator%=(const T &input) {
    value %= input;
    return *this;
  };
  device_reference &operator&=(const T &input) {
    value &= input;
    return *this;
  };
  device_reference &operator|=(const T &input) {
    value |= input;
    return *this;
  };
  device_reference &operator^=(const T &input) {
    value ^= input;
    return *this;
  };
  device_reference &operator<<=(const T &input) {
    value <<= input;
    return *this;
  };
  device_reference &operator>>=(const T &input) {
    value >>= input;
    return *this;
  };
  void swap(device_reference &input) {
    T tmp = (*this);
    *this = (input);
    input = (tmp);
  }
  T &value;
};

template <typename T>
void swap(device_reference<T> &x, device_reference<T> &y) {
  x.swap(y);
}

template <typename T> void swap(T &x, T &y) {
  T tmp = x;
  x = y;
  y = tmp;
}

namespace internal {
// struct for checking if iterator is heterogeneous or not
template <typename Iter,
          typename Void = void> // for non-heterogeneous iterators
struct is_hetero_iterator : std::false_type {};

template <typename Iter> // for heterogeneous iterators
struct is_hetero_iterator<
    Iter, typename std::enable_if<Iter::is_hetero::value, void>::type>
    : std::true_type {};
} // namespace internal

#ifdef DPCT_USM_LEVEL_NONE
template <typename T, sycl::access::mode Mode, typename Allocator>
class device_iterator;

template <typename T, sycl::access::mode Mode, typename Allocator>
class device_ptr {
protected:
  sycl::buffer<T, 1, Allocator> buffer;
  std::size_t idx;

public:
  using value_type = T;
  using difference_type = std::make_signed<std::size_t>::type;
  using pointer = T *;
  using reference = T &;
  using iterator_category = std::random_access_iterator_tag;
  using is_hetero = std::true_type;                // required
  static constexpr sycl::access::mode mode = Mode; // required

  device_ptr(sycl::buffer<T, 1> in, std::size_t i = 0) : buffer(in), idx(i) {}
#ifdef __USE_DPCT
  template <typename OtherT>
  device_ptr(OtherT *ptr)
      : buffer(dpct::detail::mem_mgr::instance()
                   .translate_ptr(ptr)
                   .buffer.template reinterpret<T, 1>(sycl::range<1>(
                       dpct::detail::mem_mgr::instance().translate_ptr(ptr).size /
                       sizeof(T)))),
        idx() {}
#endif
  // needed for device_malloc
  device_ptr(const std::size_t n) : buffer(sycl::range<1>(n)), idx() {}
  device_ptr() {}
  device_ptr(const device_ptr &in) : buffer(in.buffer), idx(in.idx) {}
  pointer get() const {
    auto res =
        (const_cast<device_ptr *>(this)
             ->buffer.template get_access<sycl::access::mode::read_write>())
            .get_pointer();
    return res + idx;
  }
  operator T *() {
    auto res = (buffer.template get_access<sycl::access::mode::read_write>())
                   .get_pointer();
    return res + idx;
  }
  device_ptr operator+(difference_type forward) const {
    return device_ptr{buffer, idx + forward};
  }
  device_ptr operator-(difference_type backward) const {
    return device_ptr{buffer, idx - backward};
  }
  difference_type operator-(const device_ptr &it) const { return idx - it.idx; }
  template <typename OtherIterator>
  typename std::enable_if<internal::is_hetero_iterator<OtherIterator>::value,
                          difference_type>::type
  operator-(const OtherIterator &it) const {
    return idx - std::distance(dpstd::begin(buffer), it);
  }

  std::size_t get_idx() { return idx; } // required

  sycl::buffer<T, 1, Allocator> get_buffer() { return buffer; } // required
};
#else
template <typename T> class device_iterator;

template <typename T> class device_ptr {
protected:
  T *ptr;

public:
  using value_type = T;
  using difference_type = std::make_signed<std::size_t>::type;
  using pointer = T *;
  using reference = T &;
  using iterator_category = std::random_access_iterator_tag;
  using is_hetero = std::false_type; // required

  device_ptr(T *p) : ptr(p) {}
  // needed for device_malloc
  device_ptr(const std::size_t n) {
    cl::sycl::queue default_queue;
    ptr = cl::sycl::malloc_device(n, default_queue.get_device(),
                                  default_queue.get_context());
  }
  device_ptr() : ptr(nullptr) {}
  //        template<typename OtherT>
  //        device_ptr(const device_ptr<OtherT>& in) : Base(in) { }
  device_ptr &operator=(const device_iterator<T> &in) {
    ptr = static_cast<device_ptr<T>>(in).ptr;
    return *this;
  }
  pointer get() const { return ptr; }
  operator T *() { return ptr; }
  device_ptr &operator++() {
    ++ptr;
    return *this;
  }
  device_ptr &operator--() {
    --ptr;
    return *this;
  }
  device_ptr operator++(int) {
    device_ptr it(*this);
    ++(*this);
    return it;
  }
  device_ptr operator--(int) {
    device_ptr it(*this);
    --(*this);
    return it;
  }
  device_ptr operator+(difference_type forward) const {
    return device_ptr{ptr + forward};
  }
  device_ptr operator-(difference_type backward) const {
    return device_ptr{ptr - backward};
  }
  difference_type operator-(const device_ptr &it) const { return ptr - it.ptr; }
};
#endif

#ifdef DPCT_USM_LEVEL_NONE
template <typename T, sycl::access::mode Mode = sycl::access::mode::read_write,
          typename Allocator = sycl::buffer_allocator>
class device_iterator : public device_ptr<T, Mode, Allocator> {
  using Base = device_ptr<T, Mode, Allocator>;

public:
  using value_type = T;
  using difference_type = std::make_signed<std::size_t>::type;
  using pointer = T *;
  using reference = T &;
  using iterator_category = std::random_access_iterator_tag;
  using is_hetero = std::true_type;                // required
  static constexpr sycl::access::mode mode = Mode; // required

  device_iterator() : Base() {}
  device_iterator(sycl::buffer<T, 1, Allocator> vec, std::size_t index)
      : Base(vec, index) {}
  template <cl::sycl::access::mode inMode>
  device_iterator(const device_iterator<T, inMode, Allocator> &in)
      : Base(in.buffer, in.idx) {} // required for iter_mode
  device_iterator &operator=(const device_iterator &in) {
    Base::buffer = in.buffer;
    Base::idx = in.idx;
    return *this;
  }

  reference operator*() const {
    auto ptr = (const_cast<device_iterator *>(this)
                    ->buffer.template get_access<mode>())
                   .get_pointer();
    return *(ptr + Base::idx);
  }

  reference operator[](difference_type i) const { return *(*this + i); }
  device_iterator &operator++() {
    ++Base::idx;
    return *this;
  }
  device_iterator &operator--() {
    --Base::idx;
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
    const auto new_idx = Base::idx + forward;
    return {Base::buffer, new_idx};
  }
  device_iterator &operator+=(difference_type forward) {
    Base::idx += forward;
    return *this;
  }
  device_iterator operator-(difference_type backward) const {
    return {Base::buffer, Base::idx - backward};
  }
  device_iterator &operator-=(difference_type backward) {
    Base::idx -= backward;
    return *this;
  }
  friend device_iterator operator+(difference_type forward,
                                   const device_iterator &it) {
    return it + forward;
  }
  difference_type operator-(const device_iterator &it) const {
    return Base::idx - it.idx;
  }
  template <typename OtherIterator>
  typename std::enable_if<internal::is_hetero_iterator<OtherIterator>::value,
                          difference_type>::type
  operator-(const OtherIterator &it) const {
    return Base::idx - std::distance(dpstd::begin(Base::buffer), it);
  }
  bool operator==(const device_iterator &it) const { return *this - it == 0; }
  bool operator!=(const device_iterator &it) const { return !(*this == it); }
  bool operator<(const device_iterator &it) const { return *this - it < 0; }
  bool operator>(const device_iterator &it) const { return it < *this; }
  bool operator<=(const device_iterator &it) const { return !(*this > it); }
  bool operator>=(const device_iterator &it) const { return !(*this < it); }

  std::size_t get_idx() { return Base::idx; } // required

  sycl::buffer<T, 1, Allocator> get_buffer() {
    return Base::buffer;
  } // required
};
#else
template <typename T> class device_iterator : public device_ptr<T> {
  using Base = device_ptr<T>;

protected:
  std::size_t idx;

public:
  using value_type = T;
  using difference_type = std::make_signed<std::size_t>::type;
  using pointer = Base;
  using reference = T &;
  using iterator_category = std::random_access_iterator_tag;
  using is_hetero = std::false_type; // required
  static constexpr sycl::access::mode mode =
      cl::sycl::access::mode::read_write; // required

  device_iterator() : Base(nullptr), idx(0) {}
  device_iterator(T *vec, std::size_t index) : Base(vec), idx(index) {}
  template <cl::sycl::access::mode inMode>
  device_iterator(const device_iterator<T> &in)
      : Base(in.ptr), idx(in.idx) {} // required for iter_mode
  device_iterator &operator=(const device_iterator &in) {
    Base::operator=(in);
    idx = in.idx;
    return *this;
  }

  reference operator*() const { return *(Base::ptr + idx); }

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
    const auto new_idx = idx + forward;
    return {Base::ptr, new_idx};
  }
  device_iterator &operator+=(difference_type forward) {
    idx += forward;
    return *this;
  }
  device_iterator operator-(difference_type backward) const {
    return {Base::ptr, idx - backward};
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

  template <typename OtherIterator>
  typename std::enable_if<internal::is_hetero_iterator<OtherIterator>::value,
                          difference_type>::type
  operator-(const OtherIterator &it) const {
    return idx - it.get_idx();
  }

  bool operator==(const device_iterator &it) const { return *this - it == 0; }
  bool operator!=(const device_iterator &it) const { return !(*this == it); }
  bool operator<(const device_iterator &it) const { return *this - it < 0; }
  bool operator>(const device_iterator &it) const { return it < *this; }
  bool operator<=(const device_iterator &it) const { return !(*this > it); }
  bool operator>=(const device_iterator &it) const { return !(*this < it); }

  std::size_t get_idx() { return idx; } // required

  device_iterator &get_buffer() { return *this; } // required
};
#endif

template <typename T> device_ptr<T> device_malloc(const std::size_t n) {
  return device_ptr<T>(n / sizeof(T));
}
template <typename T>
device_ptr<T> device_new(device_ptr<T> p, const T &exemplar,
                         const std::size_t n = 1) {
  std::vector<T> result(n, exemplar);
  p.buffer = sycl::buffer<T, 1>(result.begin(), result.end());
  return p + n;
}
template <typename T>
device_ptr<T> device_new(device_ptr<T> p, const std::size_t n = 1) {
  return device_new(p, T{}, n);
}
template <typename T> device_ptr<T> device_new(const std::size_t n = 1) {
  return device_ptr<T>(n);
}

template <typename T> void device_free(device_ptr<T> ptr) {}

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

template <typename T> device_ptr<T> device_pointer_cast(T *ptr) {
  return device_ptr<T>(ptr);
}

template <typename T>
device_ptr<T> device_pointer_cast(const device_ptr<T> &ptr) {
  return device_ptr<T>(ptr);
}

template <typename T> T *raw_pointer_cast(const device_ptr<T> &ptr) {
  return ptr.get();
}

template <typename Pointer> Pointer raw_pointer_cast(const Pointer &ptr) {
  return ptr;
}

template <typename T> using device_allocator = sycl::buffer_allocator;
template <typename T> using device_malloc_allocator = sycl::buffer_allocator;
template <typename T> using device_new_allocator = sycl::buffer_allocator;

} // namespace dpct

#endif
