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

#ifndef ONEDPL_STANDARD_POLICIES_ONLY
#include <CL/sycl.hpp>
#endif

// Memory management section:
// device_pointer, device_reference, swap, device_iterator, malloc_device,
// device_new, free_device, device_delete
namespace dpct {

namespace sycl = cl::sycl;

#ifdef DPCT_USM_LEVEL_NONE
template <typename T, sycl::access::mode Mode = sycl::access::mode::read_write,
          typename Allocator = sycl::buffer_allocator>
class device_pointer;
#else
template <typename T> class device_pointer;
#endif

template <typename T> struct device_reference {
  using pointer = device_pointer<T>;
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
class device_pointer {
protected:
  sycl::buffer<T, 1, Allocator> buffer;
  std::size_t idx;

public:
  using value_type = T;
  using difference_type = std::make_signed<std::size_t>::type;
  using pointer = T *;
  using reference = T &;
  using iterator_category = std::random_access_iterator_tag;
  using is_hetero = std::true_type; // required
  using is_passed_directly = std::false_type;
  static constexpr sycl::access::mode mode = Mode; // required

  device_pointer(sycl::buffer<T, 1> in, std::size_t i = 0)
      : buffer(in), idx(i) {}
#ifdef __USE_DPCT
  template <typename OtherT>
  device_pointer(OtherT *ptr)
      : buffer(
            dpct::detail::mem_mgr::instance()
                .translate_ptr(ptr)
                .buffer.template reinterpret<T, 1>(sycl::range<1>(
                    dpct::detail::mem_mgr::instance().translate_ptr(ptr).size /
                    sizeof(T)))),
        idx() {}
#endif
  // needed for malloc_device, count is number of bytes to allocate
  device_pointer(const std::size_t count)
      : buffer(sycl::range<1>(count / sizeof(T))), idx() {}
  device_pointer() {}
  device_pointer(const device_pointer &in) : buffer(in.buffer), idx(in.idx) {}
  pointer get() const {
    auto res =
        (const_cast<device_pointer *>(this)
             ->buffer.template get_access<sycl::access::mode::read_write>())
            .get_pointer();
    return res + idx;
  }
  operator T *() {
    auto res = (buffer.template get_access<sycl::access::mode::read_write>())
                   .get_pointer();
    return res + idx;
  }
  device_pointer operator+(difference_type forward) const {
    return device_pointer{buffer, idx + forward};
  }
  device_pointer operator-(difference_type backward) const {
    return device_pointer{buffer, idx - backward};
  }
  device_pointer &operator+=(difference_type forward) {
    idx += forward;
    return *this;
  }
  device_pointer &operator-=(difference_type backward) {
    idx -= backward;
    return *this;
  }
  device_pointer &operator++() {
    idx += 1;
    return *this;
  }
  device_pointer &operator--() {
    idx -= 1;
    return *this;
  }
  device_pointer operator++(int) {
    device_pointer p(*this);
    idx += 1;
    return p;
  }
  device_pointer operator--(int) {
    device_pointer p(*this);
    idx -= 1;
    return p;
  }
  difference_type operator-(const device_pointer &it) const {
    return idx - it.idx;
  }
  template <typename OtherIterator>
  typename std::enable_if<internal::is_hetero_iterator<OtherIterator>::value,
                          difference_type>::type
  operator-(const OtherIterator &it) const {
    return idx - std::distance(oneapi::dpl::begin(buffer), it);
  }

  std::size_t get_idx() const { return idx; } // required

  sycl::buffer<T, 1, Allocator> get_buffer() { return buffer; } // required
};
#else
template <typename T> class device_iterator;

template <typename T> class device_pointer {
protected:
  T *ptr;

public:
  using value_type = T;
  using difference_type = std::make_signed<std::size_t>::type;
  using pointer = T *;
  using reference = T &;
  using const_reference = const T &;
  using iterator_category = std::random_access_iterator_tag;
  using is_hetero = std::false_type;         // required
  using is_passed_directly = std::true_type; // required

  device_pointer(T *p) : ptr(p) {}
  // needed for malloc_device, count is number of bytes to allocate
  device_pointer(const std::size_t count) {
    cl::sycl::queue default_queue;
    ptr = static_cast<T *>(cl::sycl::malloc_device(
        count, default_queue.get_device(), default_queue.get_context()));
  }
  device_pointer() : ptr(nullptr) {}
  device_pointer &operator=(const device_iterator<T> &in) {
    ptr = static_cast<device_pointer<T>>(in).ptr;
    return *this;
  }
  pointer get() const { return ptr; }
  operator T *() { return ptr; }

  reference operator[](difference_type idx) { return ptr[idx]; }
  reference operator[](difference_type idx) const { return ptr[idx]; }

  device_pointer &operator++() {
    ++ptr;
    return *this;
  }
  device_pointer &operator--() {
    --ptr;
    return *this;
  }
  device_pointer operator++(int) {
    device_pointer it(*this);
    ++(*this);
    return it;
  }
  device_pointer operator--(int) {
    device_pointer it(*this);
    --(*this);
    return it;
  }
  device_pointer operator+(difference_type forward) const {
    return device_pointer{ptr + forward};
  }
  device_pointer operator-(difference_type backward) const {
    return device_pointer{ptr - backward};
  }
  device_pointer &operator+=(difference_type forward) {
    ptr = ptr + forward;
    return *this;
  }
  device_pointer &operator-=(difference_type backward) {
    ptr = ptr - backward;
    return *this;
  }
  difference_type operator-(const device_pointer &it) const {
    return ptr - it.ptr;
  }
};
#endif

#ifdef DPCT_USM_LEVEL_NONE
template <typename T, sycl::access::mode Mode = sycl::access::mode::read_write,
          typename Allocator = sycl::buffer_allocator>
class device_iterator : public device_pointer<T, Mode, Allocator> {
  using Base = device_pointer<T, Mode, Allocator>;

public:
  using value_type = T;
  using difference_type = std::make_signed<std::size_t>::type;
  using pointer = T *;
  using reference = T &;
  using iterator_category = std::random_access_iterator_tag;
  using is_hetero = std::true_type;                // required
  using is_passed_directly = std::false_type;      // required
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
    return Base::idx - std::distance(oneapi::dpl::begin(Base::buffer), it);
  }
  bool operator==(const device_iterator &it) const { return *this - it == 0; }
  bool operator!=(const device_iterator &it) const { return !(*this == it); }
  bool operator<(const device_iterator &it) const { return *this - it < 0; }
  bool operator>(const device_iterator &it) const { return it < *this; }
  bool operator<=(const device_iterator &it) const { return !(*this > it); }
  bool operator>=(const device_iterator &it) const { return !(*this < it); }

  std::size_t get_idx() const { return Base::idx; } // required

  sycl::buffer<T, 1, Allocator> get_buffer() {
    return Base::buffer;
  } // required
};
#else
template <typename T> class device_iterator : public device_pointer<T> {
  using Base = device_pointer<T>;

protected:
  std::size_t idx;

public:
  using value_type = T;
  using difference_type = std::make_signed<std::size_t>::type;
  using pointer = typename Base::pointer;
  using reference = typename Base::reference;
  using iterator_category = std::random_access_iterator_tag;
  using is_hetero = std::false_type;         // required
  using is_passed_directly = std::true_type; // required
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

  reference operator[](difference_type i) { return Base::ptr[idx + i]; }
  reference operator[](difference_type i) const { return Base::ptr[idx + i]; }
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

  std::size_t get_idx() const { return idx; } // required

  device_iterator &get_buffer() { return *this; } // required
};
#endif

template <typename T> device_pointer<T> malloc_device(const std::size_t count) {
  return device_pointer<T>(count);
}
template <typename T>
device_pointer<T> device_new(device_pointer<T> p, const T &value,
                             const std::size_t count = 1) {
  std::vector<T> result(count, value);
  p.buffer = sycl::buffer<T, 1>(result.begin(), result.end());
  return p + count;
}
template <typename T>
device_pointer<T> device_new(device_pointer<T> p, const std::size_t count = 1) {
  return device_new(p, T{}, count);
}
template <typename T>
device_pointer<T> device_new(const std::size_t count = 1) {
  return device_pointer<T>(count);
}

template <typename T> void free_device(device_pointer<T> ptr) {}

template <typename T>
typename std::enable_if<!std::is_trivially_destructible<T>::value, void>::type
device_delete(device_pointer<T> p, const std::size_t count = 1) {
  for (std::size_t i = 0; i < count; ++i) {
    p[i].~T();
  }
}
template <typename T>
typename std::enable_if<std::is_trivially_destructible<T>::value, void>::type
device_delete(device_pointer<T>, const std::size_t count = 1) {}

template <typename T> device_pointer<T> get_device_pointer(T *ptr) {
  return device_pointer<T>(ptr);
}

template <typename T>
device_pointer<T> get_device_pointer(const device_pointer<T> &ptr) {
  return device_pointer<T>(ptr);
}

template <typename T> T *get_raw_pointer(const device_pointer<T> &ptr) {
  return ptr.get();
}

template <typename Pointer> Pointer get_raw_pointer(const Pointer &ptr) {
  return ptr;
}

} // namespace dpct

#endif
