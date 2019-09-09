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

#ifndef __DPCT_MEMORY_H
#define __DPCT_MEMORY_H

#ifdef __PSTL_BACKEND_SYCL
#include <CL/sycl.hpp>
#endif

// Memory management section:
// device_ptr, device_reference, swap, device_iterator, device_malloc,
// device_new, device_free, device_delete
namespace dpct {

namespace sycl = cl::sycl;

template <typename T> class device_ptr;

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

template <typename T>
using device_iterator =
    dpstd::__internal::sycl_iterator<sycl::access::mode::read_write, T>;

template <typename T> class device_ptr : public device_iterator<T> {
  using Base = device_iterator<T>;

public:
  template <typename OtherT>
  device_ptr(const device_iterator<OtherT> &in) : Base(in) {}
  template <typename OtherT>
  device_ptr(sycl::buffer<OtherT, 1> in) : Base(in, std::size_t{}) {}
#ifdef __USE_DPCT
  template <typename OtherT>
  device_ptr(OtherT ptr)
      : Base(cl::sycl::buffer<T, 1>(cl::sycl::range<1>(
                 dpct::memory_manager::get_instance().translate_ptr(ptr).size /
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

};

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
