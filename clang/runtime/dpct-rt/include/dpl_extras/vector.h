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

#ifndef __DPCT_VECTOR_H__
#define __DPCT_VECTOR_H__

#include <algorithm>
#include <iterator>
#include <vector>

#include <CL/sycl.hpp>

#include "memory.h"
#include <dpstd/algorithm>
#include <dpstd/execution>

#include <dpct/device.hpp>

namespace dpct {

namespace sycl = cl::sycl;

namespace internal {
template <typename Iter, typename Void = void> // for non-iterators
struct is_iterator : std::false_type {};

template <typename Iter> // For iterators
struct is_iterator<
    Iter,
    typename std::enable_if<
        !std::is_void<typename Iter::iterator_category>::value, void>::type>
    : std::true_type {};

template <typename T> // For pointers
struct is_iterator<T *> : std::true_type {};
} // end namespace internal

#ifndef DPCT_USM_LEVEL_NONE

template <typename T,
          typename Alloc = sycl::usm_allocator<T, sycl::usm::alloc::shared>>
class device_vector {
public:
  using iterator = device_iterator<T>;
  using const_iterator = const iterator;
  using reference = device_reference<T>;
  using const_reference = const reference;
  using value_type = T;
  using pointer = T *;
  using const_pointer = const T *;
  using difference_type =
      typename std::iterator_traits<iterator>::difference_type;
  using size_type = std::size_t;

private:
  Alloc _alloc;
  size_type _size;
  size_type _capacity;
  pointer _storage;

  size_type _min_capacity() const { return size_type(1); }

public:
  template <typename OtherA> operator const std::vector<T, OtherA>() & {
    auto __tmp = std::vector<T, OtherA>(this->size());
    std::copy(dpstd::execution::make_device_policy(get_default_queue()),
              this->begin(), this->end(), __tmp.begin());
    return __tmp;
  }
  device_vector()
      : _alloc(get_default_queue()), _size(0), _capacity(_min_capacity()) {
    _storage = _alloc.allocate(_capacity);
  }
  ~device_vector() /*= default*/ { _alloc.deallocate(_storage, _capacity); };
  explicit device_vector(size_type n) : _alloc(get_default_queue()), _size(n) {
    _capacity = 2 * _size;
    _storage = _alloc.allocate(_capacity);
  }
  explicit device_vector(size_type n, const T &value)
      : _alloc(get_default_queue()), _size(n) {
    _capacity = 2 * _size;
    _storage = _alloc.allocate(_capacity);
    std::fill(dpstd::execution::make_device_policy(get_default_queue()),
              begin(), end(), T(value));
  }
  device_vector(const device_vector &other) : _alloc(get_default_queue()) {
    _size = other.size();
    _capacity = other.capacity();
    _storage = _alloc.allocate(_capacity);
    std::copy(dpstd::execution::make_device_policy(get_default_queue()),
              other.begin(), other.end(), begin());
  }
  device_vector(device_vector &&other)
      : _alloc(get_default_queue()), _size(other.size()),
        _capacity(other.capacity()) {}

  template <typename InputIterator>
  device_vector(
      InputIterator first,
      typename std::enable_if<internal::is_iterator<InputIterator>::value,
                              InputIterator>::type last)
      : _alloc(get_default_queue()) {
    _size = std::distance(first, last);
    _capacity = 2 * _size;
    _storage = _alloc.allocate(_capacity);
    std::copy(dpstd::execution::make_device_policy(get_default_queue()), first,
              last, begin());
  }

  template <typename OtherAlloc>
  device_vector(const device_vector<T, OtherAlloc> &v)
      : _alloc(get_default_queue()), _storage(v.real_begin()), _size(v.size()),
        _capacity(v.capacity()) {}

  template <typename OtherAlloc>
  device_vector(std::vector<T, OtherAlloc> &v)
      : _alloc(get_default_queue()), _size(v.size()) {
    _capacity = 2 * _size;
    _storage = _alloc.allocate(_capacity);
    std::copy(dpstd::execution::make_device_policy(get_default_queue()),
              v.begin(), v.end(), this->begin());
  }

  template <typename OtherAlloc>
  device_vector &operator=(const std::vector<T, OtherAlloc> &v) {
    resize(v.size());
    std::copy(dpstd::execution::make_device_policy(get_default_queue()),
              v.begin(), v.end(), begin());
    return *this;
  }
  device_vector &operator=(const device_vector &other) {
    resize(other.size());
    std::copy(dpstd::execution::make_device_policy(get_default_queue()),
              other.begin(), other.end(), begin());
    return *this;
  }
  device_vector &operator=(device_vector &&other) {
    this->_size = std::move(other._size);
    this->_capacity = std::move(other._capacity);
    this->_storage = std::move(other._storage);
    return *this;
  }
  size_type size() const { return _size; }
  iterator begin() noexcept { return device_iterator<T>(_storage, 0); }
  iterator end() { return device_iterator<T>(_storage, size()); }
  const_iterator begin() const noexcept {
    return device_iterator<T>(_storage, 0);
  }
  const_iterator cbegin() const noexcept { return begin(); }
  const_iterator end() const { return device_iterator<T>(_storage, size()); }
  const_iterator cend() const { return end(); }
  T *real_begin() { return _storage; }
  const T *real_begin() const { return _storage; }
  void swap(device_vector &v) {
    auto temp = std::move(v._storage);
    v._storage = std::move(this->_storage);
    this->_storage = std::move(temp);
    std::swap(_size, v._size);
    std::swap(_capacity, v._capacity);
  }
  reference operator[](size_type n) { return _storage[n]; }
  const_reference operator[](size_type n) const { return _storage[n]; }
  void reserve(size_type n) {
    if (n > capacity()) {
      // allocate buffer for new size
      auto tmp = _alloc.allocate(2 * n);
      // copy content (old buffer to new buffer)
      std::copy(dpstd::execution::make_device_policy(get_default_queue()),
                begin(), end(), tmp);
      // deallocate old memory
      _alloc.deallocate(_storage, _capacity);
      _storage = tmp;
      _capacity = 2 * n;
    }
  }
  void resize(size_type new_size, const T &x = T()) {
    reserve(new_size);
    _size = new_size;
  }
  size_type max_size(void) const {
    return std::numeric_limits<size_type>::max() / sizeof(T);
  }
  size_type capacity() const { return _capacity; }
  const_reference front() const { return *begin(); }
  reference front() { return *begin(); }
  const_reference back(void) const { return *(end() - 1); }
  reference back(void) { return *(end() - 1); }
  pointer data(void) { return _storage; }
  const_pointer data(void) const { return _storage; }
  void shrink_to_fit(void) {
    if (_size != capacity()) {
      auto tmp = _alloc.allocate(_size);
      std::copy(dpstd::execution::make_device_policy(get_default_queue()),
                begin(), end(), tmp);
      _alloc.deallocate(_storage, _capacity);
      _storage = tmp;
      _capacity = _size;
    }
  }
  void assign(size_type n, const T &x) {
    resize(n);
    std::fill(dpstd::execution::make_device_policy(get_default_queue()),
              begin(), begin() + n, x);
  }
  template <typename InputIterator>
  void
  assign(InputIterator first,
         typename std::enable_if<internal::is_iterator<InputIterator>::value,
                                 InputIterator>::type last) {
    auto n = std::distance(first, last);
    resize(n);
    std::copy(dpstd::execution::make_device_policy(get_default_queue()), first,
              last, begin());
  }
  void clear(void) { _size = 0; }
  bool empty(void) const { return (size() == 0); }
  void push_back(const T &x) { insert(end(), size_type(1), x); }
  void pop_back(void) {
    if (_size > 0)
      --_size;
  }
  iterator erase(iterator first, iterator last) {
    auto n = std::distance(first, last);
    if (last == end()) {
      _size = _size - n;
      return end();
    }
    auto m = std::distance(last, end());
    auto tmp = _alloc.allocate(m);
    // copy remainder to temporary buffer.
    std::copy(dpstd::execution::make_device_policy(get_default_queue()), last,
              end(), tmp);
    // override (erase) subsequence in storage.
    std::copy(dpstd::execution::make_device_policy(get_default_queue()), tmp,
              tmp + m, first);
    _alloc.deallocate(tmp, m);
    _size -= n;
    return begin() + first.get_idx() + n;
  }
  iterator erase(iterator pos) { return erase(pos, pos + 1); }
  iterator insert(iterator position, const T &x) {
    auto n = std::distance(begin(), position);
    insert(position, size_type(1), x);
    return begin() + n;
  }
  void insert(iterator position, size_type n, const T &x) {
    if (position == end()) {
      resize(size() + n);
      std::fill(dpstd::execution::make_device_policy(get_default_queue()),
                end() - n, end(), x);
    } else {
      auto i_n = std::distance(begin(), position);
      // allocate temporary storage
      auto m = std::distance(position, end());
      auto tmp = _alloc.allocate(m);
      // copy remainder
      std::copy(dpstd::execution::make_device_policy(get_default_queue()),
                position, end(), tmp);

      resize(size() + n);
      // resizing might invalidate position
      position = begin() + position.get_idx();

      std::fill(dpstd::execution::make_device_policy(get_default_queue()),
                position, position + n, x);

      std::copy(dpstd::execution::make_device_policy(get_default_queue()), tmp,
                tmp + m, position + n);
      _alloc.deallocate(tmp, m);
    }
  }
  template <typename InputIterator>
  void
  insert(iterator position, InputIterator first,
         typename std::enable_if<internal::is_iterator<InputIterator>::value,
                                 InputIterator>::type last) {
    auto n = std::distance(first, last);
    if (position == end()) {
      resize(size() + n);
      std::copy(dpstd::execution::make_device_policy(get_default_queue()),
                first, last, end());
    } else {
      auto m = std::distance(position, end());
      auto tmp = _alloc.allocate(m);

      std::copy(dpstd::execution::make_device_policy(get_default_queue()),
                position, end(), tmp);

      resize(size() + n);
      // resizing might invalidate position
      position = begin() + position.get_idx();

      std::copy(dpstd::execution::make_device_policy(get_default_queue()),
                first, last, position);
      std::copy(dpstd::execution::make_device_policy(get_default_queue()), tmp,
                tmp + m, position + n);
      _alloc.deallocate(tmp, m);
    }
  }
  Alloc get_allocator() const { return _alloc; }
};

#else

template <typename T, typename Alloc = cl::sycl::buffer_allocator>
class device_vector {
public:
  using iterator = device_iterator<T>;
  using const_iterator = const iterator;
  using reference = device_reference<T>;
  using const_reference = const reference;
  using value_type = T;
  using pointer = T *;
  using const_pointer = const T *;
  using difference_type =
      typename std::iterator_traits<iterator>::difference_type;
  using size_type = std::size_t;

private:
  using Buffer = sycl::buffer<T, 1, Alloc>;
  using Range = sycl::range<1>;
  // Allocator stored in buffer
  Buffer _buffer;
  size_type _size;

  size_type _min_capacity() const { return size_type(1); }

public:
  template <typename OtherA> operator const std::vector<T, OtherA>() & {
    auto __tmp = std::vector<T, OtherA>(this->size());
    std::copy(dpstd::execution::dpcpp_default, this->begin(), this->end(),
              __tmp.begin());
    return __tmp;
  }
  device_vector() : _buffer(Range(_min_capacity())), _size(0) {}
  ~device_vector() = default;
  explicit device_vector(size_type n)
      : _buffer(Range(std::max(n, _min_capacity()))), _size(n) {}
  explicit device_vector(size_type n, const T &value)
      : _buffer(Range(std::max(n, _min_capacity()))), _size(n) {
    std::fill(dpstd::execution::dpcpp_default, dpstd::begin(_buffer),
              dpstd::begin(_buffer) + n, T(value));
  }
  device_vector(const device_vector &other)
      : _buffer(other._buffer), _size(other.size()) {}
  device_vector(device_vector &&other)
      : _buffer(std::move(other._buffer)), _size(other.size()) {}

  template <typename InputIterator>
  device_vector(
      InputIterator first,
      typename std::enable_if<internal::is_iterator<InputIterator>::value,
                              InputIterator>::type last)
      : _buffer(first, last), _size(std::distance(first, last)) {}

  template <typename OtherAlloc>
  device_vector(const device_vector<T, OtherAlloc> &v)
      : _buffer(v.real_begin(), v.real_begin() + v.size()), _size(v.size()) {}

  template <typename OtherAlloc>
  device_vector(std::vector<T, OtherAlloc> &v)
      : _buffer(v.begin(), v.end()), _size(v.size()) {}

  device_vector &operator=(const device_vector &other) {
    _size = other.size();
    Buffer tmp{Range(_size)};
    std::copy(dpstd::execution::dpcpp_default,
              dpstd::begin(other.get_buffer()),
              dpstd::end(other.get_buffer()), dpstd::begin(tmp));
    _buffer = tmp;
    return *this;
  }
  device_vector &operator=(device_vector &&other) {
    _size = other.size();
    this->_buffer = std::move(other._buffer);
    return *this;
  }
  template <typename OtherAlloc>
  device_vector &operator=(const std::vector<T, OtherAlloc> &v) {
    this->_buffer = Buffer(v.begin(), v.end());
    _size = v.size();
    return *this;
  }
  Buffer get_buffer() const { return _buffer; }
  size_type size() const { return _size; }
  iterator begin() noexcept { return device_iterator<T>(_buffer, 0); }
  iterator end() { return device_iterator<T>(_buffer, size()); }
  const_iterator begin() const noexcept {
    return device_iterator<T>(_buffer, 0);
  }
  const_iterator cbegin() const noexcept { return begin(); }
  const_iterator end() const { return device_iterator<T>(_buffer, size()); }
  const_iterator cend() const { return end(); }
  T *real_begin() {
    return (_buffer.template get_access<sycl::access::mode::read_write>())
        .get_pointer();
  }
  const T *real_begin() const {
    return const_cast<device_vector *>(this)
        ->_buffer.template get_access<sycl::access::mode::read_write>()
        .get_pointer();
  }
  void swap(device_vector &v) {
    Buffer temp = std::move(v._buffer);
    v._buffer = std::move(this->_buffer);
    this->_buffer = std::move(temp);
    std::swap(_size, v._size);
  }
  reference operator[](size_type n) { return *(begin() + n); }
  const_reference operator[](size_type n) const { return *(begin() + n); }
  void reserve(size_type n) {
    if (n > capacity()) {
      // create new buffer (allocate for new size)
      Buffer tmp{Range(n)};
      // copy content (old buffer to new buffer)
      std::copy(dpstd::execution::dpcpp_default, dpstd::begin(_buffer),
                dpstd::end(_buffer), dpstd::begin(tmp));
      // deallocate old memory
      _buffer = tmp;
    }
  }
  void resize(size_type new_size, const T &x = T()) {
    reserve(new_size);
    _size = new_size;
  }
  size_type max_size(void) const {
    return std::numeric_limits<size_type>::max() / sizeof(T);
  }
  size_type capacity() const { return _buffer.get_count(); }
  const_reference front() const { return *begin(); }
  reference front() { return *begin(); }
  const_reference back(void) const { return *(end() - 1); }
  reference back(void) { return *(end() - 1); }
  pointer data(void) {
    return _buffer.template get_access<sycl::access::mode::read_write>()
        .get_pointer();
  }
  const_pointer data(void) const {
    return const_cast<Buffer>(_buffer)
        .template get_access<sycl::access::mode::read_write>()
        .get_pointer();
  }
  void shrink_to_fit(void) {
    if (_size != capacity()) {
      Buffer tmp{Range(_size)};
      std::copy(dpstd::execution::dpcpp_default, dpstd::begin(_buffer),
                dpstd::end(_buffer), dpstd::begin(tmp));
      _buffer = tmp;
    }
  }
  void assign(size_type n, const T &x) {
    resize(n);
    std::fill(dpstd::execution::dpcpp_default, begin(), begin() + n, x);
  }
  template <typename InputIterator>
  void
  assign(InputIterator first,
         typename std::enable_if<internal::is_iterator<InputIterator>::value,
                                 InputIterator>::type last) {
    auto n = std::distance(first, last);
    resize(n);
    std::copy(dpstd::execution::dpcpp_default, first, last, begin());
  }
  void clear(void) { _size = 0; }
  bool empty(void) const { return (size() == 0); }
  void push_back(const T &x) { insert(end(), size_type(1), x); }
  void pop_back(void) {
    if (_size > 0)
      --_size;
  }
  iterator erase(iterator first, iterator last) {
    auto n = std::distance(first, last);
    if (last == end()) {
      _size = _size - n;
      return end();
    }
    Buffer tmp{Range(std::distance(last, end()))};
    // copy remainder to temporary buffer.
    std::copy(dpstd::execution::dpcpp_default, last, end(), dpstd::begin(tmp));
    // override (erase) subsequence in storage.
    std::copy(dpstd::execution::dpcpp_default, dpstd::begin(tmp),
              dpstd::end(tmp), first);
    resize(_size - n);
    return begin() + first.get_idx() + n;
  }
  iterator erase(iterator pos) { return erase(pos, pos + 1); }
  iterator insert(iterator position, const T &x) {
    auto n = std::distance(begin(), position);
    insert(position, size_type(1), x);
    return begin() + n;
  }
  void insert(iterator position, size_type n, const T &x) {
    if (position == end()) {
      resize(size() + n);
      std::fill(dpstd::execution::dpcpp_default, end() - n, end(), x);
    } else {
      auto i_n = std::distance(begin(), position);
      // allocate temporary storage
      Buffer tmp{Range(std::distance(position, end()))};
      // copy remainder
      std::copy(dpstd::execution::dpcpp_default, position, end(),
                dpstd::begin(tmp));

      resize(size() + n);
      // resizing might invalidate position
      position = begin() + position.get_idx();

      std::fill(dpstd::execution::dpcpp_default, position, position + n, x);

      std::copy(dpstd::execution::dpcpp_default, dpstd::begin(tmp),
                dpstd::end(tmp), position + n);
    }
  }
  template <typename InputIterator>
  void
  insert(iterator position, InputIterator first,
         typename std::enable_if<internal::is_iterator<InputIterator>::value,
                                 InputIterator>::type last) {
    auto n = std::distance(first, last);
    if (position == end()) {
      resize(size() + n);
      std::copy(dpstd::execution::dpcpp_default, first, last, end());
    } else {
      Buffer tmp{Range(std::distance(position, end()))};

      std::copy(dpstd::execution::dpcpp_default, position, end(),
                dpstd::begin(tmp));

      resize(size() + n);
      // resizing might invalidate position
      position = begin() + position.get_idx();

      std::copy(dpstd::execution::dpcpp_default, first, last, position);
      std::copy(dpstd::execution::dpcpp_default, dpstd::begin(tmp),
                dpstd::end(tmp), position + n);
    }
  }
  Alloc get_allocator() const { return _buffer.get_allocator(); }
};

#endif

} // end namespace dpct

#endif
