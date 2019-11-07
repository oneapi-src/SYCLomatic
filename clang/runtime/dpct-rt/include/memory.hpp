/******************************************************************************
*
* Copyright 2018 - 2019 Intel Corporation.
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

//===--- memory.hpp ------------------------------*- C++ -*---===//

#ifndef __DPCT_MEMORY_HPP__
#define __DPCT_MEMORY_HPP__

#include "device.hpp"
#include <CL/sycl.hpp>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <utility>

#if defined(__linux__)
#include <sys/mman.h>
#elif defined(_WIN64)
#define NOMINMAX
#include <windows.h>
#else
#error "Only support Windows and Linux."
#endif

namespace dpct {

enum memcpy_direction {
  host_to_host,
  host_to_device,
  device_to_host,
  device_to_device,
  automatic
};
enum memory_attribute {
  device = 0,
  constant,
  local,
};

// Byte type to use.
typedef uint8_t byte_t;

// Buffer type to be used in Memory Management runtime.
typedef cl::sycl::buffer<byte_t> buffer_t;

class memory_manager {
  memory_manager() {
    // Reserved address space, no real memory allocation happens here.
#if defined(__linux__)
        mapped_address_space =
            (byte_t *)mmap(nullptr, mapped_region_size, PROT_NONE,
                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
#elif defined(_WIN64)
        mapped_address_space = (byte_t *)VirtualAlloc(
            NULL,               // NULL specified as the base address parameter
            mapped_region_size, // Size of allocation
            MEM_RESERVE,        // Allocate reserved pages
            PAGE_NOACCESS);     // Protection = no access
#else
#error "Only support Windows and Linux."
#endif
        next_free = mapped_address_space;
      };

public:
  using buffer_id_t = int;

  struct allocation {
    buffer_t buffer;
    byte_t *alloc_ptr;
    size_t size;
  };

  ~memory_manager() {
#if defined(__linux__)
    munmap(mapped_address_space, mapped_region_size);
#elif defined(_WIN64)
    VirtualFree(mapped_address_space, 0, MEM_RELEASE);
#else
#error "Only support Windows and Linux."
#endif
  };

  memory_manager(const memory_manager &) = delete;
  memory_manager& operator=(const memory_manager &) = delete;
  memory_manager(memory_manager &&) = delete;
  memory_manager& operator=(memory_manager &&) = delete;

  // Allocate
  void *mem_alloc(size_t size) {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (next_free + size > mapped_address_space + mapped_region_size) {
      std::abort();
    }
    // Allocation
    cl::sycl::range<1> r(size);
    buffer_t buf(r);
    allocation A{buf, next_free, size};
    // Map allocation to device pointer
    void *result = next_free;
    m_map.emplace(next_free + size, A);
    // Update pointer to the next free space.
    next_free += (size + extra_padding + alignment - 1) & ~(alignment - 1);

    return result;
  }

  // Deallocate
  void mem_free(const void *ptr) {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = get_map_iterator(ptr);
    m_map.erase(it);
  }

  // map: device pointer -> allocation(buffer, alloc_prt, size)
  allocation translate_ptr(const void *ptr) {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = get_map_iterator(ptr);
    return it->second;
  }

  // Check if the pointer represents device pointer or not.
  bool is_device_ptr(const void *ptr) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return (mapped_address_space <= ptr) &&
           (ptr < mapped_address_space + mapped_region_size);
  }

  // Singleton to return the instance memory_manager.
  // Using singleton enables header-only library, but may be problematic for
  // thread safety.
  static memory_manager &get_instance() {
    static memory_manager m;
    return m;
  }

private:
  std::map<byte_t *, allocation> m_map;
  mutable std::mutex m_mutex;
  byte_t *mapped_address_space;
  byte_t *next_free;
  const size_t mapped_region_size = 128ull * 1024 * 1024 * 1024;
  const size_t alignment = 256;
  // This padding may be defined to some positive value to debug
  // out of bound accesses.
  const size_t extra_padding = 0;

  std::map<byte_t *, allocation>::iterator get_map_iterator(const void *ptr) {
    auto it = m_map.upper_bound((byte_t *)ptr);
    if (it == m_map.end()) {
      // Not a device pointer or out of bound.
      std::abort();
    }
    const allocation &alloc = it->second;
    if (ptr < alloc.alloc_ptr) {
      // Out of bound.
      // This may happen if there's a gap between allocations due to alignment
      // or extra padding and pointer points to this gap.
      std::abort();
    }
    return it;
  }
};

// malloc
static inline void dpct_malloc(void **ptr, size_t size, cl::sycl::queue &q) {
#ifdef DPCT_USM_LEVEL_NONE
  *ptr = memory_manager::get_instance().mem_alloc(size * sizeof(byte_t));
#else
  *ptr = cl::sycl::malloc_device(size, q.get_device(), q.get_context());
#endif // DPCT_USM_LEVEL_NONE
}

// malloc
static void dpct_malloc(void **ptr, size_t size) {
  dpct_malloc(ptr, size, get_default_queue());
}

/// Build a cl::sycl::buffer with \p Size, The pointer that \p ptr point to is
/// set as a virtual pointer, which can map to the buffer.
/// \param [out] ptr Point to pointer which need to malloc memory.
/// \param size Size in bytes.
/// \returns no return value.
template <typename T1, typename T2>
static inline void dpct_malloc(T1 **ptr, T2 size) {
  return dpct_malloc(reinterpret_cast<void **>(ptr), static_cast<size_t>(size));
}

/// free
/// \param ptr Point to free.
/// \returns no return value.
static inline void dpct_free(void *ptr) {
  if (ptr) {
#ifdef DPCT_USM_LEVEL_NONE
    memory_manager::get_instance().mem_free(ptr);
#else
    cl::sycl::free(ptr, get_default_queue().get_context());
#endif // DPCT_USM_LEVEL_NONE
  }
}

// dpct_range used to store range infomation
// dpct_range has specialization when Dimesion = 1, 2, 3
template <int Dimesion> class dpct_range;
template <> class dpct_range<0> {
public:
  dpct_range(){};
  size_t size() const { return 1; }
};
template <> class dpct_range<1> {
public:
  dpct_range() : dpct_range(0) {}
  dpct_range(size_t dim1) : range{dim1} {}
  dpct_range(cl::sycl::range<1> range) : range{range[0]} {}
  operator cl::sycl::range<1>() const { return cl::sycl::range<1>(range[0]); }
  size_t size() const { return range[0]; }
  dpct_range<0> low() const { return dpct_range<0>(); }

private:
  size_t range[1];
};
template <> class dpct_range<2> {
public:
  dpct_range() : dpct_range(0, 0) {}
  dpct_range(size_t dim1, size_t dim2) : range{dim1, dim2} {}
  dpct_range(cl::sycl::range<2> range) : range{range[0], range[1]} {}
  operator cl::sycl::range<2>() const {
    return cl::sycl::range<2>(range[0], range[1]);
  }
  size_t size() const { return range[0] * range[1]; }
  dpct_range<1> low() const { return dpct_range<1>(range[1]); }

private:
  size_t range[2];
};
template <> class dpct_range<3> {
public:
  dpct_range() : dpct_range(0, 0, 0) {}
  dpct_range(size_t dim1, size_t dim2, size_t dim3) : range{dim1, dim2, dim3} {}
  dpct_range(cl::sycl::range<3> range) : range{range[0], range[1], range[2]} {}
  operator cl::sycl::range<3>() const {
    return cl::sycl::range<3>(range[0], range[1], range[2]);
  }
  size_t size() const { return range[0] * range[1] * range[2]; }
  dpct_range<2> low() const { return dpct_range<2>(range[1], range[2]); }

private:
  size_t range[3];
};

template <class T, memory_attribute Memory, size_t Dimension>
class dpct_accessor;
template <memory_attribute Memory, class T = byte_t> class memory_traits {
public:
  static constexpr cl::sycl::access::address_space asp =
      (Memory == device)
          ? cl::sycl::access::address_space::global_space
          : ((Memory == constant)
                 ? cl::sycl::access::address_space::constant_space
                 : cl::sycl::access::address_space::local_space);
  static constexpr cl::sycl::access::target target =
      (Memory == device)
          ? cl::sycl::access::target::global_buffer
          : ((Memory == constant) ? cl::sycl::access::target::constant_buffer
                                  : cl::sycl::access::target::local);
  static constexpr cl::sycl::access::mode mode =
      (Memory == constant) ? cl::sycl::access::mode::read
                           : cl::sycl::access::mode::read_write;
  static constexpr size_t type_size = sizeof(T);
  using element_t =
      typename std::conditional<Memory == constant, const T, T>::type;
  using value_t = typename std::remove_cv<T>::type;
  template <size_t Dimension = 1>
  using accessor_t = cl::sycl::accessor<T, Dimension, mode, target>;
#ifdef DPCT_USM_LEVEL_NONE
  // If without USM, must use cl::sycl::multi_ptr.
  using pointer_t = cl::sycl::multi_ptr<T, asp>;
#else
  // Use raw pointer when USM is enabled.
  using pointer_t = T *;
#endif // DPCT_USM_LEVEL_NONE
};

// dpct accessor used as kernel function and device function parameter
template <class T, memory_attribute Memory, size_t Dimension>
class dpct_accessor {
public:
  using memory_t = memory_traits<Memory, T>;
  using element_t = typename memory_t::element_t;
  using pointer_t = typename memory_t::pointer_t;
  using accessor_t = typename memory_t::template accessor_t<Dimension>;
  dpct_accessor(pointer_t data, const dpct_range<Dimension> &range)
      : data(data), range(range) {}
  template <memory_attribute M = Memory>
  dpct_accessor(
      typename std::enable_if<M != local, const accessor_t>::type &acc)
      : dpct_accessor(acc, dpct_range<1>(acc.get_range())) {}
  dpct_accessor(const accessor_t &acc, const dpct_range<Dimension> &range)
      : dpct_accessor(acc.get_pointer(), range) {}
  dpct_accessor<T, Memory, Dimension - 1> operator[](size_t index) const {
    auto low = range.low();
    return dpct_accessor<T, Memory, Dimension - 1>(data + index * low.size(),
                                                   low);
  }

private:
  pointer_t data;
  dpct_range<Dimension> range;
};

// dpct_accessor specialization while Dimension = 1
template <class T, memory_attribute Memory> class dpct_accessor<T, Memory, 1> {
public:
  using memory_t = memory_traits<Memory, T>;
  using element_t = typename memory_t::element_t;
  using pointer_t = typename memory_t::pointer_t;
  using accessor_t = typename memory_t::template accessor_t<1>;
  dpct_accessor(pointer_t data, const dpct_range<1> &range)
      : data(data), range(range) {}
  template <memory_attribute M = Memory>
  dpct_accessor(
      typename std::enable_if<M != local, const accessor_t>::type &acc)
      : dpct_accessor(acc, dpct_range<1>(acc.get_range())) {}
  dpct_accessor(const accessor_t &acc, const dpct_range<1> &range)
      : dpct_accessor(acc.get_pointer(), range) {}
  element_t &operator[](size_t index) const { return *(data + index); }
  element_t &operator*() { return *data; }
  template <class Ty> operator Ty *() { return (Ty *)(&(*data)); }
  template <class ReinterpretT>
  dpct_accessor<ReinterpretT, Memory, 1> reinterpret() {
    return dpct_accessor<ReinterpretT, Memory, 1>(
#ifdef DPCT_USM_LEVEL_NONE
        // Need to get raw pointer if usm disabled.
        (ReinterpretT *)data.get(),
#else
        (ReinterpretT *)data,
#endif // DPCT_USM_LEVEL_NONE
        dpct_range<1>(range.size() * sizeof(T) / sizeof(ReinterpretT)));
  }

private:
  pointer_t data;
  dpct_range<1> range;
};

// dpct_accessor specialization while Dimension = 0
template <class T, memory_attribute Memory> class dpct_accessor<T, Memory, 0> {
public:
  using memory_t = memory_traits<Memory, T>;
  using element_t = typename memory_t::element_t;
  using value_t = typename memory_t::value_t;
  using pointer_t = typename memory_t::pointer_t;
  using accessor_t = typename memory_t::template accessor_t<1>;
  dpct_accessor(pointer_t data, dpct_range<0> range = dpct_range<0>())
      : data(data) {}
  dpct_accessor(const accessor_t &acc) : dpct_accessor(acc.get_pointer()) {}
  template <class Ty> operator Ty() { return static_cast<Ty>(*data); }
  operator element_t &() { return *data; }
  dpct_accessor &operator=(const value_t &val) {
    *data = val;
    return *this;
  }
  T *operator&() { return static_cast<T *>(data); }
  template <class OperandT>
  auto operator+(const OperandT &rhs) -> decltype(T() + OperandT()) {
    return *data + rhs;
  }
  template <class OperandT, memory_attribute OperandM>
  auto operator+(const dpct_accessor<OperandT, OperandM, 0> &rhs)
      -> decltype(T() + OperandT()) {
    return *data + *rhs.data;
  }
  template <class OperandT>
  auto operator-(const OperandT &rhs) -> decltype(T() - OperandT()) {
    return *data - rhs;
  }
  template <class OperandT, memory_attribute OperandM>
  auto operator-(const dpct_accessor<OperandT, OperandM, 0> &rhs)
      -> decltype(T() - OperandT()) {
    return *data - *rhs.data;
  }
  template <class OperandT>
  auto operator*(const OperandT &rhs) -> decltype(T() * OperandT()) {
    return *data * rhs;
  }
  template <class OperandT, memory_attribute OperandM>
  auto operator*(const dpct_accessor<OperandT, OperandM, 0> &rhs)
      -> decltype(T() * OperandT()) {
    return (*data) * (*rhs.data);
  }
  template <class OperandT>
  auto operator/(const OperandT &rhs) -> decltype(T() / OperandT()) {
    return *data / rhs;
  }
  template <class OperandT, memory_attribute OperandM>
  auto operator/(const dpct_accessor<OperandT, OperandM, 0> &rhs)
      -> decltype(T() / OperandT()) {
    return (*data) / (*rhs.data);
  }
  T operator-() { return -(*data); }
  T *operator->() { return data; }

private:
  pointer_t data;
};

// memcpy
static cl::sycl::event dpct_memcpy(cl::sycl::queue &q, void *to_ptr,
                                   const void *from_ptr, size_t size,
                                   memcpy_direction direction) {
#ifdef DPCT_USM_LEVEL_NONE
  auto &mm = memory_manager::get_instance();
  memcpy_direction real_direction = direction;
  switch (direction) {
  case host_to_host:
    assert(!mm.is_device_ptr(from_ptr) && !mm.is_device_ptr(to_ptr));
    break;
  case host_to_device:
    assert(!mm.is_device_ptr(from_ptr) && mm.is_device_ptr(to_ptr));
    break;
  case device_to_host:
    assert(mm.is_device_ptr(from_ptr) && !mm.is_device_ptr(to_ptr));
    break;
  case device_to_device:
    assert(mm.is_device_ptr(from_ptr) && mm.is_device_ptr(to_ptr));
    break;
  case automatic:
    bool from_device = mm.is_device_ptr(from_ptr);
    bool to_device = mm.is_device_ptr(to_ptr);
    if (from_device) {
      if (to_device) {
        real_direction = device_to_device;
      } else {
        real_direction = device_to_host;
      }
    } else {
      if (to_device) {
        real_direction = host_to_device;
      } else {
        real_direction = host_to_host;
      }
    }
    break;
  }

  switch (real_direction) {
  case host_to_host:
    std::memcpy(to_ptr, from_ptr, size);
    return cl::sycl::event();
  case host_to_device: {
    auto alloc = mm.translate_ptr(to_ptr);
    size_t offset = (byte_t *)to_ptr - alloc.alloc_ptr;
    return q.submit([&](cl::sycl::handler &cgh) {
      auto r = cl::sycl::range<1>(size);
      auto o = cl::sycl::id<1>(offset);
      cl::sycl::accessor<byte_t, 1, cl::sycl::access::mode::write,
                         cl::sycl::access::target::global_buffer>
          acc(alloc.buffer, cgh, r, o);
      cgh.copy(from_ptr, acc);
    });
  }
  case device_to_host: {
    auto alloc = mm.translate_ptr(from_ptr);
    size_t offset = (byte_t *)from_ptr - alloc.alloc_ptr;
    return q.submit([&](cl::sycl::handler &cgh) {
      auto r = cl::sycl::range<1>(size);
      auto o = cl::sycl::id<1>(offset);
      cl::sycl::accessor<byte_t, 1, cl::sycl::access::mode::read,
                         cl::sycl::access::target::global_buffer>
          acc(alloc.buffer, cgh, r, o);
      cgh.copy(acc, to_ptr);
    });
  }
  case device_to_device: {
    auto to_alloc = mm.translate_ptr(to_ptr);
    auto from_alloc = mm.translate_ptr(from_ptr);
    size_t to_offset = (byte_t *)to_ptr - to_alloc.alloc_ptr;
    size_t from_offset = (byte_t *)from_ptr - from_alloc.alloc_ptr;
    return q.submit([&](cl::sycl::handler &cgh) {
      auto r = cl::sycl::range<1>(size);
      auto to_o = cl::sycl::id<1>(to_offset);
      auto from_o = cl::sycl::id<1>(from_offset);
      cl::sycl::accessor<byte_t, 1, cl::sycl::access::mode::write,
                         cl::sycl::access::target::global_buffer>
          to_acc(to_alloc.buffer, cgh, r, to_o);
      cl::sycl::accessor<byte_t, 1, cl::sycl::access::mode::read,
                         cl::sycl::access::target::global_buffer>
          from_acc(from_alloc.buffer, cgh, r, from_o);
      cgh.copy(from_acc, to_acc);
    });
  }
  default:
    std::abort();
  }
#else
  return q.memcpy(to_ptr, from_ptr, size);
#endif // DPCT_USM_LEVEL_NONE
}

/// Synchronously copies size bytes from the address specified by from_ptr to
/// the address specified by to_ptr. The value of direction, which is used to
/// specify the copy direction, should be one of host_to_host, host_to_device,
/// device_to_host, device_to_device, or automatic. The function will return
/// after the copy is completed.
///
/// \param to_ptr Pointer to destination memory address.
/// \param from_ptr Pointer to source memory address.
/// \param size Number of bytes to be copied.
/// \param direction Direction of the copy.
/// \returns no return value.
static void dpct_memcpy(void *to_ptr, const void *from_ptr, size_t size,
                        memcpy_direction direction = automatic) {
  dpct_memcpy(get_default_queue(), to_ptr, from_ptr, size, direction).wait();
}

/// Asynchronously copies size bytes from the address specified by from_ptr to
/// the address specified by to_ptr. The value of direction, which is used to
/// specify the copy direction, should be one of host_to_host,
/// host_to_device, device_to_host, device_to_device, or automatic. The return
/// of the function does NOT guarantee the copy is completed
///
/// \param to_ptr Pointer to destination memory address.
/// \param from_ptr Pointer to source memory address.
/// \param size Number of bytes to be copied.
/// \param direction Direction of the copy.
/// \param q Queue to execute the copy task.
/// \returns no return value.
static void async_dpct_memcpy(void *to_ptr, const void *from_ptr, size_t size,
                              memcpy_direction direction = automatic,
                              cl::sycl::queue &q = dpct::get_default_queue()) {
  dpct_memcpy(q, to_ptr, from_ptr, size, direction);
}

static std::pair<buffer_t, size_t> get_buffer_and_offset(const void *ptr) {
  auto alloc = memory_manager::get_instance().translate_ptr(ptr);
  size_t offset = (byte_t *)ptr - alloc.alloc_ptr;
  return std::make_pair(alloc.buffer, offset);
}

// memset
static inline cl::sycl::event dpct_memset(cl::sycl::queue &q, void *devPtr,
                                          int value, size_t count) {
#ifdef DPCT_USM_LEVEL_NONE
  auto &mm = memory_manager::get_instance();
  assert(mm.is_device_ptr(devPtr));
  auto alloc = mm.translate_ptr(devPtr);
  size_t offset = (byte_t *)devPtr - alloc.alloc_ptr;

  return q.submit([&](cl::sycl::handler &cgh) {
    auto r = cl::sycl::range<1>(count);
    auto o = cl::sycl::id<1>(offset);
    cl::sycl::accessor<byte_t, 1, cl::sycl::access::mode::write,
                       cl::sycl::access::target::global_buffer>
        acc(alloc.buffer, cgh, r, o);
    cgh.fill(acc, (byte_t)value);
  });
#else
  return q.memset(devPtr, value, count);
#endif // DPCT_USM_LEVEL_NONE
}

/// Synchronously sets value to the first size bytes starting from dev_ptr. The
/// function will return after the memset operation is completed.
///
/// \param dev_ptr Pointer to the device memory address.
/// \param value Value to be set.
/// \param size Number of bytes to be set to the value.
/// \returns no return value.
static void dpct_memset(void *dev_ptr, int value, size_t size) {
  dpct_memset(get_default_queue(), dev_ptr, value, size).wait();
}

/// Asynchronously sets value to the first size bytes starting from dev_ptr.
/// The return of the function does NOT guarantee the memset operation is
/// completed.
///
/// \param dev_ptr Pointer to the device memory address.
/// \param value Value to be set.
/// \param size Number of bytes to be set to the value.
/// \returns no return value.
static void async_dpct_memset(void *dev_ptr, int value, size_t size,
                              cl::sycl::queue &q = dpct::get_default_queue()) {
  dpct_memset(q, dev_ptr, value, size);
}

// Variable with address space of global or constant
template <class T, memory_attribute Memory, size_t Dimension>
class global_memory {
public:
  using accessor_t =
      typename memory_traits<Memory, T>::template accessor_t<Dimension>;
  using value_t = typename memory_traits<Memory, T>::value_t;
  using dpct_accessor_t = dpct_accessor<T, Memory, Dimension>;

  /// Default constructor
  global_memory() : global_memory(dpct_range<Dimension>()) {}

  /// Constructor of scalar variable with initial value
  global_memory(const dpct_range<Dimension> &range, const value_t &val)
      : global_memory(range) {
    static_assert(Dimension == 0,
                  "only non-array type can be inited with single value");
    dpct_memcpy(memory_ptr, &val, sizeof(T), host_to_device);
  }

  /// Constructor of 1-D array with inlitializer list
  global_memory(const dpct_range<Dimension> &range,
                std::initializer_list<value_t> &&init_list)
      : global_memory(range) {
    static_assert(Dimension == 1,
                  "only 1-D array can be inited with intialization list");
    assert(init_list.size() <= range.size());
    dpct_memcpy(memory_ptr, init_list.begin(), init_list.size() * sizeof(T),
                host_to_device);
  }

  /// Constructor with range
  global_memory(const dpct_range<Dimension> &range_in)
      : size(range_in.size() * sizeof(T)), range(range_in), reference(false),
        memory_ptr(nullptr) {
    static_assert((Memory == device) || (Memory == constant),
                  "Global memory attribute should be constant or device");
    if (size)
      dpct_malloc(&memory_ptr, size);
  }

  /// Constructor with range
  template <class... Args>
  global_memory(Args... Arguments)
      : global_memory(dpct_range<Dimension>(Arguments...)) {}

  ~global_memory() {
    if (memory_ptr && !reference)
      dpct_free(memory_ptr);
  }

  /// The variable is assigned to a device pointer.
  void assign(void *src, size_t size) {
    this->~global_memory();
    new (this) global_memory(src, size);
  }

  /// Get the virtual pointer in host code.
  void *get_ptr() { return memory_ptr; }

#ifdef DPCT_USM_LEVEL_NONE
  template <size_t D = Dimension>
  typename std::enable_if<D == 0, accessor_t>::type
  get_access(cl::sycl::handler &cgh) {
    auto buffer = memory_manager::get_instance()
                      .translate_ptr(memory_ptr)
                      .buffer.template reinterpret<T, 1>(cl::sycl::range<1>(1));
    return accessor_t(buffer, cgh);
  }
  template <size_t D = Dimension>
  typename std::enable_if<D != 0, accessor_t>::type
  get_access(cl::sycl::handler &cgh) {
    return memory_manager::get_instance()
        .translate_ptr(memory_ptr)
        .buffer.template reinterpret<T, Dimension>(range)
        .template get_access<memory_traits<Memory, T>::mode,
                             memory_traits<Memory, T>::target>(cgh);
  }
#else
  dpct_accessor_t get_access(cl::sycl::handler &cgh) {
    return dpct_accessor_t((T *)memory_ptr, range);
  }
#endif // DPCT_USM_LEVEL_NONE

private:
  global_memory(void *memory_ptr, size_t size)
      : size(size), range(size / sizeof(T)), reference(true),
        memory_ptr(memory_ptr) {}

  size_t size;
  dpct_range<Dimension> range;
  bool reference;
  void *memory_ptr;
};

template <class T, size_t Dimension>
using device_memory = global_memory<T, device, Dimension>;
template <class T, size_t Dimension>
using constant_memory = global_memory<T, constant, Dimension>;
} // namespace dpct

#endif // __DPCT_MEMORY_HPP__
