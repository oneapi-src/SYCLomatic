//==---- memory.hpp -------------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief Utility functions to provide pointer-like memory operation when USM
 * level is none.
 *
 * @copyright Copyright (C) Intel Corporation
 *
 */
#ifndef __DPCT_MEMORY_HPP__
#define __DPCT_MEMORY_HPP__

#include "device.hpp"
#include <sycl/sycl.hpp>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <map>
#include <utility>
#include <thread>
#include <type_traits>

#if defined(__linux__)
#include <sys/mman.h>
#elif defined(_WIN64)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#error "Only support Windows and Linux."
#endif

namespace dpct {
/**
 * @brief Enum for memory copy direction.
 */
enum memcpy_direction {
  host_to_host,
  host_to_device,
  device_to_host,
  device_to_device,
  automatic
};
/**
 * @brief Enum for memory region.
 */
enum memory_region {
  global = 0,  // device global memory
  constant,    // device constant memory
  local,       // device local memory
  shared,      // memory which can be accessed by host and device
};
/**
 * @brief Alias of uint8_t.
 */
typedef uint8_t byte_t;

/**
 * @brief Alias of \a sycl::buffer<byte_t>, is used in mem_mgr.
 */
typedef sycl::buffer<byte_t> buffer_t;
/**
 * @class pitched_data
 * @brief Pitched 2D/3D memory data.
 */
class pitched_data {
public:
  /**
   * @brief The default constructor.
   */
  pitched_data() : pitched_data(nullptr, 0, 0, 0) {}
  /**
   * @brief The constructor with existing data.
   * @param [in] data The pointer to the data.
   * @param [in] pitch The pitch size, including padding bytes of the data.
   * @param [in] x The width of the data.
   * @param [in] y The height of the data.
   */
  pitched_data(void *data, size_t pitch, size_t x, size_t y)
      : _data(data), _pitch(pitch), _x(x), _y(y) {}
  /**
   * @brief Gets the pointer to the data.
   * @return The pointer to the data.
   */
  void *get_data_ptr() { return _data; }
  /**
   * @brief Sets the pointer to the data.
   * @param [in] data The pointer to the data.
   */
  void set_data_ptr(void *data) { _data = data; }
  /**
   * @brief Gets The pitch size, including padding bytes of the data.
   * @return The pitch size, including padding bytes of the data.
   */
  size_t get_pitch() { return _pitch; }
  /**
   * @brief Sets The pitch size, including padding bytes of the data.
   * @param [in] pitch The pitch size, including padding bytes of the data.
   */
  void set_pitch(size_t pitch) { _pitch = pitch; }
  /**
   * @brief Gets the width of the data.
   * @return The width of the data.
   */
  size_t get_x() { return _x; }
  /**
   * @brief Sets the width of the data.
   * @param [in] x The width of the data.
   */
  void set_x(size_t x) { _x = x; };
  /**
   * @brief Gets the height of the data.
   * @return The height of the data.
   */
  size_t get_y() { return _y; }
  /**
   * @brief Sets the height of the data.
   * @param [in] y The height of the data.
   */
  void set_y(size_t y) { _y = y; }

private:
  void *_data;
  size_t _pitch, _x, _y;
};

namespace detail {
/**
 * @class mem_mgr
 * @brief Memory manager to provide pointer like memory operation.
 */
class mem_mgr {
  /**
   * @brief The default constructor which initializes the address map.
   */
  mem_mgr() {
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
  /**
   * @class allocation
   * @brief Data structure to record a buffer and its virtual pointer address.
   */
  struct allocation {
    buffer_t buffer;
    byte_t *alloc_ptr;
    size_t size;
  };
  /**
   * @brief The default destructor which resets the address map.
   */
  ~mem_mgr() {
#if defined(__linux__)
    munmap(mapped_address_space, mapped_region_size);
#elif defined(_WIN64)
    VirtualFree(mapped_address_space, 0, MEM_RELEASE);
#else
#error "Only support Windows and Linux."
#endif
  };

  mem_mgr(const mem_mgr &) = delete;
  mem_mgr &operator=(const mem_mgr &) = delete;
  mem_mgr(mem_mgr &&) = delete;
  mem_mgr &operator=(mem_mgr &&) = delete;

  /**
   * @brief Performs memory allocation operation.
   * @param [in] size The memory size to be allocated.
   * @return The virtual device pointer of the memory space.
   */
  void *mem_alloc(size_t size) {
    if (!size)
      return nullptr;
    std::lock_guard<std::mutex> lock(m_mutex);
    if (next_free + size > mapped_address_space + mapped_region_size) {
      throw std::runtime_error("dpct_malloc: out of memory for virtual memory pool");
    }
    // Allocation
    sycl::range<1> r(size);
    buffer_t buf(r);
    allocation A{buf, next_free, size};
    // Map allocation to device pointer
    void *result = next_free;
    m_map.emplace(next_free + size, A);
    // Update pointer to the next free space.
    next_free += (size + extra_padding + alignment - 1) & ~(alignment - 1);

    return result;
  }

  /**
   * @brief Performs memory free operation.
   * @param [in] ptr The virtual device pointer to be free.
   */
  void mem_free(const void *ptr) {
    if (!ptr)
      return;
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = get_map_iterator(ptr);
    m_map.erase(it);
  }

  /**
   * @brief Retrieve the allocation information with a virtual device pointer.
   * @param [in] ptr The target virtual pointer.
   * @return The allocation infomation of the pointer.
   */
  allocation translate_ptr(const void *ptr) {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = get_map_iterator(ptr);
    return it->second;
  }
  /**
   * @brief Checks if the pointer is a virtual device pointer.
   * @param [in] ptr The virtual pointer to be checked.
   * @return true if the pointer is a virtual device pointer.
   */
  bool is_device_ptr(const void *ptr) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return (mapped_address_space <= ptr) &&
           (ptr < mapped_address_space + mapped_region_size);
  }
  /**
   * @brief Gets the instance of memory manager singleton.
   * @return An instance of memory manager singleton.
   */
  static mem_mgr &instance() {
    static mem_mgr m;
    return m;
  }

private:
  std::map<byte_t *, allocation> m_map;
  mutable std::mutex m_mutex;
  byte_t *mapped_address_space;
  byte_t *next_free;
  const size_t mapped_region_size = 128ull * 1024 * 1024 * 1024;
  const size_t alignment = 256;
  /// This padding may be defined to some positive value to debug
  /// out of bound accesses.
  const size_t extra_padding = 0;

  std::map<byte_t *, allocation>::iterator get_map_iterator(const void *ptr) {
    auto it = m_map.upper_bound((byte_t *)ptr);
    if (it == m_map.end()) {
      // Not a virtual pointer.
      throw std::runtime_error("can not get buffer from non-virtual pointer");
    }
    const allocation &alloc = it->second;
    if (ptr < alloc.alloc_ptr) {
      // Out of bound.
      // This may happen if there's a gap between allocations due to alignment
      // or extra padding and pointer points to this gap.
      throw std::runtime_error("invalid virtual pointer");
    }
    return it;
  }
};

template <class T, memory_region Memory, size_t Dimension> class accessor;

/**
 * @class memory_traits
 */
template <memory_region Memory, class T = byte_t> class memory_traits {
public:
  static constexpr sycl::access::target target =
      sycl::access::target::device;
  static constexpr sycl::access_mode mode =
      (Memory == constant) ? sycl::access_mode::read
                           : sycl::access_mode::read_write;
  static constexpr size_t type_size = sizeof(T);
  using element_t =
      typename std::conditional<Memory == constant, const T, T>::type;
  using value_t = typename std::remove_cv<T>::type;
  template <size_t Dimension = 1>
  using accessor_t = typename std::conditional<
      Memory == local, sycl::local_accessor<value_t, Dimension>,
      sycl::accessor<T, Dimension, mode, target>>::type;
  using pointer_t = T *;
};
/**
 * @brief Performs pointer-like moemory allocation when USM level is none.
 * @param [in] size The memory size to be allocated.
 * @param [in] q The queue to use the memory.
 * @return The virtaul device pointer to the allocated memory.
 */
static inline void *dpct_malloc(size_t size, sycl::queue &q) {
#ifdef DPCT_USM_LEVEL_NONE
  return mem_mgr::instance().mem_alloc(size * sizeof(byte_t));
#else
  return sycl::malloc_device(size, q.get_device(), q.get_context());
#endif // DPCT_USM_LEVEL_NONE
}

#define PITCH_DEFAULT_ALIGN(x) (((x) + 31) & ~(0x1F))
/**
 * @brief Performs pointer-like moemory allocation for pitched data when USM
 * level is none.
 * @param [in] pitch The pitch size, including padding bytes of the memory.
 * @param [in] x The width of the memory.
 * @param [in] y The height of the memory.
 * @param [in] z The depth of the memory.
 * @param [in] q The queue to use the memory.
 * @return The virtaul device pointer to the allocated memory.
 */
static inline void *dpct_malloc(size_t &pitch, size_t x, size_t y, size_t z,
                                sycl::queue &q) {
  pitch = PITCH_DEFAULT_ALIGN(x);
  return dpct_malloc(pitch * y * z, q);
}

/**
 * @brief Sets \p value to the first \p size elements starting from \p dev_ptr in \p q.
 * @tparam valueT The type of the element to be set.
 * @param [in] q The queue in which the operation is done.
 * @param [in] dev_ptr Pointer to the virtual device memory address.
 * @param [in] value The value to be set.
 * @param [in] size Number of elements to be set to the value.
 * @return An event representing the memset operation.
 */
template <typename valueT>
static inline sycl::event dpct_memset(sycl::queue &q, void *dev_ptr,
                                      valueT value, size_t size) {
#ifdef DPCT_USM_LEVEL_NONE
  auto &mm = mem_mgr::instance();
  assert(mm.is_device_ptr(dev_ptr));
  auto alloc = mm.translate_ptr(dev_ptr);
  size_t offset = (valueT *)dev_ptr - (valueT *)alloc.alloc_ptr;

  return q.submit([&](sycl::handler &cgh) {
    auto r = sycl::range<1>(size);
    auto o = sycl::id<1>(offset);
    auto new_buffer = alloc.buffer.reinterpret<valueT>(
        sycl::range<1>(alloc.size / sizeof(valueT)));
    sycl::accessor<valueT, 1, sycl::access_mode::write,
                   sycl::access::target::device>
        acc(new_buffer, cgh, r, o);
    cgh.fill(acc, value);
  });
#else
  return q.fill(dev_ptr, value, size);
#endif // DPCT_USM_LEVEL_NONE
}

/**
 * @brief Sets \p value to the 3D memory region pointed by \p data in \p q.
 * @tparam valueT The type of the element to be set.
 * @param [in] q The queue in which the operation is done.
 * @param [in] data Pointer to the pitched device memory region.
 * @param [in] value The value to be set.
 * @param [in] size 3D memory region by number of elements.
 * @return An event list representing the memset operations.
 */
template<typename valueT>
static inline std::vector<sycl::event>
dpct_memset(sycl::queue &q, pitched_data data, valueT value,
            sycl::range<3> size) {
  std::vector<sycl::event> event_list;
  size_t slice = data.get_pitch() * data.get_y();
  unsigned char *data_surface = (unsigned char *)data.get_data_ptr();
  for (size_t z = 0; z < size.get(2); ++z) {
    unsigned char *data_ptr = data_surface;
    for (size_t y = 0; y < size.get(1); ++y) {
      event_list.push_back(dpct_memset(q, data_ptr, value, size.get(0)));
      data_ptr += data.get_pitch();
    }
    data_surface += slice;
  }
  return event_list;
}

/**
 * @brief Sets \p val to the pitched 2D memory region pointed by \p ptr in \p q.
 * @tparam valueT The type of the element to be set.
 * @param [in] q The queue in which the operation is done.
 * @param [in] ptr Pointer to the virtual device memory.
 * @param [in] pitch The pitch size by number of elements, including padding.
 * @param [in] val The value to be set.
 * @param [in] x The width of memory region by number of elements.
 * @param [in] y The height of memory region by number of elements.
 * @return An event list representing the memset operations.
 */
template<typename valueT>
static inline std::vector<sycl::event>
dpct_memset(sycl::queue &q, void *ptr, size_t pitch, valueT val, size_t x,
            size_t y) {
  return dpct_memset(q, pitched_data(ptr, pitch, x, 1), val,
                     sycl::range<3>(x, y, 1));
}
/**
 * @brief Enum for pointer access attribute.
 */
enum class pointer_access_attribute {
  host_only = 0,
  device_only,
  host_device,
  end
};
/**
 * @brief Gets the pointer access attribute of a device pointer.
 * @param [in] q The queue which contains the memory.
 * @param [in] ptr Pointer to the virtual device memory.
 * @return The pointer access attribute.
 */
static pointer_access_attribute get_pointer_attribute(sycl::queue &q,
                                                      const void *ptr) {
#ifdef DPCT_USM_LEVEL_NONE
  return mem_mgr::instance().is_device_ptr(ptr)
             ? pointer_access_attribute::device_only
             : pointer_access_attribute::host_only;
#else
  switch (sycl::get_pointer_type(ptr, q.get_context())) {
  case sycl::usm::alloc::unknown:
    return pointer_access_attribute::host_only;
  case sycl::usm::alloc::device:
    return pointer_access_attribute::device_only;
  case sycl::usm::alloc::shared:
  case sycl::usm::alloc::host:
    return pointer_access_attribute::host_device;
  }
#endif
}
/**
 * @brief Deduces the memory copy direction.
 * @param [in] q The queue in which the operation is done.
 * @param [in] to_ptr Pointer to the destination virtual device memory.
 * @param [in] from_ptr Pointer to the source virtual device memory.
 * @param [in] dir The user-specified memory copy direction.
 * @return The deduction result when \p dir is memcpy_direction::automatic.
 * Otherwise, the result is \p dir.
 */
static memcpy_direction deduce_memcpy_direction(sycl::queue &q, void *to_ptr,
                                             const void *from_ptr,
                                             memcpy_direction dir) {
  switch (dir) {
  case memcpy_direction::host_to_host:
  case memcpy_direction::host_to_device:
  case memcpy_direction::device_to_host:
  case memcpy_direction::device_to_device:
    return dir;
  case memcpy_direction::automatic: {
    // table[to_attribute][from_attribute]
    static const memcpy_direction
        direction_table[static_cast<unsigned>(pointer_access_attribute::end)]
                       [static_cast<unsigned>(pointer_access_attribute::end)] =
                           {{memcpy_direction::host_to_host,
                             memcpy_direction::device_to_host,
                             memcpy_direction::host_to_host},
                            {memcpy_direction::host_to_device,
                             memcpy_direction::device_to_device,
                             memcpy_direction::device_to_device},
                            {memcpy_direction::host_to_host,
                             memcpy_direction::device_to_device,
                             memcpy_direction::device_to_device}};
    return direction_table[static_cast<unsigned>(get_pointer_attribute(
        q, to_ptr))][static_cast<unsigned>(get_pointer_attribute(q, from_ptr))];
  }
  default:
    throw std::runtime_error("dpct_memcpy: invalid direction value");
  }
}
/**
 * @brief Performs memcpy between 2 pointers. Supports the virtual device pointer from dpct_malloc().
 * @param [in] q The queue in which the operation is done.
 * @param [in] to_ptr Pointer to the destination virtual device memory.
 * @param [in] from_ptr Pointer to the source virtual device memory.
 * @param [in] size The memory size to be copied.
 * @param [in] direction The memory copy direction.
 * @param [in] dep_events The events this operation depends on.
 * @return An event representing the memset operations.
 */
static sycl::event
dpct_memcpy(sycl::queue &q, void *to_ptr, const void *from_ptr, size_t size,
            memcpy_direction direction,
            const std::vector<sycl::event> &dep_events = {}) {
  if (!size)
    return sycl::event{};
#ifdef DPCT_USM_LEVEL_NONE
  auto &mm = mem_mgr::instance();
  auto real_direction = deduce_memcpy_direction(q, to_ptr, from_ptr, direction);

  switch (real_direction) {
  case host_to_host:
    return q.submit([&](sycl::handler &cgh) {
      cgh.depends_on(dep_events);
      cgh.host_task([=] { std::memcpy(to_ptr, from_ptr, size); });
    });
  case host_to_device: {
    auto alloc = mm.translate_ptr(to_ptr);
    size_t offset = (byte_t *)to_ptr - alloc.alloc_ptr;
    return q.submit([&](sycl::handler &cgh) {
      cgh.depends_on(dep_events);
      auto r = sycl::range<1>(size);
      auto o = sycl::id<1>(offset);
      sycl::accessor<byte_t, 1, sycl::access_mode::write,
                          sycl::access::target::device>
          acc(alloc.buffer, cgh, r, o);
      cgh.copy(from_ptr, acc);
    });
  }
  case device_to_host: {
    auto alloc = mm.translate_ptr(from_ptr);
    size_t offset = (byte_t *)from_ptr - alloc.alloc_ptr;
    return q.submit([&](sycl::handler &cgh) {
      cgh.depends_on(dep_events);
      auto r = sycl::range<1>(size);
      auto o = sycl::id<1>(offset);
      sycl::accessor<byte_t, 1, sycl::access_mode::read,
                          sycl::access::target::device>
          acc(alloc.buffer, cgh, r, o);
      cgh.copy(acc, to_ptr);
    });
  }
  case device_to_device: {
    auto to_alloc = mm.translate_ptr(to_ptr);
    auto from_alloc = mm.translate_ptr(from_ptr);
    size_t to_offset = (byte_t *)to_ptr - to_alloc.alloc_ptr;
    size_t from_offset = (byte_t *)from_ptr - from_alloc.alloc_ptr;
    return q.submit([&](sycl::handler &cgh) {
      cgh.depends_on(dep_events);
      auto r = sycl::range<1>(size);
      auto to_o = sycl::id<1>(to_offset);
      auto from_o = sycl::id<1>(from_offset);
      sycl::accessor<byte_t, 1, sycl::access_mode::write,
                          sycl::access::target::device>
          to_acc(to_alloc.buffer, cgh, r, to_o);
      sycl::accessor<byte_t, 1, sycl::access_mode::read,
                          sycl::access::target::device>
          from_acc(from_alloc.buffer, cgh, r, from_o);
      cgh.copy(from_acc, to_acc);
    });
  }
  default:
    throw std::runtime_error("dpct_memcpy: invalid direction value");
  }
#else
  return q.memcpy(to_ptr, from_ptr, size, dep_events);
#endif // DPCT_USM_LEVEL_NONE
}

/**
 * @brief Gets the copy range and make sure it will not exceed range for 3D memory copy.
 * @param [in] size The memory size to be copied.
 * @param [in] slice ????
 * @param [in] pitch The pitch size, including padding bytes.
 * @return The copy range.
 */
static inline size_t get_copy_range(sycl::range<3> size, size_t slice,
                                    size_t pitch) {
  return slice * (size.get(2) - 1) + pitch * (size.get(1) - 1) + size.get(0);
}
/**
 * @brief Gets the offset for 3D memory copy.
 * @param [in] id Start point of copy.
 * @param [in] slice ????
 * @param [in] pitch The pitch size, including padding bytes.
 * @return The copy offset.
 */
static inline size_t get_offset(sycl::id<3> id, size_t slice,
                                    size_t pitch) {
  return slice * id.get(2) + pitch * id.get(1) + id.get(0);
}
/**
 * @brief Performs 3D memcpy between 2 pointers. Supports the virtual device pointer from dpct_malloc().
 * @param [in] q The queue in which the operation is done.
 * @param [in] to_ptr Pointer to the destination virtual device memory.
 * @param [in] from_ptr Pointer to the source virtual device memory.
 * @param [in] to_range The memory range of the destination.
 * @param [in] from_range The memory range of the source.
 * @param [in] to_id The offset of the copy destination.
 * @param [in] from_id The offset of the source.
 * @param [in] size The memory size to be copied.
 * @param [in] direction The memory copy direction.
 * @param [in] dep_events The events this operation depends on.
 * @return An event list representing the memset operations.
 */
static inline std::vector<sycl::event>
dpct_memcpy(sycl::queue &q, void *to_ptr, const void *from_ptr,
            sycl::range<3> to_range, sycl::range<3> from_range,
            sycl::id<3> to_id, sycl::id<3> from_id,
            sycl::range<3> size, memcpy_direction direction,
            const std::vector<sycl::event> &dep_events = {}) {
  // RAII for host pointer
  class host_buffer {
    void *_buf;
    size_t _size;
    sycl::queue &_q;
    const std::vector<sycl::event> &_deps; // free operation depends

  public:
    host_buffer(size_t size, sycl::queue &q,
                const std::vector<sycl::event> &deps)
        : _buf(std::malloc(size)), _size(size), _q(q), _deps(deps) {}
    void *get_ptr() const { return _buf; }
    size_t get_size() const { return _size; }
    ~host_buffer() {
      if (_buf) {
        _q.submit([&](sycl::handler &cgh) {
          cgh.depends_on(_deps);
          cgh.host_task([buf = _buf] { std::free(buf); });
        });
      }
    }
  };
  std::vector<sycl::event> event_list;

  size_t to_slice = to_range.get(1) * to_range.get(0),
         from_slice = from_range.get(1) * from_range.get(0);
  unsigned char *to_surface =
      (unsigned char *)to_ptr + get_offset(to_id, to_slice, to_range.get(0));
  const unsigned char *from_surface =
      (const unsigned char *)from_ptr +
      get_offset(from_id, from_slice, from_range.get(0));

  if (to_slice == from_slice && to_slice == size.get(1) * size.get(0)) {
    return {dpct_memcpy(q, to_surface, from_surface, to_slice * size.get(2),
                        direction, dep_events)};
  }
  direction = deduce_memcpy_direction(q, to_ptr, from_ptr, direction);
  size_t size_slice = size.get(1) * size.get(0);
  switch (direction) {
  case host_to_host:
    for (size_t z = 0; z < size.get(2); ++z) {
      unsigned char *to_ptr = to_surface;
      const unsigned char *from_ptr = from_surface;
      if (to_range.get(0) == from_range.get(0) &&
          to_range.get(0) == size.get(0)) {
        event_list.push_back(dpct_memcpy(q, to_ptr, from_ptr, size_slice,
                                         direction, dep_events));
      } else {
        for (size_t y = 0; y < size.get(1); ++y) {
          event_list.push_back(dpct_memcpy(q, to_ptr, from_ptr, size.get(0),
                                           direction, dep_events));
          to_ptr += to_range.get(0);
          from_ptr += from_range.get(0);
        }
      }
      to_surface += to_slice;
      from_surface += from_slice;
    }
    break;
  case host_to_device: {
    host_buffer buf(get_copy_range(size, to_slice, to_range.get(0)), q,
                    event_list);
    std::vector<sycl::event> host_events;
    if (to_slice == size_slice) {
      // Copy host data to a temp host buffer with the shape of target.
      host_events =
          dpct_memcpy(q, buf.get_ptr(), from_surface, to_range, from_range,
                      sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0), size,
                      host_to_host, dep_events);
    } else {
      // Copy host data to a temp host buffer with the shape of target.
      host_events = dpct_memcpy(
          q, buf.get_ptr(), from_surface, to_range, from_range,
          sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0), size, host_to_host,
          // If has padding data, not sure whether it is useless. So fill temp
          // buffer with it.
          std::vector<sycl::event>{
              dpct_memcpy(q, buf.get_ptr(), to_surface, buf.get_size(),
                          device_to_host, dep_events)});
    }
    // Copy from temp host buffer to device with only one submit.
    event_list.push_back(dpct_memcpy(q, to_surface, buf.get_ptr(),
                                     buf.get_size(), host_to_device,
                                     host_events));
    break;
  }
  case device_to_host: {
    host_buffer buf(get_copy_range(size, from_slice, from_range.get(0)), q,
                    event_list);
    // Copy from host temp buffer to host target with reshaping.
    event_list = dpct_memcpy(
        q, to_surface, buf.get_ptr(), to_range, from_range, sycl::id<3>(0, 0, 0),
        sycl::id<3>(0, 0, 0), size, host_to_host,
        // Copy from device to temp host buffer with only one submit.
        std::vector<sycl::event>{dpct_memcpy(q, buf.get_ptr(), from_surface,
                                                 buf.get_size(),
                                                 device_to_host, dep_events)});
    break;
  }
  case device_to_device:
#ifdef DPCT_USM_LEVEL_NONE
  {
    auto &mm = mem_mgr::instance();
    auto to_alloc = mm.translate_ptr(to_surface);
    auto from_alloc = mm.translate_ptr(from_surface);
    size_t to_offset = (byte_t *)to_surface - to_alloc.alloc_ptr;
    size_t from_offset = (byte_t *)from_surface - from_alloc.alloc_ptr;
    event_list.push_back(q.submit([&](sycl::handler &cgh) {
      cgh.depends_on(dep_events);
      auto to_o = sycl::id<1>(to_offset);
      auto from_o = sycl::id<1>(from_offset);
      sycl::accessor<byte_t, 1, sycl::access_mode::write,
                         sycl::access::target::device>
          to_acc(to_alloc.buffer, cgh,
                 get_copy_range(size, to_slice, to_range.get(0)), to_o);
      sycl::accessor<byte_t, 1, sycl::access_mode::read,
                         sycl::access::target::device>
          from_acc(from_alloc.buffer, cgh,
                   get_copy_range(size, from_slice, from_range.get(0)), from_o);
      cgh.parallel_for<class dpct_memcpy_3d_detail_usmnone>(
          size,
          [=](sycl::id<3> id) {
            to_acc[get_offset(id, to_slice, to_range.get(0))] =
                from_acc[get_offset(id, from_slice, from_range.get(0))];
          });
    }));
  }
#else
    event_list.push_back(q.submit([&](sycl::handler &cgh) {
      cgh.depends_on(dep_events);
      cgh.parallel_for<class dpct_memcpy_3d_detail>(
          size,
          [=](sycl::id<3> id) {
            to_surface[get_offset(id, to_slice, to_range.get(0))] =
                from_surface[get_offset(id, from_slice, from_range.get(0))];
          });
    }));
#endif
  break;
  default:
    throw std::runtime_error("dpct_memcpy: invalid direction value");
  }
  return event_list;
}

/**
 * @brief Performs 2D/3D memcpy between 2 pointers. Supports the virtual device
 * pointer from dpct_malloc().
 * @param [in] q The queue in which the operation is done.
 * @param [in] to The pitched memory as the copy destination.
 * @param [in] to_id The offset of the copy destination.
 * @param [in] from The pitched memory as the copy source.
 * @param [in] from_id The offset of the source.
 * @param [in] size The memory size to be copied.
 * @param [in] direction The memory copy direction.
 * @return An event list representing the memset operations.
 */
static inline std::vector<sycl::event>
dpct_memcpy(sycl::queue &q, pitched_data to, sycl::id<3> to_id,
            pitched_data from, sycl::id<3> from_id, sycl::range<3> size,
            memcpy_direction direction = automatic) {
  return dpct_memcpy(q, to.get_data_ptr(), from.get_data_ptr(),
                     sycl::range<3>(to.get_pitch(), to.get_y(), 1),
                     sycl::range<3>(from.get_pitch(), from.get_y(), 1), to_id, from_id,
                     size, direction);
}

/**
 * @brief Performs 2D memcpy between 2 pointers. Supports the virtual device
 * pointer from dpct_malloc().
 * @param [in] q The queue in which the operation is done.
 * @param [in] to_ptr The pointer to the copy destination.
 * @param [in] from_ptr The pointer to the copy source.
 * @param [in] to_pitch The pitch size, including padding bytes of the copy destination.
 * @param [in] from_pitch The pitch size, including padding bytes of the copy source.
 * @param [in] x The width of the copy range.
 * @param [in] y The height of the copy range.
 * @param [in] direction The memory copy direction.
 * @return An event list representing the memset operations.
 */
static inline std::vector<sycl::event>
dpct_memcpy(sycl::queue &q, void *to_ptr, const void *from_ptr,
            size_t to_pitch, size_t from_pitch, size_t x, size_t y,
            memcpy_direction direction = automatic) {
  return dpct_memcpy(q, to_ptr, from_ptr, sycl::range<3>(to_pitch, y, 1),
                     sycl::range<3>(from_pitch, y, 1),
                     sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0),
                     sycl::range<3>(x, y, 1), direction);
}

namespace deprecated {

template <typename T, sycl::usm::alloc AllocKind>
class usm_allocator {
private:
  using Alloc = sycl::usm_allocator<T, AllocKind>;
  Alloc _impl;

public:
  using value_type = typename std::allocator_traits<Alloc>::value_type;
  using pointer = typename std::allocator_traits<Alloc>::pointer;
  using const_pointer = typename std::allocator_traits<Alloc>::const_pointer;
  using void_pointer = typename std::allocator_traits<Alloc>::void_pointer;
  using const_void_pointer =
      typename std::allocator_traits<Alloc>::const_void_pointer;
  using reference = typename std::allocator_traits<Alloc>::value_type &;
  using const_reference =
      const typename std::allocator_traits<Alloc>::value_type &;
  using difference_type =
      typename std::allocator_traits<Alloc>::difference_type;
  using size_type = typename std::allocator_traits<Alloc>::size_type;
  using propagate_on_container_copy_assignment = typename std::allocator_traits<
      Alloc>::propagate_on_container_copy_assignment;
  using propagate_on_container_move_assignment = typename std::allocator_traits<
      Alloc>::propagate_on_container_move_assignment;
  using propagate_on_container_swap =
      typename std::allocator_traits<Alloc>::propagate_on_container_swap;
  using is_always_equal =
      typename std::allocator_traits<Alloc>::is_always_equal;

  template <typename U> struct rebind {
    typedef usm_allocator<U, AllocKind> other;
  };

  usm_allocator() : _impl(dpct::get_default_queue()) {}
  usm_allocator(sycl::queue &q) : _impl(q) {}
  ~usm_allocator() {}
  usm_allocator(const usm_allocator &other) : _impl(other._impl) {}
  usm_allocator(usm_allocator &&other) : _impl(std::move(other._impl)) {}
  pointer address(reference r) { return &r; }
  const_pointer address(const_reference r) { return &r; }
  pointer allocate(size_type cnt, const_void_pointer hint = nullptr) {
    return std::allocator_traits<Alloc>::allocate(_impl, cnt, hint);
  }
  void deallocate(pointer p, size_type cnt) {
    std::allocator_traits<Alloc>::deallocate(_impl, p, cnt);
  }
  size_type max_size() const {
    return std::allocator_traits<Alloc>::max_size(_impl);
  }
  bool operator==(const usm_allocator &other) const { return _impl == other._impl; }
  bool operator!=(const usm_allocator &other) const { return _impl != other._impl; }
};

} // namespace deprecated
/**
 * @brief Frees a piece of allocated device memory.
 * @param [in] ptr The pointer to the device memory to be freed.
 * @param [in] q The queue in which the operation is done.
 */
inline void dpct_free(void *ptr,
                      const sycl::queue &q) {
  if (ptr) {
#ifdef DPCT_USM_LEVEL_NONE
    detail::mem_mgr::instance().mem_free(ptr);
#else
    sycl::free(ptr, q.get_context());
#endif // DPCT_USM_LEVEL_NONE
  }
}
} // namespace detail

#ifdef DPCT_USM_LEVEL_NONE
/**
 * @brief Checks if the pointer is a virtual device pointer.
 * @tparam T The type of the pointer to be checked.
 * @param [in] ptr The virtual pointer to be checked.
 * @return true if the pointer is a virtual device pointer.
 */
template<class T>
static inline bool is_device_ptr(T ptr) {
  if constexpr (std::is_pointer<T>::value) {
    return detail::mem_mgr::instance().is_device_ptr(ptr);
  }
  return false;
}
#endif
/**
 * @brief Gets the buffer and the offset of a piece of memory pointed by \p ptr.
 * @param [in] ptr The virtual device memory pointer.
 * @return A pair containing both the buffer and the offset.
 */
static std::pair<buffer_t, size_t> get_buffer_and_offset(const void *ptr) {
  if (ptr) {
    auto alloc = detail::mem_mgr::instance().translate_ptr(ptr);
    size_t offset = (byte_t *)ptr - alloc.alloc_ptr;
    return std::make_pair(alloc.buffer, offset);
  } else {
    throw std::runtime_error(
        "NULL pointer argument in get_buffer_and_offset function is invalid");
  }
}
/**
 * @brief Gets the data pointed by \p ptr as a 1D buffer reinterpreted as type T.
 * @tparam T The target datatype.
 * @param [in] ptr The virtual device memory pointer.
 * @return The result \a sycl::buffer.
 */
template <typename T> static sycl::buffer<T> get_buffer(const void *ptr) {
  if (!ptr)
    return sycl::buffer<T>(sycl::range<1>(0));
  auto alloc = detail::mem_mgr::instance().translate_ptr(ptr);
  return alloc.buffer.reinterpret<T>(
      sycl::range<1>(alloc.size / sizeof(T)));
}
/**
 * @brief Gets the buffer pointed by \p ptr.
 * @param [in] ptr The virtual device memory pointer.
 * @return The result \a sycl::buffer.
 */
static buffer_t get_buffer(const void *ptr) {
  return detail::mem_mgr::instance().translate_ptr(ptr).buffer;
}
/**
 * @class access_wrapper
 * @brief A wrapper class contains an \a sycl::accessor and an offset.
 */
template <typename dataT,
          sycl::access_mode accessMode = sycl::access_mode::read_write>
class access_wrapper {
  sycl::accessor<byte_t, 1, accessMode> accessor;
  size_t offset;

public:
  /**
   * @brief Constructs the accessor wrapper from a virtual device pointer \p ptr.
   * @param [in] ptr The virtual device memory pointer.
   * @param [in] cgh The command group handler.
   */
  access_wrapper(const void *ptr, sycl::handler &cgh)
      : accessor(get_buffer(ptr).get_access<accessMode>(cgh)), offset(0) {
    auto alloc = detail::mem_mgr::instance().translate_ptr(ptr);
    offset = (byte_t *)ptr - alloc.alloc_ptr;
  }
  /**
   * @brief Gets the device pointer.
   * @return The device pointer with offset.
   */
  dataT get_raw_pointer() const { return (dataT)(&accessor[0] + offset); }
};
/**
 * @brief Gets the accessor for memory pointed by \p ptr.
 * @param [in] ptr Pointer to memory.
 * @param [in] cgh The command group handler.
 * @return The accessor.
 */
template <sycl::access_mode accessMode = sycl::access_mode::read_write>
static sycl::accessor<byte_t, 1, accessMode>
get_access(const void *ptr, sycl::handler &cgh) {
  if (ptr) {
    auto alloc = detail::mem_mgr::instance().translate_ptr(ptr);
    return alloc.buffer.get_access<accessMode>(cgh);
  } else {
    throw std::runtime_error(
        "NULL pointer argument in get_access function is invalid");
  }
}
/**
 * @brief Allocates memory block on the device.
 * @param [in] num_bytes Number of bytes to allocate.
 * @param [in] q Queue to execute the allocate task.
 * @return A virtual device pointer to the newly allocated memory.
 */
template <typename T>
static inline void *dpct_malloc(T num_bytes,
                                sycl::queue &q = get_default_queue()) {
  return detail::dpct_malloc(static_cast<size_t>(num_bytes), q);
}
/**
 * @brief Gets the host pointer of a buffer that is mapped to virtual pointer ptr.
 * @param [in] ptr The virtual Pointer mapped to device buffer.
 * @return A host pointer.
 */
template <typename T> static inline T *get_host_ptr(const void *ptr) {
  auto BufferOffset = get_buffer_and_offset(ptr);
  auto host_ptr =
      BufferOffset.first.get_host_access()
          .get_pointer();
  return (T *)(host_ptr + BufferOffset.second);
}
/**
 * @brief Allocates memory block for 3D array on the device.
 * @param [in] size Size of the memory block, in bytes.
 * @param [in] q Queue to execute the allocate task.
 * @return A pitched_data object which stores the memory info.
 */
static inline pitched_data
dpct_malloc(sycl::range<3> size, sycl::queue &q = get_default_queue()) {
  pitched_data pitch(nullptr, 0, size.get(0), size.get(1));
  size_t pitch_size;
  pitch.set_data_ptr(detail::dpct_malloc(pitch_size, size.get(0), size.get(1),
                                         size.get(2), q));
  pitch.set_pitch(pitch_size);
  return pitch;
}
/**
 * @brief Allocates memory block for 2D array on the device.
 * @param [out] pitch Aligned size of x in bytes.
 * @param [in] x The width of the memory block.
 * @param [in] y The height of the memory block.
 * @param [in] q Queue to execute the allocate task.
 * @return A pointer to the newly allocated memory.
 */
static inline void *dpct_malloc(size_t &pitch, size_t x, size_t y,
                                sycl::queue &q = get_default_queue()) {
  return detail::dpct_malloc(pitch, x, y, 1, q);
}
/**
 * @brief Frees a piece of allocated device memory.
 * @param [in] ptr The pointer to the device memory to be freed.
 * @param [in] q The queue in which the operation is done.
 */
static inline void dpct_free(void *ptr,
                             sycl::queue &q = get_default_queue()) {
#ifndef DPCT_USM_LEVEL_NONE
  dpct::get_current_device().queues_wait_and_throw();
#endif
  detail::dpct_free(ptr, q);
}
/**
 * @brief Frees a batch of device memory.
 * @param [in] pointers The pointers to be freed.
 * @param [in] events The events which this operation depends on.
 * @param [in] q The queue in which the operation is done.
 */
inline void async_dpct_free(const std::vector<void *> &pointers,
                            const std::vector<sycl::event> &events,
                            sycl::queue &q = get_default_queue()) {
  q.submit([&](sycl::handler &cgh) {
    cgh.depends_on(events);
    cgh.host_task([=] {
      for (auto p : pointers)
        if (p) {
          detail::dpct_free(p, q);
        }
    });
  });
}
/**
 * @brief Performs synchronous memcpy between 2 pointers. Supports the virtual
 * device pointer from dpct_malloc().
 * @param [in] to_ptr Pointer to the destination virtual device memory.
 * @param [in] from_ptr Pointer to the source virtual device memory.
 * @param [in] size The memory size to be copied.
 * @param [in] direction The memory copy direction.
 * @param [in] q The queue in which the operation is done.
 */
static void dpct_memcpy(void *to_ptr, const void *from_ptr, size_t size,
                        memcpy_direction direction = automatic,
                        sycl::queue &q = get_default_queue()) {
  detail::dpct_memcpy(q, to_ptr, from_ptr, size, direction).wait();
}
/**
 * @brief Performs asynchronous memcpy between 2 pointers. Supports the virtual
 * device pointer from dpct_malloc().
 * @param [in] to_ptr Pointer to the destination virtual device memory.
 * @param [in] from_ptr Pointer to the source virtual device memory.
 * @param [in] size The memory size to be copied.
 * @param [in] direction The memory copy direction.
 * @param [in] q The queue in which the operation is done.
 */
static void async_dpct_memcpy(void *to_ptr, const void *from_ptr, size_t size,
                              memcpy_direction direction = automatic,
                              sycl::queue &q = dpct::get_default_queue()) {
  detail::dpct_memcpy(q, to_ptr, from_ptr, size, direction);
}
/**
 * @brief Performs synchronous 2D memcpy between 2 pointers. Supports the
 * virtual device pointer from dpct_malloc().
 * @param [in] to_ptr The pointer to the copy destination.
 * @param [in] to_pitch The pitch size, including padding bytes of the copy destination.
 * @param [in] from_ptr The pointer to the copy source.
 * @param [in] from_pitch The pitch size, including padding bytes of the copy source.
 * @param [in] x The width of the copy range.
 * @param [in] y The height of the copy range.
 * @param [in] direction The memory copy direction.
 * @param [in] q The queue in which the operation is done.
 */
static inline void dpct_memcpy(void *to_ptr, size_t to_pitch,
                               const void *from_ptr, size_t from_pitch,
                               size_t x, size_t y,
                               memcpy_direction direction = automatic,
                               sycl::queue &q = dpct::get_default_queue()) {
  sycl::event::wait(detail::dpct_memcpy(q, to_ptr, from_ptr, to_pitch,
                                            from_pitch, x, y, direction));
}
/**
 * @brief Performs asynchronous 2D memcpy between 2 pointers. Supports the
 * virtual device pointer from dpct_malloc().
 * @param [in] to_ptr The pointer to the copy destination.
 * @param [in] to_pitch The pitch size, including padding bytes of the copy destination.
 * @param [in] from_ptr The pointer to the copy source.
 * @param [in] from_pitch The pitch size, including padding bytes of the copy source.
 * @param [in] x The width of the copy range.
 * @param [in] y The height of the copy range.
 * @param [in] direction The memory copy direction.
 * @param [in] q The queue in which the operation is done.
 */
static inline void
async_dpct_memcpy(void *to_ptr, size_t to_pitch, const void *from_ptr,
                  size_t from_pitch, size_t x, size_t y,
                  memcpy_direction direction = automatic,
                  sycl::queue &q = get_default_queue()) {
  detail::dpct_memcpy(q, to_ptr, from_ptr, to_pitch, from_pitch, x, y,
                      direction);
}
/**
 * @brief Performs synchronous 3D memcpy between 2 pointers. Supports the
 * virtual device pointer from dpct_malloc().
 * @param [in] to The pointer to the destination.
 * @param [in] to_pos The position of destination.
 * @param [in] from The pointer to the source.
 * @param [in] from_pos The position of the source.
 * @param [in] size The memory size to be copied.
 * @param [in] direction The memory copy direction.
 * @param [in] q The queue in which the operation is done.
 */
static inline void dpct_memcpy(pitched_data to, sycl::id<3> to_pos,
                               pitched_data from, sycl::id<3> from_pos,
                               sycl::range<3> size,
                               memcpy_direction direction = automatic,
                               sycl::queue &q = dpct::get_default_queue()) {
  sycl::event::wait(
      detail::dpct_memcpy(q, to, to_pos, from, from_pos, size, direction));
}

/**
 * @brief Performs asynchronous 3D memcpy between 2 pointers. Supports the
 * virtual device pointer from dpct_malloc().
 * @param [in] to The pointer to the destination.
 * @param [in] to_pos The position of destination.
 * @param [in] from The pointer to the source.
 * @param [in] from_pos The position of the source.
 * @param [in] size The memory size to be copied.
 * @param [in] direction The memory copy direction.
 * @param [in] q The queue in which the operation is done.
 */
static inline void
async_dpct_memcpy(pitched_data to, sycl::id<3> to_pos, pitched_data from,
                  sycl::id<3> from_pos, sycl::range<3> size,
                  memcpy_direction direction = automatic,
                  sycl::queue &q = get_default_queue()) {
  detail::dpct_memcpy(q, to, to_pos, from, from_pos, size, direction);
}
/**
 * @brief Sets 1 byte data \p value to the first \p size elements starting from
 * \p dev_ptr in \p q synchronously.
 * @param [in] dev_ptr Pointer to the virtual device memory address.
 * @param [in] value The value to be set.
 * @param [in] size Number of elements to be set to the value.
 * @param [in] q The queue in which the operation is done.
 */
static void dpct_memset(void *dev_ptr, int value, size_t size,
                        sycl::queue &q = get_default_queue()) {
  detail::dpct_memset<unsigned char>(q, dev_ptr, value, size).wait();
}

/**
 * @brief Sets 2 bytes data \p value to the first \p size elements starting from
 * \p dev_ptr in \p q synchronously.
 * @param [in] dev_ptr Pointer to the virtual device memory address.
 * @param [in] value The value to be set.
 * @param [in] size Number of elements to be set to the value.
 * @param [in] q The queue in which the operation is done.
 */
static void dpct_memset_d16(void *dev_ptr, unsigned short value, size_t size,
                        sycl::queue &q = get_default_queue()) {
  detail::dpct_memset(q, dev_ptr, value, size).wait();
}
/**
 * @brief Sets 4 bytes data \p value to the first \p size elements starting from
 * \p dev_ptr in \p q synchronously.
 * @param [in] dev_ptr Pointer to the virtual device memory address.
 * @param [in] value The value to be set.
 * @param [in] size Number of elements to be set to the value.
 * @param [in] q The queue in which the operation is done.
 */
static void dpct_memset_d32(void *dev_ptr, unsigned int value, size_t size,
                        sycl::queue &q = get_default_queue()) {
  detail::dpct_memset(q, dev_ptr, value, size).wait();
}

/**
 * @brief Sets 1 byte data \p value to the first \p size elements starting from
 * \p dev_ptr in \p q asynchronously.
 * @param [in] dev_ptr Pointer to the virtual device memory address.
 * @param [in] value The value to be set.
 * @param [in] size Number of elements to be set to the value.
 * @param [in] q The queue in which the operation is done.
 */
static void async_dpct_memset(void *dev_ptr, int value, size_t size,
                              sycl::queue &q = dpct::get_default_queue()) {
  detail::dpct_memset<unsigned char>(q, dev_ptr, value, size);
}
/**
 * @brief Sets 2 bytes data \p value to the first \p size elements starting from
 * \p dev_ptr in \p q asynchronously.
 * @param [in] dev_ptr Pointer to the virtual device memory address.
 * @param [in] value The value to be set.
 * @param [in] size Number of elements to be set to the value.
 * @param [in] q The queue in which the operation is done.
 */
static void async_dpct_memset_d16(void *dev_ptr, unsigned short value, size_t size,
                              sycl::queue &q = dpct::get_default_queue()) {
  detail::dpct_memset(q, dev_ptr, value, size);
}
/**
 * @brief Sets 4 bytes data \p value to the first \p size elements starting from
 * \p dev_ptr in \p q asynchronously.
 * @param [in] dev_ptr Pointer to the virtual device memory address.
 * @param [in] value The value to be set.
 * @param [in] size Number of elements to be set to the value.
 * @param [in] q The queue in which the operation is done.
 */
static void async_dpct_memset_d32(void *dev_ptr, unsigned int value, size_t size,
                              sycl::queue &q = dpct::get_default_queue()) {
  detail::dpct_memset(q, dev_ptr, value, size);
}

/**
 * @brief Sets 1 byte data \p val to the pitched 2D memory region pointed by \p ptr in \p q
 * synchronously.
 * @param [in] ptr Pointer to the virtual device memory.
 * @param [in] pitch The pitch size by number of elements, including padding.
 * @param [in] val The value to be set.
 * @param [in] x The width of memory region by number of elements.
 * @param [in] y The height of memory region by number of elements.
 * @param [in] q The queue in which the operation is done.
 */
static inline void dpct_memset(void *ptr, size_t pitch, int val, size_t x,
                               size_t y,
                               sycl::queue &q = get_default_queue()) {
  sycl::event::wait(detail::dpct_memset<unsigned char>(q, ptr, pitch, val, x, y));
}
/**
 * @brief Sets 2 bytes data \p val to the pitched 2D memory region pointed by \p ptr in \p q
 * synchronously.
 * @param [in] ptr Pointer to the virtual device memory.
 * @param [in] pitch The pitch size by number of elements, including padding.
 * @param [in] val The value to be set.
 * @param [in] x The width of memory region by number of elements.
 * @param [in] y The height of memory region by number of elements.
 * @param [in] q The queue in which the operation is done.
 */
static inline void dpct_memset_d16(void *ptr, size_t pitch, unsigned short val, size_t x,
                               size_t y,
                               sycl::queue &q = get_default_queue()) {
  sycl::event::wait(detail::dpct_memset(q, ptr, pitch, val, x, y));
}
/**
 * @brief Sets 4 bytes data \p val to the pitched 2D memory region pointed by \p ptr in \p q
 * synchronously.
 * @param [in] ptr Pointer to the virtual device memory.
 * @param [in] pitch The pitch size by number of elements, including padding.
 * @param [in] val The value to be set.
 * @param [in] x The width of memory region by number of elements.
 * @param [in] y The height of memory region by number of elements.
 * @param [in] q The queue in which the operation is done.
 */
static inline void dpct_memset_d32(void *ptr, size_t pitch, unsigned int val, size_t x,
                               size_t y,
                               sycl::queue &q = get_default_queue()) {
  sycl::event::wait(detail::dpct_memset(q, ptr, pitch, val, x, y));
}

/**
 * @brief Sets 1 byte data \p val to the pitched 2D memory region pointed by \p ptr in \p q
 * asynchronously.
 * @param [in] ptr Pointer to the virtual device memory.
 * @param [in] pitch The pitch size by number of elements, including padding.
 * @param [in] val The value to be set.
 * @param [in] x The width of memory region by number of elements.
 * @param [in] y The height of memory region by number of elements.
 * @param [in] q The queue in which the operation is done.
 */
static inline void async_dpct_memset(void *ptr, size_t pitch, int val, size_t x,
                                     size_t y,
                                     sycl::queue &q = get_default_queue()) {
  detail::dpct_memset<unsigned char>(q, ptr, pitch, val, x, y);
}

/**
 * @brief Sets 2 bytes data \p val to the pitched 2D memory region pointed by \p ptr in \p q
 * asynchronously.
 * @param [in] ptr Pointer to the virtual device memory.
 * @param [in] pitch The pitch size by number of elements, including padding.
 * @param [in] val The value to be set.
 * @param [in] x The width of memory region by number of elements.
 * @param [in] y The height of memory region by number of elements.
 * @param [in] q The queue in which the operation is done.
 */
static inline void async_dpct_memset_d16(void *ptr, size_t pitch,
                                         unsigned short val, size_t x, size_t y,
                                         sycl::queue &q = get_default_queue()) {
  detail::dpct_memset(q, ptr, pitch, val, x, y);
}

/**
 * @brief Sets 4 bytes data \p val to the pitched 2D memory region pointed by \p ptr in \p q
 * asynchronously.
 * @param [in] ptr Pointer to the virtual device memory.
 * @param [in] pitch The pitch size by number of elements, including padding.
 * @param [in] val The value to be set.
 * @param [in] x The width of memory region by number of elements.
 * @param [in] y The height of memory region by number of elements.
 * @param [in] q The queue in which the operation is done.
 */
static inline void async_dpct_memset_d32(void *ptr, size_t pitch,
                                         unsigned int val, size_t x, size_t y,
                                         sycl::queue &q = get_default_queue()) {
  detail::dpct_memset(q, ptr, pitch, val, x, y);
}

/**
 * @brief Sets 1 byte data \p value to the 3D memory region pointed by \p data in \p q
 * synchronously.
 * @param [in] data Pointer to the pitched device memory region.
 * @param [in] value The value to be set.
 * @param [in] size 3D memory region by number of elements.
 * @param [in] q The queue in which the operation is done.
 */
static inline void dpct_memset(pitched_data pitch, int val,
                               sycl::range<3> size,
                               sycl::queue &q = get_default_queue()) {
  sycl::event::wait(detail::dpct_memset<unsigned char>(q, pitch, val, size));
}

/**
 * @brief Sets 1 byte data \p value to the 3D memory region pointed by \p data in \p q
 * asynchronously.
 * @param [in] data Pointer to the pitched device memory region.
 * @param [in] value The value to be set.
 * @param [in] size 3D memory region by number of elements.
 * @param [in] q The queue in which the operation is done.
 */
static inline void async_dpct_memset(pitched_data pitch, int val,
                                     sycl::range<3> size,
                                     sycl::queue &q = get_default_queue()) {
  detail::dpct_memset<unsigned char>(q, pitch, val, size);
}
/**
 * @class accessor
 * @brief Accessor template used as device function parameter.
 */
template <class T, memory_region Memory, size_t Dimension> class accessor;
/**
 * @class accessor<T, Memory, 3>
 * @brief 3D accessor used as device function parameter.
 */
template <class T, memory_region Memory> class accessor<T, Memory, 3> {
public:
  using memory_t = detail::memory_traits<Memory, T>;
  using element_t = typename memory_t::element_t;
  using pointer_t = typename memory_t::pointer_t;
  using accessor_t = typename memory_t::template accessor_t<3>;
  accessor(pointer_t data, const sycl::range<3> &in_range)
      : _data(data), _range(in_range) {}
  template <memory_region M = Memory>
  accessor(typename std::enable_if<M != local, const accessor_t>::type &acc)
      : accessor(acc, acc.get_range()) {}
  accessor(const accessor_t &acc, const sycl::range<3> &in_range)
      : accessor(
            acc.template get_multi_ptr<sycl::access::decorated::no>().get(),
            in_range) {}
  accessor<T, Memory, 2> operator[](size_t index) const {
    sycl::range<2> sub(_range.get(1), _range.get(2));
    return accessor<T, Memory, 2>(_data + index * sub.size(), sub);
  }

  pointer_t get_ptr() const { return _data; }

private:
  pointer_t _data;
  sycl::range<3> _range;
};
/**
 * @class class accessor<T, Memory, 2>
 * @brief 2D accessor used as device function parameter.
 */
template <class T, memory_region Memory> class accessor<T, Memory, 2> {
public:
  using memory_t = detail::memory_traits<Memory, T>;
  using element_t = typename memory_t::element_t;
  using pointer_t = typename memory_t::pointer_t;
  using accessor_t = typename memory_t::template accessor_t<2>;
  accessor(pointer_t data, const sycl::range<2> &in_range)
      : _data(data), _range(in_range) {}
  template <memory_region M = Memory>
  accessor(typename std::enable_if<M != local, const accessor_t>::type &acc)
      : accessor(acc, acc.get_range()) {}
  accessor(const accessor_t &acc, const sycl::range<2> &in_range)
      : accessor(
            acc.template get_multi_ptr<sycl::access::decorated::no>().get(),
            in_range) {}

  pointer_t operator[](size_t index) const {
    return _data + _range.get(1) * index;
  }

  pointer_t get_ptr() const { return _data; }

private:
  pointer_t _data;
  sycl::range<2> _range;
};

namespace detail {
/**
 * @class device_memory
 * @brief Device memory class with address space of shared, global or constant.
 */
template <class T, memory_region Memory, size_t Dimension>
class device_memory {
public:
  using accessor_t =
      typename detail::memory_traits<Memory, T>::template accessor_t<Dimension>;
  using value_t = typename detail::memory_traits<Memory, T>::value_t;
  using dpct_accessor_t = dpct::accessor<T, Memory, Dimension>;

  /**
   * @brief The default constructor. Create an array with size 1.
   */
  device_memory() : device_memory(sycl::range<Dimension>(1)) {}

  /**
   * @brief Constructor of 1-D array with initializer list.
   * @param [in] in_range The dimension of the array.
   * @param [in] init_list The initializer list.
   */
  device_memory(
      const sycl::range<Dimension> &in_range,
      std::initializer_list<value_t> &&init_list)
      : device_memory(in_range) {
    assert(init_list.size() <= in_range.size());
    _host_ptr = (value_t *)std::malloc(_size);
    std::memset(_host_ptr, 0, _size);
    std::memcpy(_host_ptr, init_list.begin(), init_list.size() * sizeof(T));
  }
  /**
   * @brief Constructor of 2-D array with initializer list.
   * @param [in] in_range The dimension of the array.
   * @param [in] init_list The initializer list.
   */
  template <size_t D = Dimension>
  device_memory(
      const typename std::enable_if<D == 2, sycl::range<2>>::type &in_range,
      std::initializer_list<std::initializer_list<value_t>> &&init_list)
      : device_memory(in_range) {
    assert(init_list.size() <= in_range[0]);
    _host_ptr = (value_t *)std::malloc(_size);
    std::memset(_host_ptr, 0, _size);
    auto tmp_data = _host_ptr;
    for (auto sub_list : init_list) {
      assert(sub_list.size() <= in_range[1]);
      std::memcpy(tmp_data, sub_list.begin(), sub_list.size() * sizeof(T));
      tmp_data += in_range[1];
    }
  }
  /**
   * @brief Constructor with range.
   * @param [in] in_range The dimension of the array.
   */
  device_memory(const sycl::range<Dimension> &range_in)
      : _size(range_in.size() * sizeof(T)), _range(range_in), _reference(false),
        _host_ptr(nullptr), _device_ptr(nullptr) {
    static_assert(
        (Memory == global) || (Memory == constant) || (Memory == shared),
        "device memory region should be global, constant or shared");
    // Make sure that singleton class mem_mgr and dev_mgr will destruct later
    // than this.
    detail::mem_mgr::instance();
    dev_mgr::instance();
  }

  /**
   * @brief Constructor with range.
   * @param [in] Arguments The arguments to construct the \a sycl::range.
   */
  template <class... Args>
  device_memory(Args... Arguments)
      : device_memory(sycl::range<Dimension>(Arguments...)) {}
  /**
   * @brief The defaul destructor.
   */
  ~device_memory() {
    if (_device_ptr && !_reference)
      dpct::dpct_free(_device_ptr);
    if (_host_ptr)
      std::free(_host_ptr);
  }
  /**
   * @brief Allocates memory with default queue, and init memory if the initial
   * value is given.
   */
  void init() {
    init(dpct::get_default_queue());
  }
  /**
   * @brief Allocate memory with specified queue, and init memory if the initial
   * value is given.
   * @param [in] q The queue in which the operation is done.
   */
  void init(sycl::queue &q) {
    if (_device_ptr)
      return;
    if (!_size)
      return;
    allocate_device(q);
    if (_host_ptr)
      detail::dpct_memcpy(q, _device_ptr, _host_ptr, _size, host_to_device);
  }

  /**
   * @brief Updates the device memory.
   * @param [in] src The pointer to the source.
   * @param [in] size The size of source memory.
   */
  void assign(value_t *src, size_t size) {
    this->~device_memory();
    new (this) device_memory(src, size);
  }
  /**
   * @brief Gets the memory pointer of the memory object, which is a virtual
   * pointer when USM level is none.
   * @return The pointer.
   */
  value_t *get_ptr() {
    return get_ptr(get_default_queue());
  }
  /**
   * @brief Gets the memory pointer of the memory object, which is a virtual
   * pointer when USM level is none.
   * @param [in] q The queue in which the operation is done.
   * @return The pointer.
   */
  value_t *get_ptr(sycl::queue &q) {
    init(q);
    return _device_ptr;
  }
  /**
   * @brief Gets the device memory object size in bytes.
   * @return The device memory object size in bytes
   */
  size_t get_size() { return _size; }

  template <size_t D = Dimension>
  typename std::enable_if<D == 1, T>::type &operator[](size_t index) {
    init();
#ifdef DPCT_USM_LEVEL_NONE
    return dpct::get_buffer<typename std::enable_if<D == 1, T>::type>(
               _device_ptr)
        .template get_access<sycl::access_mode::read_write>()[index];
#else
    return _device_ptr[index];
#endif // DPCT_USM_LEVEL_NONE
  }

#ifdef DPCT_USM_LEVEL_NONE
  /**
   * @brief Gets sycl::accessor for the device memory object when USM level is none.
   * @param [in] cgh The command group handler.
   * @return The sycl::accessor of the device memory.
   */
  accessor_t get_access(sycl::handler &cgh) {
    return get_buffer(_device_ptr)
        .template reinterpret<T, Dimension>(_range)
        .template get_access<detail::memory_traits<Memory, T>::mode,
                             detail::memory_traits<Memory, T>::target>(cgh);
  }
#else
  /**
   * @brief Gets dpct::accessor with dimension info for the device memory object
   * when USM is used and the dimension is greater than 1.
   * @param [in] cgh The command group handler.
   * @return The sycl::accessor of the device memory.
   */
  template <size_t D = Dimension>
  typename std::enable_if<D != 1, dpct_accessor_t>::type
  get_access(sycl::handler &cgh) {
    return dpct_accessor_t((T *)_device_ptr, _range);
  }
#endif // DPCT_USM_LEVEL_NONE

private:
  device_memory(value_t *memory_ptr, size_t size)
      : _size(size), _range(size / sizeof(T)), _reference(true),
        _device_ptr(memory_ptr) {}

  void allocate_device(sycl::queue &q) {
#ifndef DPCT_USM_LEVEL_NONE
    if (Memory == shared) {
      _device_ptr = (value_t *)sycl::malloc_shared(
          _size, q.get_device(), q.get_context());
      return;
    }
#ifdef SYCL_EXT_ONEAPI_USM_DEVICE_READ_ONLY
    if (Memory == constant) {
      _device_ptr = (value_t *)sycl::malloc_device(
          _size, q.get_device(), q.get_context(),
          sycl::ext::oneapi::property::usm::device_read_only());
      return;
    }
#endif
#endif
    _device_ptr = (value_t *)detail::dpct_malloc(_size, q);
  }

  size_t _size;
  sycl::range<Dimension> _range;
  bool _reference;
  value_t *_host_ptr;
  value_t *_device_ptr;
};

/**
 * @class device_memory<T, Memory, 1>
 * @brief 1D device memory class with address space of shared, global or
 * constant.
 */
template <class T, memory_region Memory>
class device_memory<T, Memory, 0> : public device_memory<T, Memory, 1> {
public:
  using base = device_memory<T, Memory, 1>;
  using value_t = typename base::value_t;
  using accessor_t =
      typename detail::memory_traits<Memory, T>::template accessor_t<0>;
  /**
   * @brief Constructor with initial value.
   * @param [in] val The initial value.
   */
  device_memory(const value_t &val) : base(sycl::range<1>(1), {val}) {}

  /**
   * @brief Default constructor.
   */
  device_memory() : base(1) {}

#ifdef DPCT_USM_LEVEL_NONE
  /**
   * @brief Gets accessor for the device memory object when USM level is none.
   * @param [in] cgh The command group handler.
   * @return The accessor of the device memory.
   */
  accessor_t get_access(sycl::handler &cgh) {
    auto buf = get_buffer(base::get_ptr())
                   .template reinterpret<T, 1>(sycl::range<1>(1));
    return accessor_t(buf, cgh);
  }
#endif // DPCT_USM_LEVEL_NONE
};
}

template <class T, size_t Dimension>
using global_memory = detail::device_memory<T, global, Dimension>;
template <class T, size_t Dimension>
using constant_memory = detail::device_memory<T, constant, Dimension>;
template <class T, size_t Dimension>
using shared_memory = detail::device_memory<T, shared, Dimension>;

// dpct::deprecated:: is for functionality that was introduced for compatibility
// purpose, but relies on deprecated C++ features, which are either removed or
// will be removed in the future standards.
// Direct use of deprecated functionality in this namespace should be avoided.
namespace deprecated {

template <typename T>
using usm_host_allocator = detail::deprecated::usm_allocator<T, sycl::usm::alloc::host>;

template <typename T>
using usm_device_allocator = detail::deprecated::usm_allocator<T, sycl::usm::alloc::shared>;
} // namespace deprecated

/**
 * @class pointer_attributes
 * @brief Class to collect attributes of a pointer.
 */
class pointer_attributes {
public:
  /**
   * @brief Initializes with a pointer.
   * @param [in] ptr The pointer to analyze.
   * @param [in] q The queue in which the operation is done.
   */
  void init(const void *ptr,
              sycl::queue &q = dpct::get_default_queue()) {
#ifdef DPCT_USM_LEVEL_NONE
    throw std::runtime_error(
          "dpct::pointer_attributes: only works for USM pointer.");
#else
    memory_type = sycl::get_pointer_type(ptr, q.get_context());
    if (memory_type == sycl::usm::alloc::unknown) {
      device_id = -1;
      return;
    }
    device_pointer = (memory_type !=
                        sycl::usm::alloc::unknown) ? ptr : nullptr;
    host_pointer = (memory_type !=
                        sycl::usm::alloc::unknown) &&
                   (memory_type != sycl::usm::alloc::device) ? ptr : nullptr;
    sycl::device device_obj = sycl::get_pointer_device(ptr, q.get_context());
    device_id = dpct::dev_mgr::instance().get_device_id(device_obj);
#endif
  }
  /**
   * @brief Gets the memory type of the pointer.
   * @return The memory type of the pointer.
   */
  sycl::usm::alloc get_memory_type() {
    return memory_type;
  }
  /**
   * @brief Gets the device pointer.
   * @return The device pointer.
   */
  const void *get_device_pointer() {
    return device_pointer;
  }
  /**
   * @brief Gets the host pointer.
   * @return The host pointer.
   */
  const void *get_host_pointer() {
    return host_pointer;
  }
  /**
   * @brief Gets whether is a shared memory.
   * @return true if the pointer points to shared memory.
   */
  bool is_memory_shared() {
    return memory_type == sycl::usm::alloc::shared;
  }
  /**
   * @brief Gets the device ID of a device pointer.
   * @return The device ID.
   */
  unsigned int get_device_id() {
    return device_id;
  }

private:
  sycl::usm::alloc memory_type = sycl::usm::alloc::unknown;
  const void *device_pointer = nullptr;
  const void *host_pointer = nullptr;
  unsigned int device_id = -1;
};
} // namespace dpct
#endif // __DPCT_MEMORY_HPP__
