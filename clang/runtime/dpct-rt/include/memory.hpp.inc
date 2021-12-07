// DPCT_LABEL_BEGIN|License|
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
//==---- memory.hpp -------------------------------*- C++ -*----------------==//
//
// Copyright (C) 2018 - 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
// DPCT_LABEL_END

#ifndef __DPCT_MEMORY_HPP__
#define __DPCT_MEMORY_HPP__
// DPCT_COMMENT
// DPCT_COMMENT Example1:
// DPCT_COMMENT // DPCT_LABEL_BEGIN|FeatureNameDef|[Namespace]
// DPCT_COMMENT // DPCT_DEPENDENCY_EMPTY
// DPCT_COMMENT // DPCT_CODE
// DPCT_COMMENT some code
// DPCT_COMMENT // DPCT_LABEL_END
// DPCT_COMMENT
// DPCT_COMMENT Example2:
// DPCT_COMMENT // DPCT_LABEL_BEGIN|FeatureNameDef|[Namespace]
// DPCT_COMMENT // DPCT_DEPENDENCY_BEGIN
// DPCT_COMMENT // FileID|FeatureNameRef
// DPCT_COMMENT [// FileID|FeatureNameRef]
// DPCT_COMMENT ...
// DPCT_COMMENT // DPCT_DEPENDENCY_END
// DPCT_COMMENT // DPCT_CODE
// DPCT_COMMENT some code
// DPCT_COMMENT // DPCT_LABEL_END
// DPCT_COMMENT
// DPCT_COMMENT For header file including dependency, please use predefined feature name:
// DPCT_COMMENT   local_include_dependency: dpct helper files
// DPCT_COMMENT   non_local_include_dependency: other headler files

#include "device.hpp"
// DPCT_LABEL_BEGIN|non_local_include_dependency|
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
#include <CL/sycl.hpp>
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
#define NOMINMAX
#include <windows.h>
#else
#error "Only support Windows and Linux."
#endif
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|local_include_dependency|
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
// DPCT_LABEL_END

namespace dpct {

// DPCT_LABEL_BEGIN|memcpy_direction|dpct
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
enum memcpy_direction {
  host_to_host,
  host_to_device,
  device_to_host,
  device_to_device,
  automatic
};
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|memory_region|dpct
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
enum memory_region {
  global = 0,  // device global memory
  constant,    // device constant memory
  local,       // device local memory
  shared,      // memory which can be accessed by host and device
};
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|typedef_byte_t|dpct
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
typedef uint8_t byte_t;
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|typedef_buffer_t|dpct
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
/// Buffer type to be used in Memory Management runtime.
typedef cl::sycl::buffer<byte_t> buffer_t;
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|pitched_data|dpct
// DPCT_DEPENDENCY_BEGIN
// Memory|pitched_data_1
// DPCT_DEPENDENCY_END
// DPCT_CODE
/// Pitched 2D/3D memory data.
class pitched_data {
public:
  pitched_data() : pitched_data(nullptr, 0, 0, 0) {}
  pitched_data(void *data, size_t pitch, size_t x, size_t y)
      : _data(data), _pitch(pitch), _x(x), _y(y) {}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|pitched_data_get_data_ptr|dpct
// DPCT_PARENT_FEATURE|pitched_data
// DPCT_DEPENDENCY_BEGIN
// Memory|pitched_data
// DPCT_DEPENDENCY_END
// DPCT_CODE
  void *get_data_ptr() { return _data; }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|pitched_data_set_data_ptr|dpct
// DPCT_PARENT_FEATURE|pitched_data
// DPCT_DEPENDENCY_BEGIN
// Memory|pitched_data
// DPCT_DEPENDENCY_END
// DPCT_CODE
  void set_data_ptr(void *data) { _data = data; }
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|pitched_data_get_pitch|dpct
// DPCT_PARENT_FEATURE|pitched_data
// DPCT_DEPENDENCY_BEGIN
// Memory|pitched_data
// DPCT_DEPENDENCY_END
// DPCT_CODE
  size_t get_pitch() { return _pitch; }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|pitched_data_set_pitch|dpct
// DPCT_PARENT_FEATURE|pitched_data
// DPCT_DEPENDENCY_BEGIN
// Memory|pitched_data
// DPCT_DEPENDENCY_END
// DPCT_CODE
  void set_pitch(size_t pitch) { _pitch = pitch; }
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|pitched_data_get_x|dpct
// DPCT_PARENT_FEATURE|pitched_data
// DPCT_DEPENDENCY_BEGIN
// Memory|pitched_data
// DPCT_DEPENDENCY_END
// DPCT_CODE
  size_t get_x() { return _x; }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|pitched_data_set_x|dpct
// DPCT_PARENT_FEATURE|pitched_data
// DPCT_DEPENDENCY_BEGIN
// Memory|pitched_data
// DPCT_DEPENDENCY_END
// DPCT_CODE
  void set_x(size_t x) { _x = x; };
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|pitched_data_get_y|dpct
// DPCT_PARENT_FEATURE|pitched_data
// DPCT_DEPENDENCY_BEGIN
// Memory|pitched_data
// DPCT_DEPENDENCY_END
// DPCT_CODE
  size_t get_y() { return _y; }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|pitched_data_set_y|dpct
// DPCT_PARENT_FEATURE|pitched_data
// DPCT_DEPENDENCY_BEGIN
// Memory|pitched_data
// DPCT_DEPENDENCY_END
// DPCT_CODE
  void set_y(size_t y) { _y = y; }
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|pitched_data_1|dpct
// DPCT_DEPENDENCY_BEGIN
// Memory|pitched_data
// DPCT_DEPENDENCY_END
// DPCT_CODE
private:
  void *_data;
  size_t _pitch, _x, _y;
};
// DPCT_LABEL_END

namespace detail {
// DPCT_LABEL_BEGIN|mem_mgr|dpct::detail
// DPCT_DEPENDENCY_BEGIN
// Memory|typedef_byte_t
// Memory|typedef_buffer_t
// Memory|mem_mgr_1
// Memory|mem_mgr_2
// DPCT_DEPENDENCY_END
// DPCT_CODE
class mem_mgr {
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

  struct allocation {
    buffer_t buffer;
    byte_t *alloc_ptr;
    size_t size;
  };

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
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|mem_mgr_mem_alloc|dpct::detail
// DPCT_PARENT_FEATURE|mem_mgr
// DPCT_DEPENDENCY_BEGIN
// Memory|mem_mgr
// DPCT_DEPENDENCY_END
// DPCT_CODE
  /// Allocate
  void *mem_alloc(size_t size) {
    if (!size)
      return nullptr;
    std::lock_guard<std::mutex> lock(m_mutex);
    if (next_free + size > mapped_address_space + mapped_region_size) {
      throw std::runtime_error("dpct_malloc: out of memory for virtual memory pool");
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
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|mem_mgr_mem_free|dpct::detail
// DPCT_PARENT_FEATURE|mem_mgr
// DPCT_DEPENDENCY_BEGIN
// Memory|mem_mgr
// Memory|mem_mgr_get_map_iterator
// DPCT_DEPENDENCY_END
// DPCT_CODE
  /// Deallocate
  void mem_free(const void *ptr) {
    if (!ptr)
      return;
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = get_map_iterator(ptr);
    m_map.erase(it);
  }
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|mem_mgr_translate_ptr|dpct::detail
// DPCT_PARENT_FEATURE|mem_mgr
// DPCT_DEPENDENCY_BEGIN
// Memory|mem_mgr
// Memory|mem_mgr_get_map_iterator
// DPCT_DEPENDENCY_END
// DPCT_CODE
  /// map: device pointer -> allocation(buffer, alloc_ptr, size)
  allocation translate_ptr(const void *ptr) {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = get_map_iterator(ptr);
    return it->second;
  }
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|mem_mgr_is_device_ptr|dpct::detail
// DPCT_PARENT_FEATURE|mem_mgr
// DPCT_DEPENDENCY_BEGIN
// Memory|mem_mgr
// DPCT_DEPENDENCY_END
// DPCT_CODE
  /// Check if the pointer represents device pointer or not.
  bool is_device_ptr(const void *ptr) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return (mapped_address_space <= ptr) &&
           (ptr < mapped_address_space + mapped_region_size);
  }
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|mem_mgr_1|dpct::detail
// DPCT_PARENT_FEATURE|mem_mgr
// DPCT_DEPENDENCY_BEGIN
// Memory|mem_mgr
// Memory|mem_mgr_2
// DPCT_DEPENDENCY_END
// DPCT_CODE
  /// Returns the instance of memory manager singleton.
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
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|mem_mgr_get_map_iterator|dpct::detail
// DPCT_PARENT_FEATURE|mem_mgr
// DPCT_DEPENDENCY_BEGIN
// Memory|mem_mgr
// DPCT_DEPENDENCY_END
// DPCT_CODE
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
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|mem_mgr_2|dpct::detail
// DPCT_PARENT_FEATURE|mem_mgr
// DPCT_DEPENDENCY_BEGIN
// Memory|mem_mgr
// Memory|mem_mgr_1
// DPCT_DEPENDENCY_END
// DPCT_CODE
};
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|memory_traits|dpct::detail
// DPCT_DEPENDENCY_BEGIN
// Memory|typedef_byte_t
// Memory|memory_region
// DPCT_DEPENDENCY_END
// DPCT_CODE
template <class T, memory_region Memory, size_t Dimension> class accessor;
template <memory_region Memory, class T = byte_t> class memory_traits {
public:
  static constexpr cl::sycl::access::address_space asp =
      (Memory == local)
          ? cl::sycl::access::address_space::local_space
          : ((Memory == constant)
                 ? cl::sycl::access::address_space::constant_space
                 : cl::sycl::access::address_space::global_space);
  static constexpr cl::sycl::access::target target =
      (Memory == local)
          ? cl::sycl::access::target::local
          : ((Memory == constant) ? cl::sycl::access::target::constant_buffer
                                  : cl::sycl::access::target::device);
  static constexpr cl::sycl::access_mode mode =
      (Memory == constant) ? cl::sycl::access_mode::read
                           : cl::sycl::access_mode::read_write;
  static constexpr size_t type_size = sizeof(T);
  using element_t =
      typename std::conditional<Memory == constant, const T, T>::type;
  using value_t = typename std::remove_cv<T>::type;
  template <size_t Dimension = 1>
  using accessor_t = cl::sycl::accessor<T, Dimension, mode, target>;
  using pointer_t = T *;
};
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|dpct_malloc_detail|dpct::detail
// DPCT_DEPENDENCY_BEGIN
// Memory|mem_mgr_mem_alloc|UsmNone
// DPCT_DEPENDENCY_END
// DPCT_CODE
static inline void *dpct_malloc(size_t size, cl::sycl::queue &q) {
#ifdef DPCT_USM_LEVEL_NONE
  return mem_mgr::instance().mem_alloc(size * sizeof(byte_t));
#else
  return cl::sycl::malloc_device(size, q.get_device(), q.get_context());
#endif // DPCT_USM_LEVEL_NONE
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|dpct_malloc_pitch_detail|dpct::detail
// DPCT_DEPENDENCY_BEGIN
// Memory|dpct_malloc_detail
// DPCT_DEPENDENCY_END
// DPCT_CODE
#define PITCH_DEFAULT_ALIGN(x) (((x) + 31) & ~(0x1F))
static inline void *dpct_malloc(size_t &pitch, size_t x, size_t y, size_t z,
                                cl::sycl::queue &q) {
  pitch = PITCH_DEFAULT_ALIGN(x);
  return dpct_malloc(pitch * y * z, q);
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|dpct_memset_detail|dpct::detail
// DPCT_DEPENDENCY_BEGIN
// Memory|mem_mgr_is_device_ptr|UsmNone
// Memory|mem_mgr_translate_ptr|UsmNone
// Memory|typedef_byte_t|UsmNone
// DPCT_DEPENDENCY_END
// DPCT_CODE
/// Set \p value to the first \p size bytes starting from \p dev_ptr in \p q.
///
/// \param q The queue in which the operation is done.
/// \param dev_ptr Pointer to the device memory address.
/// \param value Value to be set.
/// \param size Number of bytes to be set to the value.
/// \returns An event representing the memset operation.
static inline cl::sycl::event dpct_memset(cl::sycl::queue &q, void *dev_ptr,
                                          int value, size_t size) {
#ifdef DPCT_USM_LEVEL_NONE
  auto &mm = mem_mgr::instance();
  assert(mm.is_device_ptr(dev_ptr));
  auto alloc = mm.translate_ptr(dev_ptr);
  size_t offset = (byte_t *)dev_ptr - alloc.alloc_ptr;

  return q.submit([&](cl::sycl::handler &cgh) {
    auto r = cl::sycl::range<1>(size);
    auto o = cl::sycl::id<1>(offset);
    cl::sycl::accessor<byte_t, 1, cl::sycl::access_mode::write,
                       cl::sycl::access::target::device>
        acc(alloc.buffer, cgh, r, o);
    cgh.fill(acc, (byte_t)value);
  });
#else
  return q.memset(dev_ptr, value, size);
#endif // DPCT_USM_LEVEL_NONE
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|dpct_memset_3d_detail|dpct::detail
// DPCT_DEPENDENCY_BEGIN
// Memory|pitched_data_get_pitch
// Memory|pitched_data_get_y
// Memory|pitched_data_get_data_ptr
// Memory|dpct_memset_detail
// DPCT_DEPENDENCY_END
// DPCT_CODE
/// Set \p value to the 3D memory region pointed by \p data in \p q. \p size
/// specifies the 3D memory size to set.
///
/// \param q The queue in which the operation is done.
/// \param data Pointer to the device memory region.
/// \param value Value to be set.
/// \param size Memory region size.
/// \returns An event list representing the memset operations..
static inline std::vector<cl::sycl::event>
dpct_memset(cl::sycl::queue &q, pitched_data data, int value,
            cl::sycl::range<3> size) {
  std::vector<cl::sycl::event> event_list;
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
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|dpct_memset_2d_detail|dpct::detail
// DPCT_DEPENDENCY_BEGIN
// Memory|pitched_data
// Memory|dpct_memset_3d_detail
// DPCT_DEPENDENCY_END
// DPCT_CODE
/// memset 2D matrix with pitch.
static inline std::vector<cl::sycl::event>
dpct_memset(cl::sycl::queue &q, void *ptr, size_t pitch, int val, size_t x,
            size_t y) {
  return dpct_memset(q, pitched_data(ptr, pitch, x, 1), val,
                     cl::sycl::range<3>(x, y, 1));
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|dpct_memcpy_detail|dpct::detail
// DPCT_DEPENDENCY_BEGIN
// Memory|memcpy_direction
// Memory|mem_mgr_is_device_ptr|UsmNone
// Memory|mem_mgr_translate_ptr|UsmNone
// Memory|typedef_byte_t|UsmNone
// Memory|typedef_buffer_t|UsmNone
// DPCT_DEPENDENCY_END
// DPCT_CODE
static cl::sycl::event dpct_memcpy(cl::sycl::queue &q, void *to_ptr,
                                   const void *from_ptr, size_t size,
                                   memcpy_direction direction) {
  if (!size)
    return cl::sycl::event{};
#ifdef DPCT_USM_LEVEL_NONE
  auto &mm = mem_mgr::instance();
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
  bool is_cpu = q.get_device().is_cpu();

  switch (real_direction) {
  case host_to_host:
    std::memcpy(to_ptr, from_ptr, size);
    return cl::sycl::event();
  case host_to_device: {
    auto alloc = mm.translate_ptr(to_ptr);
    size_t offset = (byte_t *)to_ptr - alloc.alloc_ptr;
    if(is_cpu) {
      buffer_t from_buffer((byte_t *)from_ptr, cl::sycl::range<1>(size), {cl::sycl::property::buffer::use_host_ptr()});
      return q.submit([&](cl::sycl::handler &cgh) {
        auto r = cl::sycl::range<1>(size);
        auto o = cl::sycl::id<1>(offset);
        auto from_acc = from_buffer.get_access<cl::sycl::access_mode::read>(cgh);
        cl::sycl::accessor<byte_t, 1, cl::sycl::access_mode::write,
                           cl::sycl::access::target::device>
            acc(alloc.buffer, cgh, r, o);
        cgh.parallel_for<class memcopyh2d>(r, [=](cl::sycl::id<1> idx) {
          acc[idx] = from_acc[idx];
          });
       });
    } else {
      return q.submit([&](cl::sycl::handler &cgh) {
        auto r = cl::sycl::range<1>(size);
        auto o = cl::sycl::id<1>(offset);
         cl::sycl::accessor<byte_t, 1, cl::sycl::access_mode::write,
                           cl::sycl::access::target::device>
            acc(alloc.buffer, cgh, r, o);
        cgh.copy(from_ptr, acc);
      });
    }
  }
  case device_to_host: {
    auto alloc = mm.translate_ptr(from_ptr);
    size_t offset = (byte_t *)from_ptr - alloc.alloc_ptr;
    if(is_cpu) {
      buffer_t to_buffer((byte_t *)to_ptr, cl::sycl::range<1>(size), {cl::sycl::property::buffer::use_host_ptr()});
      return q.submit([&](cl::sycl::handler &cgh) {
        auto r = cl::sycl::range<1>(size);
        auto o = cl::sycl::id<1>(offset);
        auto to_acc = to_buffer.get_access<cl::sycl::access_mode::write>(cgh);
        cl::sycl::accessor<byte_t, 1, cl::sycl::access_mode::read,
                           cl::sycl::access::target::device>
            acc(alloc.buffer, cgh, r, o);
        cgh.parallel_for<class memcopyd2h>(r, [=](cl::sycl::id<1> idx) {
          to_acc[idx] = acc[idx];
          });
      });
    } else {
      return q.submit([&](cl::sycl::handler &cgh) {
        auto r = cl::sycl::range<1>(size);
        auto o = cl::sycl::id<1>(offset);
        cl::sycl::accessor<byte_t, 1, cl::sycl::access_mode::read,
                           cl::sycl::access::target::device>
            acc(alloc.buffer, cgh, r, o);
        cgh.copy(acc, to_ptr);
      });
    }
  }
  case device_to_device: {
    auto to_alloc = mm.translate_ptr(to_ptr);
    auto from_alloc = mm.translate_ptr(from_ptr);
    size_t to_offset = (byte_t *)to_ptr - to_alloc.alloc_ptr;
    size_t from_offset = (byte_t *)from_ptr - from_alloc.alloc_ptr;
    if(is_cpu) {
      return q.submit([&](cl::sycl::handler &cgh) {
        auto r = cl::sycl::range<1>(size);
        auto to_o = cl::sycl::id<1>(to_offset);
        auto from_o = cl::sycl::id<1>(from_offset);
        cl::sycl::accessor<byte_t, 1, cl::sycl::access_mode::write,
                           cl::sycl::access::target::device>
            to_acc(to_alloc.buffer, cgh, r, to_o);
        cl::sycl::accessor<byte_t, 1, cl::sycl::access_mode::read,
                           cl::sycl::access::target::device>
            from_acc(from_alloc.buffer, cgh, r, from_o);
        cgh.parallel_for<class memcopyd2d>(r, [=](cl::sycl::id<1> idx) {
          to_acc[idx] = from_acc[idx];
          });
      });
    }else {
      return q.submit([&](cl::sycl::handler &cgh) {
        auto r = cl::sycl::range<1>(size);
        auto to_o = cl::sycl::id<1>(to_offset);
        auto from_o = cl::sycl::id<1>(from_offset);
        cl::sycl::accessor<byte_t, 1, cl::sycl::access_mode::write,
                           cl::sycl::access::target::device>
            to_acc(to_alloc.buffer, cgh, r, to_o);
        cl::sycl::accessor<byte_t, 1, cl::sycl::access_mode::read,
                           cl::sycl::access::target::device>
            from_acc(from_alloc.buffer, cgh, r, from_o);
        cgh.copy(from_acc, to_acc);
      });
    }
  }
  default:
    throw std::runtime_error("dpct_memcpy: invalid direction value");
  }
#else
  return q.memcpy(to_ptr, from_ptr, size);
#endif // DPCT_USM_LEVEL_NONE
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|dpct_memcpy_3d_detail|dpct::detail
// DPCT_DEPENDENCY_BEGIN
// Memory|memcpy_direction
// Memory|dpct_memcpy_detail
// DPCT_DEPENDENCY_END
// DPCT_CODE
/// copy 3D matrix specified by \p size from 3D matrix specified by \p from_ptr
/// and \p from_range to another specified by \p to_ptr and \p to_range.
static inline std::vector<cl::sycl::event>
dpct_memcpy(cl::sycl::queue &q, void *to_ptr, const void *from_ptr,
            cl::sycl::range<3> to_range, cl::sycl::range<3> from_range,
            cl::sycl::id<3> to_id, cl::sycl::id<3> from_id,
            cl::sycl::range<3> size, memcpy_direction direction) {
  std::vector<cl::sycl::event> event_list;

  size_t to_slice = to_range.get(1) * to_range.get(0),
         from_slice = from_range.get(1) * from_range.get(0);
  unsigned char *to_surface = (unsigned char *)to_ptr +
                              to_id.get(2) * to_slice +
                              to_id.get(1) * to_range.get(0) + to_id.get(0);
  const unsigned char *from_surface =
      (const unsigned char *)from_ptr + from_id.get(2) * from_slice +
      from_id.get(1) * from_range.get(0) + from_id.get(0);

  if (to_slice == from_slice && to_slice == size.get(1) * size.get(0)) {
    return {dpct_memcpy(q, to_surface, from_surface, to_slice * size.get(2),
                       direction)};
  }
  for (size_t z = 0; z < size.get(2); ++z) {
    unsigned char *to_ptr = to_surface;
    const unsigned char *from_ptr = from_surface;
    if (to_range.get(0) == from_range.get(0) &&
        to_range.get(0) == size.get(0)) {
      event_list.push_back(dpct_memcpy(q, to_ptr, from_ptr,
                                       size.get(0) * size.get(1), direction));
    } else {
      for (size_t y = 0; y < size.get(1); ++y) {
        event_list.push_back(
            dpct_memcpy(q, to_ptr, from_ptr, size.get(0), direction));
        to_ptr += to_range.get(0);
        from_ptr += from_range.get(0);
      }
    }
    to_surface += to_slice;
    from_surface += from_slice;
  }
  return event_list;
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|dpct_memcpy_2d_3d_pitch_detail|dpct::detail
// DPCT_DEPENDENCY_BEGIN
// Memory|memcpy_direction
// Memory|dpct_memcpy_3d_detail
// Memory|pitched_data_get_data_ptr
// Memory|pitched_data_get_pitch
// Memory|pitched_data_get_y
// DPCT_DEPENDENCY_END
// DPCT_CODE
/// memcpy 2D/3D matrix specified by pitched_data.
static inline std::vector<cl::sycl::event>
dpct_memcpy(cl::sycl::queue &q, pitched_data to, cl::sycl::id<3> to_id,
            pitched_data from, cl::sycl::id<3> from_id, cl::sycl::range<3> size,
            memcpy_direction direction = automatic) {
  return dpct_memcpy(q, to.get_data_ptr(), from.get_data_ptr(),
                     cl::sycl::range<3>(to.get_pitch(), to.get_y(), 1),
                     cl::sycl::range<3>(from.get_pitch(), from.get_y(), 1), to_id, from_id,
                     size, direction);
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|dpct_memcpy_2d_pitch_detail|dpct::detail
// DPCT_DEPENDENCY_BEGIN
// Memory|memcpy_direction
// Memory|dpct_memcpy_3d_detail
// DPCT_DEPENDENCY_END
// DPCT_CODE
/// memcpy 2D matrix with pitch.
static inline std::vector<cl::sycl::event>
dpct_memcpy(cl::sycl::queue &q, void *to_ptr, const void *from_ptr,
            size_t to_pitch, size_t from_pitch, size_t x, size_t y,
            memcpy_direction direction = automatic) {
  return dpct_memcpy(q, to_ptr, from_ptr, cl::sycl::range<3>(to_pitch, y, 1),
                     cl::sycl::range<3>(from_pitch, y, 1),
                     cl::sycl::id<3>(0, 0, 0), cl::sycl::id<3>(0, 0, 0),
                     cl::sycl::range<3>(x, y, 1), direction);
}
// DPCT_LABEL_END
} // namespace detail

// DPCT_LABEL_BEGIN|is_device_ptr|dpct
// DPCT_DEPENDENCY_BEGIN
// Memory|mem_mgr_is_device_ptr
// DPCT_DEPENDENCY_END
// DPCT_CODE
#ifdef DPCT_USM_LEVEL_NONE
/// Check if the pointer \p ptr represents device pointer or not.
///
/// \param ptr The pointer to be checked.
/// \returns true if \p ptr is a device pointer.
template<class T>
static inline bool is_device_ptr(T ptr) {
  if constexpr (std::is_pointer<T>::value) {
    return detail::mem_mgr::instance().is_device_ptr(ptr);
  }
  return false;
}
#endif
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|get_buffer_and_offset|dpct
// DPCT_DEPENDENCY_BEGIN
// Memory|typedef_buffer_t
// Memory|mem_mgr_translate_ptr
// DPCT_DEPENDENCY_END
// DPCT_CODE
/// Get the buffer and the offset of a piece of memory pointed to by \p ptr.
///
/// \param ptr Pointer to a piece of memory.
/// If NULL is passed as an argument, an exception will be thrown.
/// \returns a pair containing both the buffer and the offset.
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
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|get_buffer_T|dpct
// DPCT_DEPENDENCY_BEGIN
// Memory|mem_mgr_translate_ptr
// DPCT_DEPENDENCY_END
// DPCT_CODE
/// Get the data pointed from \p ptr as a 1D buffer reinterpreted as type T.
template <typename T> static cl::sycl::buffer<T> get_buffer(const void *ptr) {
  auto alloc = detail::mem_mgr::instance().translate_ptr(ptr);
  return alloc.buffer.reinterpret<T>(
      cl::sycl::range<1>(alloc.size / sizeof(T)));
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|get_buffer_byte_t|dpct
// DPCT_DEPENDENCY_BEGIN
// Memory|typedef_buffer_t
// Memory|mem_mgr_translate_ptr
// DPCT_DEPENDENCY_END
// DPCT_CODE
/// Get the buffer of a piece of memory pointed to by \p ptr.
///
/// \param ptr Pointer to a piece of memory.
/// \returns the buffer.
static buffer_t get_buffer(const void *ptr) {
  return detail::mem_mgr::instance().translate_ptr(ptr).buffer;
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|access_wrapper|dpct
// DPCT_DEPENDENCY_BEGIN
// Memory|typedef_byte_t
// Memory|mem_mgr_translate_ptr
// Memory|get_buffer_byte_t
// DPCT_DEPENDENCY_END
// DPCT_CODE
/// A wrapper class contains an accessor and an offset.
template <typename dataT,
          cl::sycl::access_mode accessMode = cl::sycl::access_mode::read_write>
class access_wrapper {
  cl::sycl::accessor<byte_t, 1, accessMode> accessor;
  size_t offset;

public:
  /// Construct the accessor wrapper for memory pointed by \p ptr.
  ///
  /// \param ptr Pointer to memory.
  /// \param cgh The command group handler.
  access_wrapper(const void *ptr, cl::sycl::handler &cgh)
      : accessor(get_buffer(ptr).get_access<accessMode>(cgh)), offset(0) {
    auto alloc = detail::mem_mgr::instance().translate_ptr(ptr);
    offset = (byte_t *)ptr - alloc.alloc_ptr;
  }

  /// Get the device pointer.
  ///
  /// \returns a device pointer with offset.
  dataT get_raw_pointer() const { return (dataT)(&accessor[0] + offset); }
};
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|get_access|dpct
// DPCT_DEPENDENCY_BEGIN
// Memory|typedef_byte_t
// Memory|mem_mgr_translate_ptr
// DPCT_DEPENDENCY_END
// DPCT_CODE
/// Get the accessor for memory pointed by \p ptr.
///
/// \param ptr Pointer to memory.
/// If NULL is passed as an argument, an exception will be thrown.
/// \param cgh The command group handler.
/// \returns an accessor.
template <cl::sycl::access_mode accessMode = cl::sycl::access_mode::read_write>
static cl::sycl::accessor<byte_t, 1, accessMode>
get_access(const void *ptr, cl::sycl::handler &cgh) {
  if (ptr) {
    auto alloc = detail::mem_mgr::instance().translate_ptr(ptr);
    return alloc.buffer.get_access<accessMode>(cgh);
  } else {
    throw std::runtime_error(
        "NULL pointer argument in get_access function is invalid");
  }
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|dpct_malloc|dpct
// DPCT_DEPENDENCY_BEGIN
// Device|get_default_queue
// Memory|dpct_malloc_detail
// DPCT_DEPENDENCY_END
// DPCT_CODE
/// Allocate memory block on the device.
/// \param num_bytes Number of bytes to allocate.
/// \param q Queue to execute the allocate task.
/// \returns A pointer to the newly allocated memory.
template <typename T>
static inline void *dpct_malloc(T num_bytes,
                                cl::sycl::queue &q = get_default_queue()) {
  return detail::dpct_malloc(static_cast<size_t>(num_bytes), q);
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|get_host_ptr|dpct
// DPCT_DEPENDENCY_BEGIN
// Memory|get_buffer_and_offset
// DPCT_DEPENDENCY_END
// DPCT_CODE
/// Get the host pointer from a buffer that is mapped to virtual pointer ptr.
/// \param ptr Virtual Pointer mapped to device buffer
/// \returns A host pointer
template <typename T> static inline T *get_host_ptr(const void *ptr) {
  auto BufferOffset = get_buffer_and_offset(ptr);
  auto host_ptr =
      BufferOffset.first.get_access<cl::sycl::access_mode::read_write>()
          .get_pointer();
  return (T *)(host_ptr + BufferOffset.second);
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|dpct_malloc_3d|dpct
// DPCT_DEPENDENCY_BEGIN
// Device|get_default_queue
// Memory|dpct_malloc_pitch_detail
// Memory|pitched_data_set_data_ptr
// Memory|pitched_data_set_pitch
// DPCT_DEPENDENCY_END
// DPCT_CODE
/// Allocate memory block for 3D array on the device.
/// \param size Size of of the memory block, in bytes.
/// \param q Queue to execute the allocate task.
/// \returns A pitched_data object which stores the memory info.
static inline pitched_data
dpct_malloc(cl::sycl::range<3> size, cl::sycl::queue &q = get_default_queue()) {
  pitched_data pitch(nullptr, 0, size.get(0), size.get(1));
  size_t pitch_size;
  pitch.set_data_ptr(detail::dpct_malloc(pitch_size, size.get(0), size.get(1),
                                         size.get(2), q));
  pitch.set_pitch(pitch_size);
  return pitch;
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|dpct_malloc_2d|dpct
// DPCT_DEPENDENCY_BEGIN
// Device|get_default_queue
// Memory|dpct_malloc_pitch_detail
// DPCT_DEPENDENCY_END
// DPCT_CODE
/// Allocate memory block for 2D array on the device.
/// \param [out] pitch Aligned size of x in bytes.
/// \param x Range in dim x.
/// \param y Range in dim y.
/// \param q Queue to execute the allocate task.
/// \returns A pointer to the newly allocated memory.
static inline void *dpct_malloc(size_t &pitch, size_t x, size_t y,
                                cl::sycl::queue &q = get_default_queue()) {
  return detail::dpct_malloc(pitch, x, y, 1, q);
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|dpct_free|dpct
// DPCT_DEPENDENCY_BEGIN
// Device|get_default_queue
// Memory|mem_mgr_mem_free|UsmNone
// DPCT_DEPENDENCY_END
// DPCT_CODE
/// free
/// \param ptr Point to free.
/// \param q Queue to execute the free task.
/// \returns no return value.
static inline void dpct_free(void *ptr,
                             cl::sycl::queue &q = get_default_queue()) {
  if (ptr) {
#ifdef DPCT_USM_LEVEL_NONE
    detail::mem_mgr::instance().mem_free(ptr);
#else
    cl::sycl::free(ptr, q.get_context());
#endif // DPCT_USM_LEVEL_NONE
  }
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|async_dpct_free|dpct
// DPCT_DEPENDENCY_BEGIN
// Device|get_default_queue
// Device|get_current_device
// Device|device_ext_add_task
// Device|device_ext_async_dpct_free_friend_decl
// DPCT_DEPENDENCY_END
// DPCT_CODE
#ifndef DPCT_USM_LEVEL_NONE
/// Free the device memory pointed by a batch of pointers in \p pointers which
/// are related to \p q after \p events completed.
///
/// \param pointers The pointers point to the device memory requested to be freed.
/// \param events The events to be waited.
/// \param q The sycl::queue the memory relates to.
inline void async_dpct_free(std::vector<void *> pointers,
                            std::vector<cl::sycl::event> events,
                            cl::sycl::queue &q = get_default_queue()) {
  std::thread t(
      [](std::vector<void *> pointers, std::vector<cl::sycl::event> events,
         cl::sycl::context ctxt) {
        cl::sycl::event::wait(events);
        for (auto p : pointers)
          cl::sycl::free(p, ctxt);
      },
      std::move(pointers), std::move(events), q.get_context());
  get_current_device().add_task(std::move(t));
}
#endif
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|dpct_memcpy|dpct
// DPCT_DEPENDENCY_BEGIN
// Device|get_default_queue
// Memory|memcpy_direction
// Memory|dpct_memcpy_detail
// DPCT_DEPENDENCY_END
// DPCT_CODE
/// Synchronously copies \p size bytes from the address specified by \p from_ptr
/// to the address specified by \p to_ptr. The value of \p direction is used to
/// set the copy direction, it can be \a host_to_host, \a host_to_device,
/// \a device_to_host, \a device_to_device or \a automatic. The function will
/// return after the copy is completed.
///
/// \param to_ptr Pointer to destination memory address.
/// \param from_ptr Pointer to source memory address.
/// \param size Number of bytes to be copied.
/// \param direction Direction of the copy.
/// \param q Queue to execute the copy task.
/// \returns no return value.
static void dpct_memcpy(void *to_ptr, const void *from_ptr, size_t size,
                        memcpy_direction direction = automatic,
                        cl::sycl::queue &q = get_default_queue()) {
  detail::dpct_memcpy(q, to_ptr, from_ptr, size, direction).wait();
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|async_dpct_memcpy|dpct
// DPCT_DEPENDENCY_BEGIN
// Device|get_default_queue
// Memory|memcpy_direction
// Memory|dpct_memcpy_detail
// DPCT_DEPENDENCY_END
// DPCT_CODE
/// Asynchronously copies \p size bytes from the address specified by \p
/// from_ptr to the address specified by \p to_ptr. The value of \p direction is
/// used to set the copy direction, it can be \a host_to_host, \a
/// host_to_device, \a device_to_host, \a device_to_device or \a automatic. The
/// return of the function does NOT guarantee the copy is completed.
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
  detail::dpct_memcpy(q, to_ptr, from_ptr, size, direction);
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|dpct_memcpy_2d|dpct
// DPCT_DEPENDENCY_BEGIN
// Device|get_default_queue
// Memory|memcpy_direction
// Memory|dpct_memcpy_2d_pitch_detail
// DPCT_DEPENDENCY_END
// DPCT_CODE
/// Synchronously copies 2D matrix specified by \p x and \p y from the address
/// specified by \p from_ptr to the address specified by \p to_ptr, while \p
/// from_pitch and \p to_pitch are the range of dim x in bytes of the matrix
/// specified by \p from_ptr and \p to_ptr. The value of \p direction is used to
/// set the copy direction, it can be \a host_to_host, \a host_to_device, \a
/// device_to_host, \a device_to_device or \a automatic. The function will
/// return after the copy is completed.
///
/// \param to_ptr Pointer to destination memory address.
/// \param to_pitch Range of dim x in bytes of destination matrix.
/// \param from_ptr Pointer to source memory address.
/// \param from_pitch Range of dim x in bytes of source matrix.
/// \param x Range of dim x of matrix to be copied.
/// \param y Range of dim y of matrix to be copied.
/// \param direction Direction of the copy.
/// \param q Queue to execute the copy task.
/// \returns no return value.
static inline void dpct_memcpy(void *to_ptr, size_t to_pitch,
                               const void *from_ptr, size_t from_pitch,
                               size_t x, size_t y,
                               memcpy_direction direction = automatic,
                               cl::sycl::queue &q = dpct::get_default_queue()) {
  cl::sycl::event::wait(detail::dpct_memcpy(q, to_ptr, from_ptr, to_pitch,
                                            from_pitch, x, y, direction));
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|async_dpct_memcpy_2d|dpct
// DPCT_DEPENDENCY_BEGIN
// Device|get_default_queue
// Memory|memcpy_direction
// Memory|dpct_memcpy_2d_pitch_detail
// DPCT_DEPENDENCY_END
// DPCT_CODE
/// Asynchronously copies 2D matrix specified by \p x and \p y from the address
/// specified by \p from_ptr to the address specified by \p to_ptr, while \p
/// \p from_pitch and \p to_pitch are the range of dim x in bytes of the matrix
/// specified by \p from_ptr and \p to_ptr. The value of \p direction is used to
/// set the copy direction, it can be \a host_to_host, \a host_to_device, \a
/// device_to_host, \a device_to_device or \a automatic. The return of the
/// function does NOT guarantee the copy is completed.
///
/// \param to_ptr Pointer to destination memory address.
/// \param to_pitch Range of dim x in bytes of destination matrix.
/// \param from_ptr Pointer to source memory address.
/// \param from_pitch Range of dim x in bytes of source matrix.
/// \param x Range of dim x of matrix to be copied.
/// \param y Range of dim y of matrix to be copied.
/// \param direction Direction of the copy.
/// \param q Queue to execute the copy task.
/// \returns no return value.
static inline void
async_dpct_memcpy(void *to_ptr, size_t to_pitch, const void *from_ptr,
                  size_t from_pitch, size_t x, size_t y,
                  memcpy_direction direction = automatic,
                  cl::sycl::queue &q = get_default_queue()) {
  detail::dpct_memcpy(q, to_ptr, from_ptr, to_pitch, from_pitch, x, y,
                      direction);
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|dpct_memcpy_3d|dpct
// DPCT_DEPENDENCY_BEGIN
// Device|get_default_queue
// Memory|memcpy_direction
// Memory|dpct_memcpy_2d_3d_pitch_detail
// Memory|pitched_data
// DPCT_DEPENDENCY_END
// DPCT_CODE
/// Synchronously copies a subset of a 3D matrix specified by \p to to another
/// 3D matrix specified by \p from. The from and to position info are specified
/// by \p from_pos and \p to_pos The copied matrix size is specfied by \p size.
/// The value of \p direction is used to set the copy direction, it can be \a
/// host_to_host, \a host_to_device, \a device_to_host, \a device_to_device or
/// \a automatic. The function will return after the copy is completed.
///
/// \param to Destination matrix info.
/// \param to_pos Position of destination.
/// \param from Source matrix info.
/// \param from_pos Position of destination.
/// \param size Range of the submatrix to be copied.
/// \param direction Direction of the copy.
/// \param q Queue to execute the copy task.
/// \returns no return value.
static inline void dpct_memcpy(pitched_data to, cl::sycl::id<3> to_pos,
                               pitched_data from, cl::sycl::id<3> from_pos,
                               cl::sycl::range<3> size,
                               memcpy_direction direction = automatic,
                               cl::sycl::queue &q = dpct::get_default_queue()) {
  cl::sycl::event::wait(
      detail::dpct_memcpy(q, to, to_pos, from, from_pos, size, direction));
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|async_dpct_memcpy_3d|dpct
// DPCT_DEPENDENCY_BEGIN
// Device|get_default_queue
// Memory|memcpy_direction
// Memory|dpct_memcpy_2d_3d_pitch_detail
// DPCT_DEPENDENCY_END
// DPCT_CODE
/// Asynchronously copies a subset of a 3D matrix specified by \p to to another
/// 3D matrix specified by \p from. The from and to position info are specified
/// by \p from_pos and \p to_pos The copied matrix size is specfied by \p size.
/// The value of \p direction is used to set the copy direction, it can be \a
/// host_to_host, \a host_to_device, \a device_to_host, \a device_to_device or
/// \a automatic. The return of the function does NOT guarantee the copy is
/// completed.
///
/// \param to Destination matrix info.
/// \param to_pos Position of destination.
/// \param from Source matrix info.
/// \param from_pos Position of destination.
/// \param size Range of the submatrix to be copied.
/// \param direction Direction of the copy.
/// \param q Queue to execute the copy task.
/// \returns no return value.
static inline void
async_dpct_memcpy(pitched_data to, cl::sycl::id<3> to_pos, pitched_data from,
                  cl::sycl::id<3> from_pos, cl::sycl::range<3> size,
                  memcpy_direction direction = automatic,
                  cl::sycl::queue &q = get_default_queue()) {
  detail::dpct_memcpy(q, to, to_pos, from, from_pos, size, direction);
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|dpct_memset|dpct
// DPCT_DEPENDENCY_BEGIN
// Device|get_default_queue
// Memory|dpct_memset_detail
// DPCT_DEPENDENCY_END
// DPCT_CODE
/// Synchronously sets \p value to the first \p size bytes starting from \p
/// dev_ptr. The function will return after the memset operation is completed.
///
/// \param dev_ptr Pointer to the device memory address.
/// \param value Value to be set.
/// \param size Number of bytes to be set to the value.
/// \param q The queue in which the operation is done.
/// \returns no return value.
static void dpct_memset(void *dev_ptr, int value, size_t size,
                        cl::sycl::queue &q = get_default_queue()) {
  detail::dpct_memset(q, dev_ptr, value, size).wait();
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|async_dpct_memset|dpct
// DPCT_DEPENDENCY_BEGIN
// Device|get_default_queue
// Memory|dpct_memset_detail
// DPCT_DEPENDENCY_END
// DPCT_CODE
/// Asynchronously sets \p value to the first \p size bytes starting from \p
/// dev_ptr. The return of the function does NOT guarantee the memset operation
/// is completed.
///
/// \param dev_ptr Pointer to the device memory address.
/// \param value Value to be set.
/// \param size Number of bytes to be set to the value.
/// \returns no return value.
static void async_dpct_memset(void *dev_ptr, int value, size_t size,
                              cl::sycl::queue &q = dpct::get_default_queue()) {
  detail::dpct_memset(q, dev_ptr, value, size);
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|dpct_memset_2d|dpct
// DPCT_DEPENDENCY_BEGIN
// Device|get_default_queue
// Memory|dpct_memset_2d_detail
// DPCT_DEPENDENCY_END
// DPCT_CODE
/// Sets \p value to the 2D memory region pointed by \p ptr in \p q. \p x and
/// \p y specify the setted 2D memory size. \p pitch is the bytes in linear
/// dimension, including padding bytes. The function will return after the
/// memset operation is completed.
///
/// \param ptr Pointer to the device memory region.
/// \param pitch Bytes in linear dimension, including padding bytes.
/// \param value Value to be set.
/// \param x The setted memory size in linear dimension.
/// \param y The setted memory size in second dimension.
/// \param q The queue in which the operation is done.
/// \returns no return value.
static inline void dpct_memset(void *ptr, size_t pitch, int val, size_t x,
                               size_t y,
                               cl::sycl::queue &q = get_default_queue()) {
  cl::sycl::event::wait(detail::dpct_memset(q, ptr, pitch, val, x, y));
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|async_dpct_memset_2d|dpct
// DPCT_DEPENDENCY_BEGIN
// Device|get_default_queue
// Memory|dpct_memset_2d_detail
// DPCT_DEPENDENCY_END
// DPCT_CODE
/// Sets \p value to the 2D memory region pointed by \p ptr in \p q. \p x and
/// \p y specify the setted 2D memory size. \p pitch is the bytes in linear
/// dimension, including padding bytes. The return of the function does NOT
/// guarantee the memset operation is completed.
///
/// \param ptr Pointer to the device memory region.
/// \param pitch Bytes in linear dimension, including padding bytes.
/// \param value Value to be set.
/// \param x The setted memory size in linear dimension.
/// \param y The setted memory size in second dimension.
/// \param q The queue in which the operation is done.
/// \returns no return value.
static inline void async_dpct_memset(void *ptr, size_t pitch, int val, size_t x,
                                     size_t y,
                                     cl::sycl::queue &q = get_default_queue()) {
  detail::dpct_memset(q, ptr, pitch, val, x, y);
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|dpct_memset_3d|dpct
// DPCT_DEPENDENCY_BEGIN
// Device|get_default_queue
// Memory|dpct_memset_3d_detail
// DPCT_DEPENDENCY_END
// DPCT_CODE
/// Sets \p value to the 3D memory region specified by \p pitch in \p q. \p size
/// specify the setted 3D memory size. The function will return after the
/// memset operation is completed.
///
/// \param pitch Specify the 3D memory region.
/// \param value Value to be set.
/// \param size The setted 3D memory size.
/// \param q The queue in which the operation is done.
/// \returns no return value.
static inline void dpct_memset(pitched_data pitch, int val,
                               cl::sycl::range<3> size,
                               cl::sycl::queue &q = get_default_queue()) {
  cl::sycl::event::wait(detail::dpct_memset(q, pitch, val, size));
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|async_dpct_memset_3d|dpct
// DPCT_DEPENDENCY_BEGIN
// Device|get_default_queue
// Memory|dpct_memset_3d_detail
// DPCT_DEPENDENCY_END
// DPCT_CODE
/// Sets \p value to the 3D memory region specified by \p pitch in \p q. \p size
/// specify the setted 3D memory size. The return of the function does NOT
/// guarantee the memset operation is completed.
///
/// \param pitch Specify the 3D memory region.
/// \param value Value to be set.
/// \param size The setted 3D memory size.
/// \param q The queue in which the operation is done.
/// \returns no return value.
static inline void async_dpct_memset(pitched_data pitch, int val,
                                     cl::sycl::range<3> size,
                                     cl::sycl::queue &q = get_default_queue()) {
  detail::dpct_memset(q, pitch, val, size);
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|dpct_accessor|dpct
// DPCT_DEPENDENCY_BEGIN
// Memory|memory_traits
// Memory|memory_region
// DPCT_DEPENDENCY_END
// DPCT_CODE
/// dpct accessor used as device function parameter.
template <class T, memory_region Memory, size_t Dimension> class accessor;
template <class T, memory_region Memory> class accessor<T, Memory, 3> {
public:
  using memory_t = detail::memory_traits<Memory, T>;
  using element_t = typename memory_t::element_t;
  using pointer_t = typename memory_t::pointer_t;
  using accessor_t = typename memory_t::template accessor_t<3>;
  accessor(pointer_t data, const cl::sycl::range<3> &in_range)
      : _data(data), _range(in_range) {}
  template <memory_region M = Memory>
  accessor(typename std::enable_if<M != local, const accessor_t>::type &acc)
      : accessor(acc, acc.get_range()) {}
  accessor(const accessor_t &acc, const cl::sycl::range<3> &in_range)
      : accessor(acc.get_pointer(), in_range) {}
  accessor<T, Memory, 2> operator[](size_t index) const {
    cl::sycl::range<2> sub(_range.get(1), _range.get(2));
    return accessor<T, Memory, 2>(_data + index * sub.size(), sub);
  }

private:
  pointer_t _data;
  cl::sycl::range<3> _range;
};
template <class T, memory_region Memory> class accessor<T, Memory, 2> {
public:
  using memory_t = detail::memory_traits<Memory, T>;
  using element_t = typename memory_t::element_t;
  using pointer_t = typename memory_t::pointer_t;
  using accessor_t = typename memory_t::template accessor_t<2>;
  accessor(pointer_t data, const cl::sycl::range<2> &in_range)
      : _data(data), _range(in_range) {}
  template <memory_region M = Memory>
  accessor(typename std::enable_if<M != local, const accessor_t>::type &acc)
      : accessor(acc, acc.get_range()) {}
  accessor(const accessor_t &acc, const cl::sycl::range<2> &in_range)
      : accessor(acc.get_pointer(), in_range) {}

  pointer_t operator[](size_t index) const {
    return _data + _range.get(1) * index;
  }

private:
  pointer_t _data;
  cl::sycl::range<2> _range;
};
// DPCT_LABEL_END

namespace detail {
// DPCT_LABEL_BEGIN|device_memory|dpct::detail
// DPCT_DEPENDENCY_BEGIN
// Memory|device_memory_1
// Memory|device_memory_2
// Memory|device_memory_3
// Memory|device_memory_4
// Memory|device_memory_5
// Memory|device_memory_6
// Memory|device_memory_value_t_alias
// Memory|memory_region
// DPCT_DEPENDENCY_END
// DPCT_CODE
/// Device variable with address space of shared, global or constant.
template <class T, memory_region Memory, size_t Dimension>
class device_memory {
public:
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|device_memory_accessor_t_alias|dpct::detail
// DPCT_PARENT_FEATURE|device_memory
// DPCT_DEPENDENCY_BEGIN
// Memory|device_memory
// Memory|memory_traits
// DPCT_DEPENDENCY_END
// DPCT_CODE
  using accessor_t =
      typename detail::memory_traits<Memory, T>::template accessor_t<Dimension>;
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|device_memory_value_t_alias|dpct::detail
// DPCT_PARENT_FEATURE|device_memory
// DPCT_DEPENDENCY_BEGIN
// Memory|device_memory
// Memory|memory_traits
// DPCT_DEPENDENCY_END
// DPCT_CODE
  using value_t = typename detail::memory_traits<Memory, T>::value_t;
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|device_memory_dpct_accessor_t_alias|dpct::detail
// DPCT_PARENT_FEATURE|device_memory
// DPCT_DEPENDENCY_BEGIN
// Memory|device_memory
// Memory|dpct_accessor
// DPCT_DEPENDENCY_END
// DPCT_CODE
  using dpct_accessor_t = dpct::accessor<T, Memory, Dimension>;
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|device_memory_1|dpct::detail
// DPCT_PARENT_FEATURE|device_memory
// DPCT_DEPENDENCY_BEGIN
// Memory|device_memory
// Memory|mem_mgr
// Device|dev_mgr
// Memory|dpct_free
// DPCT_DEPENDENCY_END
// DPCT_CODE
  device_memory() : device_memory(cl::sycl::range<Dimension>(1)) {}

  /// Constructor of 1-D array with initializer list
  template <size_t D = Dimension>
  device_memory(
      const typename std::enable_if<D == 1, cl::sycl::range<1>>::type &in_range,
      std::initializer_list<value_t> &&init_list)
      : device_memory(in_range) {
    assert(init_list.size() <= in_range.size());
    _host_ptr = (value_t *)std::malloc(_size);
    std::memset(_host_ptr, 0, _size);
    std::memcpy(_host_ptr, init_list.begin(), init_list.size() * sizeof(T));
  }

  /// Constructor of 2-D array with initializer list
  template <size_t D = Dimension>
  device_memory(
      const typename std::enable_if<D == 2, cl::sycl::range<2>>::type &in_range,
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

  /// Constructor with range
  device_memory(const cl::sycl::range<Dimension> &range_in)
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

  /// Constructor with range
  template <class... Args>
  device_memory(Args... Arguments)
      : device_memory(cl::sycl::range<Dimension>(Arguments...)) {}

  ~device_memory() {
    if (_device_ptr && !_reference)
      dpct_free(_device_ptr);
    if (_host_ptr)
      std::free(_host_ptr);
  }
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|device_memory_init|dpct::detail
// DPCT_PARENT_FEATURE|device_memory
// DPCT_DEPENDENCY_BEGIN
// Memory|device_memory
// Memory|device_memory_init_q
// Device|get_default_queue
// DPCT_DEPENDENCY_END
// DPCT_CODE
  /// Allocate memory with default queue, and init memory if has initial value.
  void init() {
    init(dpct::get_default_queue());
  }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|device_memory_init_q|dpct::detail
// DPCT_PARENT_FEATURE|device_memory
// DPCT_DEPENDENCY_BEGIN
// Memory|device_memory
// Memory|dpct_memcpy_detail
// Memory|memcpy_direction
// Memory|device_memory_allocate_device
// DPCT_DEPENDENCY_END
// DPCT_CODE
  /// Allocate memory with specficed queue, and init memory if has initial value.
  void init(cl::sycl::queue &q) {
    if (_device_ptr)
      return;
    if (!_size)
      return;
    allocate_device(q);
    if (_host_ptr)
      detail::dpct_memcpy(q, _device_ptr, _host_ptr, _size, host_to_device);
  }
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|device_memory_assign|dpct::detail
// DPCT_PARENT_FEATURE|device_memory
// DPCT_DEPENDENCY_BEGIN
// Memory|device_memory
// DPCT_DEPENDENCY_END
// DPCT_CODE
  /// The variable is assigned to a device pointer.
  void assign(value_t *src, size_t size) {
    this->~device_memory();
    new (this) device_memory(src, size);
  }
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|device_memory_get_ptr|dpct::detail
// DPCT_PARENT_FEATURE|device_memory
// DPCT_DEPENDENCY_BEGIN
// Memory|device_memory
// Memory|device_memory_get_ptr_q
// Device|get_default_queue
// DPCT_DEPENDENCY_END
// DPCT_CODE
  /// Get memory pointer of the memory object, which is virtual pointer when
  /// usm is not used, and device pointer when usm is used .
  value_t *get_ptr() {
    return get_ptr(get_default_queue());
  }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|device_memory_get_ptr_q|dpct::detail
// DPCT_PARENT_FEATURE|device_memory
// DPCT_DEPENDENCY_BEGIN
// Memory|device_memory
// Memory|device_memory_init_q
// DPCT_DEPENDENCY_END
// DPCT_CODE
  /// Get memory pointer of the memory object, which is virtual pointer when
  /// usm is not used, and device pointer when usm is used .
  value_t *get_ptr(cl::sycl::queue &q) {
    init(q);
    return _device_ptr;
  }
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|device_memory_get_size|dpct::detail
// DPCT_PARENT_FEATURE|device_memory
// DPCT_DEPENDENCY_BEGIN
// Memory|device_memory
// DPCT_DEPENDENCY_END
// DPCT_CODE
  /// Get the device memory object size in bytes.
  size_t get_size() { return _size; }
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|device_memory_2|dpct::detail
// DPCT_PARENT_FEATURE|device_memory
// DPCT_DEPENDENCY_BEGIN
// Memory|device_memory
// Memory|get_buffer_T|UsmNone
// Memory|device_memory_init
// DPCT_DEPENDENCY_END
// DPCT_CODE
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
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|device_memory_get_access|dpct::detail
// DPCT_PARENT_FEATURE|device_memory
// DPCT_DEPENDENCY_BEGIN
// Memory|device_memory
// Memory|get_buffer_byte_t|UsmNone
// Memory|device_memory_dpct_accessor_t_alias|UsmRestricted
// Memory|device_memory_get_access_sp|UsmNone
// Memory|device_memory_accessor_t_alias|UsmNone
// DPCT_DEPENDENCY_END
// DPCT_CODE
#ifdef DPCT_USM_LEVEL_NONE
  /// Get cl::sycl::accessor for the device memory object when usm is not used.
  accessor_t get_access(cl::sycl::handler &cgh) {
    return get_buffer(_device_ptr)
        .template reinterpret<T, Dimension>(_range)
        .template get_access<detail::memory_traits<Memory, T>::mode,
                             detail::memory_traits<Memory, T>::target>(cgh);
  }
#else
  /// Get dpct::accessor with dimension info for the device memory object
  /// when usm is used and dimension is greater than 1.
  template <size_t D = Dimension>
  typename std::enable_if<D != 1, dpct_accessor_t>::type
  get_access(cl::sycl::handler &cgh) {
    return dpct_accessor_t((T *)_device_ptr, _range);
  }
#endif // DPCT_USM_LEVEL_NONE
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|device_memory_3|dpct::detail
// DPCT_PARENT_FEATURE|device_memory
// DPCT_DEPENDENCY_BEGIN
// Memory|device_memory
// DPCT_DEPENDENCY_END
// DPCT_CODE
private:
  device_memory(value_t *memory_ptr, size_t size)
      : _size(size), _range(size / sizeof(T)), _reference(true),
        _device_ptr(memory_ptr) {}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|device_memory_allocate_device|dpct::detail
// DPCT_PARENT_FEATURE|device_memory
// DPCT_DEPENDENCY_BEGIN
// Memory|device_memory
// Memory|dpct_malloc_detail
// DPCT_DEPENDENCY_END
// DPCT_CODE
  void allocate_device(cl::sycl::queue &q) {
#ifndef DPCT_USM_LEVEL_NONE
    if (Memory == shared) {
      _device_ptr = (value_t *)cl::sycl::malloc_shared(
          _size, q.get_device(), q.get_context());
      return;
    }
#endif
    _device_ptr = (value_t *)detail::dpct_malloc(_size, q);
  }
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|device_memory_4|dpct::detail
// DPCT_PARENT_FEATURE|device_memory
// DPCT_DEPENDENCY_BEGIN
// Memory|device_memory
// DPCT_DEPENDENCY_END
// DPCT_CODE
  size_t _size;
  cl::sycl::range<Dimension> _range;
  bool _reference;
  value_t *_host_ptr;
  value_t *_device_ptr;
};
template <class T, memory_region Memory>
class device_memory<T, Memory, 0> : public device_memory<T, Memory, 1> {
public:
  using base = device_memory<T, Memory, 1>;
  using value_t = typename base::value_t;
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|device_memory_accessor_t_alias_1|dpct::detail
// DPCT_PARENT_FEATURE|device_memory
// DPCT_DEPENDENCY_BEGIN
// Memory|device_memory
// DPCT_DEPENDENCY_END
// DPCT_CODE
  using accessor_t =
      typename detail::memory_traits<Memory, T>::template accessor_t<0>;
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|device_memory_5|dpct::detail
// DPCT_PARENT_FEATURE|device_memory
// DPCT_DEPENDENCY_BEGIN
// Memory|device_memory
// DPCT_DEPENDENCY_END
// DPCT_CODE
  /// Constructor with initial value.
  device_memory(const value_t &val) : base(cl::sycl::range<1>(1), {val}) {}

  /// Default constructor
  device_memory() : base(1) {}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|device_memory_get_access_sp|dpct::detail
// DPCT_PARENT_FEATURE|device_memory
// DPCT_DEPENDENCY_BEGIN
// Memory|device_memory
// Memory|device_memory_accessor_t_alias_1
// Memory|get_buffer_byte_t
// Memory|device_memory_get_ptr
// Memory|device_memory_get_access
// DPCT_DEPENDENCY_END
// DPCT_CODE
#ifdef DPCT_USM_LEVEL_NONE
  /// Get cl::sycl::accessor for the device memory object when usm is not used.
  accessor_t get_access(cl::sycl::handler &cgh) {
    auto buf = get_buffer(base::get_ptr())
                   .template reinterpret<T, 1>(cl::sycl::range<1>(1));
    return accessor_t(buf, cgh);
  }
#endif // DPCT_USM_LEVEL_NONE
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|device_memory_6|dpct::detail
// DPCT_PARENT_FEATURE|device_memory
// DPCT_DEPENDENCY_BEGIN
// Memory|device_memory
// DPCT_DEPENDENCY_END
// DPCT_CODE
};
// DPCT_LABEL_END
}

// DPCT_LABEL_BEGIN|global_memory_alias|dpct
// DPCT_DEPENDENCY_BEGIN
// Memory|device_memory
// Memory|memory_region
// DPCT_DEPENDENCY_END
// DPCT_CODE
template <class T, size_t Dimension>
using global_memory = detail::device_memory<T, global, Dimension>;
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|constant_memory_alias|dpct
// DPCT_DEPENDENCY_BEGIN
// Memory|device_memory
// Memory|memory_region
// DPCT_DEPENDENCY_END
// DPCT_CODE
template <class T, size_t Dimension>
using constant_memory = detail::device_memory<T, constant, Dimension>;
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|shared_memory_alias|dpct
// DPCT_DEPENDENCY_BEGIN
// Memory|device_memory
// Memory|memory_region
// DPCT_DEPENDENCY_END
// DPCT_CODE
template <class T, size_t Dimension>
using shared_memory = detail::device_memory<T, shared, Dimension>;
// DPCT_LABEL_END
} // namespace dpct

#endif // __DPCT_MEMORY_HPP__
