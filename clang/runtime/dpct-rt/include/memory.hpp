/******************************************************************************
*
* Copyright 2018 - 2020 Intel Corporation.
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
#include <map>
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
  shared,
};

// Byte type to use.
typedef uint8_t byte_t;

// Buffer type to be used in Memory Management runtime.
typedef cl::sycl::buffer<byte_t> buffer_t;

// Pitched 2D/3D memory data.
class pitched_data {
public:
  pitched_data() : pitched_data(nullptr, 0, 0, 0) {}
  pitched_data(void *data, size_t pitch, size_t x, size_t y)
      : data(data), pitch(pitch), x(x), y(y) {}
  size_t pitch;
  void *data;
  size_t x, y;
};

namespace detail {
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

  // Allocate
  void *mem_alloc(size_t size) {
    if (!size)
      return nullptr;
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
    if (!ptr)
      return;
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

  // Singleton to return the instance mem_mgr.
  // Using singleton enables header-only library, but may be problematic for
  // thread safety.
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

template <class T, memory_attribute Memory, size_t Dimension> class accessor;
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
  using pointer_t = T *;
};

// malloc
static inline void dpct_malloc(void **ptr, size_t size, cl::sycl::queue &q) {
#ifdef DPCT_USM_LEVEL_NONE
  *ptr = mem_mgr::instance().mem_alloc(size * sizeof(byte_t));
#else
  *ptr = cl::sycl::malloc_device(size, q.get_device(), q.get_context());
#endif // DPCT_USM_LEVEL_NONE
}

// malloc
static void dpct_malloc(void **ptr, size_t size) {
  dpct_malloc(ptr, size, get_default_queue());
}

#define PITCH_DEFAULT_ALIGN(x) (((x) + 31) & ~(0x1F))
static inline void dpct_malloc(void **ptr, size_t *pitch, size_t x, size_t y,
                               size_t z) {
  *pitch = PITCH_DEFAULT_ALIGN(x);
  dpct_malloc(ptr, *pitch * y * z);
}

/// Synchronously sets value to the first size bytes starting from dev_ptr in
/// \param q. The function will return after the memset operation is completed.
///
/// \param q The queue in which the operation is done.
/// \param dev_ptr Pointer to the device memory address.
/// \param value Value to be set.
/// \param size Number of bytes to be set to the value.
/// \returns no return value.
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
    cl::sycl::accessor<byte_t, 1, cl::sycl::access::mode::write,
                       cl::sycl::access::target::global_buffer>
        acc(alloc.buffer, cgh, r, o);
    cgh.fill(acc, (byte_t)value);
  });
#else
  return q.memset(dev_ptr, value, size);
#endif // DPCT_USM_LEVEL_NONE
}

/// Sets value to the 3D memory region pointed by \p data in \p q. \p size
/// specify the setted 3D memory size. The function will return after the memset
/// operation is completed.
///
/// \param q The queue in which the operation is done.
/// \param data Pointer to the device memory region.
/// \param value Value to be set.
/// \param size Specify the setted memory region.
/// \returns no return value.
static inline cl::sycl::vector_class<cl::sycl::event>
dpct_memset(cl::sycl::queue &q, pitched_data data, int value,
            cl::sycl::range<3> size) {
  cl::sycl::vector_class<cl::sycl::event> event_list;
  size_t slice = data.pitch * data.y;
  unsigned char *data_surface = (unsigned char *)data.data;
  for (size_t z = 0; z < size.get(2); ++z) {
    unsigned char *data_ptr = data_surface;
    for (size_t y = 0; y < size.get(1); ++y) {
      event_list.push_back(dpct_memset(q, data_ptr, value, size.get(0)));
      data_ptr += data.pitch;
    }
    data_surface += slice;
  }
  return event_list;
}

/// memset 2D matrix with pitch.
static inline cl::sycl::vector_class<cl::sycl::event>
dpct_memset(cl::sycl::queue &q, void *ptr, size_t pitch, int val, size_t x,
            size_t y) {
  return dpct_memset(q, pitched_data(ptr, pitch, x, 1), val,
                     cl::sycl::range<3>(x, y, 1));
}

// memcpy
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
        auto from_acc = from_buffer.get_access<cl::sycl::access::mode::read>(cgh);
        cl::sycl::accessor<byte_t, 1, cl::sycl::access::mode::write,
                           cl::sycl::access::target::global_buffer>
            acc(alloc.buffer, cgh, r, o);
        cgh.parallel_for<class memcopyh2d>(r, [=](cl::sycl::id<1> idx) {
          acc[idx] = from_acc[idx];
          });
       });
    } else {
      return q.submit([&](cl::sycl::handler &cgh) {
        auto r = cl::sycl::range<1>(size);
        auto o = cl::sycl::id<1>(offset);
         cl::sycl::accessor<byte_t, 1, cl::sycl::access::mode::write,
                           cl::sycl::access::target::global_buffer>
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
        auto to_acc = to_buffer.get_access<cl::sycl::access::mode::write>(cgh);
        cl::sycl::accessor<byte_t, 1, cl::sycl::access::mode::read,
                           cl::sycl::access::target::global_buffer>
            acc(alloc.buffer, cgh, r, o);
        cgh.parallel_for<class memcopyd2h>(r, [=](cl::sycl::id<1> idx) {
          to_acc[idx] = acc[idx];
          });
      });
    } else {
      return q.submit([&](cl::sycl::handler &cgh) {
        auto r = cl::sycl::range<1>(size);
        auto o = cl::sycl::id<1>(offset);
        cl::sycl::accessor<byte_t, 1, cl::sycl::access::mode::read,
                           cl::sycl::access::target::global_buffer>
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
        cl::sycl::accessor<byte_t, 1, cl::sycl::access::mode::write,
                           cl::sycl::access::target::global_buffer>
            to_acc(to_alloc.buffer, cgh, r, to_o);
        cl::sycl::accessor<byte_t, 1, cl::sycl::access::mode::read,
                           cl::sycl::access::target::global_buffer>
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
        cl::sycl::accessor<byte_t, 1, cl::sycl::access::mode::write,
                           cl::sycl::access::target::global_buffer>
            to_acc(to_alloc.buffer, cgh, r, to_o);
        cl::sycl::accessor<byte_t, 1, cl::sycl::access::mode::read,
                           cl::sycl::access::target::global_buffer>
            from_acc(from_alloc.buffer, cgh, r, from_o);
        cgh.copy(from_acc, to_acc);
      });
    }
  }
  default:
    std::abort();
  }
#else
  return q.memcpy(to_ptr, from_ptr, size);
#endif // DPCT_USM_LEVEL_NONE
}

/// copy 3D matrix specified by \p size from 3D matrix specified by \p from_ptr
/// and \p from_range to another specified by \p to_ptr and \p to_range.
static inline cl::sycl::vector_class<cl::sycl::event>
dpct_memcpy(cl::sycl::queue &q, void *to_ptr, const void *from_ptr,
            cl::sycl::range<3> to_range, cl::sycl::range<3> from_range,
            cl::sycl::id<3> to_id, cl::sycl::id<3> from_id,
            cl::sycl::range<3> size, memcpy_direction direction) {
  cl::sycl::vector_class<cl::sycl::event> event_list;

  size_t to_slice = to_range.get(1) * to_range.get(0),
         from_slice = from_range.get(1) * from_range.get(0);
  unsigned char *to_surface = (unsigned char *)to_ptr +
                              to_id.get(2) * to_slice +
                              to_id.get(1) * to_range.get(0) + to_id.get(0);
  const unsigned char *from_surface =
      (const unsigned char *)from_ptr + from_id.get(2) * from_slice +
      from_id.get(1) * from_range.get(0) + from_id.get(0);

  for (size_t z = 0; z < size.get(2); ++z) {
    unsigned char *to_ptr = to_surface;
    const unsigned char *from_ptr = from_surface;
    for (size_t y = 0; y < size.get(1); ++y) {
      event_list.push_back(
          dpct_memcpy(q, to_ptr, from_ptr, size.get(0), direction));
      to_ptr += to_range.get(0);
      from_ptr += from_range.get(0);
    }
    to_surface += to_slice;
    from_surface += from_slice;
  }
  return event_list;
}

/// memcpy 2D/3D matrix specified by pitched_data.
static inline cl::sycl::vector_class<cl::sycl::event>
dpct_memcpy(cl::sycl::queue &q, pitched_data to, cl::sycl::id<3> to_id,
            pitched_data from, cl::sycl::id<3> from_id, cl::sycl::range<3> size,
            memcpy_direction direction = automatic) {
  return dpct_memcpy(q, to.data, from.data,
                     cl::sycl::range<3>(to.pitch, to.y, 1),
                     cl::sycl::range<3>(from.pitch, from.y, 1), to_id, from_id,
                     size, direction);
}

/// memcpy 2D matrix with pitch.
static inline cl::sycl::vector_class<cl::sycl::event>
dpct_memcpy(cl::sycl::queue &q, void *to_ptr, const void *from_ptr,
            size_t to_pitch, size_t from_pitch, size_t x, size_t y,
            memcpy_direction direction = automatic) {
  return dpct_memcpy(q, to_ptr, from_ptr, cl::sycl::range<3>(to_pitch, y, 1),
                     cl::sycl::range<3>(from_pitch, y, 1),
                     cl::sycl::id<3>(0, 0, 0), cl::sycl::id<3>(0, 0, 0),
                     cl::sycl::range<3>(x, y, 1), direction);
}
} // namespace detail

/// Get the buffer and the offset of a piece of memory pointed to by \param ptr.
///
/// \param ptr Pointer to a piece of memory.
/// \returns a pair containing both the buffer and the offset.
static std::pair<buffer_t, size_t> get_buffer_and_offset(const void *ptr) {
  auto alloc = detail::mem_mgr::instance().translate_ptr(ptr);
  size_t offset = (byte_t *)ptr - alloc.alloc_ptr;
  return std::make_pair(alloc.buffer, offset);
}

/// Get the data pointed by ptr as a 1D buffer reinterpreted as type T.
template <typename T> static cl::sycl::buffer<T> get_buffer(const void *ptr) {
  auto alloc = detail::mem_mgr::instance().translate_ptr(ptr);
  return alloc.buffer.reinterpret<T>(
      cl::sycl::range<1>(alloc.size / sizeof(T)));
}

/// Get the buffer of a piece of memory pointed to by \param ptr.
///
/// \param ptr Pointer to a piece of memory.
/// \returns the buffer.
static buffer_t get_buffer(const void *ptr) {
  return detail::mem_mgr::instance().translate_ptr(ptr).buffer;
}

/// Build a cl::sycl::buffer with \p Size, The pointer that \p ptr point to is
/// set as a virtual pointer, which can map to the buffer.
/// \param [out] ptr Point to pointer which need to malloc memory.
/// \param size Size in bytes.
/// \returns no return value.
template <typename T1, typename T2>
static inline void dpct_malloc(T1 **ptr, T2 size) {
  return detail::dpct_malloc(reinterpret_cast<void **>(ptr),
                             static_cast<size_t>(size));
}

/// Malloc 3D array on device with size of \p size.
/// \param [out] pitch Pointer to pitch info which store the memory info.
/// \param [out] size Malloc memory size.
static inline void dpct_malloc(pitched_data *pitch, cl::sycl::range<3> size) {
  *pitch = pitched_data(nullptr, 0, size.get(0), size.get(1));
  detail::dpct_malloc(&pitch->data, &pitch->pitch, size.get(0), size.get(1),
                      size.get(2));
}

/// Malloc 2D array on device with range of pitch, y. Pitch is the aligned
/// size of x.
/// \param [out] ptr Point to pointer which need to malloc memory.
/// \param [out] pitch Aligned size of x in bytes.
/// \param x Range in dim x.
/// \param y Range in dim y.
/// \param z Range in dim z.
/// \returns no return value.
static inline void dpct_malloc(void **ptr, size_t *pitch, size_t x, size_t y) {
  return detail::dpct_malloc(ptr, pitch, x, y, 1);
}

/// free
/// \param ptr Point to free.
/// \returns no return value.
static inline void dpct_free(void *ptr) {
  if (ptr) {
#ifdef DPCT_USM_LEVEL_NONE
    detail::mem_mgr::instance().mem_free(ptr);
#else
    cl::sycl::free(ptr, get_default_queue().get_context());
#endif // DPCT_USM_LEVEL_NONE
  }
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
  detail::dpct_memcpy(get_default_queue(), to_ptr, from_ptr, size, direction)
      .wait();
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
  detail::dpct_memcpy(q, to_ptr, from_ptr, size, direction);
}

/// Synchronously copies 2D matrix specified by x and y from the address
/// specified by from_ptr to the address specified by to_ptr, while from_pitch
/// and to_pitch are the range of dim x in bytes of the matrix specified by
/// from_ptr and to_ptr, The value of direction, which is used to specify the
/// copy direction, should be one of host_to_host, host_to_device,
/// device_to_host, device_to_device, or automatic. The function will return
/// after the copy is completed.
///
/// \param to_ptr Pointer to destination memory address.
/// \param to_pitch Range of dim x in bytes of destination matrix.
/// \param from_ptr Pointer to source memory address.
/// \param from_pitch Range of dim x in bytes of source matrix.
/// \param x Range of dim x of matrix to be copied.
/// \param y Range of dim y of matrix to be copied.
/// \param direction Direction of the copy.
/// \returns no return value.
static inline void dpct_memcpy(void *to_ptr, size_t to_pitch,
                               const void *from_ptr, size_t from_pitch,
                               size_t x, size_t y,
                               memcpy_direction direction = automatic) {
  cl::sycl::event::wait(detail::dpct_memcpy(get_default_queue(), to_ptr,
                                            from_ptr, to_pitch, from_pitch, x,
                                            y, direction));
}

/// Asynchronously copies 2D matrix specified by x and y from the address
/// specified by from_ptr to the address specified by to_ptr, while from_pitch
/// and to_pitch are the range of dim x in bytes of the matrix specified by
/// from_ptr and to_ptr, The value of direction, which is used to specify the
/// copy direction, should be one of host_to_host, host_to_device,
/// device_to_host, device_to_device, or automatic. The return of the function
/// does NOT guarantee the copy is completed
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

/// Synchronously copies a subset of a 3D matrix specified by \p to to another
/// 3D matrix specified by \p from. The from and to position info are specified
/// by \p from_pos and \p to_pos The copied matrix size is specfied by \p size.
/// Copy direction, should be one of host_to_host, host_to_device,
/// device_to_host, device_to_device, or automatic is specified by the param.
/// The function will return after the copy is completed.
///
/// \param to Destination matrix info.
/// \param to_pos Position of destination.
/// \param from Source matrix info.
/// \param from_pos Position of destination.
/// \param size Range of the submatrix to be copied.
/// \param direction Direction of the copy.
/// \returns no return value.
static inline void dpct_memcpy(pitched_data to, cl::sycl::id<3> to_pos,
                               pitched_data from, cl::sycl::id<3> from_pos,
                               cl::sycl::range<3> size,
                               memcpy_direction direction = automatic) {
  cl::sycl::event::wait(detail::dpct_memcpy(get_default_queue(), to, to_pos,
                                            from, from_pos, size, direction));
}

/// Asynchronously copies a subset of a 3D matrix specified by \p to to another
/// 3D matrix specified by \p from. The from and to position info are specified
/// by \p from_pos and \p to_pos The copied matrix size is specfied by \p size.
/// Copy direction, should be one of host_to_host, host_to_device,
/// device_to_host, device_to_device, or automatic is specified by the param.
/// The function will return after the copy is completed.
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

/// Synchronously sets value to the first size bytes starting from dev_ptr. The
/// function will return after the memset operation is completed.
///
/// \param dev_ptr Pointer to the device memory address.
/// \param value Value to be set.
/// \param size Number of bytes to be set to the value.
/// \returns no return value.
static void dpct_memset(void *dev_ptr, int value, size_t size) {
  detail::dpct_memset(get_default_queue(), dev_ptr, value, size).wait();
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
  detail::dpct_memset(q, dev_ptr, value, size);
}

/// Sets value to the 2D memory region pointed by \p ptr in \p q. \p x and \p y
/// specify the setted 2D memory size. \p pitch is the bytes in linear
/// dimension, including padding bytes. The function will return after the
/// memset operation is completed.
///
/// \param ptr Pointer to the device memory region.
/// \param pitch Bytes in linear dimension, including padding bytes.
/// \param value Value to be set.
/// \param x The setted memory size in linear dimension.
/// \param y The setted memory size in second dimension.
/// \returns no return value.
static inline void dpct_memset(void *ptr, size_t pitch, int val, size_t x,
                               size_t y) {
  cl::sycl::event::wait(
      detail::dpct_memset(get_default_queue(), ptr, pitch, val, x, y));
}

/// Sets value to the 2D memory region pointed by \p ptr in \p q. \p x and \p y
/// specify the setted 2D memory size. \p pitch is the bytes in linear
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

/// Sets value to the 3D memory region specified by \p pitch in \p q. \p size
/// specify the setted 3D memory size. The function will return after the
/// memset operation is completed.
///
/// \param pitch Specify the 3D memory region.
/// \param value Value to be set.
/// \param size The setted 3D memory size.
/// \returns no return value.
static inline void dpct_memset(pitched_data pitch, int val,
                               cl::sycl::range<3> size) {
  cl::sycl::event::wait(
      detail::dpct_memset(get_default_queue(), pitch, val, size));
}

/// Sets value to the 3D memory region specified by \p pitch in \p q. \p size
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

// dpct accessor used as kernel function and device function parameter
template <class T, memory_attribute Memory, size_t Dimension> class accessor;
template <class T, memory_attribute Memory> class accessor<T, Memory, 3> {
public:
  using memory_t = detail::memory_traits<Memory, T>;
  using element_t = typename memory_t::element_t;
  using pointer_t = typename memory_t::pointer_t;
  using accessor_t = typename memory_t::template accessor_t<3>;
  accessor(pointer_t data, const cl::sycl::range<3> &in_range)
      : data(data), range(in_range) {}
  template <memory_attribute M = Memory>
  accessor(typename std::enable_if<M != local, const accessor_t>::type &acc)
      : accessor(acc, acc.get_range()) {}
  accessor(const accessor_t &acc, const cl::sycl::range<3> &in_range)
      : accessor(acc.get_pointer(), in_range) {}
  accessor<T, Memory, 2> operator[](size_t index) const {
    cl::sycl::range<2> sub(range.get(1), range.get(2));
    return accessor<T, Memory, 2>(data + index * sub.size(), sub);
  }

private:
  pointer_t data;
  cl::sycl::range<3> range;
};
template <class T, memory_attribute Memory> class accessor<T, Memory, 2> {
public:
  using memory_t = detail::memory_traits<Memory, T>;
  using element_t = typename memory_t::element_t;
  using pointer_t = typename memory_t::pointer_t;
  using accessor_t = typename memory_t::template accessor_t<2>;
  accessor(pointer_t data, const cl::sycl::range<2> &in_range)
      : data(data), range(in_range) {}
  template <memory_attribute M = Memory>
  accessor(typename std::enable_if<M != local, const accessor_t>::type &acc)
      : accessor(acc, acc.get_range()) {}
  accessor(const accessor_t &acc, const cl::sycl::range<2> &in_range)
      : accessor(acc.get_pointer(), in_range) {}

  pointer_t operator[](size_t index) const {
    return data + range.get(1) * index;
  }

private:
  pointer_t data;
  cl::sycl::range<2> range;
};

// Variable with address space of global or constant
template <class T, memory_attribute Memory, size_t Dimension>
class global_memory {
public:
  using accessor_t =
      typename detail::memory_traits<Memory, T>::template accessor_t<Dimension>;
  using value_t = typename detail::memory_traits<Memory, T>::value_t;
  using dpct_accessor_t = dpct::accessor<T, Memory, Dimension>;

  global_memory() : global_memory(cl::sycl::range<Dimension>(1)) {}

  /// Constructor of 1-D array with initializer list
  template <size_t D = Dimension>
  global_memory(
      const typename std::enable_if<D == 1, cl::sycl::range<1>>::type &in_range,
      std::initializer_list<value_t> &&init_list)
      : global_memory(in_range) {
    assert(init_list.size() <= in_range.size());
    dpct_memcpy(memory_ptr, init_list.begin(), init_list.size() * sizeof(T),
                host_to_device);
  }

  /// Constructor of 2-D array with initializer list
  template <size_t D = Dimension>
  global_memory(
      const typename std::enable_if<D == 2, cl::sycl::range<2>>::type &in_range,
      std::initializer_list<std::initializer_list<value_t>> &&init_list)
      : global_memory(in_range) {
    assert(init_list.size() <= in_range[0]);
    auto data = (byte_t *)std::malloc(size);
    auto tmp_data = data;
    std::memset(data, 0, size);
    for (auto sub_list : init_list) {
      assert(sub_list.size() <= in_range[1]);
      std::memcpy(tmp_data, sub_list.begin(), sub_list.size() * sizeof(T));
      tmp_data += in_range[1] * sizeof(T);
    }
    dpct_memcpy(memory_ptr, data, size, host_to_device);
    free(data);
  }

  /// Constructor with range
  global_memory(const cl::sycl::range<Dimension> &range_in)
      : size(range_in.size() * sizeof(T)), range(range_in), reference(false),
        memory_ptr(nullptr) {
    static_assert(
        (Memory == device) || (Memory == constant) || (Memory == shared),
        "Global memory attribute should be constant, device or shared");
    if (size) {
#ifndef DPCT_USM_LEVEL_NONE
      if (Memory == shared) {
        memory_ptr = (value_t *)cl::sycl::malloc_shared(
            size, get_default_queue().get_device(),
            get_default_queue().get_context());
        return;
      }
#endif // DPCT_USM_LEVEL_NONE
      dpct_malloc(&memory_ptr, size);
    }
  }

  /// Constructor with range
  template <class... Args>
  global_memory(Args... Arguments)
      : global_memory(cl::sycl::range<Dimension>(Arguments...)) {}

  ~global_memory() {
    if (memory_ptr && !reference)
      dpct_free(memory_ptr);
  }

  /// The variable is assigned to a device pointer.
  void assign(value_t *src, size_t size) {
    this->~global_memory();
    new (this) global_memory(src, size);
  }

  /// Get memory pointer of the memory object, which is virtual pointer when
  /// usm is not used, and device pointer when usm is used .
  value_t *get_ptr() { return memory_ptr; }

  /// Get the device memory object size in bytes.
  size_t get_size() { return size; }

#ifdef DPCT_USM_LEVEL_NONE
  template <size_t D = Dimension>
  typename std::enable_if<D == 1, T>::type &operator[](size_t index) const {
    return dpct::get_buffer<typename std::enable_if<D == 1, T>::type>(
               memory_ptr)
        .template get_access<sycl::access::mode::read_write>()[index];
  }
  /// Get cl::sycl::accessor for the device memory object when usm is not used.
  accessor_t get_access(cl::sycl::handler &cgh) {
    return get_buffer(memory_ptr)
        .template reinterpret<T, Dimension>(range)
        .template get_access<detail::memory_traits<Memory, T>::mode,
                             detail::memory_traits<Memory, T>::target>(cgh);
  }
#else
  template <size_t D = Dimension>
  typename std::enable_if<D == 1, T>::type &operator[](size_t index) const {
    return memory_ptr[index];
  }
  /// Get dpct::accessor with dimension info for the device memory object
  /// when usm is used and dimension is greater than 1.
  template <size_t D = Dimension>
  typename std::enable_if<D != 1, dpct_accessor_t>::type
  get_access(cl::sycl::handler &cgh) {
    return dpct_accessor_t((T *)memory_ptr, range);
  }
#endif // DPCT_USM_LEVEL_NONE

private:
  global_memory(value_t *memory_ptr, size_t size)
      : size(size), range(size / sizeof(T)), reference(true),
        memory_ptr(memory_ptr) {}

  size_t size;
  cl::sycl::range<Dimension> range;
  bool reference;
  value_t *memory_ptr;
};
template <class T, memory_attribute Memory>
class global_memory<T, Memory, 0> : public global_memory<T, Memory, 1> {
public:
  using base = global_memory<T, Memory, 1>;
  using value_t = typename base::value_t;
  using accessor_t =
      typename detail::memory_traits<Memory, T>::template accessor_t<0>;

  /// Constructor with initial value.
  global_memory(const value_t &val) : base(cl::sycl::range<1>(1), {val}) {}

  /// Default constructor
  global_memory() : base(1) {}

#ifdef DPCT_USM_LEVEL_NONE
  /// Get cl::sycl::accessor for the device memory object when usm is not used.
  accessor_t get_access(cl::sycl::handler &cgh) {
    auto buf = get_buffer(base::get_ptr())
                   .template reinterpret<T, 1>(cl::sycl::range<1>(1));
    return accessor_t(buf, cgh);
  }
#endif // DPCT_USM_LEVEL_NONE
};

template <class T, size_t Dimension>
using device_memory = global_memory<T, device, Dimension>;
template <class T, size_t Dimension>
using constant_memory = global_memory<T, constant, Dimension>;
template <class T, size_t Dimension>
using shared_memory = global_memory<T, shared, Dimension>;
} // namespace dpct

#endif // __DPCT_MEMORY_HPP__
