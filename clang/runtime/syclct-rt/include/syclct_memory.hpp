//===--- syclct_memory.hpp ------------------------------*- C++ -*---===//
//
// Copyright (C) 2018 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

#ifndef SYCLCT_MEMORY_H
#define SYCLCT_MEMORY_H

#include "syclct_device.hpp"
#include <CL/sycl.hpp>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <utility>

// TODO: Add Windows version.
#include <sys/mman.h>

// DESIGN CONSIDERATIONS
// All known helper memory management classes do the following:
// - create SYCL buffers behind the scene.
// - return some kind of fake pointers to allow, which allow address
//   arithmetics and back-mapping to SYCL buffers and offset inside the buffer.
// Note, such implementation assumes that CUDA program is not using Unified
// Memory. I.e. device pointers are not dereferenced on the host.
//
// This functionality is pretty much straight forward to implement and enables
// memory allocation, deallocation, and converting SYCL buffers functionality.
//
// The trickier part is memory copies (to and from device), as naturally these
// operations in CUDA are done by imperative API (i.e. explicit copy
// operations), while in SYCL they are managed by declarative API (buffers
// passed to kernels) and managed by runtime.
//
// It seems that the most practical approach to overcome this gap is to use
// lower level OpenCL buffer API. Alternatives, which use pure SYCL API are
// known to be less efficient (require either or both of memory and compute
// overhead).

namespace syclct {

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
  shared,
};

// Byte type to use.
typedef uint8_t byte_t;

// Buffer type to be used in Memory Management runtime.
typedef cl::sycl::buffer<byte_t> buffer_t;

// TODO:
// - integration with error handling - error code to be returned.
// - integration with stream support - proper queue to be used.
// - extend mmaped space when the limit is reached.

// There may be a lot of different strategies for allocating and mapping
// fake device pointers to SYCL/OpenCL buffers. For example:
// - continuous address allocation
// - encoding buffer number in higher bits of the address and offset in the
//   lower bits
//
// Current algorithm allocates huge (128Gb) continuous address space and uses it
// for allocation of device pointers. In current version address space is not
// reused after freeing and not extended when the limit is reached. For mapping
// pointers to buffers std::map is used, which has log(N) complexity, where N is
// number of currently live allocations. This looks reasonable, given that
// number of buffers is typically not big (while quite often buffers may be big
// themselves).
class memory_manager {
public:
  using buffer_id_t = int;

  struct allocation {
    buffer_t buffer;
    byte_t *alloc_ptr;
    size_t size;
  };

  memory_manager() {
    // Reserved address space, no real memory allocation happens here.
    mapped_address_space =
        (byte_t *)mmap(nullptr, mapped_region_size, PROT_NONE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    next_free = mapped_address_space;
  };

  ~memory_manager() { munmap(mapped_address_space, mapped_region_size); };

  memory_manager(const memory_manager &) = delete;

  // Allocate
  void *mem_alloc(size_t size, cl::sycl::queue &queue) {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (next_free + size > mapped_address_space + mapped_region_size) {
      // TODO: proper error reporting.
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
  void mem_free(void *ptr) {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = get_map_iterator(ptr);
    m_map.erase(it);
  }

  // map: device pointer -> allocation(buffer, alloc_prt, size)
  allocation translate_ptr(void *ptr) {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = get_map_iterator(ptr);
    return it->second;
  }

  // Check if the pointer represents device pointer or not.
  bool is_device_ptr(void *ptr) const {
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
  std::unordered_map<buffer_id_t, allocation> m_map_old;
  std::map<byte_t *, allocation> m_map;
  mutable std::mutex m_mutex;
  byte_t *mapped_address_space;
  byte_t *next_free;
  const size_t mapped_region_size = 128ull * 1024 * 1024 * 1024;
  const size_t alignment = 256;
  // This padding may be defined to some positive value to debug
  // out of bound accesses.
  const size_t extra_padding = 0;

  std::map<byte_t *, allocation>::iterator get_map_iterator(void *ptr) {
    auto it = m_map.upper_bound((byte_t *)ptr);
    if (it == m_map.end()) {
      // Not a device pointer or out of bound.
      // TODO: proper error reporting.
      std::abort();
    }
    const allocation &alloc = it->second;
    if (ptr < alloc.alloc_ptr) {
      // Out of bound.
      // This may happen if there's a gap between allocations due to alignment
      // or extra padding and pointer points to this gap.
      // TODO: proper error reporting.
      std::abort();
    }
    return it;
  }
};

// malloc
// TODO: ret values to adjust for error handling.
static void sycl_malloc(void **ptr, size_t size, cl::sycl::queue q) {
  *ptr = memory_manager::get_instance().mem_alloc(size * sizeof(byte_t), q);
}

static void sycl_malloc(void **ptr, size_t size) {
  cl::sycl::queue q =
      syclct::get_device_manager().current_device().default_queue();
  sycl_malloc(ptr, size, q);
}

// free
// TODO: ret values to adjust for error handling.
static void sycl_free(void *ptr) {
  memory_manager::get_instance().mem_free(ptr);
}

// syclct_range used to store range infomation
// syclct_range has specialization when Dimesion = 1, 2, 3
template <size_t Dimesion> class syclct_range;
template <> class syclct_range<0> {
public:
  syclct_range(){};
  size_t size() const { return 1; }
};
template <> class syclct_range<1> {
public:
  syclct_range(size_t dim1) : range{dim1} {}
  syclct_range(cl::sycl::range<1> range) : range{range[0]} {}
  operator cl::sycl::range<1>() { return cl::sycl::range<1>(range[0]); }
  size_t size() const { return range[0]; }
  syclct_range<0> low() const { return syclct_range<0>(); }

private:
  size_t range[1];
};
template <> class syclct_range<2> {
public:
  syclct_range(size_t dim1, size_t dim2) : range{dim1, dim2} {}
  syclct_range(cl::sycl::range<2> range) : range{range[0], range[1]} {}
  operator cl::sycl::range<2>() {
    return cl::sycl::range<2>(range[0], range[1]);
  }
  size_t size() const { return range[0] * range[1]; }
  syclct_range<1> low() const { return syclct_range<1>(range[1]); }

private:
  size_t range[2];
};
template <> class syclct_range<3> {
public:
  syclct_range(size_t dim1, size_t dim2, size_t dim3)
      : range{dim1, dim2, dim3} {}
  syclct_range(cl::sycl::range<3> range)
      : range{range[0], range[1], range[2]} {}
  operator cl::sycl::range<3>() {
    return cl::sycl::range<3>(range[0], range[1], range[2]);
  }
  size_t size() const { return range[0] * range[1] * range[2]; }
  syclct_range<2> low() const { return syclct_range<2>(range[1], range[2]); }

private:
  size_t range[3];
};

// sycl memory traits
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
  template <size_t Dimension = 1>
  using accessor_t = cl::sycl::accessor<T, Dimension, mode, target>;
  using pointer_t = cl::sycl::multi_ptr<T, asp>;
  using element_t =
      typename std::conditional<Memory == constant, const T, T>::type;
};

// syclct accessor used as kernel function and device function parameter
template <class T, memory_attribute Memory, size_t Dimension>
class syclct_accessor {
public:
  using memory_t = memory_traits<Memory, T>;
  using element_t = typename memory_t::element_t;
  using pointer_t = typename memory_t::pointer_t;
  using accessor_t = typename memory_t::template accessor_t<Dimension>;
  syclct_accessor(pointer_t data, const syclct_range<Dimension> &range)
      : data(data), range(range){};
  syclct_accessor(const accessor_t &acc)
      : syclct_accessor((pointer_t)acc.get_pointer(),
                        syclct_range<Dimension>(acc.get_range())) {}
  syclct_accessor<T, Memory, Dimension - 1> operator[](size_t index) const {
    auto low = range.low();
    return syclct_accessor<T, Memory, Dimension - 1>(data + index * low.size(),
                                                     low);
  }

private:
  pointer_t data;
  syclct_range<Dimension> range;
};

// syclct_accessor specialization while Dimension = 1
template <class T, memory_attribute Memory>
class syclct_accessor<T, Memory, 1> {
public:
  using memory_t = memory_traits<Memory, T>;
  using element_t = typename memory_t::element_t;
  using pointer_t = typename memory_t::pointer_t;
  using accessor_t = typename memory_t::template accessor_t<1>;
  syclct_accessor(pointer_t data, const syclct_range<1> &range)
      : data(data), range(range){};
  syclct_accessor(const accessor_t &acc)
      : syclct_accessor((pointer_t)acc.get_pointer(),
                        syclct_range<1>(acc.get_range())) {}
  element_t &operator[](size_t index) const { return *(data + index); }
  operator pointer_t() { return data; }
  operator T *() { return data; }
  template <class ReinterpretT>
  syclct_accessor<ReinterpretT, Memory, 1> reinterpret() {
    return syclct_accessor<ReinterpretT, Memory, 1>(
        (typename memory_traits<Memory, ReinterpretT>::element_t *)data.get(),
        syclct_range<1>(range.size() * sizeof(T) / sizeof(ReinterpretT)));
  }

private:
  pointer_t data;
  syclct_range<1> range;
};

// syclct_accessor specialization while Dimension = 0
template <class T, memory_attribute Memory>
class syclct_accessor<T, Memory, 0> {
public:
  using memory_t = memory_traits<Memory, T>;
  using element_t = typename memory_t::element_t;
  using pointer_t = typename memory_t::pointer_t;
  using accessor_t = typename memory_t::template accessor_t<0>;
  syclct_accessor(pointer_t data) : data(data) {}
  syclct_accessor(const accessor_t &acc)
      : syclct_accessor((pointer_t)acc.get_pointer(), syclct_range<0>()) {}
  operator element_t &() const { return *data; }

private:
  pointer_t data;
};

//__shared__ uses local memory
template <class T, size_t Dimension> class shared_memory {
public:
  using accessor_t =
      typename memory_traits<shared, T>::template accessor_t<Dimension>;
  shared_memory(cl::sycl::range<Dimension> range, cl::sycl::handler &cgh)
      : acc(range, cgh) {}
  accessor_t get_access(cl::sycl::handler &cgh) { return acc; }

private:
  accessor_t acc;
};
using extern_shared_memory = shared_memory<byte_t, 1>;

//__constant__ and __device__ both use global memory
template <class T, memory_attribute Memory, size_t Dimension>
class global_memory {
public:
  using accessor_t =
      typename memory_traits<Memory, T>::template accessor_t<Dimension>;

  global_memory(cl::sycl::range<Dimension> range) : range(range) {
    static_assert((Memory == device) || (Memory == constant),
                  "Global memory attribute should be constant or device");
    sycl_malloc((void **)&memory_ptr, range.size() * sizeof(T));
  }
  virtual ~global_memory() { sycl_free(memory_ptr); }
  void *get_ptr() { return memory_ptr; }
  accessor_t get_access(cl::sycl::handler &cgh) {
    return memory_manager::get_instance()
        .translate_ptr(memory_ptr)
        .buffer.template reinterpret<T, Dimension>(range)
        .template get_access<memory_traits<Memory, T>::mode,
                             memory_traits<Memory, T>::target>(cgh);
  }

private:
  cl::sycl::range<Dimension> range;
  void *memory_ptr;
};
template <class T, size_t Dimension>
using constant_memory = global_memory<T, constant, Dimension>;
template <class T, size_t Dimension>
using device_memory = global_memory<T, device, Dimension>;

// memcpy
static void sycl_memcpy(void *to_ptr, void *from_ptr, size_t size,
                        memcpy_direction direction, cl::sycl::queue q) {
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
    memcpy(to_ptr, from_ptr, size);
    break;
  case host_to_device: {
    auto alloc = mm.translate_ptr(to_ptr);
    size_t offset = (byte_t *)to_ptr - alloc.alloc_ptr;
    q.submit([&](cl::sycl::handler &cgh) {
      auto r = cl::sycl::range<1>(size);
      auto o = cl::sycl::id<1>(offset);
      cl::sycl::accessor<byte_t, 1, cl::sycl::access::mode::write,
                         cl::sycl::access::target::global_buffer>
          acc(alloc.buffer, cgh, r, o);
      cgh.copy(from_ptr, acc);
    });
  } break;
  case device_to_host: {
    auto alloc = mm.translate_ptr(from_ptr);
    size_t offset = (byte_t *)from_ptr - alloc.alloc_ptr;
    q.submit([&](cl::sycl::handler &cgh) {
      auto r = cl::sycl::range<1>(size);
      auto o = cl::sycl::id<1>(offset);
      cl::sycl::accessor<byte_t, 1, cl::sycl::access::mode::read,
                         cl::sycl::access::target::global_buffer>
          acc(alloc.buffer, cgh, r, o);
      cgh.copy(acc, to_ptr);
    });
  } break;
  case device_to_device: {
    auto to_alloc = mm.translate_ptr(to_ptr);
    auto from_alloc = mm.translate_ptr(from_ptr);
    size_t to_offset = (byte_t *)to_ptr - to_alloc.alloc_ptr;
    size_t from_offset = (byte_t *)from_ptr - from_alloc.alloc_ptr;
    q.submit([&](cl::sycl::handler &cgh) {
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
  } break;
  default:
    std::abort();
  }
}

static void sycl_memcpy(void *to_ptr, void *from_ptr, size_t size,
                        memcpy_direction direction) {
  sycl_memcpy(to_ptr, from_ptr, size, direction,
              syclct::get_device_manager().current_device().default_queue());
}

// sycl_memcpy_to_symbol copies size bytes from the memory area pointed to by
// from_ptr to the memory area pointed to by offset bytes from the start of
// symbol symbol. where direction specifies the direction of the copy, and must
// be one of host_to_device, device_to_device, or automatic. Passing automatic
// is recommended, in which case the type of transfer is inferred from the
// pointer values.
static void sycl_memcpy_to_symbol(void *symbol, void *from_ptr, size_t size,
                                  size_t offset = 0,
                                  memcpy_direction direction = host_to_device) {
  switch (direction) {
  case host_to_device:
  case device_to_device:
  case automatic:
    break;
  default:
    std::abort();
  }

  sycl_memcpy((void *)((size_t)symbol + offset), from_ptr, size, direction);
}

// sycl_memcpy_from_symbol copies size bytes from the symbol address
// (from_symbol) plus offest to destination memory area (dst).
// Direction could be either device_to_host or device_to_device.
static void
sycl_memcpy_from_symbol(void *dst, void *from_symbol, size_t size,
                        size_t offset = 0,
                        memcpy_direction direction = device_to_host) {
  switch (direction) {
  case device_to_host:
  case device_to_device:
    break;
  default:
    std::abort();
  }

  sycl_memcpy(dst, (void *)(((char *)from_symbol) + offset), size, direction);
}

// In following functions buffer_t is returned instead of buffer_t*, because of
// SYCL 1.2.1 #4.3.2 Common reference semantics, which explains why it's
// ok to take a copy of buffer. On the other side, returning a pointer to
// buffer would cause obligations for not moving referenced buffer.

// FIXME: this function is not used in translation and causes segfault
// with latest OpenCL CPU runtime and ComputeCpp 1.0.1.
// TODO: this is commented out and to be removed in the future.
// The reason to leave it as a commented out - we need to file a bug against
// OpenCL CPU runtime, which has a bug triggered by this code.
/*
static buffer_t get_buffer(void *ptr) {
  auto &mm = memory_manager::get_instance();
  auto& alloc = mm.translate_ptr(ptr);
  size_t offset = (byte_t*)ptr - alloc.alloc_ptr;
  if (offset == 0) {
    return alloc.buffer;
  } else {
    // TODO: taking subbuffers has some requirements for allignment/element
    //       count in the new buffer.
    // This causes incorrect work in case of bad offsets. This needs to be
    // investigated.
    assert(offset < alloc.size);
    const cl::sycl::id<1> id(offset);
    const cl::sycl::range<1> range(alloc.size-offset);
    buffer_t sub_buffer = buffer_t(alloc.buffer, id, range);
    return sub_buffer;
  }
}
*/

static std::pair<buffer_t, size_t> get_buffer_and_offset(void *ptr) {
  auto alloc = memory_manager::get_instance().translate_ptr(ptr);
  size_t offset = (byte_t *)ptr - alloc.alloc_ptr;
  return std::make_pair(alloc.buffer, offset);
}

// memset
static void sycl_memset(void *devPtr, int value, size_t count,
                        cl::sycl::queue q) {
  auto &mm = memory_manager::get_instance();
  assert(mm.is_device_ptr(devPtr));
  auto alloc = mm.translate_ptr(devPtr);
  size_t offset = (byte_t *)devPtr - alloc.alloc_ptr;

  q.submit([&](cl::sycl::handler &cgh) {
    auto r = cl::sycl::range<1>(count);
    auto o = cl::sycl::id<1>(offset);
    cl::sycl::accessor<byte_t, 1, cl::sycl::access::mode::write,
                       cl::sycl::access::target::global_buffer>
        acc(alloc.buffer, cgh, r, o);
    cgh.fill(acc, (byte_t)value);
  });
}

static void sycl_memset(void *devPtr, int value, size_t count) {
  sycl_memset(devPtr, value, count,
              syclct::get_device_manager().current_device().default_queue());
}

} // namespace syclct

#endif // SYCLCT_MEMORY_H
