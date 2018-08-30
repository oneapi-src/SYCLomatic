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

#include <cassert>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <CL/sycl.hpp>
#include "syclct_device.hpp"

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

enum memcpy_direction{
  to_device,
  to_host
};

// Buffer type to be used in Memory Management runtime.
typedef cl::sycl::buffer<uint8_t> bufferT;

// TODO:
// - thread safety.
// - integration with error handling - error code to be returned.
// - integration with stream support - proper queue to be used.

// There may be a lot of different strategies for allocating and mapping
// fake device pointers to SYCL/OpenCL buffers. For example:
// - continuous address allocation
// - encoding buffer number in higher bits of the address and offset in the lower bits
// I'm choosing the later one for the reason of ease of implementation and
// more efficient mapping algorithm.
class memory_manager {
public:

  using buffer_id_t = int;

  struct fake_device_pointer {
    // |== MAX_BUFFERS ==|======== MAX_OFFSET ========|
    // |   Buffer Id     |       Offset in buffer     |
    // |=================|============================|
    uintptr_t m_fake_ptr;

    static const unsigned int AddressBits = sizeof(void *) * 8;
    static const unsigned int BufferIdSize = 16u;
    static const uintptr_t MaxNumberBuffers = (1ULL << BufferIdSize) - 1;
    static const uintptr_t MaxOffset = (1ULL << (AddressBits - BufferIdSize)) - 1;

    buffer_id_t get_buffer_id() const {
      return m_fake_ptr >> (AddressBits - BufferIdSize);
    }

    size_t get_offset() const {
      return m_fake_ptr & MaxOffset;
    }

    // don't need this probably.
    operator void *() const { return reinterpret_cast<void *>(m_fake_ptr); }
    operator uintptr_t() const { return m_fake_ptr; }

    fake_device_pointer(buffer_id_t id) {
      m_fake_ptr = ((uintptr_t)id) << (AddressBits - BufferIdSize);
    }

    fake_device_pointer(void *ptr) : m_fake_ptr(reinterpret_cast<uintptr_t>(ptr)) {}

  };

  struct allocation {
    bufferT buffer;
    cl_mem memobj;
    size_t size;
  };

  memory_manager()
      : m_map({}){};

  memory_manager(const memory_manager &) = delete;

  buffer_id_t generate_id() {
    static buffer_id_t counter = 0;
    return ++counter;
  }

  // Create new fake pointer for buffer.
  fake_device_pointer add_pointer(allocation &&a) {
    auto next_number = m_map.size();
    buffer_id_t b_id = generate_id();
    m_map.emplace(b_id, a);
    if (next_number > fake_device_pointer::MaxNumberBuffers) {
      // TODO: nullptr
      return fake_device_pointer(0);
    }
    return fake_device_pointer(b_id);
  }

  // allocate
  // FIXME: Error checking
  void* mem_alloc(size_t size, cl::sycl::queue &queue) {
    cl_int error;
    cl_mem mem = clCreateBuffer(
        queue.get_context().get(), CL_MEM_READ_WRITE, size, NULL, &error);

    // TODO: since SYCL 1.2.1 buffer construction requires context instead of
    // queue. Need to clean up the interface to require context as well.
    allocation A {bufferT(mem, queue.get_context()), mem, size};

    return add_pointer(std::move(A));
  }

  // deallocate
  void mem_free(void *ptr) {
    fake_device_pointer f(ptr);
    buffer_id_t b_id = f.get_buffer_id();
    m_map.erase(b_id);
  }

  // map: fake ptr -> buffer
  allocation& translate_ptr(void *ptr) {
    fake_device_pointer fdp(ptr);
    auto b_id = fdp.get_buffer_id();
    auto it = m_map.find(b_id);
    if (it != m_map.end()) {
      return it->second;
    }
    std::abort();
  }

  // map: fake ptr -> offset in the buffer
  size_t get_ptr_offset(void *ptr) {
    fake_device_pointer fdp(ptr);
    return fdp.get_offset();
  }

  // Singleton to return the instance memory_manager.
  // Using singleton enables header-only library, but may be problematic for
  // thread safety.
  static memory_manager &get_instance() {
    static memory_manager m;
    return m;
  }

private:
  std::unordered_map<buffer_id_t, allocation> m_map;
};


// malloc
// TODO: ret values to adjust for error handling.
void sycl_malloc(void **ptr, size_t size, cl::sycl::queue q) {
  *ptr = memory_manager::get_instance().mem_alloc(size * sizeof(uint8_t), q);
}

void sycl_malloc(void **ptr, size_t size) {
  cl::sycl::queue q = syclct::get_device_manager().current_device().default_queue();
  sycl_malloc(ptr, size, q);
}

// free
// TODO: ret values to adjust for error handling.
void sycl_free(void *ptr) {
  memory_manager::get_instance().mem_free(ptr);
}

// memcpy
// TODO: ret values to adjust for error handling.
void sycl_memcpy(void *to_ptr, void *from_ptr, size_t size, memcpy_direction direction, cl::sycl::queue q) {
  cl_int rc;
  if (direction == memcpy_direction::to_device) {
    memory_manager::allocation &a =  memory_manager::get_instance().translate_ptr(to_ptr);
    size_t offset = memory_manager::fake_device_pointer(to_ptr).get_offset();
    rc = clEnqueueWriteBuffer(q.get(), a.memobj, CL_TRUE, offset, size * sizeof(uint8_t), from_ptr, 0, NULL, NULL);
  }
  else if (direction == memcpy_direction::to_host) {
    memory_manager::allocation &a = memory_manager::get_instance().translate_ptr(from_ptr);
    size_t offset = memory_manager::fake_device_pointer(from_ptr).get_offset();
    rc = clEnqueueReadBuffer(q.get(), a.memobj, CL_TRUE, offset, size * sizeof(uint8_t), to_ptr, 0, NULL, NULL);
  } else {
    // Oooops!
  }

  // TODO: error checking and reporting back.
}

void sycl_memcpy(void *to_ptr, void *from_ptr, size_t size, memcpy_direction direction) {
  sycl_memcpy(to_ptr, from_ptr, size, direction, syclct::get_device_manager().current_device().default_queue());
}

// In following functions bufferT is return instead of bufferT*, because of
// SYCL 1.2.1 #4.3.2 Common reference semantics, which explains why it's
// ok to take a copy of buffer. On the othe side, returning a pointer to
// buffer would cause obligations for not moving referenced buffer.

bufferT get_buffer(void *ptr) {
  memory_manager::allocation& alloc = memory_manager::get_instance().translate_ptr(ptr);
  size_t offset = memory_manager::get_instance().get_ptr_offset(ptr);
  if (offset == 0) {
    return alloc.buffer;
  } else {
    // TODO: taking subbuffers has some requirements for allignment/element count in the new buffer.
    // This causes incorrect work in case of bad offsets. This needs to be investigated.
    assert(offset < alloc.size);
    const cl::sycl::id<1> id(offset);
    const cl::sycl::range<1> range(alloc.size-offset);
    bufferT sub_buffer = bufferT(alloc.buffer, id, range);
    return sub_buffer;
  }
}

std::pair<bufferT, size_t> get_buffer_and_offset(void *ptr) {
  memory_manager::allocation& alloc = memory_manager::get_instance().translate_ptr(ptr);
  size_t offset = memory_manager::get_instance().get_ptr_offset(ptr);
  return std::make_pair(alloc.buffer, offset);
}

} // namespace syclct

#endif // SYCLCT_MEMORY_H
