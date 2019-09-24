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

//===--- types.hpp ------------------------------*- C++ -*---===//#pragma once

#ifndef __DPCT_TYPES_HPP__
#define __DPCT_TYPES_HPP__

#include <CL/sycl.hpp>

#include "util.hpp"

namespace dpct {

/// memory copy directions
enum memcpy_direction {
  host_to_host,
  host_to_device,
  device_to_host,
  device_to_device,
  automatic
};

/// memory address space attribute.
enum memory_attribute {
  device = 0,
  constant,
  local,
};

/// image data types.
enum dpct_image_data_type {
  data_matrix,
  data_linear,
  data_pitch,
  data_unsupport
};

/// Image channel info, include channel number, order, data width and type
struct dpct_image_channel {
  cl::sycl::image_channel_order _order;
  cl::sycl::image_channel_type _type;
  unsigned _elem_size;
};

/// 2D or 3D matrix data for image.
class dpct_matrix {
  dpct_image_channel _channel;
  /// <x, y, z>
  cl::sycl::range<3> _range = cl::sycl::range<3>(1, 1, 1);
  int _dims = 0;
  void *_src = nullptr;

  /// Set range of each dimension.
  template <class... Rest>
  void set_range(int dim_idx, size_t first, Rest &&... rest) {
    if (!first)
      return set_range(dim_idx);
    _range[dim_idx] = first;
  }

  inline void set_range(int dim_idx) { _dims = dim_idx; }

  template <int... DimIdx>
  cl::sycl::range<sizeof...(DimIdx)>
  get_range(internal::integer_sequence<DimIdx...>) {
    return cl::sycl::range<sizeof...(DimIdx)>(_range[DimIdx]...);
  }

public:
  /// Constructor with channel info and dimension size info.
  template <class... Args>
  dpct_matrix(dpct_image_channel channel, Args &&... args)
      : _channel(channel) {
    set_range(0, std::forward<Args>(args)...);
    _src = std::malloc(_range.size() * _channel._elem_size);
  }
  /// Construct a new image class with the matrix data.
  template <int Dimension> cl::sycl::image<Dimension> *allocate_image() {
    return new cl::sycl::image<Dimension>(
        _src, _channel._order, _channel._type,
        get_range(internal::make_index_sequence<Dimension>()),
        cl::sycl::property::image::use_host_ptr());
  }
  /// Free the data.
  void free() {
    if (_src)
      std::free(_src);
    _src = nullptr;
  }

  /// Get data pointer with offset
  inline void *get_data(const cl::sycl::id<3> &offset) {
    return internal::compute_offset(_src, _range, offset);
  }

  /// Get data pointer with offset
  inline void *get_data(size_t off_x, size_t off_y, size_t off_z) {
    return get_data(cl::sycl::id<3>(off_x * _channel._elem_size, off_y, off_z));
  }
  /// Get channel info.
  inline dpct_image_channel get_channel() { return _channel; }
  /// Get matrix dims.
  inline int get_dims() { return _dims; }

  inline size_t get_range(unsigned dim) { return _range[dim]; }

  ~dpct_matrix() { free(); }
};
using dpct_matrix_p = dpct_matrix *;

/// pitch info.
/// ptr Pointer to the data.
/// pitch Aligned size in bytes.
/// x, y Range of dim x/y.
struct dpct_pitch {
  void *ptr;
  size_t pitch;
  size_t x, y;
};

/// create pitch with specficed data.
/// \param ptr Member ptr in struct.
/// \param pitch Member pitch in struct.
/// \param x Member x in struct.
/// \param y Member y in struct.
static inline dpct_pitch dpct_create_pitch(void *ptr, size_t pitch, size_t x,
                                           size_t y) {
  dpct_pitch out{ptr, pitch, x, y};
  return out;
}

/// Image data info.
class dpct_image_data {
public:
  union {
    dpct_matrix_p matrix;
    struct {
      void *data;
      dpct_image_channel chn;
      size_t size;
    } linear;
    struct {
      dpct_pitch data;
      dpct_image_channel chn;
    } two_dim;
  } data;
  dpct_image_data_type type = data_unsupport;
};

/// 2D/3D memory copy parameter
struct dpct_memcpy_param {
  struct {
    dpct_image_data data;
    cl::sycl::id<3> offset;

    inline dpct_matrix_p &matrix() {
      data.type = dpct_image_data_type::data_matrix;
      return data.data.matrix;
    }
    inline dpct_pitch &pitch() {
      data.type = dpct_image_data_type::data_pitch;
      return data.data.two_dim.data;
    }
    inline dpct_pitch to_pitch() {
      if (data.type == dpct_image_data_type::data_pitch) {
        dpct_pitch out = pitch();
        out.ptr = internal::compute_offset(
            out.ptr, cl::sycl::range<3>(out.pitch, out.y, 1), offset);
        return out;
      }
      return dpct_create_pitch(data.data.matrix->get_data(offset),
                               data.data.matrix->get_channel()._elem_size *
                                   data.data.matrix->get_range(0),
                               data.data.matrix->get_range(0),
                               data.data.matrix->get_range(1));
    }
  } from, to;
  cl::sycl::range<3> copy_size = cl::sycl::range<3>(0, 0, 0);
  memcpy_direction direction = automatic;
};

} // namespace dpct

#endif // __DPCT_TYPES_HPP__
