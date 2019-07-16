/******************************************************************************
* INTEL CONFIDENTIAL
*
* Copyright 2018-2019 Intel Corporation.
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

//===--- syclct_texture.hpp ------------------------------*- C++ -*---===//

#ifndef SYCLCT_TEXTURE_H
#define SYCLCT_TEXTURE_H

#include <CL/sycl.hpp>

#include "syclct_memory.hpp"
#include "syclct_util.hpp"

namespace syclct {
template <class T, int Dimension>
using texture_acc_t = cl::sycl::accessor<cl::sycl::vec<T, 4>, Dimension,
                                         cl::sycl::access::mode::read,
                                         cl::sycl::access::target::image>;
struct syclct_channel_desc {
  cl::sycl::image_channel_order _order;
  cl::sycl::image_channel_type _type;
};

// This class is wrapper of cl::sycl::image info.
template <int Dimension> class syclct_array {
  syclct_channel_desc _channel;
  cl::sycl::range<Dimension> _range;
  cl::sycl::image<Dimension> *_image = nullptr;

  void copy_from_host(void *ptr) {
    free();
    _image = new cl::sycl::image<Dimension>(ptr, _channel._order,
                                            _channel._type, _range);
  }
  void copy_from_device(void *ptr) {
    copy_from_host(memory_manager::get_instance()
                       .translate_ptr(ptr)
                       .buffer.get_access<cl::sycl::access::mode::read_write>()
                       .get_pointer());
  }

public:
  template <class T>
  texture_acc_t<T, Dimension> get_access(cl::sycl::handler &cgh) {
    return _image->template get_access<cl::sycl::vec<T, 4>,
                                       cl::sycl::access::mode::read>(cgh);
  }

  // Initialize channel info and array element size.
  void init(syclct_channel_desc channel, cl::sycl::range<Dimension> range) {
    _channel = channel;
    _range = range;
  }

  void free() {
    if (_image)
      delete _image;
    _image = nullptr;
  }

  // Copy data from device or host buffer.
  void copy_from(void *ptr, memcpy_direction direct = automatic) {
    free();
    switch (direct) {
    case syclct::host_to_host:
    case syclct::host_to_device:
      return copy_from_host(ptr);
    case syclct::device_to_host:
    case syclct::device_to_device:
      return copy_from_device(ptr);
    case syclct::automatic:
      if (memory_manager::get_instance().is_device_ptr(ptr))
        return copy_from_device(ptr);
      return copy_from_host(ptr);
    default:
      break;
    }
  }
};

// Texture object.
template <class T, int Dimension, int ChannelNum> class syclct_texture_accessor;
template <class T, int Dimension, int ChannelNum> class syclct_texture {
  cl::sycl::addressing_mode _addr_mode;
  cl::sycl::filtering_mode _filter_mode;
  cl::sycl::coordinate_normalization_mode _norm_mode;
  syclct_array<Dimension> *_array;

public:
  syclct_texture_accessor<T, Dimension, ChannelNum>
  get_access(cl::sycl::handler &cgh) {
    return syclct_texture_accessor<T, Dimension, ChannelNum>(
        cl::sycl::sampler(_norm_mode, _addr_mode, _filter_mode),
        _array->template get_access<T>(cgh));
  }

  void bind(syclct_array<Dimension> &data) { _array = &data; }

  void set_addr_mode(int mode) {
    switch (mode) {
    case 0:
      _addr_mode = cl::sycl::addressing_mode::repeat;
      return;
    case 1:
      _addr_mode = cl::sycl::addressing_mode::clamp_to_edge;
      return;
    case 2:
      _addr_mode = cl::sycl::addressing_mode::mirrored_repeat;
      return;
    case 3:
      _addr_mode = cl::sycl::addressing_mode::clamp;
      return;
    default:
      return;
    }
  }

  void set_filter_mode(int mode) {
    if (mode == 0) {
      _filter_mode = cl::sycl::filtering_mode::nearest;
    } else if (mode == 1) {
      _filter_mode = cl::sycl::filtering_mode::linear;
    }
  }

  void set_coord_norm_mode(int mode) {
    if (mode == 0) {
      _norm_mode = cl::sycl::coordinate_normalization_mode::normalized;
    } else {
      _norm_mode = cl::sycl::coordinate_normalization_mode::unnormalized;
    }
  }
};

// Extract data from vector types according to channel num.
template <class T, int ChannelNum> class FetchTextureData;
template <class T> class FetchTextureData<T, 1> {
public:
  using data_t = T;
  data_t operator()(cl::sycl::vec<T, 4> &&original_data) {
    return original_data.r();
  }
};
template <class T> class FetchTextureData<T, 2> {
public:
  using data_t = cl::sycl::vec<T, 2>;
  data_t operator()(cl::sycl::vec<T, 4> &&original_data) {
    return data_t(original_data.r(), original_data.g());
  }
};
template <class T>
class FetchTextureData<T, 3> : public FetchTextureData<T, 4> {};
template <class T> class FetchTextureData<T, 4> {
public:
  using data_t = cl::sycl::vec<T, 4>;
  data_t operator()(cl::sycl::vec<T, 4> &&original_data) {
    return original_data;
  }
};

// Wrap sampler and image accessor together.
template <class T, int Dimension, int ChannelNum>
class syclct_texture_accessor {
  cl::sycl::sampler _sampler;
  texture_acc_t<T, Dimension> _img_acc;

public:
  syclct_texture_accessor(cl::sycl::sampler sampler,
                          texture_acc_t<T, Dimension> acc)
      : _sampler(sampler), _img_acc(acc) {}

  using data_t = typename FetchTextureData<T, ChannelNum>::data_t;
  template <class Coords> data_t read(const Coords &coords) {
    return FetchTextureData<T, ChannelNum>()(_img_acc.read(coords, _sampler));
  }
};

syclct_channel_desc cread_channel_desc(int x, int y, int z, int w,
                                       int channel_kind) {
  syclct_channel_desc desc;

#define CHANNEL_WIDTH(kind, type, n)                                           \
  if (channel_kind == kind)                                                    \
    desc._type = cl::sycl::image_channel_type::type##n;
#define SIGNED_CHANNEL_WIDTH(n) CHANNEL_WIDTH(0, signed_int, n)
#define UNSIGNED_CHANNEL_WIDTH(n) CHANNEL_WIDTH(1, unsigned_int, n)
#define FP_CHANNEL_WIDTH(n) CHANNEL_WIDTH(2, fp, n)

  if (x == 32) {
    SIGNED_CHANNEL_WIDTH(32)
    else UNSIGNED_CHANNEL_WIDTH(32) else FP_CHANNEL_WIDTH(32)
  } else if (x == 16) {
    SIGNED_CHANNEL_WIDTH(16)
    else UNSIGNED_CHANNEL_WIDTH(16) else FP_CHANNEL_WIDTH(16)
  } else if (x == 8) {
    SIGNED_CHANNEL_WIDTH(8)
    else UNSIGNED_CHANNEL_WIDTH(8)
  }
  if (y == 0)
    desc._order = cl::sycl::image_channel_order::r;
  else if (z == 0)
    desc._order = cl::sycl::image_channel_order::rg;
  else if (w == 0)
    desc._order = cl::sycl::image_channel_order::rgb;
  else
    desc._order = cl::sycl::image_channel_order::rgba;

#undef CHANNEL_WIDTH
#undef SIGNED_CHANNEL_WIDTH
#undef UNSIGNED_CHANNEL_WIDTH
#undef FP_CHANNEL_WIDTH

  return desc;
}
} // namespace syclct

#endif // !SYCLCT_TEXTURE_H
