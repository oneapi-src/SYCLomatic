/******************************************************************************
* INTEL CONFIDENTIAL
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

//===--- image.hpp ------------------------------*- C++ -*---===//

#ifndef __DPCT_IMAGE_HPP__
#define __DPCT_IMAGE_HPP__

#include <CL/sycl.hpp>

#include "memory.hpp"
#include "util.hpp"

namespace dpct {
// Texture object type traits.
template <class T> struct image_trait {
  using acc_data_t = cl::sycl::vec<T, 4>;
  template <int Dimension>
  using accessor_t =
      cl::sycl::accessor<acc_data_t, Dimension, cl::sycl::access::mode::read,
                         cl::sycl::access::target::image>;
  using data_t = T;
  static data_t fetch_data(acc_data_t &&original_data) {
    return original_data.r();
  }
};
template <class T>
struct image_trait<cl::sycl::vec<T, 1>> : public image_trait<T> {};
template <class T>
struct image_trait<cl::sycl::vec<T, 2>> : public image_trait<T> {
  using data_t = cl::sycl::vec<T, 2>;
  static data_t fetch_data(cl::sycl::vec<T, 4> &&original_data) {
    return data_t(original_data.r(), original_data.g());
  }
};
template <class T>
struct image_trait<cl::sycl::vec<T, 3>>
    : public image_trait<cl::sycl::vec<T, 4>> {};
template <class T>
struct image_trait<cl::sycl::vec<T, 4>> : public image_trait<T> {
  using data_t = cl::sycl::vec<T, 4>;
  static data_t fetch_data(cl::sycl::vec<T, 4> &&original_data) {
    return original_data;
  }
};

struct dpct_image_channel {
  cl::sycl::image_channel_order _order;
  cl::sycl::image_channel_type _type;
  unsigned _elem_size;
};
// This class prepare 2D or 3D data for image class.
class dpct_image_data {
  dpct_image_channel _channel;
  int _range[3] = {0};
  void *_src = nullptr;

  // Set range of each dimension.
  template <class... Rest>
  size_t set_range(int dim, int first, Rest &&... rest) {
    if (!first)
      return set_range(dim);
    _range[dim] = first;
    return first * set_range(++dim, std::forward<Rest>(rest)...);
  }
  // If the dims are not used, set its range to 1.
  inline size_t set_range(int dim) {
    while (dim < 3)
      _range[dim++] = 1;
    return 1;
  }

  template <int... DimIdx>
  cl::sycl::range<sizeof...(DimIdx)> get_range(integer_sequence<DimIdx...>) {
    return cl::sycl::range<sizeof...(DimIdx)>(_range[DimIdx]...);
  }

public:
  template <int Dimension> cl::sycl::image<Dimension> *allocate_image() {
    return new cl::sycl::image<Dimension>(
        _src, _channel._order, _channel._type,
        get_range(make_index_sequence<Dimension>()),
        cl::sycl::property::image::use_host_ptr());
  }

  // Initialize channel info and array element size.
  template <class... Args>
  void malloc(dpct_image_channel channel, Args &&... args) {
    _channel = channel;
    auto size = set_range(0, std::forward<Args>(args)...);
    _src = std::malloc(size * _channel._elem_size);
  }

  void free() {
    if (_src)
      std::free(_src);
    _src = nullptr;
  }

  // Copy data from /param ptr.
  void copy_from(size_t off_x, size_t off_y, size_t off_z, void *ptr,
                 size_t count) {
    char *dst = (char *)_src +
                (off_x * _range[1] * _range[2] + off_y * _range[2] + off_z) *
                    _channel._elem_size;
    dpct_memcpy(dst, ptr, count, automatic);
  }

  ~dpct_image_data() { free(); }
};

// Texture object.
template <class T, int Dimension> class dpct_image_accessor;
template <class T, int Dimension> class dpct_image {
  cl::sycl::addressing_mode _addr_mode;
  cl::sycl::filtering_mode _filter_mode;
  cl::sycl::coordinate_normalization_mode _norm_mode;
  cl::sycl::image<Dimension> *_image;

public:
  ~dpct_image() { detach(); }

public:
  using acc_data_t = typename image_trait<T>::acc_data_t;
  dpct_image_accessor<T, Dimension> get_access(cl::sycl::handler &cgh) {
    return dpct_image_accessor<T, Dimension>(
        cl::sycl::sampler(_norm_mode, _addr_mode, _filter_mode),
        _image->template get_access<acc_data_t, cl::sycl::access::mode::read>(
            cgh));
  }

  void attach(dpct_image_data &data) {
    detach();
    _image = data.allocate_image<Dimension>();
  }
  void attach(void *ptr, const dpct_image_channel &chn_desc, size_t count) {
    detach();
    if (memory_manager::get_instance().is_device_ptr(ptr))
      ptr = memory_manager::get_instance()
                .translate_ptr(ptr)
                .buffer.get_access<cl::sycl::access::mode::read_write>()
                .get_pointer();
    _image = new cl::sycl::image<Dimension>(
        ptr, chn_desc._order, chn_desc._type,
        cl::sycl::range<1>(count / chn_desc._elem_size));
  }
  void detach() {
    if (_image)
      delete _image;
    _image = nullptr;
  }

  inline void set_addr_mode(cl::sycl::addressing_mode mode) {
    _addr_mode = mode;
  }

  inline void set_filter_mode(cl::sycl::filtering_mode mode) {
    _filter_mode = mode;
  }

  inline void set_coord_norm_mode(bool mode) {
    _norm_mode = mode ? cl::sycl::coordinate_normalization_mode::unnormalized
                      : cl::sycl::coordinate_normalization_mode::normalized;
  }

  inline cl::sycl::addressing_mode get_addr_mode() { return _addr_mode; }

  inline cl::sycl::filtering_mode get_filter_mode() { return _filter_mode; }

  inline cl::sycl::coordinate_normalization_mode get_coord_norm_mode() {
    return _norm_mode;
  }
};

// Wrap sampler and image accessor together.
template <class T, int Dimension> class dpct_image_accessor {
public:
  using accessor_t = typename image_trait<T>::template accessor_t<Dimension>;
  using data_t = typename image_trait<T>::data_t;
  cl::sycl::sampler _sampler;
  accessor_t _img_acc;

public:
  dpct_image_accessor(cl::sycl::sampler sampler, accessor_t acc)
      : _sampler(sampler), _img_acc(acc) {}

  template <class Coords> data_t read(const Coords &coords) {
    return image_trait<T>::fetch_data(_img_acc.read(coords, _sampler));
  }
};

enum dpct_channel_format_kind {
  channel_signed,
  channel_unsigned,
  channel_float,
};

inline dpct_image_channel
create_image_channel(int x, int y, int z, int w,
                     dpct_channel_format_kind channel_kind) {
  dpct_image_channel channel;

  if (x == 32) {
    channel._elem_size = 4;
    if (channel_kind == channel_signed)
      channel._type = cl::sycl::image_channel_type::signed_int32;
    else if (channel_kind == channel_unsigned)
      channel._type = cl::sycl::image_channel_type::unsigned_int32;
    else if (channel_kind == channel_float)
      channel._type = cl::sycl::image_channel_type::fp32;
  } else if (x == 16) {
    channel._elem_size = 2;
    if (channel_kind == channel_signed)
      channel._type = cl::sycl::image_channel_type::signed_int16;
    else if (channel_kind == channel_unsigned)
      channel._type = cl::sycl::image_channel_type::unsigned_int16;
    else if (channel_kind == channel_float)
      channel._type = cl::sycl::image_channel_type::fp16;
  } else if (x == 8) {
    channel._elem_size = 1;
    if (channel_kind == channel_signed)
      channel._type = cl::sycl::image_channel_type::signed_int8;
    else if (channel_kind == channel_unsigned)
      channel._type = cl::sycl::image_channel_type::unsigned_int8;
  }
  if (y == 0) {
    channel._order = cl::sycl::image_channel_order::r;
  } else if (z == 0) {
    channel._order = cl::sycl::image_channel_order::rg;
    channel._elem_size *= 2;
  } else if (w == 0) {
    channel._order = cl::sycl::image_channel_order::rgb;
    channel._elem_size *= 3;
  } else {
    channel._order = cl::sycl::image_channel_order::rgba;
    channel._elem_size *= 4;
  }

  return channel;
}

template <class T, int Dimension>
inline void dpct_attach_image(dpct_image<T, Dimension> &texture,
                              dpct_image_data &a) {
  texture.attach(a);
}
template <class T>
inline void dpct_attach_image(dpct_image<T, 1> &texture, void *ptr,
                              const dpct_image_channel &desc, size_t size) {
  texture.attach(ptr, desc, size);
}
template <class T, int Dimension>
inline void dpct_detach_image(dpct_image<T, Dimension> &texture) {
  texture.detach();
}
template <class... Args>
inline void dpct_malloc_image(dpct_image_data *a, dpct_image_channel *desc,
                              Args &&... args) {
  a->malloc(*desc, std::forward<Args>(args)...);
}
inline void dpct_memcpy_to_image(dpct_image_data &a, size_t off_x, size_t off_y,
                                 void *ptr, size_t count) {
  a.copy_from(off_x, off_y, 0, ptr, count);
}
inline void dpct_free(dpct_image_data &a) { a.free(); }
template <class T, class CoordT>
inline typename image_trait<T>::data_t
    dpct_read_image(dpct_image_accessor<T, 1> &acc, CoordT x) {
  return acc.read(x);
}
template <class T, class CoordT>
inline typename image_trait<T>::data_t
    dpct_read_image(dpct_image_accessor<T, 2> &acc, CoordT x, CoordT y) {
  return acc.read(cl::sycl::vec<CoordT, 2>(x, y));
}
template <class T, class CoordT>
inline typename image_trait<T>::data_t
    dpct_read_image(dpct_image_accessor<T, 3> &acc, CoordT x, CoordT y,
                    float z) {
  return acc.read(cl::sycl::vec<CoordT, 4>(x, y, z, 0));
}
} // namespace dpct

#endif // __DPCT_IMAGE_HPP__