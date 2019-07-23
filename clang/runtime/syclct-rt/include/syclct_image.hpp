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
// Texture object type traits.
template <class T> struct texture_trait {
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
struct texture_trait<cl::sycl::vec<T, 1>> : public texture_trait<T> {};
template <class T>
struct texture_trait<cl::sycl::vec<T, 2>> : public texture_trait<T> {
  using data_t = cl::sycl::vec<T, 2>;
  static data_t fetch_data(cl::sycl::vec<T, 4> &&original_data) {
    return data_t(original_data.r(), original_data.g());
  }
};
template <class T>
struct texture_trait<cl::sycl::vec<T, 3>>
    : public texture_trait<cl::sycl::vec<T, 4>> {};
template <class T>
struct texture_trait<cl::sycl::vec<T, 4>> : public texture_trait<T> {
  using data_t = cl::sycl::vec<T, 4>;
  static data_t fetch_data(cl::sycl::vec<T, 4> &&original_data) {
    return original_data;
  }
};

struct syclct_channel_desc {
  cl::sycl::image_channel_order _order;
  cl::sycl::image_channel_type _type;
};
// This class is wrapper of cl::sycl::image info.
class syclct_array {
  syclct_channel_desc _channel;
  int _range[3] = {0};
  void *_src = nullptr;

  void set_host_src(void *ptr) { _src = ptr; }
  void set_device_src(void *ptr) {
    set_host_src(memory_manager::get_instance()
                     .translate_ptr(ptr)
                     .buffer.get_access<cl::sycl::access::mode::read_write>()
                     .get_pointer());
  }
  template <class... Rest> void set_range(int idx, int first, Rest &&... rest) {
    _range[idx] = first;
    set_range(idx + 1, std::forward<Rest>(rest)...);
  }
  inline void set_range(int idx) {}

  template <int... DimIdx>
  cl::sycl::range<sizeof...(DimIdx)> get_range(integer_sequence<DimIdx...>) {
    return cl::sycl::range<sizeof...(DimIdx)>(_range[DimIdx]...);
  }

public:
  template <int Dimension> cl::sycl::image<Dimension> *allocate_image() {
    return new cl::sycl::image<Dimension>(
        _src, _channel._order, _channel._type,
        get_range(make_index_sequence<Dimension>()));
  }

  // Initialize channel info and array element size.
  template <class... Args>
  void init(syclct_channel_desc channel, Args &&... args) {
    _channel = channel;
    set_range(0, std::forward<Args>(args)...);
  }

  void free() { _src = nullptr; }

  // Copy data from device or host buffer.
  void set_src(void *ptr) {
    if (memory_manager::get_instance().is_device_ptr(ptr))
      return set_device_src(ptr);
    return set_host_src(ptr);
  }
};

// Texture object.
template <class T, int Dimension> class syclct_texture_accessor;
template <class T, int Dimension> class syclct_texture {
  cl::sycl::addressing_mode _addr_mode;
  cl::sycl::filtering_mode _filter_mode;
  cl::sycl::coordinate_normalization_mode _norm_mode;
  cl::sycl::image<Dimension> *_image;

public:
  ~syclct_texture() { unbind(); }

public:
  using acc_data_t = typename texture_trait<T>::acc_data_t;
  syclct_texture_accessor<T, Dimension> get_access(cl::sycl::handler &cgh) {
    return syclct_texture_accessor<T, Dimension>(
        cl::sycl::sampler(_norm_mode, _addr_mode, _filter_mode),
        _image->template get_access<acc_data_t, cl::sycl::access::mode::read>(
            cgh));
  }

  void bind(syclct_array &data) { _image = data.allocate_image<Dimension>(); }
  void unbind() {
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
template <class T, int Dimension> class syclct_texture_accessor {
public:
  using accessor_t = typename texture_trait<T>::template accessor_t<Dimension>;
  using data_t = typename texture_trait<T>::data_t;
  cl::sycl::sampler _sampler;
  accessor_t _img_acc;

public:
  syclct_texture_accessor(cl::sycl::sampler sampler, accessor_t acc)
      : _sampler(sampler), _img_acc(acc) {}

  template <class Coords> data_t read(const Coords &coords) {
    return texture_trait<T>::fetch_data(_img_acc.read(coords, _sampler));
  }
};

enum syclct_channel_format_kind {
  channel_signed,
  channel_unsigned,
  channel_float,
};

inline syclct_channel_desc
create_channel_desc(int x, int y, int z, int w,
                    syclct_channel_format_kind channel_kind) {
  syclct_channel_desc desc;

#define CHANNEL_WIDTH(kind, type, n)                                           \
  if (channel_kind == kind)                                                    \
    desc._type = cl::sycl::image_channel_type::type##n;
#define SIGNED_CHANNEL_WIDTH(n) CHANNEL_WIDTH(channel_signed, signed_int, n)
#define UNSIGNED_CHANNEL_WIDTH(n)                                              \
  CHANNEL_WIDTH(channel_unsigned, unsigned_int, n)
#define FP_CHANNEL_WIDTH(n) CHANNEL_WIDTH(channel_float, fp, n)

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

template <class T, int Dimension>
inline void syclct_bind_texture(syclct_texture<T, Dimension> &texture,
                                syclct_array &a) {
  texture.bind(a);
}
template <class T, int Dimension>
inline void syclct_unbind_texture(syclct_texture<T, Dimension> &texture) {
  texture.unbind();
}
template <class... Args>
inline void syclct_malloc_array(syclct_array *a, syclct_channel_desc *desc,
                                Args &&... args) {
  a->init(*desc, std::forward<Args>(args)...);
}
inline void syclct_memcpy_to_array(syclct_array &a, void *ptr) {
  a.set_src(ptr);
}
inline void syclct_free_array(syclct_array &a) { a.free(); }
template <class T, class CoordT>
inline typename texture_trait<T>::data_t
syclct_read_texture(syclct_texture_accessor<T, 1> &acc, CoordT x) {
  return acc.read(x);
}
template <class T,class CoordT>
inline typename texture_trait<T>::data_t
syclct_read_texture(syclct_texture_accessor<T, 2> &acc, CoordT x, CoordT y) {
  return acc.read(cl::sycl::vec<CoordT, 2>(x, y));
}
template <class T, class CoordT>
inline typename texture_trait<T>::data_t
syclct_read_texture(syclct_texture_accessor<T, 3> &acc, CoordT x, CoordT y,
                    float z) {
  return acc.read(cl::sycl::vec<CoordT, 4>(x, y, z, 0));
}
} // namespace syclct

#endif // !SYCLCT_TEXTURE_H
