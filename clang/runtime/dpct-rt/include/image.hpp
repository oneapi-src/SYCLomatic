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

//===--- image.hpp ------------------------------*- C++ -*---===//

#ifndef __DPCT_IMAGE_HPP__
#define __DPCT_IMAGE_HPP__

#include <CL/sycl.hpp>

#include "memory.hpp"
#include "util.hpp"

namespace dpct {

enum dpct_channel_data_kind {
  channel_signed,
  channel_unsigned,
  channel_float,
};

/// Image object type traits, with accessor type and sampled data type defined.
/// The data type of an image accessor must be one of cl_int4, cl_uint4,
/// cl_float4 and cl_half4. The data type of accessors with 8bits/16bits channel
/// width will be 32 bits. cl_harf is an exception.
template <class T> struct image_trait {
  using acc_data_t = cl::sycl::vec<T, 4>;
  template <int Dimension>
  using accessor_t =
      cl::sycl::accessor<acc_data_t, Dimension, cl::sycl::access::mode::read,
                         cl::sycl::access::target::image>;
  using data_t = T;
  using elem_t = T;
  static constexpr dpct_channel_data_kind channel_kind =
      std::is_integral<T>::value
          ? (std::is_signed<T>::value ? channel_signed : channel_unsigned)
          : channel_float;
  static constexpr int channel_nums = 1;
};
template <>
struct image_trait<cl::sycl::cl_uchar> : public image_trait<cl::sycl::cl_uint> {
  using data_t = cl::sycl::cl_uchar;
  using elem_t = data_t;
};
template <>
struct image_trait<cl::sycl::cl_ushort>
    : public image_trait<cl::sycl::cl_uint> {
  using data_t = cl::sycl::cl_ushort;
  using elem_t = data_t;
};
template <>
struct image_trait<cl::sycl::cl_char> : public image_trait<cl::sycl::cl_int> {
  using data_t = cl::sycl::cl_char;
  using elem_t = data_t;
};
template <>
struct image_trait<cl::sycl::cl_short> : public image_trait<cl::sycl::cl_int> {
  using data_t = cl::sycl::cl_short;
  using elem_t = data_t;
};

template <class T>
struct image_trait<cl::sycl::vec<T, 1>> : public image_trait<T> {};

template <class T>
struct image_trait<cl::sycl::vec<T, 2>> : public image_trait<T> {
  using data_t = cl::sycl::vec<T, 2>;
  static constexpr int channel_nums = 2;
};

template <class T>
struct image_trait<cl::sycl::vec<T, 3>>
    : public image_trait<cl::sycl::vec<T, 4>> {
  static constexpr int channel_nums = 3;
};

template <class T>
struct image_trait<cl::sycl::vec<T, 4>> : public image_trait<T> {
  using data_t = cl::sycl::vec<T, 4>;
  static constexpr int channel_nums = 4;
};

// Functor to fetch data from read result of an image accessor.
template <class T> struct fetch_data {
  using return_t = typename image_trait<T>::data_t;
  using acc_data_t = typename image_trait<T>::acc_data_t;

  return_t operator()(acc_data_t &&original_data) {
    return (return_t)original_data.r();
  }
};
template <class T>
struct fetch_data<cl::sycl::vec<T, 1>> : public fetch_data<T> {};
template <class T> struct fetch_data<cl::sycl::vec<T, 2>> {
  using return_t = typename image_trait<cl::sycl::vec<T, 2>>::data_t;
  using acc_data_t = typename image_trait<cl::sycl::vec<T, 2>>::acc_data_t;

  return_t operator()(acc_data_t &&origin_data) {
    return return_t(origin_data.r(), origin_data.g());
  }
};
template <class T>
struct fetch_data<cl::sycl::vec<T, 3>>
    : public fetch_data<cl::sycl::vec<T, 4>> {};
template <class T> struct fetch_data<cl::sycl::vec<T, 4>> {
  using return_t = typename image_trait<cl::sycl::vec<T, 4>>::data_t;
  using acc_data_t = typename image_trait<cl::sycl::vec<T, 4>>::acc_data_t;

  return_t operator()(acc_data_t &&origin_data) {
    return return_t(origin_data.r(), origin_data.g(), origin_data.b(),
                    origin_data.a());
  }
};

// Image channel info, include channel number, order, data width and type
struct dpct_image_channel {
  cl::sycl::image_channel_order _order;
  cl::sycl::image_channel_type _type;
  unsigned _elem_size;
};

/// 2D or 3D matrix data for image.
class dpct_image_matrix {
  dpct_image_channel _channel;
  int _range[3] = {0};
  int _dims = 0;
  void *_src = nullptr;

  /// Set range of each dimension.
  template <class... Rest>
  size_t set_range(int dim_idx, int first, Rest &&... rest) {
    if (!first)
      return set_range(dim_idx);
    _range[dim_idx] = first;
    return first * set_range(++dim_idx, std::forward<Rest>(rest)...);
  }

  inline size_t set_range(int dim_idx) {
    _dims = dim_idx;
    while (dim_idx < 3)
      _range[dim_idx++] = 1;
    return 1;
  }

  template <int... DimIdx>
  cl::sycl::range<sizeof...(DimIdx)> get_range(integer_sequence<DimIdx...>) {
    return cl::sycl::range<sizeof...(DimIdx)>(_range[DimIdx]...);
  }

public:
  /// Constructor with channel info and dimension size info.
  template <class... Args>
  dpct_image_matrix(dpct_image_channel channel, Args &&... args)
      : _channel(channel) {
    auto size = set_range(0, std::forward<Args>(args)...);
    _src = std::malloc(size * _channel._elem_size);
  }
  /// Construct a new image class with the matrix data.
  template <int Dimension> cl::sycl::image<Dimension> *allocate_image() {
    return new cl::sycl::image<Dimension>(
        _src, _channel._order, _channel._type,
        get_range(make_index_sequence<Dimension>()),
        cl::sycl::property::image::use_host_ptr());
  }
  /// Free the data.
  void free() {
    if (_src)
      std::free(_src);
    _src = nullptr;
  }

  /// Get data pointer with offset
  inline void *get_data(size_t off_x, size_t off_y, size_t off_z) {
    return (char *)_src +
           (off_x * _range[1] * _range[2] + off_y * _range[2] + off_z) *
               _channel._elem_size;
  }
  /// Get channel info.
  inline dpct_image_channel get_channel() { return _channel; }
  /// Get matrix dims.
  inline int get_dims() { return _dims; }

  ~dpct_image_matrix() { free(); }
};
using dpct_image_matrix_p = dpct_image_matrix *;

enum dpct_image_data_type { data_matrix, data_linear, data_unsupport };

/// Image data info.
class dpct_image_data {
public:
  dpct_image_data_type type;
  union {
    dpct_image_matrix *matrix;
    struct {
      void *data;
      dpct_image_channel chn;
      size_t size;
    } linear;
  } data;
};

/// Image sampling info, include addressing mode, filtering mode and
/// normalization info.
class dpct_image_info {
  cl::sycl::addressing_mode _addr_mode;
  cl::sycl::filtering_mode _filter_mode;
  cl::sycl::coordinate_normalization_mode _norm_mode;

public:
  inline cl::sycl::addressing_mode &addr_mode() { return _addr_mode; }
  inline cl::sycl::filtering_mode &filter_mode() { return _filter_mode; }
  inline cl::sycl::coordinate_normalization_mode &coord_norm_mode() {
    return _norm_mode;
  }
};

/// Image base class.
class dpct_image_base : public dpct_image_info {
  dpct_image_data _data;

public:
  virtual ~dpct_image_base() = 0;

  // Set image info.
  void set_info(dpct_image_info *info) {
    *(static_cast<dpct_image_info *>(this)) = *info;
  }
  // Set data info.
  virtual void set_data(dpct_image_data *data) { _data = *data; }
  const dpct_image_info *get_info() { return this; }
  const dpct_image_data *get_data() { return &_data; }
};
inline dpct_image_base::~dpct_image_base() {}
using dpct_image_base_p = dpct_image_base *;

template <class T, int Dimension> class dpct_image_accessor;

/// Image class, wrapper of cl::sycl::image.
template <class T, int Dimension> class dpct_image : public dpct_image_base {
  cl::sycl::image<Dimension> *_image = nullptr;

  // Functor for attaching data to image class.
  template <class DataT, int Dim> struct attach_data {
    void operator()(dpct_image<DataT, Dim> *image, dpct_image_data *data) {
      assert(data->type == data_matrix);
      image->attach(data->data.matrix);
    }
  };
  template <class DataT> struct attach_data<DataT, 1> {
    void operator()(dpct_image<DataT, 1> *image, dpct_image_data *data) {
      if (data->type == data_linear)
        image->attach(data->data.linear.data, data->data.linear.chn,
                      data->data.linear.size);
      else if (data->type == data_matrix)
        image->attach(data->data.matrix);
    }
  };

public:
  ~dpct_image() { detach(); }

public:
  using acc_data_t = typename image_trait<T>::acc_data_t;
  // Get image accessor.
  dpct_image_accessor<T, Dimension> get_access(cl::sycl::handler &cgh) {
    return dpct_image_accessor<T, Dimension>(
        cl::sycl::sampler(coord_norm_mode(), addr_mode(), filter_mode()),
        _image->template get_access<acc_data_t, cl::sycl::access::mode::read>(
            cgh));
  }
  // Set data info, attach the data to this class.
  void set_data(dpct_image_data *data) override {
    dpct_image_base::set_data(data);
    attach_data<T, Dimension>()(this, data);
  }
  // Attach matrix data to this class.
  void attach(dpct_image_matrix *data) {
    detach();
    _image = data->allocate_image<Dimension>();
  }
  // Attach linear data to this class.
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
  // Detach data.
  void detach() {
    if (_image)
      delete _image;
    _image = nullptr;
  }
};

/// Wrap sampler and image accessor together.
template <class T, int Dimension> class dpct_image_accessor {
public:
  using accessor_t = typename image_trait<T>::template accessor_t<Dimension>;
  using data_t = typename image_trait<T>::data_t;
  cl::sycl::sampler _sampler;
  accessor_t _img_acc;

public:
  dpct_image_accessor(cl::sycl::sampler sampler, accessor_t acc)
      : _sampler(sampler), _img_acc(acc) {}

  // Read data from accessor.
  template <class Coords> data_t read(const Coords &coords) {
    return fetch_data<T>()(_img_acc.read(coords, _sampler));
  }
};

static inline dpct_image_channel
create_image_channel(int elem_size, int channel_nums,
                     dpct_channel_data_kind channel_kind) {
  dpct_image_channel channel;
  channel._elem_size = elem_size;
  if (elem_size == 4) {
    if (channel_kind == channel_signed)
      channel._type = cl::sycl::image_channel_type::signed_int32;
    else if (channel_kind == channel_unsigned)
      channel._type = cl::sycl::image_channel_type::unsigned_int32;
    else if (channel_kind == channel_float)
      channel._type = cl::sycl::image_channel_type::fp32;
  } else if (elem_size == 2) {
    if (channel_kind == channel_signed)
      channel._type = cl::sycl::image_channel_type::signed_int16;
    else if (channel_kind == channel_unsigned)
      channel._type = cl::sycl::image_channel_type::unsigned_int16;
    else if (channel_kind == channel_float)
      channel._type = cl::sycl::image_channel_type::fp16;
  } else {
    if (channel_kind == channel_signed)
      channel._type = cl::sycl::image_channel_type::signed_int8;
    else if (channel_kind == channel_unsigned)
      channel._type = cl::sycl::image_channel_type::unsigned_int8;
  }
  channel._elem_size *= channel_nums;
  if (channel_nums >= 4) {
    channel._order = cl::sycl::image_channel_order::rgba;
  } else if (channel_nums == 3) {
    channel._order = cl::sycl::image_channel_order::rgb;
  } else if (channel_nums == 2) {
    channel._order = cl::sycl::image_channel_order::rg;
  } else {
    channel._order = cl::sycl::image_channel_order::r;
  }
  return channel;
}

/// Create image channel info.
/// \param r Channel r width in bits.
/// \param g Channel g width in bits. Should be same with \p r, or zero.
/// \param b Channel b width in bits. Should be same with \p g, or zero.
/// \param a Channel a width in bits. Should be same with \p b, or zero.
/// \channel_kind Channel data type kind: signed int, unsigned int or float.
static inline dpct_image_channel
create_image_channel(int r, int g, int b, int a,
                     dpct_channel_data_kind channel_kind) {
  if (a) {
    return create_image_channel(r / 32, 4, channel_kind);
  } else if (b) {
    return create_image_channel(r / 32, 3, channel_kind);
  } else if (g) {
    return create_image_channel(r / 32, 2, channel_kind);
  } else {
    return create_image_channel(r / 32, 1, channel_kind);
  }
}

template <class T> static inline dpct_image_channel create_image_channel() {
  return create_image_channel(sizeof(typename image_trait<T>::elem_t),
                              image_trait<T>::channel_nums,
                              image_trait<T>::channel_kind);
}

/// Attach a matrix to an image class.
/// \param image The image class to be attached.
/// \param a The matrix data class pointer.
template <class T, int Dimension>
inline void dpct_attach_image(dpct_image<T, Dimension> &image,
                              dpct_image_matrix *a) {
  image.attach(a);
}

/// Attach a memory block to an image class.
/// \param image The image class to be attached.
/// \param ptr The pointer that point to the memory block.
/// \param desc Channel info.
/// \param size Memory block size in bytes.
template <class T>
inline void dpct_attach_image(dpct_image<T, 1> &image, void *ptr,
                              const dpct_image_channel &chn, size_t size) {
  image.attach(ptr, chn, size);
}

/// Attach a memory block to an image class.
/// \param image The image class to be attached.
/// \param ptr The pointer that point to the memory block.
/// \param size Memory block size in bytes.
template <class T>
inline void dpct_attach_image(dpct_image<T, 1> &image, void *ptr, size_t size) {
  image.attach(ptr, create_image_channel<T>(), size);
}

/// Detach data from an image class.
/// \param image The image class to be detached.
template <class T, int Dimension>
inline void dpct_detach_image(dpct_image<T, Dimension> *image) {
  image->detach();
}

/// Detach data from an image class.
/// \param image The image class to be detached.
template <class T, int Dimension>
inline void dpct_detach_image(dpct_image<T, Dimension> &image) {
  return dpct_detach_image(&image);
}

/// Malloc matrix data.
/// \param [out] a Point to a matrix pointer.
/// \param chn Pointer to channel info.
/// \param args Varidic arguments of range.
template <class... Args>
inline void dpct_malloc_matrix(dpct_image_matrix **a, dpct_image_channel *chn,
                               Args &&... args) {
  *a = new dpct_image_matrix(*chn, std::forward<Args>(args)...);
}

/// Copy data to matrix.
/// \param a Pointer to matrix.
/// \param off_x Destination offset at dim x.
/// \param off_y Destination offset at dim y.
/// \param ptr Point to source data.
/// \param count Size in bytes.
inline void dpct_memcpy_to_matrix(dpct_image_matrix *a, size_t off_x,
                                  size_t off_y, void *ptr, size_t count) {
  dpct_memcpy(ptr, a->get_data(off_x, off_y, 0), count);
}

/// Free a matrix.
/// \param a Pointer to matrix.
inline void dpct_free(dpct_image_matrix *a) { delete a; }

/// Read data from image accessor.
/// \param acc Image accessor.
/// \param x Coordinate.
template <class T, class CoordT>
inline typename image_trait<T>::data_t
dpct_read_image(dpct_image_accessor<T, 1> &acc, CoordT x) {
  return acc.read(x);
}

/// Read data from image accessor.
/// \param [out] data Point to the memory that expect to have the read value.
/// \param acc Image accessor.
/// \param x Coordinate.
template <class T, class CoordT>
inline void dpct_read_image(typename image_trait<T>::data_t *data,
                            dpct_image_accessor<T, 1> &acc, CoordT x) {
  *data = dpct_read_image(acc, x);
}

/// Read data from image accessor.
/// \param acc Image accessor.
/// \param x Coordinate.
/// \param y Coordinate.
template <class T, class CoordT>
inline typename image_trait<T>::data_t
dpct_read_image(dpct_image_accessor<T, 2> &acc, CoordT x, CoordT y) {
  return acc.read(cl::sycl::vec<CoordT, 2>(x, y));
}

/// Read data from image accessor.
/// \param [out] data Point to the memory that expect to have the read value.
/// \param acc Image accessor.
/// \param x Coordinate.
/// \param y Coordinate.
template <class T, class CoordT>
inline void dpct_read_image(typename image_trait<T>::data_t *data,
                            dpct_image_accessor<T, 2> &acc, CoordT x,
                            CoordT y) {
  *data = dpct_read_image(acc, x, y);
}

/// Read data from image accessor.
/// \param acc Image accessor.
/// \param x Coordinate.
/// \param y Coordinate.
/// \param z Coordinate.
template <class T, class CoordT>
inline typename image_trait<T>::data_t
dpct_read_image(dpct_image_accessor<T, 3> &acc, CoordT x, CoordT y, CoordT z) {
  return acc.read(cl::sycl::vec<CoordT, 4>(x, y, z, 0));
}

/// Read data from image accessor.
/// \param [out] data Point to the memory that expect to have the read value.
/// \param acc Image accessor.
/// \param x Coordinate.
/// \param y Coordinate.
/// \param z Coordinate.
template <class T, class CoordT>
inline void dpct_read_image(typename image_trait<T>::data_t *data,
                            dpct_image_accessor<T, 4> &acc, CoordT x, CoordT y,
                            CoordT z) {
  *data = dpct_read_image(acc, x, y, z);
}

/// Create image according with given type \p T and \p dims.
template <class T> static dpct_image_base *dpct_create_image(int dims) {
  switch (dims) {
  case 1:
    return new dpct_image<T, 1>();
  case 2:
    return new dpct_image<T, 2>();
  case 3:
    return new dpct_image<T, 3>();
  default:
    return nullptr;
  }
}

/// Create image with given data type \p T, channel order and dims
template <class T>
static dpct_image_base *dpct_create_image(cl::sycl::image_channel_order order,
                                          int dims) {
  switch (order) {
  case cl::sycl::image_channel_order::r:
    return dpct_create_image<T>(dims);
  case cl::sycl::image_channel_order::rg:
    return dpct_create_image<cl::sycl::vec<T, 2>>(dims);
  case cl::sycl::image_channel_order::rgb:
    return dpct_create_image<cl::sycl::vec<T, 3>>(dims);
  case cl::sycl::image_channel_order::rgba:
    return dpct_create_image<cl::sycl::vec<T, 4>>(dims);
  default:
    return nullptr;
  }
}

/// Create image with channel info and specified dimensions.
static dpct_image_base *dpct_create_image(dpct_image_channel chn, int dims) {
  switch (chn._type) {
  case cl::sycl::image_channel_type::fp16:
    return dpct_create_image<cl::sycl::cl_half>(chn._order, dims);
  case cl::sycl::image_channel_type::fp32:
    return dpct_create_image<cl::sycl::cl_float>(chn._order, dims);
  case cl::sycl::image_channel_type::signed_int8:
    return dpct_create_image<cl::sycl::cl_char>(chn._order, dims);
  case cl::sycl::image_channel_type::signed_int16:
    return dpct_create_image<cl::sycl::cl_short>(chn._order, dims);
  case cl::sycl::image_channel_type::signed_int32:
    return dpct_create_image<cl::sycl::cl_int>(chn._order, dims);
  case cl::sycl::image_channel_type::unsigned_int8:
    return dpct_create_image<cl::sycl::cl_uchar>(chn._order, dims);
  case cl::sycl::image_channel_type::unsigned_int16:
    return dpct_create_image<cl::sycl::cl_ushort>(chn._order, dims);
  case cl::sycl::image_channel_type::unsigned_int32:
    return dpct_create_image<cl::sycl::cl_uint>(chn._order, dims);
  default:
    return nullptr;
  }
}

/// Create image according to image data and image info.
/// \param [out] image_p Point to a pointer of image base class.
/// \param data Pointer to image data.
/// \param info Pointer to image info.
inline void dpct_create_image(dpct_image_base **image_p, dpct_image_data *data,
                              dpct_image_info *info) {
  dpct_image_channel channel;
  int dims = 1;
  if (data->type == dpct::data_matrix) {
    channel = data->data.matrix->get_channel();
    dims = data->data.matrix->get_dims();
  } else if (data->type == dpct::data_linear) {
    channel = data->data.linear.chn;
  } else {
    assert(data->type != dpct::data_unsupport);
  }

  if (auto image = dpct_create_image(channel, dims)) {
    image->set_info(info);
    image->set_data(data);
    *image_p = image;
  }
}

/// Free an image class.
/// \param image Pointer of an Image base class.
inline void dpct_free(dpct_image_base *image) { delete image; }

/// Get image info from an image class.
/// \param [out] info Point to image info.
/// \param image Point to image class.
inline void dpct_get_image_info(dpct_image_info *info, dpct_image_base *image) {
  *info = *image->get_info();
}

/// Get image data from an image class.
/// \param [out] info Point to image data.
/// \param image Point to image class.
inline void dpct_get_image_data(dpct_image_data *data, dpct_image_base *image) {
  *data = *image->get_data();
}
} // namespace dpct

#endif // !__DPCT_IMAGE_HPP__
