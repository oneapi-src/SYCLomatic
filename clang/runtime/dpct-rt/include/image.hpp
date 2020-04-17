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

//===--- image.hpp ------------------------------*- C++ -*---===//

#ifndef __DPCT_IMAGE_HPP__
#define __DPCT_IMAGE_HPP__

#include <CL/sycl.hpp>

#include "memory.hpp"
#include "util.hpp"

namespace dpct {

enum channel_data_kind {
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
  static constexpr channel_data_kind channel_kind =
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

/// Image channel info, include channel number, order, data width and type
struct image_channel {
  cl::sycl::image_channel_type type;
  /// Channels number.
  unsigned channel_nums;
  /// Total size of all channels in bytes.
  unsigned elem_size;

  cl::sycl::image_channel_order get_channel_order() const {
    switch (channel_nums) {
    case 1:
      return cl::sycl::image_channel_order::r;
    case 2:
      return cl::sycl::image_channel_order::rg;
    case 3:
      return cl::sycl::image_channel_order::rgb;
    case 4:
      return cl::sycl::image_channel_order::rgba;
    default:
      return cl::sycl::image_channel_order::r;
    }
  }
  /// Get the size for each channel in bits.
  unsigned get_channel_size() const { return elem_size * 8 / channel_nums; }

  /// Set channel size.
  /// \param in_channel_nums Channels number to set.
  /// \param channel_size Size for each channel in bits.
  void set_channel_size(unsigned in_channel_nums,
                        unsigned channel_size) {
    if (in_channel_nums < channel_nums)
      return;
    channel_nums = in_channel_nums;
    elem_size = channel_size * channel_nums / 8;
  }
};
/// 2D or 3D matrix data for image.
class image_matrix {
  image_channel _channel;
  int _range[3] = {1, 1, 1};
  int _dims = 0;
  void *_src = nullptr;

  /// Set range of each dimension.
  template <int Dims> void set_range(cl::sycl::range<Dims> range) {
    for (int i = 0; i < Dims; ++i)
      _range[i] = range[i];
    _dims = Dims;
  }

  template <int... DimIdx>
  cl::sycl::range<sizeof...(DimIdx)> get_range(integer_sequence<DimIdx...>) {
    return cl::sycl::range<sizeof...(DimIdx)>(_range[DimIdx]...);
  }

public:
  /// Constructor with channel info and dimension size info.
  template <int Dim>
  image_matrix(image_channel channel, cl::sycl::range<Dim>range) : _channel(channel) {
    set_range(range);
    _src = std::malloc(range.size() * _channel.elem_size);
  }
  /// Construct a new image class with the matrix data.
  template <int Dimension> cl::sycl::image<Dimension> *allocate_image() {
    return new cl::sycl::image<Dimension>(
        _src, _channel.get_channel_order(), _channel.type,
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
               _channel.elem_size;
  }
  /// Get channel info.
  inline image_channel get_channel() { return _channel; }
  /// Get range of the image.
  cl::sycl::range<3> get_range() {
    return cl::sycl::range<3>(_range[0], _range[1], _range[2]);
  }
  /// Get matrix dims.
  inline int get_dims() { return _dims; }
  /// Convert to pitched data.
  pitched_data to_pitched_data() {
    return pitched_data(_src, _range[0], _range[0], _range[1]);
  }

  ~image_matrix() { free(); }
};
using image_matrix_p = image_matrix *;

enum image_data_type { data_matrix, data_linear, data_unsupport };

/// Image data info.
class image_data {
public:
  image_data_type type;
  union {
    image_matrix *matrix;
    struct {
      void *data;
      image_channel chn;
      size_t size;
    } linear;
  } data;
};

/// Image sampling info, include addressing mode, filtering mode and
/// normalization info.
class image_info {
  cl::sycl::addressing_mode _addr_mode;
  cl::sycl::filtering_mode _filter_mode;
  bool _normalized;

public:
  inline cl::sycl::addressing_mode &addr_mode() { return _addr_mode; }
  inline cl::sycl::filtering_mode &filter_mode() { return _filter_mode; }
  inline bool &coord_normalized() {
    return _normalized;
  }
};

/// Image base class.
class image_base : public image_info {
  image_data _data;

public:
  virtual ~image_base() = 0;

  // Set image info.
  void set_info(image_info *info) {
    *(static_cast<image_info *>(this)) = *info;
  }
  // Set data info.
  virtual void set_data(image_data *data) { _data = *data; }
  const image_info *get_info() { return this; }
  const image_data *get_data() { return &_data; }
};
inline image_base::~image_base() {}
using image_base_p = image_base *;

template <class T, int Dimension> class image_accessor;

/// Image class, wrapper of cl::sycl::image.
template <class T, int Dimension> class image : public image_base {
  cl::sycl::image<Dimension> *_image = nullptr;

  // Functor for attaching data to image class.
  template <class DataT, int Dim> struct attach_data {
    void operator()(image<DataT, Dim> *in_image, image_data *data) {
      assert(data->type == data_matrix);
      in_image->attach(data->data.matrix);
    }
  };
  template <class DataT> struct attach_data<DataT, 1> {
    void operator()(image<DataT, 1> *in_image, image_data *data) {
      if (data->type == data_linear)
        in_image->attach(data->data.linear.data, data->data.linear.chn,
                      data->data.linear.size);
      else if (data->type == data_matrix)
        in_image->attach(data->data.matrix);
    }
  };

public:
  ~image() { detach(); }

public:
  using acc_data_t = typename image_trait<T>::acc_data_t;
  // Get image accessor.
  image_accessor<T, Dimension> get_access(cl::sycl::handler &cgh) {
    return image_accessor<T, Dimension>(
        cl::sycl::sampler(
            coord_normalized()
                ? cl::sycl::coordinate_normalization_mode::normalized
                : cl::sycl::coordinate_normalization_mode::unnormalized,
            addr_mode(), filter_mode()),
        _image->template get_access<acc_data_t, cl::sycl::access::mode::read>(
            cgh));
  }
  // Set data info, attach the data to this class.
  void set_data(image_data *data) override {
    image_base::set_data(data);
    attach_data<T, Dimension>()(this, data);
  }
  // Attach matrix data to this class.
  void attach(image_matrix *data) {
    detach();
    _image = data->allocate_image<Dimension>();
  }
  // Attach linear data to this class.
  void attach(void *ptr, const image_channel &chn_desc, size_t count) {
    detach();
    if (detail::mem_mgr::instance().is_device_ptr(ptr))
      ptr = get_buffer(ptr).get_access<cl::sycl::access::mode::read_write>()
                .get_pointer();
    _image = new cl::sycl::image<Dimension>(
        ptr, chn_desc.get_channel_order(), chn_desc.type,
        cl::sycl::range<1>(count / chn_desc.elem_size));
  }
  // Detach data.
  void detach() {
    if (_image)
      delete _image;
    _image = nullptr;
  }
};

/// Wrap sampler and image accessor together.
template <class T, int Dimension> class image_accessor {
public:
  using accessor_t = typename image_trait<T>::template accessor_t<Dimension>;
  using data_t = typename image_trait<T>::data_t;
  cl::sycl::sampler _sampler;
  accessor_t _img_acc;

public:
  image_accessor(cl::sycl::sampler sampler, accessor_t acc)
      : _sampler(sampler), _img_acc(acc) {}

  // Read data from accessor.
  template <class Coords> data_t read(const Coords &coords) {
    return fetch_data<T>()(_img_acc.read(coords, _sampler));
  }
};

static inline image_channel
create_image_channel(int elem_size, int channel_nums,
                     channel_data_kind channel_kind) {
  image_channel channel;
  channel.elem_size = elem_size;
  if (elem_size == 4) {
    if (channel_kind == channel_signed)
      channel.type = cl::sycl::image_channel_type::signed_int32;
    else if (channel_kind == channel_unsigned)
      channel.type = cl::sycl::image_channel_type::unsigned_int32;
    else if (channel_kind == channel_float)
      channel.type = cl::sycl::image_channel_type::fp32;
  } else if (elem_size == 2) {
    if (channel_kind == channel_signed)
      channel.type = cl::sycl::image_channel_type::signed_int16;
    else if (channel_kind == channel_unsigned)
      channel.type = cl::sycl::image_channel_type::unsigned_int16;
    else if (channel_kind == channel_float)
      channel.type = cl::sycl::image_channel_type::fp16;
  } else {
    if (channel_kind == channel_signed)
      channel.type = cl::sycl::image_channel_type::signed_int8;
    else if (channel_kind == channel_unsigned)
      channel.type = cl::sycl::image_channel_type::unsigned_int8;
  }
  channel.elem_size *= channel_nums;
  channel.channel_nums = channel_nums;
  return channel;
}

/// Create image channel info.
/// \param r Channel r width in bits.
/// \param g Channel g width in bits. Should be same with \p r, or zero.
/// \param b Channel b width in bits. Should be same with \p g, or zero.
/// \param a Channel a width in bits. Should be same with \p b, or zero.
/// \channel_kind Channel data type kind: signed int, unsigned int or float.
static inline image_channel
create_image_channel(int r, int g, int b, int a,
                     channel_data_kind channel_kind) {
  if (a) {
    return create_image_channel(r / 8, 4, channel_kind);
  } else if (b) {
    return create_image_channel(r / 8, 3, channel_kind);
  } else if (g) {
    return create_image_channel(r / 8, 2, channel_kind);
  } else {
    return create_image_channel(r / 8, 1, channel_kind);
  }
}

/// Create image channel info according to template argument \p T.
template <class T> static inline image_channel create_image_channel() {
  return create_image_channel(sizeof(typename image_trait<T>::elem_t),
                              image_trait<T>::channel_nums,
                              image_trait<T>::channel_kind);
}

/// Attach a matrix to an image class.
/// \param image The image class to be attached.
/// \param a The matrix data class pointer.
template <class T, int Dimension>
inline void attach_image(image<T, Dimension> &in_image,
                              image_matrix *a) {
  in_image.attach(a);
}

/// Attach a memory block to an image class.
/// \param image The image class to be attached.
/// \param ptr The pointer that point to the memory block.
/// \param desc Channel info.
/// \param size Memory block size in bytes.
template <class T>
inline void attach_image(image<T, 1> &in_image, void *ptr,
                              const image_channel &chn, size_t size) {
  in_image.attach(ptr, chn, size);
}

/// Attach a memory block to an image class.
/// \param image The image class to be attached.
/// \param ptr The pointer that point to the memory block.
/// \param size Memory block size in bytes.
template <class T>
inline void attach_image(image<T, 1> &in_image, void *ptr, size_t size) {
  in_image.attach(ptr, create_image_channel<T>(), size);
}

/// Detach data from an image class.
/// \param image The image class to be detached.
template <class T, int Dimension>
inline void detach_image(image<T, Dimension> *in_image) {
  in_image->detach();
}

/// Detach data from an image class.
/// \param image The image class to be detached.
template <class T, int Dimension>
inline void detach_image(image<T, Dimension> &in_image) {
  return detach_image(&in_image);
}

/// Read data from image accessor.
/// \param acc Image accessor.
/// \param x Coordinate.
template <class T, class CoordT>
inline typename image_trait<T>::data_t
read_image(image_accessor<T, 1> &acc, CoordT x) {
  return acc.read(x);
}

/// Read data from image accessor.
/// \param [out] data Point to the memory that expect to have the read value.
/// \param acc Image accessor.
/// \param x Coordinate.
template <class T, class CoordT>
inline void read_image(typename image_trait<T>::data_t *data,
                            image_accessor<T, 1> &acc, CoordT x) {
  *data = read_image(acc, x);
}

/// Read data from image accessor.
/// \param acc Image accessor.
/// \param x Coordinate.
/// \param y Coordinate.
template <class T, class CoordT>
inline typename image_trait<T>::data_t
read_image(image_accessor<T, 2> &acc, CoordT x, CoordT y) {
  return acc.read(cl::sycl::vec<CoordT, 2>(x, y));
}

/// Read data from image accessor.
/// \param [out] data Point to the memory that expect to have the read value.
/// \param acc Image accessor.
/// \param x Coordinate.
/// \param y Coordinate.
template <class T, class CoordT>
inline void read_image(typename image_trait<T>::data_t *data,
                            image_accessor<T, 2> &acc, CoordT x,
                            CoordT y) {
  *data = read_image(acc, x, y);
}

/// Read data from image accessor.
/// \param acc Image accessor.
/// \param x Coordinate.
/// \param y Coordinate.
/// \param z Coordinate.
template <class T, class CoordT>
inline typename image_trait<T>::data_t
read_image(image_accessor<T, 3> &acc, CoordT x, CoordT y, CoordT z) {
  return acc.read(cl::sycl::vec<CoordT, 4>(x, y, z, 0));
}

/// Read data from image accessor.
/// \param [out] data Point to the memory that expect to have the read value.
/// \param acc Image accessor.
/// \param x Coordinate.
/// \param y Coordinate.
/// \param z Coordinate.
template <class T, class CoordT>
inline void read_image(typename image_trait<T>::data_t *data,
                            image_accessor<T, 4> &acc, CoordT x, CoordT y,
                            CoordT z) {
  *data = read_image(acc, x, y, z);
}

/// Create image according with given type \p T and \p dims.
template <class T> static image_base *create_image(int dims) {
  switch (dims) {
  case 1:
    return new image<T, 1>();
  case 2:
    return new image<T, 2>();
  case 3:
    return new image<T, 3>();
  default:
    return nullptr;
  }
}

/// Create image with given data type \p T, channel order and dims
template <class T>
static image_base *create_image(unsigned channel_nums, int dims) {
  switch (channel_nums) {
  case 1:
    return create_image<T>(dims);
  case 2:
    return create_image<cl::sycl::vec<T, 2>>(dims);
  case 3:
    return create_image<cl::sycl::vec<T, 3>>(dims);
  case 4:
    return create_image<cl::sycl::vec<T, 4>>(dims);
  default:
    return nullptr;
  }
}

/// Create image with channel info and specified dimensions.
static image_base *create_image(image_channel chn, int dims) {
  switch (chn.type) {
  case cl::sycl::image_channel_type::fp16:
    return create_image<cl::sycl::cl_half>(chn.channel_nums, dims);
  case cl::sycl::image_channel_type::fp32:
    return create_image<cl::sycl::cl_float>(chn.channel_nums, dims);
  case cl::sycl::image_channel_type::signed_int8:
    return create_image<cl::sycl::cl_char>(chn.channel_nums, dims);
  case cl::sycl::image_channel_type::signed_int16:
    return create_image<cl::sycl::cl_short>(chn.channel_nums, dims);
  case cl::sycl::image_channel_type::signed_int32:
    return create_image<cl::sycl::cl_int>(chn.channel_nums, dims);
  case cl::sycl::image_channel_type::unsigned_int8:
    return create_image<cl::sycl::cl_uchar>(chn.channel_nums, dims);
  case cl::sycl::image_channel_type::unsigned_int16:
    return create_image<cl::sycl::cl_ushort>(chn.channel_nums, dims);
  case cl::sycl::image_channel_type::unsigned_int32:
    return create_image<cl::sycl::cl_uint>(chn.channel_nums, dims);
  default:
    return nullptr;
  }
}

/// Create image according to image data and image info.
/// \param [out] image_p Point to a pointer of image base class.
/// \param data Pointer to image data.
/// \param info Pointer to image info.
inline void create_image(image_base **image_p, image_data *data,
                              image_info *info) {
  image_channel channel;
  int dims = 1;
  if (data->type == dpct::data_matrix) {
    channel = data->data.matrix->get_channel();
    dims = data->data.matrix->get_dims();
  } else if (data->type == dpct::data_linear) {
    channel = data->data.linear.chn;
  } else {
    assert(data->type != dpct::data_unsupport);
  }

  if (auto _image = create_image(channel, dims)) {
    _image->set_info(info);
    _image->set_data(data);
    *image_p = _image;
  }
}

/// Free an image class.
/// \param image Pointer of an Image base class.
inline void dpct_free(image_base *in_image) { delete in_image; }

/// Get image info from an image class.
/// \param [out] info Point to image info.
/// \param image Point to image class.
inline void get_image_info(image_info *info, image_base *in_image) {
  *info = *in_image->get_info();
}

/// Get image data from an image class.
/// \param [out] info Point to image data.
/// \param image Point to image class.
inline void get_image_data(image_data *data, image_base *in_image) {
  *data = *in_image->get_data();
}
} // namespace dpct

#endif // !__DPCT_IMAGE_HPP__
