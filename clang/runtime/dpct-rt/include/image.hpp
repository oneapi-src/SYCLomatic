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

/// Image channel info, include channel number, order, data width and type
class image_channel {
public:
  channel_data_kind kind = channel_signed;
  /// Channels number.
  unsigned channel_nums = 0;
  /// Total size of all channels in bytes.
  unsigned elem_size = 0;

  cl::sycl::image_channel_type get_channel_type() const {
    auto channel_size = elem_size / channel_nums;
    if (channel_size == 4) {
      if (kind == channel_signed)
        return cl::sycl::image_channel_type::signed_int32;
      else if (kind == channel_unsigned)
        return cl::sycl::image_channel_type::unsigned_int32;
      else if (kind == channel_float)
        return cl::sycl::image_channel_type::fp32;
    } else if (channel_size == 2) {
      if (kind == channel_signed)
        return cl::sycl::image_channel_type::signed_int16;
      else if (kind == channel_unsigned)
        return cl::sycl::image_channel_type::unsigned_int16;
      else if (kind == channel_float)
        return cl::sycl::image_channel_type::fp16;
    } else {
      if (kind == channel_signed)
        return cl::sycl::image_channel_type::signed_int8;
      else if (kind == channel_unsigned)
        return cl::sycl::image_channel_type::unsigned_int8;
    }
    assert(false && "unexpected channel data kind and channel size");
    return cl::sycl::image_channel_type::signed_int32;
  }

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

class image_wrapper_base;

namespace detail {
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
  template <int Dimension>
  using array_accessor_t =
      cl::sycl::accessor<acc_data_t, Dimension, cl::sycl::access::mode::read,
                         cl::sycl::access::target::image_array>;
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

static inline image_channel
create_image_channel(int elem_size, int channel_nums,
                     channel_data_kind channel_kind) {
  image_channel channel;
  channel.kind = channel_kind;
  channel.set_channel_size(channel_nums, elem_size * 8);
  return channel;
}

/// Create image according with given type \p T and \p dims.
template <class T> static image_wrapper_base *create_image_wrapper(int dims);

/// Create image with given data type \p T, channel order and dims
template <class T>
static image_wrapper_base *create_image_wrapper(unsigned channel_nums, int dims);

/// Create image with channel info and specified dimensions.
static image_wrapper_base *create_image_wrapper(image_channel chn, int dims);

// Functor for attaching data to image class.
template <class T, int Dimension, bool IsImageArray> struct attach_data;

} // namespace detail

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
    return detail::create_image_channel(r / 8, 4, channel_kind);
  } else if (b) {
    return detail::create_image_channel(r / 8, 3, channel_kind);
  } else if (g) {
    return detail::create_image_channel(r / 8, 2, channel_kind);
  } else {
    return detail::create_image_channel(r / 8, 1, channel_kind);
  }
}

/// Create image channel info according to template argument \p T.
template <class T> static inline image_channel create_image_channel() {
  return detail::create_image_channel(sizeof(typename detail::image_trait<T>::elem_t),
                              detail::image_trait<T>::channel_nums,
                              detail::image_trait<T>::channel_kind);
}

/// 2D or 3D matrix data for image.
class image_matrix {
  image_channel _channel;
  int _range[3] = {1, 1, 1};
  int _dims = 0;
  void *_host_data = nullptr;

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
    _host_data = std::malloc(range.size() * _channel.elem_size);
  }
  /// Construct a new image class with the matrix data.
  template <int Dimension> cl::sycl::image<Dimension> *create_image() {
    return create_image<Dimension>(_channel);
  }
  /// Construct a new image class with the matrix data.
  template <int Dimension>
  cl::sycl::image<Dimension> *create_image(image_channel chn) {
    return new cl::sycl::image<Dimension>(
        _host_data, chn.get_channel_order(), chn.get_channel_type(),
        get_range(make_index_sequence<Dimension>()),
        cl::sycl::property::image::use_host_ptr());
  }
  /// Free the data.
  void free_data() {
    if (_host_data)
      std::free(_host_data);
    _host_data = nullptr;
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
    return pitched_data(_host_data, _range[0], _range[0], _range[1]);
  }

  ~image_matrix() { free_data(); }
};
using image_matrix_p = image_matrix *;

enum image_data_type { data_matrix, data_linear, data_pitch, data_unsupport };

/// Image data info.
/// This class doesn't manage the data pointer.
class image_data {
public:
  image_data() { type = data_unsupport; }
  image_data(image_matrix_p matrix) { set_data(matrix); }
  image_data(void *data_ptr, size_t x_size, image_channel channel) {
    set_data(data_ptr, x_size, channel);
  }
  image_data(void *data_ptr, size_t x_size, size_t y_size, size_t pitch_size,
             image_channel channel) {
    set_data(data_ptr, x_size, y_size, pitch_size, channel);
  }
  void set_data(image_matrix_p matrix) {
    type = data_matrix;
    data = matrix;
  }
  void set_data(void *data_ptr, size_t x_size, image_channel channel) {
    type = data_linear;
    data = data_ptr;
    x = x_size;
    chn = channel;
  }
  void set_data(void *data_ptr, size_t x_size, size_t y_size, size_t pitch_size,
                image_channel channel) {
    type = data_pitch;
    data = data_ptr;
    x = x_size;
    y = y_size;
    pitch = pitch_size;
  }
  image_data_type type;
  void *data = nullptr;
  size_t x, y, pitch;
  image_channel chn;
};

/// Image sampling info, include addressing mode, filtering mode and
/// normalization info.
class sampling_info {
  cl::sycl::addressing_mode _addr_mode = cl::sycl::addressing_mode::clamp_to_edge;
  cl::sycl::filtering_mode _filter_mode = cl::sycl::filtering_mode::nearest;
  bool _normalized = false;

public:
  inline cl::sycl::addressing_mode &addr_mode() { return _addr_mode; }
  inline cl::sycl::filtering_mode &filter_mode() { return _filter_mode; }
  inline bool &coord_normalized() {
    return _normalized;
  }
  cl::sycl::sampler get_sampler() {
    return cl::sycl::sampler(
        coord_normalized()
            ? cl::sycl::coordinate_normalization_mode::normalized
            : cl::sycl::coordinate_normalization_mode::unnormalized,
        addr_mode(), filter_mode());
  }
};

/// Image base class.
class image_wrapper_base {
  sampling_info _smpl_info;
  image_data _data;

public:
  virtual ~image_wrapper_base() = 0;

  // Set image sampling info.
  void set_sampling_info(sampling_info info) {
    _smpl_info = info;
  }
  // Set data info.
  virtual void attach(image_data data) { _data = data; }
  sampling_info get_sampling_info() { return _smpl_info; }
  const image_data &get_data() { return _data; }
  image_channel &channel() { return _data.chn; }
  inline cl::sycl::addressing_mode &addr_mode() {
    return _smpl_info.addr_mode();
  }
  inline cl::sycl::filtering_mode &filter_mode() {
    return _smpl_info.filter_mode();
  }
  inline bool &coord_normalized() { return _smpl_info.coord_normalized(); }
  cl::sycl::sampler get_sampler() { return _smpl_info.get_sampler(); }
};
inline image_wrapper_base::~image_wrapper_base() {}
using image_wrapper_base_p = image_wrapper_base *;

template <class T, int Dimension, bool IsImageArray> class image_accessor_ext;

template <class T, int Dimension, bool IsImageArray> struct attach_data;
/// Image class, wrapper of cl::sycl::image.
template <class T, int Dimension, bool IsImageArray = false> class image_wrapper : public image_wrapper_base {
  cl::sycl::image<Dimension> *_image = nullptr;

public:
  using acc_data_t = typename detail::image_trait<T>::acc_data_t;
  using accessor_t =
      typename image_accessor_ext<T, IsImageArray ? (Dimension - 1) : Dimension,
                              IsImageArray>::accessor_t;

  image_wrapper() { channel() = create_image_channel<T>(); }
  ~image_wrapper() { detach(); }
  // Get image accessor.
  accessor_t get_access(cl::sycl::handler &cgh) {
    return accessor_t(*_image, cgh);
  }
  // Set data info, attach the data to this class.
  void attach(image_data data) override {
    image_wrapper_base::attach(data);
    detail::attach_data<T, Dimension, IsImageArray>()(*this, get_data());
  }
  // Attach matrix data to this class.
  void attach(image_matrix *matrix) {
    detach();
    _image = matrix->create_image<Dimension>();
  }
  // Attach matrix data to this class.
  void attach(image_matrix *matrix, image_channel chn_desc) {
    detach();
    _image = matrix->create_image<Dimension>(chn_desc);
  }
  // Attach linear data to this class.
  void attach(void *ptr, size_t count) {
    attach(ptr, count, channel());
  }
  // Attach linear data to this class.
  void attach(void *ptr, size_t count, image_channel chn_desc) {
    detach();
    if (detail::mem_mgr::instance().is_device_ptr(ptr))
      ptr = get_buffer(ptr)
                .get_access<cl::sycl::access::mode::read_write>()
                .get_pointer();
    _image = new cl::sycl::image<Dimension>(
        ptr, chn_desc.get_channel_order(), chn_desc.get_channel_type(),
        cl::sycl::range<1>(count / chn_desc.elem_size));
  }
  // Attach 2D data to this class.
  void attach(void *data, size_t x, size_t y, size_t pitch) {
    attach(data, x, y, pitch, channel());
  }
  // Attach 2D data to this class.
  void attach(void *data, size_t x, size_t y, size_t pitch, const image_channel &chn_desc) {
    detach();
    if (detail::mem_mgr::instance().is_device_ptr(data))
      data = get_buffer(data)
                .get_access<cl::sycl::access::mode::read_write>()
                .get_pointer();
    cl::sycl::range<1> pitch_range(pitch);
    _image = new cl::sycl::image<Dimension>(
        data, chn_desc.get_channel_order(), chn_desc.get_channel_type(),
        cl::sycl::range<2>(x / chn_desc.elem_size, y), pitch_range);
  }
  // Detach data.
  void detach() {
    if (_image)
      delete _image;
    _image = nullptr;
  }
};

/// Wrap sampler and image accessor together.
template <class T, int Dimension, bool IsImageArray = false>
class image_accessor_ext {
public:
  using accessor_t =
      typename detail::image_trait<T>::template accessor_t<Dimension>;
  using data_t = typename detail::image_trait<T>::data_t;
  cl::sycl::sampler _sampler;
  accessor_t _img_acc;

public:
  image_accessor_ext(cl::sycl::sampler sampler, accessor_t acc)
      : _sampler(sampler), _img_acc(acc) {}

  // Read data from accessor.
  template <bool Available = Dimension == 3>
  typename std::enable_if<Available, data_t>::type read(float x, float y,
                                                        float z) {
    return detail::fetch_data<T>()(
        _img_acc.read(cl::sycl::float4(x, y, z, 0), _sampler));
  }
  // Read data from accessor.
  template <bool Available = Dimension == 3>
  typename std::enable_if<Available, data_t>::type read(int x, int y, int z) {
    return detail::fetch_data<T>()(
        _img_acc.read(cl::sycl::int4(x, y, z, 0), _sampler));
  }
  // Read data from accessor.
  template <bool Available = Dimension == 2>
  typename std::enable_if<Available, data_t>::type read(float x, float y) {
    return detail::fetch_data<T>()(
        _img_acc.read(cl::sycl::float2(x, y), _sampler));
  }
  // Read data from accessor.
  template <bool Available = Dimension == 2>
  typename std::enable_if<Available, data_t>::type read(int x, int y) {
    return detail::fetch_data<T>()(
        _img_acc.read(cl::sycl::int2(x, y), _sampler));
  }
  // Read data from accessor.
  template <bool Available = Dimension == 1>
  typename std::enable_if<Available, data_t>::type read(float x) {
    return detail::fetch_data<T>()(_img_acc.read(x, _sampler));
  }
  // Read data from accessor.
  template <bool Available = Dimension == 1>
  typename std::enable_if<Available, data_t>::type read(int x) {
    return detail::fetch_data<T>()(_img_acc.read(x, _sampler));
  }
};

template <class T, int Dimension> class image_accessor_ext<T, Dimension, true> {
public:
  using accessor_t =
      typename detail::image_trait<T>::template array_accessor_t<Dimension>;
  using data_t = typename detail::image_trait<T>::data_t;
  cl::sycl::sampler _sampler;
  accessor_t _img_acc;

public:
  image_accessor_ext(cl::sycl::sampler sampler, accessor_t acc)
      : _sampler(sampler), _img_acc(acc) {}

  // Read data from accessor.
  template <bool Available = Dimension == 2>
  typename std::enable_if<Available, data_t>::type read(int index, float x,
                                                        float y) {
    return detail::fetch_data<T>()(
        _img_acc[index].read(cl::sycl::float2(x, y), _sampler));
  }
  // Read data from accessor.
  template <bool Available = Dimension == 2>
  typename std::enable_if<Available, data_t>::type read(int index, int x, int y) {
    return detail::fetch_data<T>()(
        _img_acc[index].read(cl::sycl::int2(x, y), _sampler));
  }
  // Read data from accessor.
  template <bool Available = Dimension == 1>
  typename std::enable_if<Available, data_t>::type read(int index, float x) {
    return detail::fetch_data<T>()(
        _img_acc[index].read(x, _sampler));
  }
  // Read data from accessor.
  template <bool Available = Dimension == 1>
  typename std::enable_if<Available, data_t>::type read(int index, int x) {
    return detail::fetch_data<T>()(
        _img_acc[index].read(x, _sampler));
  }
};

/// Create image according to image data and sampling info.
/// \return Pointer to image wrapper base class.
/// \param data Image data used to create image wrapper.
/// \param info Image sampling info used to create image wrapper.
inline image_wrapper_base *create_image_wrapper(image_data data,
                              sampling_info info) {
  image_channel channel;
  int dims = 1;
  if (data.type == dpct::data_matrix) {
    auto matrix = (image_matrix_p)data.data;
    channel = matrix->get_channel();
    dims = matrix->get_dims();
  } else {
    channel = data.chn;
  }

  if (auto ret = detail::create_image_wrapper(channel, dims)) {
    ret->set_sampling_info(info);
    ret->attach(data);
    return ret;
  }
  return nullptr;
}

namespace detail {
// Functor for attaching data to image class.
template <class T, int Dimension, bool IsImageArray> struct attach_data {
  void operator()(image_wrapper<T, Dimension, IsImageArray> &in_image,
                  image_data data) {
    assert(data.type == data_matrix);
    in_image.attach((image_matrix_p)data.data);
  }
};
template <class T, bool IsImageArray> struct attach_data<T, 1, IsImageArray> {
  void operator()(image_wrapper<T, 1, IsImageArray> &in_image, image_data data) {
    if (data.type == data_linear)
      in_image.attach(data.data, data.x, data.chn);
    else if (data.type == data_matrix)
      in_image.attach((image_matrix_p)data.data);
  }
};
template <class T, bool IsImageArray> struct attach_data<T, 2, IsImageArray> {
  void operator()(image_wrapper<T, 2, IsImageArray> &in_image, image_data data) {
    if (data.type == data_matrix)
      in_image.attach((image_matrix_p)data.data);
    else if (data.type == data_pitch)
      in_image.attach(data.data, data.x, data.y, data.pitch, data.chn);
  }
};

/// Create image according with given type \p T and \p dims.
template <class T> static image_wrapper_base *create_image_wrapper(int dims) {
  switch (dims) {
  case 1:
    return new image_wrapper<T, 1>();
  case 2:
    return new image_wrapper<T, 2>();
  case 3:
    return new image_wrapper<T, 3>();
  default:
    return nullptr;
  }
}
/// Create image with given data type \p T, channel order and dims
template <class T>
static image_wrapper_base *create_image_wrapper(unsigned channel_nums, int dims) {
  switch (channel_nums) {
  case 1:
    return create_image_wrapper<T>(dims);
  case 2:
    return create_image_wrapper<cl::sycl::vec<T, 2>>(dims);
  case 3:
    return create_image_wrapper<cl::sycl::vec<T, 3>>(dims);
  case 4:
    return create_image_wrapper<cl::sycl::vec<T, 4>>(dims);
  default:
    return nullptr;
  }
}

/// Create image with channel info and specified dimensions.
static image_wrapper_base *create_image_wrapper(image_channel chn, int dims) {
  switch (chn.get_channel_type()) {
  case cl::sycl::image_channel_type::fp16:
    return create_image_wrapper<cl::sycl::cl_half>(chn.channel_nums, dims);
  case cl::sycl::image_channel_type::fp32:
    return create_image_wrapper<cl::sycl::cl_float>(chn.channel_nums, dims);
  case cl::sycl::image_channel_type::signed_int8:
    return create_image_wrapper<cl::sycl::cl_char>(chn.channel_nums, dims);
  case cl::sycl::image_channel_type::signed_int16:
    return create_image_wrapper<cl::sycl::cl_short>(chn.channel_nums, dims);
  case cl::sycl::image_channel_type::signed_int32:
    return create_image_wrapper<cl::sycl::cl_int>(chn.channel_nums, dims);
  case cl::sycl::image_channel_type::unsigned_int8:
    return create_image_wrapper<cl::sycl::cl_uchar>(chn.channel_nums, dims);
  case cl::sycl::image_channel_type::unsigned_int16:
    return create_image_wrapper<cl::sycl::cl_ushort>(chn.channel_nums, dims);
  case cl::sycl::image_channel_type::unsigned_int32:
    return create_image_wrapper<cl::sycl::cl_uint>(chn.channel_nums, dims);
  default:
    return nullptr;
  }
}
} // namespace detail

} // namespace dpct

#endif // !__DPCT_IMAGE_HPP__
