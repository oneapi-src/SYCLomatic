//==---- image.hpp --------------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __DPCT_IMAGE_HPP__
#define __DPCT_IMAGE_HPP__

#include <sycl/sycl.hpp>

#include "memory.hpp"
#include "util.hpp"

namespace dpct {

enum class image_channel_data_type {
  signed_int,
  unsigned_int,
  fp,
};

class image_channel;
class image_wrapper_base;
namespace detail {
/// Image object type traits, with accessor type and sampled data type defined.
/// The data type of an image accessor must be one of sycl::int4, sycl::uint4,
/// sycl::float4 and sycl::half4. The data type of accessors with 8bits/16bits
/// channel width will be 32 bits. sycl::half is an exception.
template <class T> struct image_trait {
  using acc_data_t = sycl::vec<T, 4>;
  template <int dimensions>
  using accessor_t =
      sycl::accessor<acc_data_t, dimensions, sycl::access_mode::read,
                         sycl::access::target::image>;
  template <int dimensions>
  using array_accessor_t =
      sycl::accessor<acc_data_t, dimensions, sycl::access_mode::read,
                         sycl::access::target::image_array>;
  using data_t = T;
  using elem_t = T;
  static constexpr image_channel_data_type data_type =
      std::is_integral<T>::value
          ? (std::is_signed<T>::value ? image_channel_data_type::signed_int
                                      : image_channel_data_type::unsigned_int)
          : image_channel_data_type::fp;
  static constexpr int channel_num = 1;
};
template <>
struct image_trait<std::uint8_t> : public image_trait<std::uint32_t> {
  using data_t = std::uint8_t;
  using elem_t = data_t;
};
template <>
struct image_trait<std::uint16_t>
    : public image_trait<std::uint32_t> {
  using data_t = std::uint16_t;
  using elem_t = data_t;
};
template <>
struct image_trait<std::int8_t> : public image_trait<std::int32_t> {
  using data_t = std::int8_t;
  using elem_t = data_t;
};
template <>
struct image_trait<std::int16_t> : public image_trait<std::int32_t> {
  using data_t = std::int16_t;
  using elem_t = data_t;
};
template <>
struct image_trait<char>
    : public image_trait<typename std::conditional<
          std::is_signed<char>::value, signed char, unsigned char>::type> {};

template <class T>
struct image_trait<sycl::vec<T, 1>> : public image_trait<T> {};

template <class T>
struct image_trait<sycl::vec<T, 2>> : public image_trait<T> {
  using data_t = sycl::vec<T, 2>;
  static constexpr int channel_num = 2;
};

template <class T>
struct image_trait<sycl::vec<T, 3>>
    : public image_trait<sycl::vec<T, 4>> {
  static constexpr int channel_num = 3;
};

template <class T>
struct image_trait<sycl::vec<T, 4>> : public image_trait<T> {
  using data_t = sycl::vec<T, 4>;
  static constexpr int channel_num = 4;
};

/// Functor to fetch data from read result of an image accessor.
template <class T> struct fetch_data {
  using return_t = typename image_trait<T>::data_t;
  using acc_data_t = typename image_trait<T>::acc_data_t;

  return_t operator()(acc_data_t &&original_data) {
    return (return_t)original_data.r();
  }
};
template <class T>
struct fetch_data<sycl::vec<T, 1>> : public fetch_data<T> {};
template <class T> struct fetch_data<sycl::vec<T, 2>> {
  using return_t = typename image_trait<sycl::vec<T, 2>>::data_t;
  using acc_data_t = typename image_trait<sycl::vec<T, 2>>::acc_data_t;

  return_t operator()(acc_data_t &&origin_data) {
    return return_t(origin_data.r(), origin_data.g());
  }
};
template <class T>
struct fetch_data<sycl::vec<T, 3>>
    : public fetch_data<sycl::vec<T, 4>> {};
template <class T> struct fetch_data<sycl::vec<T, 4>> {
  using return_t = typename image_trait<sycl::vec<T, 4>>::data_t;
  using acc_data_t = typename image_trait<sycl::vec<T, 4>>::acc_data_t;

  return_t operator()(acc_data_t &&origin_data) {
    return return_t(origin_data.r(), origin_data.g(), origin_data.b(),
                    origin_data.a());
  }
};

/// Create image according with given type \p T and \p dims.
template <class T> static image_wrapper_base *create_image_wrapper(int dims);

/// Create image with given data type \p T, channel order and dims
template <class T>
static image_wrapper_base *create_image_wrapper(unsigned channel_num, int dims);

/// Create image with channel info and specified dimensions.
static image_wrapper_base *create_image_wrapper(image_channel channel, int dims);

} // namespace detail

/// Image channel info, include channel number, order, data width and type
class image_channel {
  image_channel_data_type _type = image_channel_data_type::signed_int;
  /// Number of channels.
  unsigned _channel_num = 0;
  /// Total size of all channels in bytes.
  unsigned _total_size = 0;
  /// Size of each channel in bytes.
  unsigned _channel_size = 0;

public:
  /// Create image channel info according to template argument \p T.
  template <class T> static image_channel create() {
    image_channel channel;
    channel.set_channel_size(detail::image_trait<T>::channel_num,
                             sizeof(typename detail::image_trait<T>::elem_t) *
                                 8);
    channel.set_channel_data_type(detail::image_trait<T>::data_type);
    return channel;
  }

  image_channel() = default;

  image_channel_data_type get_channel_data_type() { return _type; }
  void set_channel_data_type(image_channel_data_type type) { _type = type; }

  unsigned get_total_size() { return _total_size; }

  unsigned get_channel_num() { return _channel_num; }
  void set_channel_num(unsigned channel_num) {
    _channel_num = channel_num;
    _total_size = _channel_size * _channel_num;
  }

  /// image_channel constructor.
  /// \param r Channel r width in bits.
  /// \param g Channel g width in bits. Should be same with \p r, or zero.
  /// \param b Channel b width in bits. Should be same with \p g, or zero.
  /// \param a Channel a width in bits. Should be same with \p b, or zero.
  /// \param data_type Image channel data type: signed_nt, unsigned_int or fp.
  image_channel(int r, int g, int b, int a, image_channel_data_type data_type) {
    _type = data_type;
    if (a) {
      assert(r == a && "SYCL doesn't support different channel size");
      assert(r == b && "SYCL doesn't support different channel size");
      assert(r == g && "SYCL doesn't support different channel size");
      set_channel_size(4, a);
    } else if (b) {
      assert(r == b && "SYCL doesn't support different channel size");
      assert(r == g && "SYCL doesn't support different channel size");
      set_channel_size(3, b);
    } else if (g) {
      assert(r == g && "SYCL doesn't support different channel size");
      set_channel_size(2, g);
    } else {
      set_channel_size(1, r);
    }
  }

  sycl::image_channel_type get_channel_type() const {
    if (_channel_size == 4) {
      if (_type == image_channel_data_type::signed_int)
        return sycl::image_channel_type::signed_int32;
      else if (_type == image_channel_data_type::unsigned_int)
        return sycl::image_channel_type::unsigned_int32;
      else if (_type == image_channel_data_type::fp)
        return sycl::image_channel_type::fp32;
    } else if (_channel_size == 2) {
      if (_type == image_channel_data_type::signed_int)
        return sycl::image_channel_type::signed_int16;
      else if (_type == image_channel_data_type::unsigned_int)
        return sycl::image_channel_type::unsigned_int16;
      else if (_type == image_channel_data_type::fp)
        return sycl::image_channel_type::fp16;
    } else {
      if (_type == image_channel_data_type::signed_int)
        return sycl::image_channel_type::signed_int8;
      else if (_type == image_channel_data_type::unsigned_int)
        return sycl::image_channel_type::unsigned_int8;
    }
    assert(false && "unexpected channel data kind and channel size");
    return sycl::image_channel_type::signed_int32;
  }
  void set_channel_type(sycl::image_channel_type type) {
    switch (type) {
    case sycl::image_channel_type::unsigned_int8:
      _type = image_channel_data_type::unsigned_int;
      _channel_size = 1;
      break;
    case sycl::image_channel_type::unsigned_int16:
      _type = image_channel_data_type::unsigned_int;
      _channel_size = 2;
      break;
    case sycl::image_channel_type::unsigned_int32:
      _type = image_channel_data_type::unsigned_int;
      _channel_size = 4;
      break;
    case sycl::image_channel_type::signed_int8:
      _type = image_channel_data_type::signed_int;
      _channel_size = 1;
      break;
    case sycl::image_channel_type::signed_int16:
      _type = image_channel_data_type::signed_int;
      _channel_size = 2;
      break;
    case sycl::image_channel_type::signed_int32:
      _type = image_channel_data_type::signed_int;
      _channel_size = 4;
      break;
    case sycl::image_channel_type::fp16:
      _type = image_channel_data_type::fp;
      _channel_size = 2;
      break;
    case sycl::image_channel_type::fp32:
      _type = image_channel_data_type::fp;
      _channel_size = 4;
      break;
    default:
      break;
    }
    _total_size = _channel_size * _channel_num;
  }

  sycl::image_channel_order get_channel_order() const {
    switch (_channel_num) {
    case 1:
      return sycl::image_channel_order::r;
    case 2:
      return sycl::image_channel_order::rg;
    case 3:
      return sycl::image_channel_order::rgb;
    case 4:
      return sycl::image_channel_order::rgba;
    default:
      return sycl::image_channel_order::r;
    }
  }
  /// Get the size for each channel in bits.
  unsigned get_channel_size() const { return _channel_size * 8; }

  /// Set channel size.
  /// \param in_channel_num Channels number to set.
  /// \param channel_size Size for each channel in bits.
  void set_channel_size(unsigned in_channel_num,
                        unsigned channel_size) {
    if (in_channel_num < _channel_num)
      return;
    _channel_num = in_channel_num;
    _channel_size = channel_size / 8;
    _total_size = _channel_size * _channel_num;
  }
};

/// 2D or 3D matrix data for image.
class image_matrix {
  image_channel _channel;
  int _range[3] = {1, 1, 1};
  int _dims = 0;
  void *_host_data = nullptr;

  /// Set range of each dimension.
  template <int dimensions> void set_range(sycl::range<dimensions> range) {
    for (int i = 0; i < dimensions; ++i)
      _range[i] = range[i];
    _dims = dimensions;
  }

  template <int... DimIdx>
  sycl::range<sizeof...(DimIdx)> get_range(integer_sequence<DimIdx...>) {
    return sycl::range<sizeof...(DimIdx)>(_range[DimIdx]...);
  }

public:
  /// Constructor with channel info and dimension size info.
  template <int dimensions>
  image_matrix(image_channel channel, sycl::range<dimensions> range)
      : _channel(channel) {
    set_range(range);
    _host_data = std::malloc(range.size() * _channel.get_total_size());
  }
  image_matrix(sycl::image_channel_type channel_type, unsigned channel_num,
               size_t x, size_t y) {
    _channel.set_channel_type(channel_type);
    _channel.set_channel_num(channel_num);
    _dims = 1;
    _range[0] = x;
    if (y) {
      _dims = 2;
      _range[1] = y;
    }
    _host_data = std::malloc(_range[0] * _range[1] * _channel.get_total_size());
  }

  /// Construct a new image class with the matrix data.
  template <int dimensions> sycl::image<dimensions> *create_image() {
    return create_image<dimensions>(_channel);
  }
  /// Construct a new image class with the matrix data.
  template <int dimensions>
  sycl::image<dimensions> *create_image(image_channel channel) {
    return new sycl::image<dimensions>(
        _host_data, channel.get_channel_order(), channel.get_channel_type(),
        get_range(make_index_sequence<dimensions>()),
        sycl::property::image::use_host_ptr());
  }

  /// Get channel info.
  inline image_channel get_channel() { return _channel; }
  /// Get range of the image.
  sycl::range<3> get_range() {
    return sycl::range<3>(_range[0], _range[1], _range[2]);
  }
  /// Get matrix dims.
  inline int get_dims() { return _dims; }
  /// Convert to pitched data.
  pitched_data to_pitched_data() {
    return pitched_data(_host_data, _range[0] * _channel.get_total_size(),
                        _range[0], _range[1]);
  }

  ~image_matrix() {
    if (_host_data)
      std::free(_host_data);
    _host_data = nullptr;
  }
};
using image_matrix_p = image_matrix *;

enum class image_data_type { matrix, linear, pitch, unsupport };

/// Image data info.
class image_data {
public:
  image_data() { _type = image_data_type::unsupport; }
  image_data(image_matrix_p matrix_data) { set_data(matrix_data); }
  image_data(void *data_ptr, size_t x_size, image_channel channel) {
    set_data(data_ptr, x_size, channel);
  }
  image_data(void *data_ptr, size_t x_size, size_t y_size, size_t pitch_size,
             image_channel channel) {
    set_data(data_ptr, x_size, y_size, pitch_size, channel);
  }
  void set_data(image_matrix_p matrix_data) {
    _type = image_data_type::matrix;
    _data = matrix_data;
    _channel = matrix_data->get_channel();
  }
  void set_data(void *data_ptr, size_t x_size, image_channel channel) {
    _type = image_data_type::linear;
    _data = data_ptr;
    _x = x_size;
    _channel = channel;
  }
  void set_data(void *data_ptr, size_t x_size, size_t y_size, size_t pitch_size,
                image_channel channel) {
    _type = image_data_type::pitch;
    _data = data_ptr;
    _x = x_size;
    _y = y_size;
    _pitch = pitch_size;
    _channel = channel;
  }

  image_data_type get_data_type() const { return _type; }
  void set_data_type(image_data_type type) { _type = type; }

  void *get_data_ptr() const { return _data; }
  void set_data_ptr(void *data) { _data = data; }

  size_t get_x() const { return _x; }
  void set_x(size_t x) { _x = x; }

  size_t get_y() const { return _y; }
  void set_y(size_t y) { _y = y; }

  size_t get_pitch() const { return _pitch; }
  void set_pitch(size_t pitch) { _pitch = pitch; }

  image_channel get_channel() const { return _channel; }
  void set_channel(image_channel channel) { _channel = channel; }

  image_channel_data_type get_channel_data_type() {
    return _channel.get_channel_data_type();
  }
  void set_channel_data_type(image_channel_data_type type) {
    _channel.set_channel_data_type(type);
  }

  unsigned get_channel_size() { return _channel.get_channel_size(); }
  void set_channel_size(unsigned channel_num, unsigned channel_size) {
    return _channel.set_channel_size(channel_num, channel_size);
  }

  unsigned get_channel_num() { return _channel.get_channel_num(); }
  void set_channel_num(unsigned num) {
    return _channel.set_channel_num(num);
  }

  sycl::image_channel_type get_channel_type() {
    return _channel.get_channel_type();
  }
  void set_channel_type(sycl::image_channel_type type) {
    return _channel.set_channel_type(type);
  }

private:
  image_data_type _type;
  void *_data = nullptr;
  size_t _x, _y, _pitch;
  image_channel _channel;
};

/// Image sampling info, include addressing mode, filtering mode and
/// normalization info.
class sampling_info {
  sycl::addressing_mode _addressing_mode =
      sycl::addressing_mode::clamp_to_edge;
  sycl::filtering_mode _filtering_mode = sycl::filtering_mode::nearest;
  sycl::coordinate_normalization_mode _coordinate_normalization_mode =
      sycl::coordinate_normalization_mode::unnormalized;

public:
  sycl::addressing_mode get_addressing_mode() { return _addressing_mode; }
  void set(sycl::addressing_mode addressing_mode) { _addressing_mode = addressing_mode; }

  sycl::filtering_mode get_filtering_mode() { return _filtering_mode; }
  void set(sycl::filtering_mode filtering_mode) { _filtering_mode = filtering_mode; }

  sycl::coordinate_normalization_mode get_coordinate_normalization_mode() {
    return _coordinate_normalization_mode;
  }
  void set(sycl::coordinate_normalization_mode coordinate_normalization_mode) {
    _coordinate_normalization_mode = coordinate_normalization_mode;
  }

  bool is_coordinate_normalized() {
    return _coordinate_normalization_mode ==
           sycl::coordinate_normalization_mode::normalized;
  }
  void set_coordinate_normalization_mode(int is_normalized) {
    _coordinate_normalization_mode =
        is_normalized ? sycl::coordinate_normalization_mode::normalized
                      : sycl::coordinate_normalization_mode::unnormalized;
  }
  void
  set(sycl::addressing_mode addressing_mode,
      sycl::filtering_mode filtering_mode,
      sycl::coordinate_normalization_mode coordinate_normalization_mode) {
    set(addressing_mode);
    set(filtering_mode);
    set(coordinate_normalization_mode);
  }
  void set(sycl::addressing_mode addressing_mode,
           sycl::filtering_mode filtering_mode, int is_normalized) {
    set(addressing_mode);
    set(filtering_mode);
    set_coordinate_normalization_mode(is_normalized);
  }

  sycl::sampler get_sampler() {
    return sycl::sampler(_coordinate_normalization_mode, _addressing_mode,
                             _filtering_mode);
  }
};

/// Image base class.
class image_wrapper_base {
  sampling_info _sampling_info;
  image_data _data;

public:
  virtual ~image_wrapper_base() = 0;

  void attach(image_data data) { set_data(data); }
  /// Attach matrix data to this class.
  void attach(image_matrix *matrix) {
    detach();
    image_wrapper_base::set_data(image_data(matrix));
  }
  /// Attach matrix data to this class.
  void attach(image_matrix *matrix, image_channel channel) {
    attach(matrix);
    image_wrapper_base::set_channel(channel);
  }
  /// Attach linear data to this class.
  void attach(const void *ptr, size_t count) {
    attach(ptr, count, get_channel());
  }
  /// Attach linear data to this class.
  void attach(const void *ptr, size_t count, image_channel channel) {
    detach();
    image_wrapper_base::set_data(image_data(const_cast<void *>(ptr), count, channel));
  }
  /// Attach 2D data to this class.
  void attach(const void *data, size_t x, size_t y, size_t pitch) {
    attach(data, x, y, pitch, get_channel());
  }
  /// Attach 2D data to this class.
  void attach(const void *data, size_t x, size_t y, size_t pitch,
              image_channel channel) {
    detach();
    image_wrapper_base::set_data(
        image_data(const_cast<void *>(data), x, y, pitch, channel));
  }
  /// Detach data.
  virtual void detach() {}

  sampling_info get_sampling_info() { return _sampling_info; }
  void set_sampling_info(sampling_info info) {
    _sampling_info = info;
  }
  const image_data &get_data() { return _data; }
  void set_data(image_data data) { _data = data; }

  image_channel get_channel() { return _data.get_channel(); }
  void set_channel(image_channel channel) { _data.set_channel(channel); }

  image_channel_data_type get_channel_data_type() {
    return _data.get_channel_data_type();
  }
  void set_channel_data_type(image_channel_data_type type) {
    _data.set_channel_data_type(type);
  }

  unsigned get_channel_size() { return _data.get_channel_size(); }
  void set_channel_size(unsigned channel_num, unsigned channel_size) {
    return _data.set_channel_size(channel_num, channel_size);
  }

  sycl::addressing_mode get_addressing_mode() {
    return _sampling_info.get_addressing_mode();
  }
  void set(sycl::addressing_mode addressing_mode) {
    _sampling_info.set(addressing_mode);
  }

  sycl::filtering_mode get_filtering_mode() {
    return _sampling_info.get_filtering_mode();
  }
  void set(sycl::filtering_mode filtering_mode) {
    _sampling_info.set(filtering_mode);
  }

  sycl::coordinate_normalization_mode get_coordinate_normalization_mode() {
    return _sampling_info.get_coordinate_normalization_mode();
  }
  void
  set(sycl::coordinate_normalization_mode coordinate_normalization_mode) {
    _sampling_info.set(coordinate_normalization_mode);
  }

  bool is_coordinate_normalized() {
    return _sampling_info.is_coordinate_normalized();
  }
  void set_coordinate_normalization_mode(int is_normalized) {
    _sampling_info.set_coordinate_normalization_mode(is_normalized);
  }
  void
  set(sycl::addressing_mode addressing_mode,
      sycl::filtering_mode filtering_mode,
      sycl::coordinate_normalization_mode coordinate_normalization_mode) {
    set(addressing_mode);
    set(filtering_mode);
    set(coordinate_normalization_mode);
  }
  void set(sycl::addressing_mode addressing_mode,
           sycl::filtering_mode filtering_mode, int is_normalized) {
    set(addressing_mode);
    set(filtering_mode);
    set_coordinate_normalization_mode(is_normalized);
  }

  unsigned get_channel_num() { return _data.get_channel_num(); }
  void set_channel_num(unsigned num) {
    return _data.set_channel_num(num);
  }

  sycl::image_channel_type get_channel_type() {
    return _data.get_channel_type();
  }
  void set_channel_type(sycl::image_channel_type type) {
    return _data.set_channel_type(type);
  }

  sycl::sampler get_sampler() {
    sycl::sampler smp = _sampling_info.get_sampler();
    /// linear memory only used for sycl::filtering_mode::nearest.
    if (_data.get_data_type() == image_data_type::linear) {
      smp = sycl::sampler(smp.get_coordinate_normalization_mode(),
                          smp.get_addressing_mode(),
                          sycl::filtering_mode::nearest);
    }
    return smp;
  }
};
inline image_wrapper_base::~image_wrapper_base() {}
using image_wrapper_base_p = image_wrapper_base *;

template <class T, int dimensions, bool IsImageArray> class image_accessor_ext;

/// Image class, wrapper of sycl::image.
template <class T, int dimensions, bool IsImageArray = false> class image_wrapper : public image_wrapper_base {
  sycl::image<dimensions> *_image = nullptr;

#ifndef DPCT_USM_LEVEL_NONE
  std::vector<char> _host_buffer;
#endif

  void create_image(sycl::queue q) {
    auto &data = get_data();
    if (data.get_data_type() == image_data_type::matrix) {
      _image = static_cast<image_matrix_p>(data.get_data_ptr())
          ->create_image<dimensions>(data.get_channel());
      return;
    }
    auto ptr = data.get_data_ptr();
    auto channel = data.get_channel();

    if (detail::get_pointer_attribute(q, ptr) == detail::pointer_access_attribute::device_only) {
#ifdef DPCT_USM_LEVEL_NONE
      ptr = get_buffer(ptr)
                .template get_access<sycl::access_mode::read_write>()
                .get_pointer();
#else
      auto sz = data.get_x();
      if (data.get_data_type() == image_data_type::pitch)
        sz *= channel.get_total_size() * data.get_y();
      _host_buffer.resize(sz);
      q.memcpy(_host_buffer.data(), ptr, sz).wait();
      ptr = _host_buffer.data();
#endif
    }

    if constexpr (dimensions == 1) {
      assert(data.get_data_type() == image_data_type::linear);
      _image = new sycl::image<1>(
        ptr, channel.get_channel_order(), channel.get_channel_type(),
        sycl::range<1>(data.get_x() / channel.get_total_size()));
    } else if constexpr (dimensions == 2) {
      assert(data.get_data_type() == image_data_type::pitch);
      _image = new sycl::image<2>(ptr, channel.get_channel_order(),
                                  channel.get_channel_type(),
                                  sycl::range<2>(data.get_x(), data.get_y()),
                                  sycl::range<1>(data.get_pitch()));
    } else {
      throw std::runtime_error("3D image only support matrix data");
    }
    return;
  }

public:
  using acc_data_t = typename detail::image_trait<T>::acc_data_t;
  using accessor_t =
      typename image_accessor_ext<T, IsImageArray ? (dimensions - 1) : dimensions,
                              IsImageArray>::accessor_t;

  image_wrapper() { set_channel(image_channel::create<T>()); }
  ~image_wrapper() { detach(); }

  /// Get image accessor.
  accessor_t get_access(sycl::handler &cgh, sycl::queue &q = get_default_queue()) {
    if (!_image)
      create_image(q);
    return accessor_t(*_image, cgh);
  }

  /// Detach data.
  void detach() override {
    if (_image)
      delete _image;
    _image = nullptr;
  }
};

/// Wrap sampler and image accessor together.
template <class T, int dimensions, bool IsImageArray = false>
class image_accessor_ext {
public:
  using accessor_t =
      typename detail::image_trait<T>::template accessor_t<dimensions>;
  using data_t = typename detail::image_trait<T>::data_t;
  sycl::sampler _sampler;
  accessor_t _img_acc;

public:
  image_accessor_ext(sycl::sampler sampler, accessor_t acc)
      : _sampler(sampler), _img_acc(acc) {}

  /// Read data from accessor.
  template <bool Available = dimensions == 3>
  typename std::enable_if<Available, data_t>::type read(float x, float y,
                                                        float z) {
    return detail::fetch_data<T>()(
        _img_acc.read(sycl::float4(x, y, z, 0), _sampler));
  }
  /// Read data from accessor.
  template <class Coord0, class Coord1, class Coord2,
            bool Available = dimensions == 3 &&
                             std::is_integral<Coord0>::value
                                 &&std::is_integral<Coord1>::value
                                     &&std::is_integral<Coord2>::value>
  typename std::enable_if<Available, data_t>::type read(Coord0 x, Coord1 y,
                                                        Coord2 z) {
    return detail::fetch_data<T>()(
        _img_acc.read(sycl::int4(x, y, z, 0), _sampler));
  }
  /// Read data from accessor.
  template <bool Available = dimensions == 2>
  typename std::enable_if<Available, data_t>::type read(float x, float y) {
    return detail::fetch_data<T>()(
        _img_acc.read(sycl::float2(x, y), _sampler));
  }
  /// Read data from accessor.
  template <class Coord0, class Coord1,
            bool Available = dimensions == 2 &&
                             std::is_integral<Coord0>::value
                                 &&std::is_integral<Coord1>::value>
  typename std::enable_if<Available, data_t>::type read(Coord0 x, Coord1 y) {
    return detail::fetch_data<T>()(
        _img_acc.read(sycl::int2(x, y), _sampler));
  }
  /// Read data from accessor.
  template <bool Available = dimensions == 1>
  typename std::enable_if<Available, data_t>::type read(float x) {
    return detail::fetch_data<T>()(_img_acc.read(x, _sampler));
  }
  /// Read data from accessor.
  template <class CoordT,
            bool Available = dimensions == 1 && std::is_integral<CoordT>::value>
  typename std::enable_if<Available, data_t>::type read(CoordT x) {
    return detail::fetch_data<T>()(_img_acc.read(x, _sampler));
  }
};

template <class T, int dimensions> class image_accessor_ext<T, dimensions, true> {
public:
  using accessor_t =
      typename detail::image_trait<T>::template array_accessor_t<dimensions>;
  using data_t = typename detail::image_trait<T>::data_t;
  sycl::sampler _sampler;
  accessor_t _img_acc;

public:
  image_accessor_ext(sycl::sampler sampler, accessor_t acc)
      : _sampler(sampler), _img_acc(acc) {}

  /// Read data from accessor.
  template <bool Available = dimensions == 2>
  typename std::enable_if<Available, data_t>::type read(int index, float x,
                                                        float y) {
    return detail::fetch_data<T>()(
        _img_acc[index].read(sycl::float2(x, y), _sampler));
  }
  /// Read data from accessor.
  template <bool Available = dimensions == 2>
  typename std::enable_if<Available, data_t>::type read(int index, int x, int y) {
    return detail::fetch_data<T>()(
        _img_acc[index].read(sycl::int2(x, y), _sampler));
  }
  /// Read data from accessor.
  template <bool Available = dimensions == 1>
  typename std::enable_if<Available, data_t>::type read(int index, float x) {
    return detail::fetch_data<T>()(
        _img_acc[index].read(x, _sampler));
  }
  /// Read data from accessor.
  template <bool Available = dimensions == 1>
  typename std::enable_if<Available, data_t>::type read(int index, int x) {
    return detail::fetch_data<T>()(
        _img_acc[index].read(x, _sampler));
  }
};

/// Create image wrapper according to image data and sampling info.
/// \return Pointer to image wrapper base class.
/// \param data Image data used to create image wrapper.
/// \param info Image sampling info used to create image wrapper.
/// \returns Pointer to base class of created image wrapper object.
static inline image_wrapper_base *create_image_wrapper(image_data data,
                              sampling_info info) {
  image_channel channel;
  int dims = 1;
  if (data.get_data_type() == image_data_type::matrix) {
    auto matrix = (image_matrix_p)data.get_data_ptr();
    channel = matrix->get_channel();
    dims = matrix->get_dims();
  } else {
    if (data.get_data_type() == image_data_type::pitch) {
      dims = 2;
    }
    channel = data.get_channel();
  }

  if (auto ret = detail::create_image_wrapper(channel, dims)) {
    ret->set_sampling_info(info);
    ret->set_data(data);
    return ret;
  }
  return nullptr;
}

#ifdef SYCL_EXT_ONEAPI_BINDLESS_IMAGES
namespace detail {
inline std::pair<image_data, sampling_info> &get_img_info_map(
    const sycl::ext::oneapi::experimental::sampled_image_handle handle) {
  static std::unordered_map<sycl::ext::oneapi::experimental::
                                sampled_image_handle::raw_image_handle_type,
                            std::pair<image_data, sampling_info>>
      img_info_map;
  return img_info_map[handle.raw_handle];
}
static inline size_t
get_ele_size(const sycl::ext::oneapi::experimental::image_descriptor &decs) {
  size_t channel_num, channel_size;
  switch (decs.channel_order) {
  case sycl::image_channel_order::r:
    channel_num = 1;
    break;
  case sycl::image_channel_order::rg:
    channel_num = 2;
    break;
  case sycl::image_channel_order::rgb:
    channel_num = 3;
    break;
  case sycl::image_channel_order::rgba:
    channel_num = 4;
    break;
  default:
    throw std::runtime_error("Unsupported channel_order in get_ele_size!");
    break;
  }
  switch (decs.channel_type) {
  case sycl::image_channel_type::signed_int8:
  case sycl::image_channel_type::unsigned_int8:
    channel_size = 1;
    break;
  case sycl::image_channel_type::fp16:
  case sycl::image_channel_type::signed_int16:
  case sycl::image_channel_type::unsigned_int16:
    channel_size = 2;
    break;
  case sycl::image_channel_type::fp32:
  case sycl::image_channel_type::signed_int32:
  case sycl::image_channel_type::unsigned_int32:
    channel_size = 4;
    break;
  default:
    throw std::runtime_error("Unsupported channel_type in get_ele_size!");
    break;
  }
  return channel_num * channel_size;
}
} // namespace detail

/// Create bindless image according to image data and sampling info.
/// \param data Image data used to create bindless image.
/// \param info Image sampling info used to create bindless image.
/// \returns Sampled image handle of created bindless image.
static inline sycl::ext::oneapi::experimental::sampled_image_handle
create_bindless_image(image_data data, sampling_info info,
                      sycl::queue &q = get_default_queue()) {
  auto desc = sycl::ext::oneapi::experimental::image_descriptor();
  auto samp = sycl::ext::oneapi::experimental::bindless_image_sampler(
      info.get_addressing_mode(), info.get_coordinate_normalization_mode(),
      info.get_filtering_mode());

  switch (data.get_data_type()) {
  case image_data_type::linear: {
    desc.width = data.get_x() / data.get_channel().get_total_size();
    desc.channel_order = data.get_channel().get_channel_order();
    desc.channel_type = data.get_channel_type();
    desc.num_levels = 1;
    // linear memory only use sycl::filtering_mode::nearest.
    samp.filtering = sycl::filtering_mode::nearest;
    // TODO: Use pointer to create image when bindless image support.
    auto mem = new sycl::ext::oneapi::experimental::image_mem(desc, q);
    auto img =
        sycl::ext::oneapi::experimental::create_image(*mem, samp, desc, q);
    auto ptr = data.get_data_ptr();
#ifdef DPCT_USM_LEVEL_NONE
    ptr = get_buffer(data.get_data_ptr())
              .template get_access<sycl::access_mode::read_write>()
              .get_pointer();
#endif
    q.ext_oneapi_copy(ptr, mem->get_handle(), desc);
    q.wait_and_throw();
    detail::get_img_info_map(img) = {data, info};
    return img;
  }
  case image_data_type::pitch: {
    desc.width = data.get_x();
    desc.height = data.get_y();
    desc.channel_order = data.get_channel().get_channel_order();
    desc.channel_type = data.get_channel_type();
    desc.num_levels = 1;
#ifdef DPCT_USM_LEVEL_NONE
    auto mem = new sycl::ext::oneapi::experimental::image_mem(desc, q);
    auto img =
        sycl::ext::oneapi::experimental::create_image(*mem, samp, desc, q);
    auto ptr = get_buffer(data.get_data_ptr())
                   .template get_access<sycl::access_mode::read_write>()
                   .get_pointer();
    q.ext_oneapi_copy(ptr, mem->get_handle(), desc);
    q.wait_and_throw();
#else
    auto img = sycl::ext::oneapi::experimental::create_image(
        data.get_data_ptr(), data.get_pitch(), samp, desc, q);
#endif
    detail::get_img_info_map(img) = {data, info};
    return img;
  }
  case image_data_type::matrix: {
    const auto mem = static_cast<sycl::ext::oneapi::experimental::image_mem *>(
        data.get_data_ptr());
    auto img = sycl::ext::oneapi::experimental::create_image(
        *mem, samp, mem->get_descriptor(), q);
    detail::get_img_info_map(img) = {data, info};
    return img;
  }
  default:
    throw std::runtime_error(
        "Unsupported image_data_type in create_bindless_image!");
    break;
  }
  // Must not reach here.
  return sycl::ext::oneapi::experimental::create_image(data.get_data_ptr(), 0,
                                                       samp, desc, q);
}

using image_mem_p = sycl::ext::oneapi::experimental::image_mem *;

/// TODO: ???
static inline image_data
get_data(const sycl::ext::oneapi::experimental::sampled_image_handle img) {
  return detail::get_img_info_map(img).first;
}

/// TODO: ???
static inline sampling_info get_sampling_info(
    const sycl::ext::oneapi::experimental::sampled_image_handle img) {
  return detail::get_img_info_map(img).second;
}

/// TODO: ???
static inline image_channel
get_channel(const sycl::ext::oneapi::experimental::image_mem *mem) {
  image_channel channel;
  channel.set_channel_num(mem->get_num_channels());
  channel.set_channel_type(mem->get_channel_type());
  return channel;
}

/// TODO: ???
template <class T, int dimensions> class bindless_image_wrapper {
public:
  bindless_image_wrapper() : _channel(image_channel::create<T>()) {}
  ~bindless_image_wrapper() { destroy_image_handle(_img, get_default_queue()); }
  void attach(void *data, const image_channel &channel, size_t size = UINT_MAX,
              sycl::queue &q = get_default_queue()) {
    detach(q);
    auto desc = sycl::ext::oneapi::experimental::image_descriptor(
        {size}, channel.get_channel_order(), channel.get_channel_type());
    auto samp = sycl::ext::oneapi::experimental::bindless_image_sampler(
        _addressing_mode, _coordinate_normalization_mode, _filtering_mode);
    // TODO: Use pointer to create image when bindless image support.
    auto mem = new sycl::ext::oneapi::experimental::image_mem(desc, q);
    _img = sycl::ext::oneapi::experimental::create_image(*mem, samp, desc, q);
    q.ext_oneapi_copy(data, mem->get_handle(), desc);
    q.wait_and_throw();
  }
  void attach(void *data, size_t size = UINT_MAX,
              sycl::queue &q = get_default_queue()) {
    attach(data, _channel, size, q);
  }
  void attach(void *data, const image_channel &channel, size_t width,
              size_t height, size_t pitch,
              sycl::queue &q = get_default_queue()) {
    detach(q);
    auto desc = sycl::ext::oneapi::experimental::image_descriptor(
        {width, height}, channel.get_channel_order(),
        channel.get_channel_type());
    auto samp = sycl::ext::oneapi::experimental::bindless_image_sampler(
        _addressing_mode, _coordinate_normalization_mode, _filtering_mode);
    _img = sycl::ext::oneapi::experimental::create_image(data, pitch, samp,
                                                         desc, q);
  }
  void attach(void *data, size_t width, size_t height, size_t pitch,
              sycl::queue &q = get_default_queue()) {
    attach(data, _channel, width, height, pitch, q);
  }
  void attach(sycl::ext::oneapi::experimental::image_mem *mem,
              const image_channel &channel,
              sycl::queue &q = get_default_queue()) {
    detach(q);
    auto samp = sycl::ext::oneapi::experimental::bindless_image_sampler(
        _addressing_mode, _coordinate_normalization_mode, _filtering_mode);
    _img = sycl::ext::oneapi::experimental::create_image(
        *mem, samp, mem->get_descriptor(), q);
  }
  void attach(sycl::ext::oneapi::experimental::image_mem *mem,
              sycl::queue &q = get_default_queue()) {
    attach(mem, _channel, q);
  }
  void detach(sycl::queue &q = get_default_queue()) {
    destroy_image_handle(_img, q);
  }
  void set_channel(image_channel channel) { _channel = channel; }
  void set_channel_size(unsigned channel_num, unsigned channel_size) {
    return _channel.set_channel_size(channel_num, channel_size);
  }
  void set_channel_data_type(image_channel_data_type type) {
    _channel.set_channel_data_type(type);
  }
  void set(sycl::addressing_mode addressing_mode) {
    _addressing_mode = addressing_mode;
  }
  void set(sycl::coordinate_normalization_mode addressing_mode) {
    _coordinate_normalization_mode = addressing_mode;
  }
  void set(sycl::filtering_mode filtering_mode) {
    _filtering_mode = filtering_mode;
  }
  inline sycl::ext::oneapi::experimental::sampled_image_handle get_handle() {
    return _img;
  }

private:
  image_channel _channel;
  sycl::addressing_mode _addressing_mode = sycl::addressing_mode::clamp_to_edge;
  sycl::coordinate_normalization_mode _coordinate_normalization_mode =
      sycl::coordinate_normalization_mode::unnormalized;
  sycl::filtering_mode _filtering_mode = sycl::filtering_mode::nearest;
  sycl::ext::oneapi::experimental::sampled_image_handle _img;
};

/// TODO: ???
static inline void
async_dpct_memcpy(sycl::ext::oneapi::experimental::image_mem *src,
                  size_t w_offset_src, size_t h_offset_src, void *dest,
                  size_t p, size_t w, size_t h,
                  sycl::queue &q = get_default_queue()) {
  auto src_offset = sycl::range<3>(w_offset_src, h_offset_src, 0);
  auto dest_offset = sycl::range<3>(0, 0, 0);
  auto dest_extend = sycl::range<3>(0, 0, 0);
  auto copy_extend =
      sycl::range<3>(p / detail::get_ele_size(src->get_descriptor()), h, 0);
  q.ext_oneapi_copy(src->get_handle(), src_offset, src->get_descriptor(), dest,
                    dest_offset, dest_extend, copy_extend);
}

/// TODO: ???
static inline void
async_dpct_memcpy(sycl::ext::oneapi::experimental::image_mem *src,
                  size_t w_offset_src, size_t h_offset_src, void *dest,
                  size_t s, sycl::queue &q = get_default_queue()) {
  auto w =
      src->get_descriptor().width * detail::get_ele_size(src->get_descriptor());
  auto h = s / w;
  async_dpct_memcpy(src, w_offset_src, h_offset_src, dest, w, w, h, q);
}

/// TODO: ???
static inline void dpct_memcpy(sycl::ext::oneapi::experimental::image_mem *src,
                               size_t w_offset_src, size_t h_offset_src,
                               void *dest, size_t p, size_t w, size_t h,
                               sycl::queue &q = get_default_queue()) {
  async_dpct_memcpy(src, w_offset_src, h_offset_src, dest, p, w, h, q);
  q.wait_and_throw();
}

/// TODO: ???
static inline void dpct_memcpy(sycl::ext::oneapi::experimental::image_mem *src,
                               size_t w_offset_src, size_t h_offset_src,
                               void *dest, size_t s,
                               sycl::queue &q = get_default_queue()) {
  auto w =
      src->get_descriptor().width * detail::get_ele_size(src->get_descriptor());
  auto h = s / w;
  dpct_memcpy(src, w_offset_src, h_offset_src, dest, w, w, h, q);
}

/// TODO: ???
static inline void
async_dpct_memcpy(void *src, sycl::ext::oneapi::experimental::image_mem *dest,
                  size_t w_offset_dest, size_t h_offset_dest, size_t p,
                  size_t w, size_t h, sycl::queue &q = get_default_queue()) {
  auto src_offset = sycl::range<3>(0, 0, 0);
  auto src_extend = sycl::range<3>(0, 0, 0);
  auto dest_offset = sycl::range<3>(w_offset_dest, h_offset_dest, 0);
  auto copy_extend =
      sycl::range<3>(p / detail::get_ele_size(dest->get_descriptor()), h, 0);
  q.ext_oneapi_copy(src, src_offset, src_extend, dest->get_handle(),
                    dest_offset, dest->get_descriptor(), copy_extend);
}

/// TODO: ???
static inline void
async_dpct_memcpy(void *src, sycl::ext::oneapi::experimental::image_mem *dest,
                  size_t w_offset_dest, size_t h_offset_dest, size_t s,
                  sycl::queue &q = get_default_queue()) {
  auto w = dest->get_descriptor().width *
           detail::get_ele_size(dest->get_descriptor());
  auto h = s / w;
  async_dpct_memcpy(src, dest, w_offset_dest, h_offset_dest, w, w, h, q);
}

/// TODO: ???
static inline void dpct_memcpy(void *src,
                               sycl::ext::oneapi::experimental::image_mem *dest,
                               size_t w_offset_dest, size_t h_offset_dest,
                               size_t p, size_t w, size_t h,
                               sycl::queue &q = get_default_queue()) {
  async_dpct_memcpy(src, dest, w_offset_dest, h_offset_dest, p, w, h, q);
  q.wait_and_throw();
}

/// TODO: ???
static inline void dpct_memcpy(void *src,
                               sycl::ext::oneapi::experimental::image_mem *dest,
                               size_t w_offset_dest, size_t h_offset_dest,
                               size_t s, sycl::queue &q = get_default_queue()) {
  auto w = dest->get_descriptor().width *
           detail::get_ele_size(dest->get_descriptor());
  auto h = s / w;
  dpct_memcpy(src, dest, w_offset_dest, h_offset_dest, w, w, h, q);
}

/// TODO: ???
static inline void dpct_memcpy(sycl::ext::oneapi::experimental::image_mem *src,
                               size_t w_offset_src, size_t h_offset_src,
                               sycl::ext::oneapi::experimental::image_mem *dest,
                               size_t w_offset_dest, size_t h_offset_dest,
                               size_t w, size_t h,
                               sycl::queue &q = get_default_queue()) {
  auto temp = (void *)sycl::malloc_device(w * h, q);
  // TODO: need change when bindless support device to device copy.
  dpct_memcpy(src, w_offset_src, h_offset_src, temp, w, w, h, q);
  dpct_memcpy(temp, dest, w_offset_dest, h_offset_dest, w, w, h, q);
  sycl::free(temp, q);
}

/// TODO: ???
static inline void dpct_memcpy(sycl::ext::oneapi::experimental::image_mem *src,
                               size_t w_offset_src, size_t h_offset_src,
                               sycl::ext::oneapi::experimental::image_mem *dest,
                               size_t w_offset_dest, size_t h_offset_dest,
                               size_t s, sycl::queue &q = get_default_queue()) {
  auto w =
      src->get_descriptor().width * detail::get_ele_size(src->get_descriptor());
  auto h = s / w;
  dpct_memcpy(src, w_offset_src, h_offset_src, dest, w_offset_dest,
              h_offset_dest, w, h, q);
}
#endif

namespace detail {
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
static image_wrapper_base *create_image_wrapper(unsigned channel_num, int dims) {
  switch (channel_num) {
  case 1:
    return create_image_wrapper<T>(dims);
  case 2:
    return create_image_wrapper<sycl::vec<T, 2>>(dims);
  case 3:
    return create_image_wrapper<sycl::vec<T, 3>>(dims);
  case 4:
    return create_image_wrapper<sycl::vec<T, 4>>(dims);
  default:
    return nullptr;
  }
}

/// Create image with channel info and specified dimensions.
static image_wrapper_base *create_image_wrapper(image_channel channel, int dims) {
  switch (channel.get_channel_type()) {
  case sycl::image_channel_type::fp16:
    return create_image_wrapper<sycl::half>(channel.get_channel_num(), dims);
  case sycl::image_channel_type::fp32:
    return create_image_wrapper<float>(channel.get_channel_num(), dims);
  case sycl::image_channel_type::signed_int8:
    return create_image_wrapper<std::int8_t>(channel.get_channel_num(), dims);
  case sycl::image_channel_type::signed_int16:
    return create_image_wrapper<std::int16_t>(channel.get_channel_num(), dims);
  case sycl::image_channel_type::signed_int32:
    return create_image_wrapper<std::int32_t>(channel.get_channel_num(), dims);
  case sycl::image_channel_type::unsigned_int8:
    return create_image_wrapper<std::uint8_t>(channel.get_channel_num(), dims);
  case sycl::image_channel_type::unsigned_int16:
    return create_image_wrapper<std::uint16_t>(channel.get_channel_num(), dims);
  case sycl::image_channel_type::unsigned_int32:
    return create_image_wrapper<std::uint32_t>(channel.get_channel_num(), dims);
  default:
    return nullptr;
  }
}
} // namespace detail
} // namespace dpct

#endif // !__DPCT_IMAGE_HPP__
