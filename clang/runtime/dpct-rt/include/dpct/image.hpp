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

/**
 * @file
 * @brief Helper functions to get/set the descriptor of image data and helper
 * functions to fetch/store image data.
 *
 * @copyright Copyright (C) Intel Corporation
 *
 */

namespace dpct {

/**
 * @brief Enum for image channel data type.
 */
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

/**
 * @class image_channel
 * @brief Image channel info, including channel number, order, data width and type.
 */
class image_channel {
  image_channel_data_type _type = image_channel_data_type::signed_int;
  /// Number of channels.
  unsigned _channel_num = 0;
  /// Total size of all channels in bytes.
  unsigned _total_size = 0;
  /// Size of each channel in bytes.
  unsigned _channel_size = 0;

public:
  /**
   * @brief Create image channel info.
   * @tparam T The data type of the image channel.
   * @return The image_channel created.
   */
  template <class T> static image_channel create() {
    image_channel channel;
    channel.set_channel_size(detail::image_trait<T>::channel_num,
                             sizeof(typename detail::image_trait<T>::elem_t) *
                                 8);
    channel.set_channel_data_type(detail::image_trait<T>::data_type);
    return channel;
  }
  /**
   * @brief The default constructor.
   */
  image_channel() = default;

  image_channel_data_type get_channel_data_type() { return _type; }
  void set_channel_data_type(image_channel_data_type type) { _type = type; }

  unsigned get_total_size() { return _total_size; }

  unsigned get_channel_num() { return _channel_num; }
  void set_channel_num(unsigned channel_num) {
    _channel_num = channel_num;
    _total_size = _channel_size * _channel_num;
  }
  /**
   * @brief Create image_channel constructor.
   * @param r Channel r width in bits.
   * @param g Channel g width in bits. Should be same with \p r, or zero.
   * @param b Channel b width in bits. Should be same with \p g, or zero.
   * @param a Channel a width in bits. Should be same with \p b, or zero.
   * @param data_type Image channel data type: signed_nt, unsigned_int or fp.
   */
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
  /**
   * @brief Gets the channel type of current image channel.
   * @return Image channel data type: signed_nt, unsigned_int or fp.
   */
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
  /**
   * @brief Sets the channel type of current image channel.
   * @param type Image channel data type: signed_nt, unsigned_int or fp.
   */
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
  /**
   * @brief Gets the channel order of current image channel.
   * @return Image channel order: r, rg, rgb, rgba.
   */
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
  /**
   * @brief Gets the size for each channel in bits.
   * @return Image channel size.
   */
  unsigned get_channel_size() const { return _channel_size * 8; }
  /**
   * @brief Sets the image channel size.
   * @param in_channel_num The channels number to set.
   * @param channel_size The size for each channel in bits.
   */
  void set_channel_size(unsigned in_channel_num,
                        unsigned channel_size) {
    if (in_channel_num < _channel_num)
      return;
    _channel_num = in_channel_num;
    _channel_size = channel_size / 8;
    _total_size = _channel_size * _channel_num;
  }
};
/**
 * @class image_matrix
 * @brief Class to store 2D/3D matrix data in host.
 */
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
  /**
   * @brief Constructor with channel info and dimension size info.
   * @tparam dimensions The dimension of the data.
   * @param image_channel The image channel information of the data.
   * @param range The range of each dimension.
   */
  template <int dimensions>
  image_matrix(image_channel channel, sycl::range<dimensions> range)
      : _channel(channel) {
    set_range(range);
    _host_data = std::malloc(range.size() * _channel.get_total_size());
  }
  /**
   * @brief Constructor for 2D image matrix.
   * @param channel_type The image channel type of the data.
   * @param channel_num The channel number of the data.
   * @param x The width of the data.
   * @param y The height of the data.
   */
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
  /**
   * @brief Creates image with the matrix data.
   * @return The created \a sycl::image.
   */
  template <int dimensions> sycl::image<dimensions> *create_image() {
    return create_image<dimensions>(_channel);
  }
  /**
   * @brief Creates image with the matrix data.
   * @param channel The image channel to be used.
   * @return The created \a sycl::image.
   */
  template <int dimensions>
  sycl::image<dimensions> *create_image(image_channel channel) {
    return new sycl::image<dimensions>(
        _host_data, channel.get_channel_order(), channel.get_channel_type(),
        get_range(make_index_sequence<dimensions>()),
        sycl::property::image::use_host_ptr());
  }
  /**
   * @brief Gets the channel info.
   * @return The image channel info for current image.
   */
  inline image_channel get_channel() { return _channel; }
  /**
   * @brief Gets the range of the image.
   * @return The 3D range of the image.
   */
  sycl::range<3> get_range() {
    return sycl::range<3>(_range[0], _range[1], _range[2]);
  }
  /**
   * @brief Gets the matrix dimension of the image.
   * @return The matrix dimension.
   */
  inline int get_dims() { return _dims; }
  /**
   * @brief Gets a pitched data from the matrix data.
   * @return The pitched data.
   */
  pitched_data to_pitched_data() {
    return pitched_data(_host_data, _range[0] * _channel.get_total_size(),
                        _range[0], _range[1]);
  }
  /**
   * @brief The default destructor.
   */
  ~image_matrix() {
    if (_host_data)
      std::free(_host_data);
    _host_data = nullptr;
  }
};

/**
 * @class image_matrix_p
 * @brief Pointer type of image_matrix.
 */
using image_matrix_p = image_matrix *;
/**
 * @brief Enum for image data type.
 */
enum class image_data_type { matrix, linear, pitch, unsupport };

#ifdef SYCL_EXT_ONEAPI_BINDLESS_IMAGES
namespace experimental {
class image_mem_wrapper;
}
#endif

/**
 * @class image_data
 * @brief Class to store image data with channel information.
 */
class image_data {
public:
  /**
   * @brief The default constructor.
   */
  image_data() { _type = image_data_type::unsupport; }
  /**
   * @brief The constructor with matrix_data.
   * @param matrix_data The matrix data to be used.
   */
  image_data(image_matrix_p matrix_data) { set_data(matrix_data); }
  /**
   * @brief The constructor for 1D data.
   * @param data_ptr The pointer to the data.
   * @param x_size The length of the data.
   * @param channel The image channel to be used.
   */
  image_data(void *data_ptr, size_t x_size, image_channel channel) {
    set_data(data_ptr, x_size, channel);
  }
  /**
   * @brief The constructor for 2D data.
   * @param data_ptr The pointer to the data.
   * @param x_size The width of the data.
   * @param y_size The height of the data.
   * @param pitch_size The pitch size of the data.
   * @param channel The image channel to be used.
   */
  image_data(void *data_ptr, size_t x_size, size_t y_size, size_t pitch_size,
             image_channel channel) {
    set_data(data_ptr, x_size, y_size, pitch_size, channel);
  }
  /**
   * @brief Sets the image data from matrix data.
   * @param matrix_data The matrix data to be used.
   */
  void set_data(image_matrix_p matrix_data) {
    _type = image_data_type::matrix;
    _data = matrix_data;
    _channel = matrix_data->get_channel();
  }
#ifdef SYCL_EXT_ONEAPI_BINDLESS_IMAGES
  void set_data(experimental::image_mem_wrapper *image_mem) {
    _type = image_data_type::matrix;
    _data = image_mem;
  }
#endif
  /**
   * @brief Sets the image data.
   * @param matrix_data The matrix data to be used.
   */
  void set_data(void *data_ptr, size_t x_size, image_channel channel) {
    _type = image_data_type::linear;
    _data = data_ptr;
    _x = x_size;
    _channel = channel;
  }
  /**
   * @brief Sets the image data.
   * @param data_ptr The pointer to the data.
   * @param x_size The width of the data.
   * @param y_size The height of the data.
   * @param pitch_size The pitch size of the data.
   * @param channel The image channel to be used.
   */
  void set_data(void *data_ptr, size_t x_size, size_t y_size, size_t pitch_size,
                image_channel channel) {
    _type = image_data_type::pitch;
    _data = data_ptr;
    _x = x_size;
    _y = y_size;
    _pitch = pitch_size;
    _channel = channel;
  }
  /**
   * @brief Gets the data type of the image.
   * @return The data type of the image.
   */
  image_data_type get_data_type() const { return _type; }
  /**
   * @brief Sets the data type of the image.
   * @param [in] type The data type of the image.
   */
  void set_data_type(image_data_type type) { _type = type; }

  /**
   * @brief Gets the data pointer of the image.
   * @return The data pointer of the image.
   */
  void *get_data_ptr() const { return _data; }
  /**
   * @brief Sets the data pointer of the image.
   * @param [in] type The data pointer of the image.
   */
  void set_data_ptr(void *data) { _data = data; }

  /**
   * @brief Gets the data width of the image.
   * @return The data width of the image.
   */
  size_t get_x() const { return _x; }
  /**
   * @brief Sets the data width of the image.
   * @param [in] x The data width of the image.
   */
  void set_x(size_t x) { _x = x; }
  /**
   * @brief Gets the data height of the image.
   * @return The data height of the image.
   */
  size_t get_y() const { return _y; }
  /**
   * @brief Sets the data height of the image.
   * @param [in] y The data height of the image.
   */
  void set_y(size_t y) { _y = y; }
  /**
   * @brief Gets the pitch size of the image data.
   * @return The pitch size of the image data.
   */
  size_t get_pitch() const { return _pitch; }
  /**
   * @brief Sets the pitch size of the image data.
   * @param [in] pitch The pitch size of the image data.
   */
  void set_pitch(size_t pitch) { _pitch = pitch; }

  /**
   * @brief Gets the channel info of the image.
   * @return The channel info of the image.
   */
  image_channel get_channel() const { return _channel; }
  /**
   * @brief Sets the channel info of the image.
   * @param [in] channel The channel info of the image.
   */
  void set_channel(image_channel channel) { _channel = channel; }

  /**
   * @brief Gets the channel data type the image.
   * @return The channel data type the image.
   */
  image_channel_data_type get_channel_data_type() {
    return _channel.get_channel_data_type();
  }
  /**
   * @brief Sets the channel data type the image.
   * @param [in] type The channel data type the image.
   */
  void set_channel_data_type(image_channel_data_type type) {
    _channel.set_channel_data_type(type);
  }

  /**
   * @brief Gets the channel size of the image.
   * @return The channel size of the image.
   */
  unsigned get_channel_size() { return _channel.get_channel_size(); }
  /**
   * @brief Sets the channel size of the image.
   * @param [in] channel_num The channel num of the image.
   * @param [in] channel_size The channel size of the image.
   */
  void set_channel_size(unsigned channel_num, unsigned channel_size) {
    return _channel.set_channel_size(channel_num, channel_size);
  }

  /**
   * @brief Gets the channel number of the image.
   * @return The channel number of the image.
   */
  unsigned get_channel_num() { return _channel.get_channel_num(); }
  /**
   * @brief Sets the channel number of the image.
   * @param [in] channel_num The channel number of the image.
   */
  void set_channel_num(unsigned num) {
    return _channel.set_channel_num(num);
  }

  /**
   * @brief Gets the channel type of the image.
   * @return The channel type of the image.
   */
  sycl::image_channel_type get_channel_type() {
    return _channel.get_channel_type();
  }
  /**
   * @brief Sets the channel type of the image.
   * @param [in] type The channel type of the image.
   */
  void set_channel_type(sycl::image_channel_type type) {
    return _channel.set_channel_type(type);
  }

private:
  image_data_type _type;
  void *_data = nullptr;
  size_t _x, _y, _pitch;
  image_channel _channel;
};

/**
 * @class sampling_info
 * @brief Class to store image sampling info, include addressing mode, filtering
 * mode and normalization info.
 */
class sampling_info {
  sycl::addressing_mode _addressing_mode =
      sycl::addressing_mode::clamp_to_edge;
  sycl::filtering_mode _filtering_mode = sycl::filtering_mode::nearest;
  sycl::coordinate_normalization_mode _coordinate_normalization_mode =
      sycl::coordinate_normalization_mode::unnormalized;
  // Dictates the method in which sampling between mipmap levels is performed.
  sycl::filtering_mode _mipmap_filtering = sycl::filtering_mode::nearest;
  // Defines the minimum mipmap level from which we can sample, with the minimum
  // value being 0.
  float _min_mipmap_level_clamp = 0.f;
  // Defines the maximum mipmap level from which we can sample. This value
  // cannot be higher than the number of allocated levels.
  float _max_mipmap_level_clamp = 0.f;
  // Dictates the anisotropic ratio used when sampling the mipmap with
  // anisotropic filtering.
  float _max_anisotropy = 0.f;

public:
<<<<<<< HEAD:clang/runtime/dpct-rt/include/dpct/image.hpp
  sycl::addressing_mode get_addressing_mode() const noexcept {
    return _addressing_mode;
  }
  void set(sycl::addressing_mode addressing_mode) noexcept {
    _addressing_mode = addressing_mode;
  }

  sycl::filtering_mode get_filtering_mode() const noexcept {
    return _filtering_mode;
  }
  void set(sycl::filtering_mode filtering_mode) noexcept {
    _filtering_mode = filtering_mode;
  }

  sycl::coordinate_normalization_mode
  get_coordinate_normalization_mode() const noexcept {
    return _coordinate_normalization_mode;
  }
  void set(sycl::coordinate_normalization_mode
               coordinate_normalization_mode) noexcept {
    _coordinate_normalization_mode = coordinate_normalization_mode;
  }

  bool is_coordinate_normalized() const noexcept {
    return _coordinate_normalization_mode ==
           sycl::coordinate_normalization_mode::normalized;
  }
  void set_coordinate_normalization_mode(int is_normalized) noexcept {
=======
  /**
   * @brief Gets the addressing mode.
   * @return The addressing mode.
   */
  sycl::addressing_mode get_addressing_mode() { return _addressing_mode; }
  /**
   * @brief Sets the addressing mode.
   * @param addressing_mode The addressing mode.
   */
  void set(sycl::addressing_mode addressing_mode) { _addressing_mode = addressing_mode; }
  /**
   * @brief Gets the filtering mode.
   * @return The filtering mode.
   */
  sycl::filtering_mode get_filtering_mode() { return _filtering_mode; }
  /**
   * @brief Sets the filtering mode.
   * @param filtering_mode The filtering mode.
   */
  void set(sycl::filtering_mode filtering_mode) { _filtering_mode = filtering_mode; }
  /**
   * @brief Gets the coordinate normalization mode.
   * @return The coordinate normalization mode.
   */
  sycl::coordinate_normalization_mode get_coordinate_normalization_mode() {
    return _coordinate_normalization_mode;
  }
  /**
   * @brief Sets the coordinate normalization mode.
   * @param coordinate_normalization_mode The coordinate normalization mode.
   */
  void set(sycl::coordinate_normalization_mode coordinate_normalization_mode) {
    _coordinate_normalization_mode = coordinate_normalization_mode;
  }
  /**
   * @brief Gets whether the coordinate normalization mode is normalized.
   * @return true if the coordinate normalization mode is normalized.
   */
  bool is_coordinate_normalized() {
    return _coordinate_normalization_mode ==
           sycl::coordinate_normalization_mode::normalized;
  }
  /**
   * @brief Sets whether the coordinate normalization mode is normalized.
   * @param is_normalized Whether the coordinate normalization mode is normalized.
   */
  void set_coordinate_normalization_mode(int is_normalized) {
>>>>>>> Add comment doxygen comments:clang/runtime/dpct-rt/include/image.hpp
    _coordinate_normalization_mode =
        is_normalized ? sycl::coordinate_normalization_mode::normalized
                      : sycl::coordinate_normalization_mode::unnormalized;
  }
<<<<<<< HEAD:clang/runtime/dpct-rt/include/dpct/image.hpp

  /// Get the method in which sampling between mipmap levels is performed.
  /// \returns The method in which sampling between mipmap levels is performed.
  sycl::filtering_mode get_mipmap_filtering() const noexcept {
    return _mipmap_filtering;
  }
  /// Set the method in which sampling between mipmap levels is performed.
  /// \param [in] filtering_mode The method in which sampling between mipmap
  /// levels is performed.
  void set_mipmap_filtering(sycl::filtering_mode filtering_mode) noexcept {
    _mipmap_filtering = filtering_mode;
  }

  /// Get the minimum mipmap level from which we can sample
  /// \returns The minimum mipmap level from which we can sample.
  float get_min_mipmap_level_clamp() const noexcept {
    return _min_mipmap_level_clamp;
  }
  /// Set the minimum mipmap level from which we can sample
  /// \param [in] min_mipmap_level_clamp The minimum mipmap level from which we
  /// can sample.
  void set_min_mipmap_level_clamp(float min_mipmap_level_clamp) noexcept {
    _min_mipmap_level_clamp = min_mipmap_level_clamp;
  }

  /// Get the maximum mipmap level from which we can sample.
  /// \returns The maximum mipmap level from which we can sample.
  float get_max_mipmap_level_clamp() const noexcept {
    return _max_mipmap_level_clamp;
  }
  /// Set the maximum mipmap level from which we can sample.
  /// \param [in] max_mipmap_level_clamp The maximum mipmap level from which we
  /// can sample.
  void set_max_mipmap_level_clamp(float max_mipmap_level_clamp) noexcept {
    _max_mipmap_level_clamp = max_mipmap_level_clamp;
  }

  /// Get the anisotropic ratio used when sampling the mipmap with
  // anisotropic filtering.
  /// \returns The anisotropic ratio used when sampling the mipmap with
  // anisotropic filtering.
  float get_max_anisotropy() const noexcept { return _max_anisotropy; }
  /// Set the anisotropic ratio used when sampling the mipmap with
  // anisotropic filtering.
  /// \param [in] max_anisotropy The anisotropic ratio used when sampling the
  /// mipmap with anisotropic filtering.
  void set_max_anisotropy(float max_anisotropy) noexcept {
    _max_anisotropy = max_anisotropy;
  }

  void set(sycl::addressing_mode addressing_mode,
           sycl::filtering_mode filtering_mode,
           sycl::coordinate_normalization_mode
               coordinate_normalization_mode) noexcept {
=======
  /**
   * @brief Sets the addressing mode, filtering mode and the coordinate
   * normalization mode.
   * @param addressing_mode The addressing mode.
   * @param filtering_mode The filtering mode.
   * @param coordinate_normalization_mode The coordinate normalization mode to
   * be used.
   */
  void
  set(sycl::addressing_mode addressing_mode,
      sycl::filtering_mode filtering_mode,
      sycl::coordinate_normalization_mode coordinate_normalization_mode) {
>>>>>>> Add comment doxygen comments:clang/runtime/dpct-rt/include/image.hpp
    set(addressing_mode);
    set(filtering_mode);
    set(coordinate_normalization_mode);
  }
  /**
   * @brief Sets the addressing mode, filtering mode.
   * @param addressing_mode The addressing mode.
   * @param filtering_mode The filtering mode.
   * @param is_normalized Whether the coordinate normalization mode is
   * normalized.
   */
  void set(sycl::addressing_mode addressing_mode,
           sycl::filtering_mode filtering_mode, int is_normalized) noexcept {
    set(addressing_mode);
    set(filtering_mode);
    set_coordinate_normalization_mode(is_normalized);
  }
<<<<<<< HEAD:clang/runtime/dpct-rt/include/dpct/image.hpp

  sycl::sampler get_sampler() const {
=======
  /**
   * @brief Gets \a sycl::sampler with the stored info.
   * @return The created \a sycl::sampler.
   */
  sycl::sampler get_sampler() {
>>>>>>> Add comment doxygen comments:clang/runtime/dpct-rt/include/image.hpp
    return sycl::sampler(_coordinate_normalization_mode, _addressing_mode,
                         _filtering_mode);
  }
};

/**
 * @class image_wrapper_base
 * @brief Base class for image wrapper which extend \a sycl::image
 */
class image_wrapper_base {
  sampling_info _sampling_info;
  image_data _data;

public:
  virtual ~image_wrapper_base() = 0;
  /**
   * @brief Attaches image data to this class.
   * @param data The image data to be used.
   */
  void attach(image_data data) { set_data(data); }
  /**
   * @brief Attaches matrix data to this class.
   * @param data The matrix data to be used.
   */
  void attach(image_matrix *matrix) {
    detach();
    image_wrapper_base::set_data(image_data(matrix));
  }
  /**
   * @brief Attaches matrix data to this class.
   * @param data The matrix data to be used.
   * @param channel The channel information of the data.
   */
  void attach(image_matrix *matrix, image_channel channel) {
    attach(matrix);
    image_wrapper_base::set_channel(channel);
  }
  /**
   * @brief Attaches linear data to this class.
   * @param ptr The pointer to the data.
   * @param count The size of the data.
   */
  void attach(const void *ptr, size_t count) {
    attach(ptr, count, get_channel());
  }
  /**
   * @brief Attaches linear data to this class.
   * @param ptr The pointer to the data.
   * @param count The size of the data.
   * @param channel The channel information of the data.
   */
  void attach(const void *ptr, size_t count, image_channel channel) {
    detach();
    image_wrapper_base::set_data(image_data(const_cast<void *>(ptr), count, channel));
  }
  /**
   * @brief Attaches 2D data to this class.
   * @param data The pointer to the data.
   * @param x The width of the data.
   * @param y The height of the data.
   * @param pitch The pitch size of the data.
   */
  void attach(const void *data, size_t x, size_t y, size_t pitch) {
    attach(data, x, y, pitch, get_channel());
  }
  /**
   * @brief Attaches 2D data to this class.
   * @param data The pointer to the data.
   * @param x The width of the data.
   * @param y The height of the data.
   * @param pitch The pitch size of the data.
   * @param channel The channel information of the data.
   */
  void attach(const void *data, size_t x, size_t y, size_t pitch,
              image_channel channel) {
    detach();
    image_wrapper_base::set_data(
        image_data(const_cast<void *>(data), x, y, pitch, channel));
  }
  /**
   * @brief Detaches data.
   */
  virtual void detach() {}

  /**
   * @brief Gets the sampling information.
   * @return The sampling information.
   */
  sampling_info get_sampling_info() { return _sampling_info; }
  void set_sampling_info(sampling_info info) {
    _sampling_info = info;
  }
  /**
   * @brief Gets the image data.
   * @return The reference to the image data.
   */
  const image_data &get_data() { return _data; }
  /**
   * @brief Sets the image data.
   * @param data The image data.
   */
  void set_data(image_data data) { _data = data; }
  /**
   * @brief Gets the image channel information.
   * @return The image channel information.
   */
  image_channel get_channel() { return _data.get_channel(); }
  /**
   * @brief Sets the image channel information.
   * @param channel The image channel information.
   */
  void set_channel(image_channel channel) { _data.set_channel(channel); }
  /**
   * @brief Gets the image channel data type.
   * @return The image channel data type.
   */
  image_channel_data_type get_channel_data_type() {
    return _data.get_channel_data_type();
  }
  /**
   * @brief Sets the image channel data type.
   * @param type The image channel data type.
   */
  void set_channel_data_type(image_channel_data_type type) {
    _data.set_channel_data_type(type);
  }
  /**
   * @brief Gets the image channel size.
   * @return The image channel size.
   */
  unsigned get_channel_size() { return _data.get_channel_size(); }
  /**
   * @brief Sets the image channel size.
   * @param channel_num The image channel number.
   * @param channel_size The image channel size.
   */
  void set_channel_size(unsigned channel_num, unsigned channel_size) {
    return _data.set_channel_size(channel_num, channel_size);
  }
  /**
   * @brief Gets the image addressing mode.
   * @return The image addressing mode.
   */
  sycl::addressing_mode get_addressing_mode() {
    return _sampling_info.get_addressing_mode();
  }
  /**
   * @brief Sets the image addressing mode.
   * @param addressing_mode The image addressing mode.
   */
  void set(sycl::addressing_mode addressing_mode) {
    _sampling_info.set(addressing_mode);
  }
  /**
   * @brief Gets the image filtering mode.
   * @return The image filtering mode.
   */
  sycl::filtering_mode get_filtering_mode() {
    return _sampling_info.get_filtering_mode();
  }
  /**
   * @brief Sets the image filtering mode.
   * @param filtering_mode The image filtering mode.
   */
  void set(sycl::filtering_mode filtering_mode) {
    _sampling_info.set(filtering_mode);
  }
  /**
   * @brief Gets the image coordinate normalization mode.
   * @return The image coordinate normalization mode.
   */
  sycl::coordinate_normalization_mode get_coordinate_normalization_mode() {
    return _sampling_info.get_coordinate_normalization_mode();
  }
  /**
   * @brief Sets the image coordinate normalization mode.
   * @param coordinate_normalization_mode The image coordinate normalization
   * mode.
   */
  void
  set(sycl::coordinate_normalization_mode coordinate_normalization_mode) {
    _sampling_info.set(coordinate_normalization_mode);
  }
  /**
   * @brief Gets whether the image is coordinate normalized.
   * @return true if the image is coordinate normalized.
   */
  bool is_coordinate_normalized() {
    return _sampling_info.is_coordinate_normalized();
  }
  /**
   * @brief Sets whether the image is coordinate normalized.
   * @param is_normalized whether the image is coordinate normalized.
   */
  void set_coordinate_normalization_mode(int is_normalized) {
    _sampling_info.set_coordinate_normalization_mode(is_normalized);
  }
  /**
   * @brief Sets the addressing mode, filtering mode and the coordinate
   * normalization mode.
   * @param addressing_mode The addressing mode.
   * @param filtering_mode The filtering mode.
   * @param coordinate_normalization_mode The coordinate normalization mode to
   * be used.
   */
  void
  set(sycl::addressing_mode addressing_mode,
      sycl::filtering_mode filtering_mode,
      sycl::coordinate_normalization_mode coordinate_normalization_mode) {
    set(addressing_mode);
    set(filtering_mode);
    set(coordinate_normalization_mode);
  }
  /**
   * @brief Sets the addressing mode, filtering mode.
   * @param addressing_mode The addressing mode.
   * @param filtering_mode The filtering mode.
   * @param is_normalized Whether the coordinate normalization mode is
   * normalized.
   */
  void set(sycl::addressing_mode addressing_mode,
           sycl::filtering_mode filtering_mode, int is_normalized) {
    set(addressing_mode);
    set(filtering_mode);
    set_coordinate_normalization_mode(is_normalized);
  }
  /**
   * @brief Gets the image channel number.
   * @return The image channel number.
   */
  unsigned get_channel_num() { return _data.get_channel_num(); }
  /**
   * @brief Sets the image channel number.
   * @param num The image channel number.
   */
  void set_channel_num(unsigned num) {
    return _data.set_channel_num(num);
  }
  /**
   * @brief Gets the image channel type.
   * @return The image channel type.
   */
  sycl::image_channel_type get_channel_type() {
    return _data.get_channel_type();
  }
  /**
   * @brief Sets the image channel type.
   * @param type The image channel type.
   */
  void set_channel_type(sycl::image_channel_type type) {
    return _data.set_channel_type(type);
  }
<<<<<<< HEAD:clang/runtime/dpct-rt/include/dpct/image.hpp

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
=======
  /**
   * @brief Gets \a sycl::sampler with the stored info.
   * @return The created \a sycl::sampler.
   */
  sycl::sampler get_sampler() { return _sampling_info.get_sampler(); }
>>>>>>> Add comment doxygen comments:clang/runtime/dpct-rt/include/image.hpp
};
inline image_wrapper_base::~image_wrapper_base() {}
using image_wrapper_base_p = image_wrapper_base *;

template <class T, int dimensions, bool IsImageArray> class image_accessor_ext;

/**
 * @class image_wrapper
 * @brief Image class, wrapper of sycl::image.
 */
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
      ptr = get_buffer(ptr).get_host_access().get_pointer();
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
  /**
   * @brief The default constructor. Initializes the channel information.
   */
  image_wrapper() { set_channel(image_channel::create<T>()); }
  /**
   * @brief The default destructor.
   */
  ~image_wrapper() { detach(); }

  /**
   * @brief Gets the image accessor.
   * @param [in] cgh The command group handler.
   * @param [in] q The queue to create the \a sycl::image.
   */
  accessor_t get_access(sycl::handler &cgh, sycl::queue &q = get_default_queue()) {
    if (!_image)
      create_image(q);
    return accessor_t(*_image, cgh);
  }

  /**
   * @brief Detaches the image data.
   */
  void detach() override {
    if (_image)
      delete _image;
    _image = nullptr;
  }
};

/**
 * @class image_accessor_ext
 * @brief Wrap sampler and image accessor together.
 */
template <class T, int dimensions, bool IsImageArray = false>
class image_accessor_ext {
public:
  using accessor_t =
      typename detail::image_trait<T>::template accessor_t<dimensions>;
  using data_t = typename detail::image_trait<T>::data_t;
  sycl::sampler _sampler;
  accessor_t _img_acc;

public:
  /**
   * @brief The default constructor.
   */
  image_accessor_ext(sycl::sampler sampler, accessor_t acc)
      : _sampler(sampler), _img_acc(acc) {}
  /**
   * @brief Read data from accessor.
   */
  template <bool Available = dimensions == 3>
  typename std::enable_if<Available, data_t>::type read(float x, float y,
                                                        float z) {
    return detail::fetch_data<T>()(
        _img_acc.read(sycl::float4(x, y, z, 0), _sampler));
  }
  /**
   * @brief Read data from accessor.
   */
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
  /**
   * @brief Read data from accessor.
   */
  template <bool Available = dimensions == 2>
  typename std::enable_if<Available, data_t>::type read(float x, float y) {
    return detail::fetch_data<T>()(
        _img_acc.read(sycl::float2(x, y), _sampler));
  }
  /**
   * @brief Read data from accessor.
   */
  template <class Coord0, class Coord1,
            bool Available = dimensions == 2 &&
                             std::is_integral<Coord0>::value
                                 &&std::is_integral<Coord1>::value>
  typename std::enable_if<Available, data_t>::type read(Coord0 x, Coord1 y) {
    return detail::fetch_data<T>()(
        _img_acc.read(sycl::int2(x, y), _sampler));
  }
  /**
   * @brief Read data from accessor.
   */
  template <bool Available = dimensions == 1>
  typename std::enable_if<Available, data_t>::type read(float x) {
    return detail::fetch_data<T>()(_img_acc.read(x, _sampler));
  }
  /**
   * @brief Read data from accessor.
   */
  template <class CoordT,
            bool Available = dimensions == 1 && std::is_integral<CoordT>::value>
  typename std::enable_if<Available, data_t>::type read(CoordT x) {
    return detail::fetch_data<T>()(_img_acc.read(x, _sampler));
  }
};

/**
 * @class image_accessor_ext
 * @brief Wrap sampler and image accessor together.
 */
template <class T, int dimensions> class image_accessor_ext<T, dimensions, true> {
public:
  using accessor_t =
      typename detail::image_trait<T>::template array_accessor_t<dimensions>;
  using data_t = typename detail::image_trait<T>::data_t;
  sycl::sampler _sampler;
  accessor_t _img_acc;

public:
  /**
   * @brief The constructor with sampler and accessor.
   * @param [in] sampler The image sampler.
   * @param [in] acc The image accessor.
   */
  image_accessor_ext(sycl::sampler sampler, accessor_t acc)
      : _sampler(sampler), _img_acc(acc) {}

  /**
   * @brief Read data from accessor.
   */
  template <bool Available = dimensions == 2>
  typename std::enable_if<Available, data_t>::type read(int index, float x,
                                                        float y) {
    return detail::fetch_data<T>()(
        _img_acc[index].read(sycl::float2(x, y), _sampler));
  }
  /**
   * @brief Read data from accessor.
   */
  template <bool Available = dimensions == 2>
  typename std::enable_if<Available, data_t>::type read(int index, int x, int y) {
    return detail::fetch_data<T>()(
        _img_acc[index].read(sycl::int2(x, y), _sampler));
  }
  /**
   * @brief Read data from accessor.
   */
  template <bool Available = dimensions == 1>
  typename std::enable_if<Available, data_t>::type read(int index, float x) {
    return detail::fetch_data<T>()(
        _img_acc[index].read(x, _sampler));
  }
  /**
   * @brief Read data from accessor.
   */
  template <bool Available = dimensions == 1>
  typename std::enable_if<Available, data_t>::type read(int index, int x) {
    return detail::fetch_data<T>()(
        _img_acc[index].read(x, _sampler));
  }
};
/**
 * @brief Creates image wrapper.
 * @param data Image data used to create image wrapper.
 * @param info Image sampling info used to create image wrapper.
 * @return Pointer to base class of created image wrapper object.
 */
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

namespace detail {
/**
 * @brief Creates image wrapper.
 * @tparam T The data type of the image.
 * @param dims The dimension of the image.
 * @return The created image_wrapper_base.
 */
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
/**
 * @brief Creates image wrapper.
 * @tparam T The data type of the image.
 * @param channel_num The channel number of the image.
 * @param dims The dimension of the image.
 * @return The created image_wrapper_base.
 */
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

/**
 * @brief Creates image wrapper.
 * @param channel The image channel to be used.
 * @param dims The dimension of the image.
 * @return The created image_wrapper_base.
 */
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
