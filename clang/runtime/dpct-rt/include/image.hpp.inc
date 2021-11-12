// DPCT_LABEL_BEGIN|License|
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
//==---- image.hpp --------------------------------*- C++ -*----------------==//
//
// Copyright (C) 2018 - 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
// DPCT_LABEL_END

#ifndef __DPCT_IMAGE_HPP__
#define __DPCT_IMAGE_HPP__
// DPCT_COMMENT
// DPCT_COMMENT Example1:
// DPCT_COMMENT // DPCT_LABEL_BEGIN|FeatureNameDef|[Namespace]
// DPCT_COMMENT // DPCT_DEPENDENCY_EMPTY
// DPCT_COMMENT // DPCT_CODE
// DPCT_COMMENT some code
// DPCT_COMMENT // DPCT_LABEL_END
// DPCT_COMMENT
// DPCT_COMMENT Example2:
// DPCT_COMMENT // DPCT_LABEL_BEGIN|FeatureNameDef|[Namespace]
// DPCT_COMMENT // DPCT_DEPENDENCY_BEGIN
// DPCT_COMMENT // FileID|FeatureNameRef
// DPCT_COMMENT [// FileID|FeatureNameRef]
// DPCT_COMMENT ...
// DPCT_COMMENT // DPCT_DEPENDENCY_END
// DPCT_COMMENT // DPCT_CODE
// DPCT_COMMENT some code
// DPCT_COMMENT // DPCT_LABEL_END
// DPCT_COMMENT
// DPCT_COMMENT For header file including dependency, please use predefined feature name:
// DPCT_COMMENT   local_include_dependency: dpct helper files
// DPCT_COMMENT   non_local_include_dependency: other headler files

// DPCT_LABEL_BEGIN|non_local_include_dependency|
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
#include <CL/sycl.hpp>
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|local_include_dependency|
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
// DPCT_LABEL_END
#include "memory.hpp"
#include "util.hpp"

namespace dpct {

// DPCT_LABEL_BEGIN|image_channel_data_type|dpct
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
enum class image_channel_data_type {
  signed_int,
  unsigned_int,
  fp,
};
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|image_channel_forward_decl|dpct
// DPCT_DEPENDENCY_BEGIN
// Image|image_channel
// DPCT_DEPENDENCY_END
// DPCT_CODE
class image_channel;
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|image_wrapper_base_forward_decl|dpct
// DPCT_DEPENDENCY_BEGIN
// Image|image_wrapper_base
// DPCT_DEPENDENCY_END
// DPCT_CODE
class image_wrapper_base;
// DPCT_LABEL_END
namespace detail {
// DPCT_LABEL_BEGIN|image_trait|dpct::detail
// DPCT_DEPENDENCY_BEGIN
// Image|image_channel_data_type
// DPCT_DEPENDENCY_END
// DPCT_CODE
/// Image object type traits, with accessor type and sampled data type defined.
/// The data type of an image accessor must be one of cl_int4, cl_uint4,
/// cl_float4 and cl_half4. The data type of accessors with 8bits/16bits channel
/// width will be 32 bits. cl_harf is an exception.
template <class T> struct image_trait {
  using acc_data_t = cl::sycl::vec<T, 4>;
  template <int dimensions>
  using accessor_t =
      cl::sycl::accessor<acc_data_t, dimensions, cl::sycl::access_mode::read,
                         cl::sycl::access::target::image>;
  template <int dimensions>
  using array_accessor_t =
      cl::sycl::accessor<acc_data_t, dimensions, cl::sycl::access_mode::read,
                         cl::sycl::access::target::image_array>;
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
template <>
struct image_trait<char>
    : public image_trait<typename std::conditional<
          std::is_signed<char>::value, signed char, unsigned char>::type> {};

template <class T>
struct image_trait<cl::sycl::vec<T, 1>> : public image_trait<T> {};

template <class T>
struct image_trait<cl::sycl::vec<T, 2>> : public image_trait<T> {
  using data_t = cl::sycl::vec<T, 2>;
  static constexpr int channel_num = 2;
};

template <class T>
struct image_trait<cl::sycl::vec<T, 3>>
    : public image_trait<cl::sycl::vec<T, 4>> {
  static constexpr int channel_num = 3;
};

template <class T>
struct image_trait<cl::sycl::vec<T, 4>> : public image_trait<T> {
  using data_t = cl::sycl::vec<T, 4>;
  static constexpr int channel_num = 4;
};
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|fetch_data|dpct::detail
// DPCT_DEPENDENCY_BEGIN
// Image|image_trait
// DPCT_DEPENDENCY_END
// DPCT_CODE
/// Functor to fetch data from read result of an image accessor.
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
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|create_image_wrapper_detail_forward_decl|dpct::detail
// DPCT_DEPENDENCY_BEGIN
// Image|create_image_wrapper_detail
// Image|image_wrapper_base_forward_decl
// Image|image_channel_forward_decl
// DPCT_DEPENDENCY_END
// DPCT_CODE
/// Create image according with given type \p T and \p dims.
template <class T> static image_wrapper_base *create_image_wrapper(int dims);

/// Create image with given data type \p T, channel order and dims
template <class T>
static image_wrapper_base *create_image_wrapper(unsigned channel_num, int dims);

/// Create image with channel info and specified dimensions.
static image_wrapper_base *create_image_wrapper(image_channel channel, int dims);
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|image_creator_forward_decl|dpct::detail
// DPCT_DEPENDENCY_BEGIN
// Image|image_creator
// DPCT_DEPENDENCY_END
// DPCT_CODE
/// Functor to create sycl::image from image_data.
template <int dimensions> struct image_creator;
// DPCT_LABEL_END

} // namespace detail

// DPCT_LABEL_BEGIN|image_channel|dpct
// DPCT_DEPENDENCY_BEGIN
// Image|image_channel_1
// Image|image_channel_2
// Image|image_channel_3
// Image|image_channel_data_type
// DPCT_DEPENDENCY_END
// DPCT_CODE
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
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|image_channel_create|dpct
// DPCT_PARENT_FEATURE|image_channel
// DPCT_DEPENDENCY_BEGIN
// Image|image_channel
// Image|image_trait
// Image|image_channel_set_channel_size
// Image|image_channel_set_channel_data_type
// DPCT_DEPENDENCY_END
// DPCT_CODE
  /// Create image channel info according to template argument \p T.
  template <class T> static image_channel create() {
    image_channel channel;
    channel.set_channel_size(detail::image_trait<T>::channel_num,
                             sizeof(typename detail::image_trait<T>::elem_t) *
                                 8);
    channel.set_channel_data_type(detail::image_trait<T>::data_type);
    return channel;
  }
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|image_channel_1|dpct
// DPCT_PARENT_FEATURE|image_channel
// DPCT_DEPENDENCY_BEGIN
// Image|image_channel
// DPCT_DEPENDENCY_END
// DPCT_CODE
  image_channel() = default;
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|image_channel_get_channel_data_type|dpct
// DPCT_PARENT_FEATURE|image_channel
// DPCT_DEPENDENCY_BEGIN
// Image|image_channel
// DPCT_DEPENDENCY_END
// DPCT_CODE
  image_channel_data_type get_channel_data_type() { return _type; }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|image_channel_set_channel_data_type|dpct
// DPCT_PARENT_FEATURE|image_channel
// DPCT_DEPENDENCY_BEGIN
// Image|image_channel
// DPCT_DEPENDENCY_END
// DPCT_CODE
  void set_channel_data_type(image_channel_data_type type) { _type = type; }
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|image_channel_get_total_size|dpct
// DPCT_PARENT_FEATURE|image_channel
// DPCT_DEPENDENCY_BEGIN
// Image|image_channel
// DPCT_DEPENDENCY_END
// DPCT_CODE
  unsigned get_total_size() { return _total_size; }
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|image_channel_get_channel_num|dpct
// DPCT_PARENT_FEATURE|image_channel
// DPCT_DEPENDENCY_BEGIN
// Image|image_channel
// DPCT_DEPENDENCY_END
// DPCT_CODE
  unsigned get_channel_num() { return _channel_num; }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|image_channel_set_channel_num|dpct
// DPCT_PARENT_FEATURE|image_channel
// DPCT_DEPENDENCY_BEGIN
// Image|image_channel
// DPCT_DEPENDENCY_END
// DPCT_CODE
  void set_channel_num(unsigned channel_num) {
    _channel_num = channel_num;
    _total_size = _channel_size * _channel_num;
  }
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|image_channel_2|dpct
// DPCT_PARENT_FEATURE|image_channel
// DPCT_DEPENDENCY_BEGIN
// Image|image_channel
// Image|image_channel_set_channel_size
// DPCT_DEPENDENCY_END
// DPCT_CODE
  /// image_channel constructor.
  /// \param r Channel r width in bits.
  /// \param g Channel g width in bits. Should be same with \p r, or zero.
  /// \param b Channel b width in bits. Should be same with \p g, or zero.
  /// \param a Channel a width in bits. Should be same with \p b, or zero.
  /// \param data_type Image channel data type: signed_nt, unsigned_int or fp.
  image_channel(int r, int g, int b, int a, image_channel_data_type data_type) {
    _type = data_type;
    if (a) {
      assert(r == a && "DPC++ doesn't support different channel size");
      assert(r == b && "DPC++ doesn't support different channel size");
      assert(r == g && "DPC++ doesn't support different channel size");
      set_channel_size(4, a);
    } else if (b) {
      assert(r == b && "DPC++ doesn't support different channel size");
      assert(r == g && "DPC++ doesn't support different channel size");
      set_channel_size(3, b);
    } else if (g) {
      assert(r == g && "DPC++ doesn't support different channel size");
      set_channel_size(2, g);
    } else {
      set_channel_size(1, r);
    }
  }
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|image_channel_get_channel_type|dpct
// DPCT_PARENT_FEATURE|image_channel
// DPCT_DEPENDENCY_BEGIN
// Image|image_channel
// DPCT_DEPENDENCY_END
// DPCT_CODE
  cl::sycl::image_channel_type get_channel_type() const {
    if (_channel_size == 4) {
      if (_type == image_channel_data_type::signed_int)
        return cl::sycl::image_channel_type::signed_int32;
      else if (_type == image_channel_data_type::unsigned_int)
        return cl::sycl::image_channel_type::unsigned_int32;
      else if (_type == image_channel_data_type::fp)
        return cl::sycl::image_channel_type::fp32;
    } else if (_channel_size == 2) {
      if (_type == image_channel_data_type::signed_int)
        return cl::sycl::image_channel_type::signed_int16;
      else if (_type == image_channel_data_type::unsigned_int)
        return cl::sycl::image_channel_type::unsigned_int16;
      else if (_type == image_channel_data_type::fp)
        return cl::sycl::image_channel_type::fp16;
    } else {
      if (_type == image_channel_data_type::signed_int)
        return cl::sycl::image_channel_type::signed_int8;
      else if (_type == image_channel_data_type::unsigned_int)
        return cl::sycl::image_channel_type::unsigned_int8;
    }
    assert(false && "unexpected channel data kind and channel size");
    return cl::sycl::image_channel_type::signed_int32;
  }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|image_channel_set_channel_type|dpct
// DPCT_PARENT_FEATURE|image_channel
// DPCT_DEPENDENCY_BEGIN
// Image|image_channel
// DPCT_DEPENDENCY_END
// DPCT_CODE
  void set_channel_type(cl::sycl::image_channel_type type) {
    switch (type) {
    case cl::sycl::image_channel_type::unsigned_int8:
      _type = image_channel_data_type::unsigned_int;
      _channel_size = 1;
      break;
    case cl::sycl::image_channel_type::unsigned_int16:
      _type = image_channel_data_type::unsigned_int;
      _channel_size = 2;
      break;
    case cl::sycl::image_channel_type::unsigned_int32:
      _type = image_channel_data_type::unsigned_int;
      _channel_size = 4;
      break;
    case cl::sycl::image_channel_type::signed_int8:
      _type = image_channel_data_type::signed_int;
      _channel_size = 1;
      break;
    case cl::sycl::image_channel_type::signed_int16:
      _type = image_channel_data_type::signed_int;
      _channel_size = 2;
      break;
    case cl::sycl::image_channel_type::signed_int32:
      _type = image_channel_data_type::signed_int;
      _channel_size = 4;
      break;
    case cl::sycl::image_channel_type::fp16:
      _type = image_channel_data_type::fp;
      _channel_size = 2;
      break;
    case cl::sycl::image_channel_type::fp32:
      _type = image_channel_data_type::fp;
      _channel_size = 4;
      break;
    default:
      break;
    }
    _total_size = _channel_size * _channel_num;
  }
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|image_channel_get_channel_order|dpct
// DPCT_PARENT_FEATURE|image_channel
// DPCT_DEPENDENCY_BEGIN
// Image|image_channel
// DPCT_DEPENDENCY_END
// DPCT_CODE
  cl::sycl::image_channel_order get_channel_order() const {
    switch (_channel_num) {
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
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|image_channel_get_channel_size|dpct
// DPCT_PARENT_FEATURE|image_channel
// DPCT_DEPENDENCY_BEGIN
// Image|image_channel
// DPCT_DEPENDENCY_END
// DPCT_CODE
  /// Get the size for each channel in bits.
  unsigned get_channel_size() const { return _channel_size * 8; }
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|image_channel_set_channel_size|dpct
// DPCT_PARENT_FEATURE|image_channel
// DPCT_DEPENDENCY_BEGIN
// Image|image_channel
// DPCT_DEPENDENCY_END
// DPCT_CODE
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
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|image_channel_3|dpct
// DPCT_PARENT_FEATURE|image_channel
// DPCT_DEPENDENCY_BEGIN
// Image|image_channel
// DPCT_DEPENDENCY_END
// DPCT_CODE
};
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|image_matrix|dpct
// DPCT_DEPENDENCY_BEGIN
// Image|image_channel
// Image|image_matrix_1
// Image|image_matrix_2
// DPCT_DEPENDENCY_END
// DPCT_CODE
/// 2D or 3D matrix data for image.
class image_matrix {
  image_channel _channel;
  int _range[3] = {1, 1, 1};
  int _dims = 0;
  void *_host_data = nullptr;
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|image_matrix_set_range|dpct
// DPCT_PARENT_FEATURE|image_matrix
// DPCT_DEPENDENCY_BEGIN
// Image|image_matrix
// DPCT_DEPENDENCY_END
// DPCT_CODE
  /// Set range of each dimension.
  template <int dimensions> void set_range(cl::sycl::range<dimensions> range) {
    for (int i = 0; i < dimensions; ++i)
      _range[i] = range[i];
    _dims = dimensions;
  }
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|image_matrix_get_range_T|dpct
// DPCT_PARENT_FEATURE|image_matrix
// DPCT_DEPENDENCY_BEGIN
// Util|make_index_sequence
// Image|image_matrix
// DPCT_DEPENDENCY_END
// DPCT_CODE
  template <int... DimIdx>
  cl::sycl::range<sizeof...(DimIdx)> get_range(integer_sequence<DimIdx...>) {
    return cl::sycl::range<sizeof...(DimIdx)>(_range[DimIdx]...);
  }
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|image_matrix_1|dpct
// DPCT_PARENT_FEATURE|image_matrix
// DPCT_DEPENDENCY_BEGIN
// Image|image_matrix
// Image|image_matrix_set_range
// Image|image_channel_get_total_size
// Image|image_channel_set_channel_type
// Image|image_channel_set_channel_num
// DPCT_DEPENDENCY_END
// DPCT_CODE
public:
  /// Constructor with channel info and dimension size info.
  template <int dimensions>
  image_matrix(image_channel channel, cl::sycl::range<dimensions> range)
      : _channel(channel) {
    set_range(range);
    _host_data = std::malloc(range.size() * _channel.get_total_size());
  }
  image_matrix(cl::sycl::image_channel_type channel_type, unsigned channel_num,
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
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|image_matrix_create_image|dpct
// DPCT_PARENT_FEATURE|image_matrix
// DPCT_DEPENDENCY_BEGIN
// Image|image_matrix
// Image|image_matrix_create_image_with_channel
// DPCT_DEPENDENCY_END
// DPCT_CODE
  /// Construct a new image class with the matrix data.
  template <int dimensions> cl::sycl::image<dimensions> *create_image() {
    return create_image<dimensions>(_channel);
  }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|image_matrix_create_image_with_channel|dpct
// DPCT_PARENT_FEATURE|image_matrix
// DPCT_DEPENDENCY_BEGIN
// Image|image_matrix
// Image|image_channel_get_channel_type
// Image|image_channel_get_channel_order
// Util|make_index_sequence
// Image|image_matrix_get_range_T
// DPCT_DEPENDENCY_END
// DPCT_CODE
  /// Construct a new image class with the matrix data.
  template <int dimensions>
  cl::sycl::image<dimensions> *create_image(image_channel channel) {
    return new cl::sycl::image<dimensions>(
        _host_data, channel.get_channel_order(), channel.get_channel_type(),
        get_range(make_index_sequence<dimensions>()),
        cl::sycl::property::image::use_host_ptr());
  }
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|image_matrix_get_channel|dpct
// DPCT_PARENT_FEATURE|image_matrix
// DPCT_DEPENDENCY_BEGIN
// Image|image_matrix
// Image|image_channel
// DPCT_DEPENDENCY_END
// DPCT_CODE
  /// Get channel info.
  inline image_channel get_channel() { return _channel; }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|image_matrix_get_range|dpct
// DPCT_PARENT_FEATURE|image_matrix
// DPCT_DEPENDENCY_BEGIN
// Image|image_matrix
// DPCT_DEPENDENCY_END
// DPCT_CODE
  /// Get range of the image.
  cl::sycl::range<3> get_range() {
    return cl::sycl::range<3>(_range[0], _range[1], _range[2]);
  }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|image_matrix_get_dims|dpct
// DPCT_PARENT_FEATURE|image_matrix
// DPCT_DEPENDENCY_BEGIN
// Image|image_matrix
// DPCT_DEPENDENCY_END
// DPCT_CODE
  /// Get matrix dims.
  inline int get_dims() { return _dims; }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|image_matrix_to_pitched_data|dpct
// DPCT_PARENT_FEATURE|image_matrix
// DPCT_DEPENDENCY_BEGIN
// Image|image_matrix
// Memory|pitched_data
// DPCT_DEPENDENCY_END
// DPCT_CODE
  /// Convert to pitched data.
  pitched_data to_pitched_data() {
    return pitched_data(_host_data, _range[0], _range[0], _range[1]);
  }
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|image_matrix_2|dpct
// DPCT_PARENT_FEATURE|image_matrix
// DPCT_DEPENDENCY_BEGIN
// Image|image_matrix
// DPCT_DEPENDENCY_END
// DPCT_CODE
  ~image_matrix() {
    if (_host_data)
      std::free(_host_data);
    _host_data = nullptr;
  }
};
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|image_matrix_p_alias|dpct
// DPCT_DEPENDENCY_BEGIN
// Image|image_matrix
// DPCT_DEPENDENCY_END
// DPCT_CODE
using image_matrix_p = image_matrix *;
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|image_data_type|dpct
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
enum class image_data_type { matrix, linear, pitch, unsupport };
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|image_data|dpct
// DPCT_DEPENDENCY_BEGIN
// Image|image_data_type
// Image|image_matrix_p_alias
// Image|image_channel
// Image|image_data_1
// Image|image_matrix_get_channel
// Image|image_data_set_data
// DPCT_DEPENDENCY_END
// DPCT_CODE
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
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|image_data_set_data|dpct
// DPCT_PARENT_FEATURE|image_data
// DPCT_DEPENDENCY_BEGIN
// Image|image_data
// DPCT_DEPENDENCY_END
// DPCT_CODE
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
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|image_data_get_data_type|dpct
// DPCT_PARENT_FEATURE|image_data
// DPCT_DEPENDENCY_BEGIN
// Image|image_data
// DPCT_DEPENDENCY_END
// DPCT_CODE
  image_data_type get_data_type() { return _type; }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|image_data_set_data_type|dpct
// DPCT_PARENT_FEATURE|image_data
// DPCT_DEPENDENCY_BEGIN
// Image|image_data
// DPCT_DEPENDENCY_END
// DPCT_CODE
  void set_data_type(image_data_type type) { _type = type; }
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|image_data_get_data_ptr|dpct
// DPCT_PARENT_FEATURE|image_data
// DPCT_DEPENDENCY_BEGIN
// Image|image_data
// DPCT_DEPENDENCY_END
// DPCT_CODE
  void *get_data_ptr() { return _data; }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|image_data_set_data_ptr|dpct
// DPCT_PARENT_FEATURE|image_data
// DPCT_DEPENDENCY_BEGIN
// Image|image_data
// DPCT_DEPENDENCY_END
// DPCT_CODE
  void set_data_ptr(void *data) { _data = data; }
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|image_data_get_x|dpct
// DPCT_PARENT_FEATURE|image_data
// DPCT_DEPENDENCY_BEGIN
// Image|image_data
// DPCT_DEPENDENCY_END
// DPCT_CODE
  size_t get_x() { return _x; }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|image_data_set_x|dpct
// DPCT_PARENT_FEATURE|image_data
// DPCT_DEPENDENCY_BEGIN
// Image|image_data
// DPCT_DEPENDENCY_END
// DPCT_CODE
  void set_x(size_t x) { _x = x; }
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|image_data_get_y|dpct
// DPCT_PARENT_FEATURE|image_data
// DPCT_DEPENDENCY_BEGIN
// Image|image_data
// DPCT_DEPENDENCY_END
// DPCT_CODE
  size_t get_y() { return _y; }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|image_data_set_y|dpct
// DPCT_PARENT_FEATURE|image_data
// DPCT_DEPENDENCY_BEGIN
// Image|image_data
// DPCT_DEPENDENCY_END
// DPCT_CODE
  void set_y(size_t y) { _y = y; }
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|image_data_get_pitch|dpct
// DPCT_PARENT_FEATURE|image_data
// DPCT_DEPENDENCY_BEGIN
// Image|image_data
// DPCT_DEPENDENCY_END
// DPCT_CODE
  size_t get_pitch() { return _pitch; }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|image_data_set_pitch|dpct
// DPCT_PARENT_FEATURE|image_data
// DPCT_DEPENDENCY_BEGIN
// Image|image_data
// DPCT_DEPENDENCY_END
// DPCT_CODE
  void set_pitch(size_t pitch) { _pitch = pitch; }
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|image_data_get_channel|dpct
// DPCT_PARENT_FEATURE|image_data
// DPCT_DEPENDENCY_BEGIN
// Image|image_data
// Image|image_channel
// DPCT_DEPENDENCY_END
// DPCT_CODE
  image_channel get_channel() { return _channel; }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|image_data_set_channel|dpct
// DPCT_PARENT_FEATURE|image_data
// DPCT_DEPENDENCY_BEGIN
// Image|image_data
// Image|image_channel
// DPCT_DEPENDENCY_END
// DPCT_CODE
  void set_channel(image_channel channel) { _channel = channel; }
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|image_data_get_channel_data_type|dpct
// DPCT_PARENT_FEATURE|image_data
// DPCT_DEPENDENCY_BEGIN
// Image|image_data
// Image|image_channel_get_channel_data_type
// DPCT_DEPENDENCY_END
// DPCT_CODE
  image_channel_data_type get_channel_data_type() {
    return _channel.get_channel_data_type();
  }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|image_data_set_channel_data_type|dpct
// DPCT_PARENT_FEATURE|image_data
// DPCT_DEPENDENCY_BEGIN
// Image|image_data
// Image|image_channel_set_channel_data_type
// DPCT_DEPENDENCY_END
// DPCT_CODE
  void set_channel_data_type(image_channel_data_type type) {
    _channel.set_channel_data_type(type);
  }
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|image_data_get_channel_size|dpct
// DPCT_PARENT_FEATURE|image_data
// DPCT_DEPENDENCY_BEGIN
// Image|image_data
// Image|image_channel_get_channel_size
// DPCT_DEPENDENCY_END
// DPCT_CODE
  unsigned get_channel_size() { return _channel.get_channel_size(); }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|image_data_set_channel_size|dpct
// DPCT_PARENT_FEATURE|image_data
// DPCT_DEPENDENCY_BEGIN
// Image|image_data
// Image|image_channel_set_channel_size
// DPCT_DEPENDENCY_END
// DPCT_CODE
  void set_channel_size(unsigned channel_num, unsigned channel_size) {
    return _channel.set_channel_size(channel_num, channel_size);
  }
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|image_data_get_channel_num|dpct
// DPCT_PARENT_FEATURE|image_data
// DPCT_DEPENDENCY_BEGIN
// Image|image_data
// Image|image_channel_get_channel_num
// DPCT_DEPENDENCY_END
// DPCT_CODE
  unsigned get_channel_num() { return _channel.get_channel_num(); }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|image_data_set_channel_num|dpct
// DPCT_PARENT_FEATURE|image_data
// DPCT_DEPENDENCY_BEGIN
// Image|image_data
// Image|image_channel_set_channel_num
// DPCT_DEPENDENCY_END
// DPCT_CODE
  void set_channel_num(unsigned num) {
    return _channel.set_channel_num(num);
  }
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|image_data_get_channel_type|dpct
// DPCT_PARENT_FEATURE|image_data
// DPCT_DEPENDENCY_BEGIN
// Image|image_data
// Image|image_channel_get_channel_type
// DPCT_DEPENDENCY_END
// DPCT_CODE
  cl::sycl::image_channel_type get_channel_type() {
    return _channel.get_channel_type();
  }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|image_data_set_channel_type|dpct
// DPCT_PARENT_FEATURE|image_data
// DPCT_DEPENDENCY_BEGIN
// Image|image_data
// Image|image_channel_set_channel_type
// DPCT_DEPENDENCY_END
// DPCT_CODE
  void set_channel_type(cl::sycl::image_channel_type type) {
    return _channel.set_channel_type(type);
  }
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|image_data_1|dpct
// DPCT_PARENT_FEATURE|image_data
// DPCT_DEPENDENCY_BEGIN
// Image|image_data
// DPCT_DEPENDENCY_END
// DPCT_CODE
private:
  image_data_type _type;
  void *_data = nullptr;
  size_t _x, _y, _pitch;
  image_channel _channel;
};
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|sampling_info|dpct
// DPCT_DEPENDENCY_BEGIN
// Image|sampling_info_1
// Image|sampling_info_2
// DPCT_DEPENDENCY_END
// DPCT_CODE
/// Image sampling info, include addressing mode, filtering mode and
/// normalization info.
class sampling_info {
  cl::sycl::addressing_mode _addressing_mode =
      cl::sycl::addressing_mode::clamp_to_edge;
  cl::sycl::filtering_mode _filtering_mode = cl::sycl::filtering_mode::nearest;
  cl::sycl::coordinate_normalization_mode _coordinate_normalization_mode =
      cl::sycl::coordinate_normalization_mode::unnormalized;
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|sampling_info_1|dpct
// DPCT_PARENT_FEATURE|sampling_info
// DPCT_DEPENDENCY_BEGIN
// Image|sampling_info
// DPCT_DEPENDENCY_END
// DPCT_CODE
public:
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|sampling_info_get_addressing_mode|dpct
// DPCT_PARENT_FEATURE|sampling_info
// DPCT_DEPENDENCY_BEGIN
// Image|sampling_info_1
// DPCT_DEPENDENCY_END
// DPCT_CODE
  cl::sycl::addressing_mode get_addressing_mode() { return _addressing_mode; }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|sampling_info_set_addressing_mode|dpct
// DPCT_PARENT_FEATURE|sampling_info
// DPCT_DEPENDENCY_BEGIN
// Image|sampling_info_1
// DPCT_DEPENDENCY_END
// DPCT_CODE
  void set(cl::sycl::addressing_mode addressing_mode) { _addressing_mode = addressing_mode; }
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|sampling_info_get_filtering_mode|dpct
// DPCT_PARENT_FEATURE|sampling_info
// DPCT_DEPENDENCY_BEGIN
// Image|sampling_info_1
// DPCT_DEPENDENCY_END
// DPCT_CODE
  cl::sycl::filtering_mode get_filtering_mode() { return _filtering_mode; }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|sampling_info_set_filtering_mode|dpct
// DPCT_PARENT_FEATURE|sampling_info
// DPCT_DEPENDENCY_BEGIN
// Image|sampling_info_1
// DPCT_DEPENDENCY_END
// DPCT_CODE
  void set(cl::sycl::filtering_mode filtering_mode) { _filtering_mode = filtering_mode; }
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|sampling_info_get_coordinate_normalization_mode|dpct
// DPCT_PARENT_FEATURE|sampling_info
// DPCT_DEPENDENCY_BEGIN
// Image|sampling_info_1
// DPCT_DEPENDENCY_END
// DPCT_CODE
  cl::sycl::coordinate_normalization_mode get_coordinate_normalization_mode() {
    return _coordinate_normalization_mode;
  }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|sampling_info_set_coordinate_normalization_mode_enum|dpct
// DPCT_PARENT_FEATURE|sampling_info
// DPCT_DEPENDENCY_BEGIN
// Image|sampling_info_1
// DPCT_DEPENDENCY_END
// DPCT_CODE
  void set(cl::sycl::coordinate_normalization_mode coordinate_normalization_mode) {
    _coordinate_normalization_mode = coordinate_normalization_mode;
  }
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|sampling_info_is_coordinate_normalized|dpct
// DPCT_PARENT_FEATURE|sampling_info
// DPCT_DEPENDENCY_BEGIN
// Image|sampling_info_1
// DPCT_DEPENDENCY_END
// DPCT_CODE
  bool is_coordinate_normalized() {
    return _coordinate_normalization_mode ==
           cl::sycl::coordinate_normalization_mode::normalized;
  }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|sampling_info_set_coordinate_normalization_mode|dpct
// DPCT_PARENT_FEATURE|sampling_info
// DPCT_DEPENDENCY_BEGIN
// Image|sampling_info_1
// DPCT_DEPENDENCY_END
// DPCT_CODE
  void set_coordinate_normalization_mode(int is_normalized) {
    _coordinate_normalization_mode =
        is_normalized ? cl::sycl::coordinate_normalization_mode::normalized
                      : cl::sycl::coordinate_normalization_mode::unnormalized;
  }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|sampling_info_set_addressing_mode_filtering_mode_coordinate_normalization_mode|dpct
// DPCT_PARENT_FEATURE|sampling_info
// DPCT_DEPENDENCY_BEGIN
// Image|sampling_info_1
// DPCT_DEPENDENCY_END
// DPCT_CODE
  void
  set(cl::sycl::addressing_mode addressing_mode,
      cl::sycl::filtering_mode filtering_mode,
      cl::sycl::coordinate_normalization_mode coordinate_normalization_mode) {
    set(addressing_mode);
    set(filtering_mode);
    set(coordinate_normalization_mode);
  }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|sampling_info_set_addressing_mode_filtering_mode_is_normalized|dpct
// DPCT_PARENT_FEATURE|sampling_info
// DPCT_DEPENDENCY_BEGIN
// Image|sampling_info_1
// DPCT_DEPENDENCY_END
// DPCT_CODE
  void set(cl::sycl::addressing_mode addressing_mode,
           cl::sycl::filtering_mode filtering_mode, int is_normalized) {
    set(addressing_mode);
    set(filtering_mode);
    set_coordinate_normalization_mode(is_normalized);
  }
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|sampling_info_get_sampler|dpct
// DPCT_PARENT_FEATURE|sampling_info
// DPCT_DEPENDENCY_BEGIN
// Image|sampling_info_1
// DPCT_DEPENDENCY_END
// DPCT_CODE
  cl::sycl::sampler get_sampler() {
    return cl::sycl::sampler(_coordinate_normalization_mode, _addressing_mode,
                             _filtering_mode);
  }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|sampling_info_2|dpct
// DPCT_PARENT_FEATURE|sampling_info
// DPCT_DEPENDENCY_BEGIN
// Image|sampling_info
// DPCT_DEPENDENCY_END
// DPCT_CODE
};
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|image_wrapper_base|dpct
// DPCT_DEPENDENCY_BEGIN
// Image|sampling_info
// Image|image_data
// Image|image_wrapper_base_1
// DPCT_DEPENDENCY_END
// DPCT_CODE
/// Image base class.
class image_wrapper_base {
  sampling_info _sampling_info;
  image_data _data;

public:
  virtual ~image_wrapper_base() = 0;
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|image_wrapper_base_attach|dpct
// DPCT_PARENT_FEATURE|image_wrapper_base
// DPCT_DEPENDENCY_BEGIN
// Image|image_wrapper_base
// Image|image_wrapper_base_set_data
// DPCT_DEPENDENCY_END
// DPCT_CODE
  void attach(image_data data) { set_data(data); }
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|image_wrapper_base_get_sampling_info|dpct
// DPCT_PARENT_FEATURE|image_wrapper_base
// DPCT_DEPENDENCY_BEGIN
// Image|image_wrapper_base
// DPCT_DEPENDENCY_END
// DPCT_CODE
  sampling_info get_sampling_info() { return _sampling_info; }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|image_wrapper_base_set_sampling_info|dpct
// DPCT_PARENT_FEATURE|image_wrapper_base
// DPCT_DEPENDENCY_BEGIN
// Image|image_wrapper_base
// DPCT_DEPENDENCY_END
// DPCT_CODE
  void set_sampling_info(sampling_info info) {
    _sampling_info = info;
  }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|image_wrapper_base_get_data|dpct
// DPCT_PARENT_FEATURE|image_wrapper_base
// DPCT_DEPENDENCY_BEGIN
// Image|image_wrapper_base
// DPCT_DEPENDENCY_END
// DPCT_CODE
  const image_data &get_data() { return _data; }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|image_wrapper_base_set_data|dpct
// DPCT_PARENT_FEATURE|image_wrapper_base
// DPCT_DEPENDENCY_BEGIN
// Image|image_wrapper_base
// DPCT_DEPENDENCY_END
// DPCT_CODE
  void set_data(image_data data) { _data = data; }
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|image_wrapper_base_get_channel|dpct
// DPCT_PARENT_FEATURE|image_wrapper_base
// DPCT_DEPENDENCY_BEGIN
// Image|image_wrapper_base
// Image|image_data_get_channel
// DPCT_DEPENDENCY_END
// DPCT_CODE
  image_channel get_channel() { return _data.get_channel(); }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|image_wrapper_base_set_channel|dpct
// DPCT_PARENT_FEATURE|image_wrapper_base
// DPCT_DEPENDENCY_BEGIN
// Image|image_wrapper_base
// Image|image_data_set_channel
// DPCT_DEPENDENCY_END
// DPCT_CODE
  void set_channel(image_channel channel) { _data.set_channel(channel); }
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|image_wrapper_base_get_channel_data_type|dpct
// DPCT_PARENT_FEATURE|image_wrapper_base
// DPCT_DEPENDENCY_BEGIN
// Image|image_wrapper_base
// Image|image_data_get_channel_data_type
// DPCT_DEPENDENCY_END
// DPCT_CODE
  image_channel_data_type get_channel_data_type() {
    return _data.get_channel_data_type();
  }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|image_wrapper_base_set_channel_data_type|dpct
// DPCT_PARENT_FEATURE|image_wrapper_base
// DPCT_DEPENDENCY_BEGIN
// Image|image_wrapper_base
// Image|image_data_set_channel_data_type
// DPCT_DEPENDENCY_END
// DPCT_CODE
  void set_channel_data_type(image_channel_data_type type) {
    _data.set_channel_data_type(type);
  }
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|image_wrapper_base_get_channel_size|dpct
// DPCT_PARENT_FEATURE|image_wrapper_base
// DPCT_DEPENDENCY_BEGIN
// Image|image_wrapper_base
// Image|image_data_get_channel_size
// DPCT_DEPENDENCY_END
// DPCT_CODE
  unsigned get_channel_size() { return _data.get_channel_size(); }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|image_wrapper_base_set_channel_size|dpct
// DPCT_PARENT_FEATURE|image_wrapper_base
// DPCT_DEPENDENCY_BEGIN
// Image|image_wrapper_base
// Image|image_data_set_channel_size
// DPCT_DEPENDENCY_END
// DPCT_CODE
  void set_channel_size(unsigned channel_num, unsigned channel_size) {
    return _data.set_channel_size(channel_num, channel_size);
  }
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|image_wrapper_base_get_addressing_mode|dpct
// DPCT_PARENT_FEATURE|image_wrapper_base
// DPCT_DEPENDENCY_BEGIN
// Image|image_wrapper_base
// Image|sampling_info_get_addressing_mode
// DPCT_DEPENDENCY_END
// DPCT_CODE
  cl::sycl::addressing_mode get_addressing_mode() {
    return _sampling_info.get_addressing_mode();
  }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|image_wrapper_base_set_addressing_mode|dpct
// DPCT_PARENT_FEATURE|image_wrapper_base
// DPCT_DEPENDENCY_BEGIN
// Image|image_wrapper_base
// Image|sampling_info_set_addressing_mode
// DPCT_DEPENDENCY_END
// DPCT_CODE
  void set(cl::sycl::addressing_mode addressing_mode) {
    _sampling_info.set(addressing_mode);
  }
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|image_wrapper_base_get_filtering_mode|dpct
// DPCT_PARENT_FEATURE|image_wrapper_base
// DPCT_DEPENDENCY_BEGIN
// Image|image_wrapper_base
// Image|sampling_info_get_filtering_mode
// DPCT_DEPENDENCY_END
// DPCT_CODE
  cl::sycl::filtering_mode get_filtering_mode() {
    return _sampling_info.get_filtering_mode();
  }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|image_wrapper_base_set_filtering_mode|dpct
// DPCT_PARENT_FEATURE|image_wrapper_base
// DPCT_DEPENDENCY_BEGIN
// Image|image_wrapper_base
// Image|sampling_info_set_filtering_mode
// DPCT_DEPENDENCY_END
// DPCT_CODE
  void set(cl::sycl::filtering_mode filtering_mode) {
    _sampling_info.set(filtering_mode);
  }
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|image_wrapper_base_get_coordinate_normalization_mode|dpct
// DPCT_PARENT_FEATURE|image_wrapper_base
// DPCT_DEPENDENCY_BEGIN
// Image|image_wrapper_base
// Image|sampling_info_get_coordinate_normalization_mode
// DPCT_DEPENDENCY_END
// DPCT_CODE
  cl::sycl::coordinate_normalization_mode get_coordinate_normalization_mode() {
    return _sampling_info.get_coordinate_normalization_mode();
  }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|image_wrapper_base_set_coordinate_normalization_mode_enum|dpct
// DPCT_PARENT_FEATURE|image_wrapper_base
// DPCT_DEPENDENCY_BEGIN
// Image|image_wrapper_base
// Image|sampling_info_set_coordinate_normalization_mode_enum
// DPCT_DEPENDENCY_END
// DPCT_CODE
  void
  set(cl::sycl::coordinate_normalization_mode coordinate_normalization_mode) {
    _sampling_info.set(coordinate_normalization_mode);
  }
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|image_wrapper_base_is_coordinate_normalized|dpct
// DPCT_PARENT_FEATURE|image_wrapper_base
// DPCT_DEPENDENCY_BEGIN
// Image|image_wrapper_base
// Image|sampling_info_is_coordinate_normalized
// DPCT_DEPENDENCY_END
// DPCT_CODE
  bool is_coordinate_normalized() {
    return _sampling_info.is_coordinate_normalized();
  }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|image_wrapper_base_set_coordinate_normalization_mode|dpct
// DPCT_PARENT_FEATURE|image_wrapper_base
// DPCT_DEPENDENCY_BEGIN
// Image|image_wrapper_base
// Image|sampling_info_set_coordinate_normalization_mode
// DPCT_DEPENDENCY_END
// DPCT_CODE
  void set_coordinate_normalization_mode(int is_normalized) {
    _sampling_info.set_coordinate_normalization_mode(is_normalized);
  }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|image_wrapper_base_set_addressing_mode_filtering_mode_coordinate_normalization_mode|dpct
// DPCT_PARENT_FEATURE|image_wrapper_base
// DPCT_DEPENDENCY_BEGIN
// Image|image_wrapper_base
// Image|image_wrapper_base_set_addressing_mode
// Image|image_wrapper_base_set_filtering_mode
// Image|image_wrapper_base_set_coordinate_normalization_mode_enum
// DPCT_DEPENDENCY_END
// DPCT_CODE
  void
  set(cl::sycl::addressing_mode addressing_mode,
      cl::sycl::filtering_mode filtering_mode,
      cl::sycl::coordinate_normalization_mode coordinate_normalization_mode) {
    set(addressing_mode);
    set(filtering_mode);
    set(coordinate_normalization_mode);
  }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|image_wrapper_base_set_addressing_mode_filtering_mode_is_normalized|dpct
// DPCT_PARENT_FEATURE|image_wrapper_base
// DPCT_DEPENDENCY_BEGIN
// Image|image_wrapper_base
// Image|image_wrapper_base_set_addressing_mode
// Image|image_wrapper_base_set_filtering_mode
// Image|image_wrapper_base_set_coordinate_normalization_mode
// DPCT_DEPENDENCY_END
// DPCT_CODE
  void set(cl::sycl::addressing_mode addressing_mode,
           cl::sycl::filtering_mode filtering_mode, int is_normalized) {
    set(addressing_mode);
    set(filtering_mode);
    set_coordinate_normalization_mode(is_normalized);
  }
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|image_wrapper_base_get_channel_num|dpct
// DPCT_PARENT_FEATURE|image_wrapper_base
// DPCT_DEPENDENCY_BEGIN
// Image|image_wrapper_base
// Image|image_data_get_channel_num
// DPCT_DEPENDENCY_END
// DPCT_CODE
  unsigned get_channel_num() { return _data.get_channel_num(); }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|image_wrapper_base_set_channel_num|dpct
// DPCT_PARENT_FEATURE|image_wrapper_base
// DPCT_DEPENDENCY_BEGIN
// Image|image_wrapper_base
// Image|image_data_set_channel_num
// DPCT_DEPENDENCY_END
// DPCT_CODE
  void set_channel_num(unsigned num) {
    return _data.set_channel_num(num);
  }
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|image_wrapper_base_get_channel_type|dpct
// DPCT_PARENT_FEATURE|image_wrapper_base
// DPCT_DEPENDENCY_BEGIN
// Image|image_wrapper_base
// Image|image_data_get_channel_type
// DPCT_DEPENDENCY_END
// DPCT_CODE
  cl::sycl::image_channel_type get_channel_type() {
    return _data.get_channel_type();
  }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|image_wrapper_base_set_channel_type|dpct
// DPCT_PARENT_FEATURE|image_wrapper_base
// DPCT_DEPENDENCY_BEGIN
// Image|image_wrapper_base
// Image|image_data_set_channel_type
// DPCT_DEPENDENCY_END
// DPCT_CODE
  void set_channel_type(cl::sycl::image_channel_type type) {
    return _data.set_channel_type(type);
  }
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|image_wrapper_base_get_sampler|dpct
// DPCT_PARENT_FEATURE|image_wrapper_base
// DPCT_DEPENDENCY_BEGIN
// Image|image_wrapper_base
// Image|sampling_info_get_sampler
// DPCT_DEPENDENCY_END
// DPCT_CODE
  cl::sycl::sampler get_sampler() { return _sampling_info.get_sampler(); }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|image_wrapper_base_1|dpct
// DPCT_PARENT_FEATURE|image_wrapper_base
// DPCT_DEPENDENCY_BEGIN
// Image|image_wrapper_base
// DPCT_DEPENDENCY_END
// DPCT_CODE
};
inline image_wrapper_base::~image_wrapper_base() {}
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|image_wrapper_base_p_alias|dpct
// DPCT_DEPENDENCY_BEGIN
// Image|image_wrapper_base
// DPCT_DEPENDENCY_END
// DPCT_CODE
using image_wrapper_base_p = image_wrapper_base *;
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|image_accessor_ext_forward_decl|dpct
// DPCT_DEPENDENCY_BEGIN
// Image|image_accessor_ext
// DPCT_DEPENDENCY_END
// DPCT_CODE
template <class T, int dimensions, bool IsImageArray> class image_accessor_ext;
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|image_wrapper|dpct
// DPCT_DEPENDENCY_BEGIN
// Image|image_wrapper_base
// Image|image_accessor_ext_forward_decl
// Image|image_creator_forward_decl
// Image|image_trait
// Image|image_wrapper_1
// Image|image_channel_create
// Image|image_wrapper_base_set_channel
// DPCT_DEPENDENCY_END
// DPCT_CODE
/// Image class, wrapper of cl::sycl::image.
template <class T, int dimensions, bool IsImageArray = false> class image_wrapper : public image_wrapper_base {
  cl::sycl::image<dimensions> *_image = nullptr;

public:
  using acc_data_t = typename detail::image_trait<T>::acc_data_t;
  using accessor_t =
      typename image_accessor_ext<T, IsImageArray ? (dimensions - 1) : dimensions,
                              IsImageArray>::accessor_t;

  image_wrapper() { set_channel(image_channel::create<T>()); }
  ~image_wrapper() { detach(); }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|image_wrapper_get_access|dpct
// DPCT_PARENT_FEATURE|image_wrapper
// DPCT_DEPENDENCY_BEGIN
// Image|image_wrapper
// Image|image_wrapper_base_get_data
// DPCT_DEPENDENCY_END
// DPCT_CODE
  /// Get image accessor.
  accessor_t get_access(cl::sycl::handler &cgh) {
    if (!_image)
      _image = detail::image_creator<dimensions>()(get_data());
    return accessor_t(*_image, cgh);
  }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|image_wrapper_attach|dpct
// DPCT_PARENT_FEATURE|image_wrapper
// DPCT_DEPENDENCY_BEGIN
// Image|image_wrapper
// Image|image_matrix
// Image|image_data
// Image|image_wrapper_base_set_data
// Image|image_channel
// Image|image_wrapper_base_set_channel
// Image|image_wrapper_base_get_channel
// Image|image_wrapper_base_attach
// DPCT_DEPENDENCY_END
// DPCT_CODE
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
  void attach(void *ptr, size_t count) {
    attach(ptr, count, get_channel());
  }
  /// Attach linear data to this class.
  void attach(void *ptr, size_t count, image_channel channel) {
    detach();
    image_wrapper_base::set_data(image_data(ptr, count, channel));
  }
  /// Attach 2D data to this class.
  void attach(void *data, size_t x, size_t y, size_t pitch) {
    attach(data, x, y, pitch, get_channel());
  }
  /// Attach 2D data to this class.
  void attach(void *data, size_t x, size_t y, size_t pitch, image_channel channel) {
    detach();
    image_wrapper_base::set_data(image_data(data, x * sizeof(T), y, pitch, channel));
  }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|image_wrapper_1|dpct
// DPCT_PARENT_FEATURE|image_wrapper
// DPCT_DEPENDENCY_BEGIN
// Image|image_wrapper
// DPCT_DEPENDENCY_END
// DPCT_CODE
  /// Detach data.
  void detach() {
    if (_image)
      delete _image;
    _image = nullptr;
  }
};
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|image_accessor_ext|dpct
// DPCT_DEPENDENCY_BEGIN
// Image|image_wrapper_base
// Image|fetch_data
// DPCT_DEPENDENCY_END
// DPCT_CODE
/// Wrap sampler and image accessor together.
template <class T, int dimensions, bool IsImageArray = false>
class image_accessor_ext {
public:
  using accessor_t =
      typename detail::image_trait<T>::template accessor_t<dimensions>;
  using data_t = typename detail::image_trait<T>::data_t;
  cl::sycl::sampler _sampler;
  accessor_t _img_acc;

public:
  image_accessor_ext(cl::sycl::sampler sampler, accessor_t acc)
      : _sampler(sampler), _img_acc(acc) {}

  /// Read data from accessor.
  template <bool Available = dimensions == 3>
  typename std::enable_if<Available, data_t>::type read(float x, float y,
                                                        float z) {
    return detail::fetch_data<T>()(
        _img_acc.read(cl::sycl::float4(x, y, z, 0), _sampler));
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
        _img_acc.read(cl::sycl::int4(x, y, z, 0), _sampler));
  }
  /// Read data from accessor.
  template <bool Available = dimensions == 2>
  typename std::enable_if<Available, data_t>::type read(float x, float y) {
    return detail::fetch_data<T>()(
        _img_acc.read(cl::sycl::float2(x, y), _sampler));
  }
  /// Read data from accessor.
  template <class Coord0, class Coord1,
            bool Available = dimensions == 2 &&
                             std::is_integral<Coord0>::value
                                 &&std::is_integral<Coord1>::value>
  typename std::enable_if<Available, data_t>::type read(Coord0 x, Coord1 y) {
    return detail::fetch_data<T>()(
        _img_acc.read(cl::sycl::int2(x, y), _sampler));
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
  cl::sycl::sampler _sampler;
  accessor_t _img_acc;

public:
  image_accessor_ext(cl::sycl::sampler sampler, accessor_t acc)
      : _sampler(sampler), _img_acc(acc) {}

  /// Read data from accessor.
  template <bool Available = dimensions == 2>
  typename std::enable_if<Available, data_t>::type read(int index, float x,
                                                        float y) {
    return detail::fetch_data<T>()(
        _img_acc[index].read(cl::sycl::float2(x, y), _sampler));
  }
  /// Read data from accessor.
  template <bool Available = dimensions == 2>
  typename std::enable_if<Available, data_t>::type read(int index, int x, int y) {
    return detail::fetch_data<T>()(
        _img_acc[index].read(cl::sycl::int2(x, y), _sampler));
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
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|create_image_wrapper|dpct
// DPCT_DEPENDENCY_BEGIN
// Image|create_image_wrapper_detail_forward_decl
// Image|image_wrapper_base
// Image|image_data
// Image|sampling_info
// Image|image_channel
// Image|image_data_get_data_type
// Image|image_matrix_p_alias
// Image|image_data_get_data_ptr
// Image|image_matrix_get_channel
// Image|image_matrix_get_dims
// Image|image_data_get_channel
// Image|create_image_wrapper_detail
// Image|image_wrapper_base_set_sampling_info
// Image|image_wrapper_base_set_data
// DPCT_DEPENDENCY_END
// DPCT_CODE
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
    channel = data.get_channel();
  }

  if (auto ret = detail::create_image_wrapper(channel, dims)) {
    ret->set_sampling_info(info);
    ret->set_data(data);
    return ret;
  }
  return nullptr;
}
// DPCT_LABEL_END

namespace detail {
// DPCT_LABEL_BEGIN|image_creator|dpct::detail
// DPCT_DEPENDENCY_BEGIN
// Image|image_wrapper_base
// Memory|mem_mgr_is_device_ptr
// Memory|get_buffer_byte_t
// Image|image_data_get_data_type
// Image|image_data_get_data_ptr
// Image|image_data_get_channel
// Image|image_data_get_x
// Image|image_data_get_y
// Image|image_data_get_pitch
// Image|image_channel_get_channel_order
// Image|image_channel_get_channel_type
// Image|image_channel_get_total_size
// Image|image_matrix_create_image_with_channel
// DPCT_DEPENDENCY_END
// DPCT_CODE
/// Functor to create sycl::image from image_data
template <> struct image_creator<1> {
  cl::sycl::image<1> *operator()(image_data data) {
    assert(data.get_data_type() == image_data_type::linear ||
           data.get_data_type() == image_data_type::matrix);
    if (data.get_data_type() == image_data_type::matrix) {
      return static_cast<image_matrix_p>(data.get_data_ptr())
          ->create_image<1>(data.get_channel());
    }
    auto ptr = data.get_data_ptr();
    if (detail::mem_mgr::instance().is_device_ptr(ptr))
      ptr = get_buffer(ptr)
                .template get_access<cl::sycl::access_mode::read_write>()
                .get_pointer();
    auto channel = data.get_channel();
    return new cl::sycl::image<1>(
        ptr, channel.get_channel_order(), channel.get_channel_type(),
        cl::sycl::range<1>(data.get_x() / channel.get_total_size()));
  }
};
template <> struct image_creator<2> {
  cl::sycl::image<2> *operator()(image_data data) {
    assert(data.get_data_type() == image_data_type::pitch ||
           data.get_data_type() == image_data_type::matrix);
    if (data.get_data_type() == image_data_type::matrix) {
      return static_cast<image_matrix_p>(data.get_data_ptr())
          ->create_image<2>(data.get_channel());
    }
    auto ptr = data.get_data_ptr();
    if (detail::mem_mgr::instance().is_device_ptr(ptr))
      ptr = get_buffer(ptr)
                .template get_access<cl::sycl::access_mode::read_write>()
                .get_pointer();
    auto channel = data.get_channel();
    return new cl::sycl::image<2>(
        ptr, channel.get_channel_order(), channel.get_channel_type(),
        cl::sycl::range<2>(data.get_x() / channel.get_total_size(),
                           data.get_y()),
        cl::sycl::range<1>(data.get_pitch()));
  }
};
template <> struct image_creator<3> {
  sycl::image<3> *operator()(image_data data) {
    assert(data.get_data_type() == image_data_type::matrix);
    return static_cast<image_matrix_p>(data.get_data_ptr())
        ->create_image<3>(data.get_channel());
  }
};
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|create_image_wrapper_detail|dpct::detail
// DPCT_DEPENDENCY_BEGIN
// Image|image_wrapper_base
// Image|image_wrapper
// Image|image_data_get_channel_type
// Image|image_data_get_channel_num
// DPCT_DEPENDENCY_END
// DPCT_CODE
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
static image_wrapper_base *create_image_wrapper(image_channel channel, int dims) {
  switch (channel.get_channel_type()) {
  case cl::sycl::image_channel_type::fp16:
    return create_image_wrapper<cl::sycl::cl_half>(channel.get_channel_num(), dims);
  case cl::sycl::image_channel_type::fp32:
    return create_image_wrapper<cl::sycl::cl_float>(channel.get_channel_num(), dims);
  case cl::sycl::image_channel_type::signed_int8:
    return create_image_wrapper<cl::sycl::cl_char>(channel.get_channel_num(), dims);
  case cl::sycl::image_channel_type::signed_int16:
    return create_image_wrapper<cl::sycl::cl_short>(channel.get_channel_num(), dims);
  case cl::sycl::image_channel_type::signed_int32:
    return create_image_wrapper<cl::sycl::cl_int>(channel.get_channel_num(), dims);
  case cl::sycl::image_channel_type::unsigned_int8:
    return create_image_wrapper<cl::sycl::cl_uchar>(channel.get_channel_num(), dims);
  case cl::sycl::image_channel_type::unsigned_int16:
    return create_image_wrapper<cl::sycl::cl_ushort>(channel.get_channel_num(), dims);
  case cl::sycl::image_channel_type::unsigned_int32:
    return create_image_wrapper<cl::sycl::cl_uint>(channel.get_channel_num(), dims);
  default:
    return nullptr;
  }
}
// DPCT_LABEL_END
} // namespace detail

} // namespace dpct

#endif // !__DPCT_IMAGE_HPP__
