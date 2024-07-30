//===--- bindless_images.hpp ----------------------*- C++ -*---------------===//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __DPCT_BINDLESS_IMAGE_HPP__
#define __DPCT_BINDLESS_IMAGE_HPP__

#ifdef _WIN32
// DirectX headers
#include <d3d11.h>
#include <dxgi.h>
#include <dxgi1_2.h>
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#endif

namespace dpct {
namespace experimental {

#ifdef SYCL_EXT_ONEAPI_BINDLESS_IMAGES

class interop_mem_wrapper;

/// The wrapper class of bindless image memory handle.
class image_mem_wrapper {
public:
  /// Create bindless image memory wrapper.
  /// \tparam dimensions The dimensions of image memory.
  /// \param [in] channel The image channel used to create bindless image
  /// \param [in] range The sizes of each dimension of bindless image memory.
  /// memory.
  template <int dimensions = 3>
  image_mem_wrapper(image_channel channel, sycl::range<dimensions> range,
                    sycl::ext::oneapi::experimental::image_type type =
                        sycl::ext::oneapi::experimental::image_type::standard,
                    unsigned int num_levels = 1)
      : _channel(channel),
        _desc(sycl::ext::oneapi::experimental::image_descriptor(
            range, _channel.get_channel_num(), _channel.get_channel_type(),
            type, num_levels)) {
    auto q = get_default_queue();
    _handle = alloc_image_mem(_desc, q);
    init_mip_level_wrappers(q);
  }
  /// Create bindless image memory wrapper.
  /// \param [in] channel The image channel used to create bindless image
  /// \param [in] width The width of bindless image memory.
  /// \param [in] height The height of bindless image memory.
  /// \param [in] depth The depth of bindless image memory.
  image_mem_wrapper(image_channel channel, size_t width, size_t height = 0,
                    size_t depth = 0)
      : image_mem_wrapper(channel, {width, height, depth}) {}
  /// Create bindless image memory wrapper.
  /// \param [in] desc The image descriptor used to create bindless image
  image_mem_wrapper(
      const sycl::ext::oneapi::experimental::image_descriptor *desc)
      : _desc(*desc) {
    _channel.set_channel_type(desc->channel_type);
    _channel.set_channel_num(desc->num_channels);
    auto q = get_default_queue();
    _handle = alloc_image_mem(_desc, q);
  }
  /// Create bindless image memory wrapper.
  /// \param [in] res_interop_mem_wrapper The interop mem wrapper of a 
  /// bindless image.
  image_mem_wrapper(
      sycl::ext::oneapi::experimental::image_descriptor desc,
      sycl::ext::oneapi::experimental::image_mem_handle img_mem_handle)
      : _desc(desc), _handle(img_mem_handle) {
    _channel.set_channel_type(_desc.channel_type);
  #if (__SYCL_COMPILER_VERSION && __SYCL_COMPILER_VERSION >= 20240725)
    _channel.set_channel_num(_desc.num_channels);
  #endif
    auto q = get_default_queue();
    init_mip_level_wrappers(q);
  }
  image_mem_wrapper(const interop_mem_wrapper *);
  image_mem_wrapper(const image_mem_wrapper &) = delete;
  image_mem_wrapper &operator=(const image_mem_wrapper &) = delete;
  /// Destroy bindless image memory wrapper.
  ~image_mem_wrapper() {
    if (_sub_wrappers) {
      std::destroy_n(_sub_wrappers, _desc.num_levels);
      std::free(_sub_wrappers);
    }
    free_image_mem(_handle, _desc.type, get_default_queue());
  }
  /// Initialize image mem wrappers for all the levels of mipmap image
  /// @param q The Queue to be used to query the mip levels 
  void init_mip_level_wrappers(sycl::queue q) {
    auto num_levels = _desc.num_levels;
    if (_desc.type == sycl::ext::oneapi::experimental::image_type::mipmap) {
      assert(num_levels > 1);
      _sub_wrappers = (image_mem_wrapper *)std::malloc(
          sizeof(image_mem_wrapper) * num_levels);
      for (unsigned i = 0; i < num_levels; ++i)
        new (_sub_wrappers + i) image_mem_wrapper(
            _channel, _desc.get_mip_level_desc(i),
            sycl::ext::oneapi::experimental::get_mip_level_mem_handle(
                _handle, i, q.get_device(), q.get_context()));
    }
  }
  /// Get the image channel of the bindless image memory.
  /// \returns The image channel of bindless image memory.
  image_channel get_channel() const noexcept { return _channel; }
  /// Get the sizes of each dimension of the bindless image memory.
  /// \returns The sizes of each dimension of bindless image memory.
  sycl::range<3> get_range() const {
    return {_desc.width, _desc.height, _desc.depth};
  }
  /// Get the image descriptor of the bindless image memory.
  /// \returns The image descriptor of bindless image memory.
  const sycl::ext::oneapi::experimental::image_descriptor &
  get_desc() const noexcept {
    return _desc;
  }
  /// Get the image handle of the bindless image memory.
  /// \returns The image handle of bindless image memory.
  sycl::ext::oneapi::experimental::image_mem_handle
  get_handle() const noexcept {
    return _handle;
  }
  /// Get the image mip level of the bindless image memory.
  /// \returns The image mip level of the bindless image memory.
  image_mem_wrapper *get_mip_level(unsigned int level) {
    assert(_desc.type == sycl::ext::oneapi::experimental::image_type::mipmap);
    return _sub_wrappers + level;
  }

private:
  image_mem_wrapper(
      const image_channel &channel,
      const sycl::ext::oneapi::experimental::image_descriptor &desc,
      const sycl::ext::oneapi::experimental::image_mem_handle &handle)
      : _channel(channel), _desc(desc), _handle(handle) {}

  image_channel _channel;
  const sycl::ext::oneapi::experimental::image_descriptor _desc;
  sycl::ext::oneapi::experimental::image_mem_handle _handle;
  image_mem_wrapper *_sub_wrappers{nullptr};
};

#ifdef _WIN32
std::unordered_set<ID3D11Resource *> uniqueD3D11Res;
#endif

/// The wrapper class of graphics interop memory handle.
class interop_mem_wrapper {
public:
#ifdef _WIN32
  interop_mem_wrapper(ID3D11Resource* d3d11_res, int reg_flags)
      : _d3d11_res(d3d11_res), _reg_flags(reg_flags) {
    
    if (uniqueD3D11Res.find(_d3d11_res) != uniqueD3D11Res.end()) {
      throw std::runtime_error("This D3D11 resource is already registered!");
    }

    query_res_info(_d3d11_res);

    _win_nt_handle = create_shared_handle(_d3d11_res);

    sycl::ext::oneapi::experimental::resource_win32_handle 
      ext_mem_win_handle{_win_nt_handle};

    sycl::ext::oneapi::experimental::external_mem_descriptor
      <sycl::ext::oneapi::experimental::resource_win32_handle> ext_mem_desc{
        ext_mem_win_handle,
        sycl::ext::oneapi::experimental::external_mem_handle_type::
            win32_nt_dx12_resource,
        _res_size_bytes};

    _interop_mem_handle_ptr =
        new sycl::ext::oneapi::experimental::interop_mem_handle;
    *_interop_mem_handle_ptr =
          sycl::ext::oneapi::experimental::import_external_memory(
              ext_mem_desc, get_default_queue());

    uniqueD3D11Res.insert(_d3d11_res);
  }
#endif

  interop_mem_wrapper(const interop_mem_wrapper &) = delete;

  interop_mem_wrapper &operator=(const interop_mem_wrapper &) = delete;

  ~interop_mem_wrapper() {
    if (!_interop_mem_handle_ptr) {
      throw std::runtime_error("Resource is not registered!");
    }

    CloseHandle(_win_nt_handle);

    sycl::ext::oneapi::experimental::release_external_memory(
        get_interop_mem_handle(), get_default_queue());
    
    uniqueD3D11Res.erase(_d3d11_res);
  }

  const sycl::ext::oneapi::experimental::interop_mem_handle
  get_interop_mem_handle() const noexcept {
    return *_interop_mem_handle_ptr;
  }

  sycl::ext::oneapi::experimental::image_mem_handle
  get_img_mem_handle() const noexcept {
    return _img_mem_wrapper_ptr->get_handle();
  }

  void *
  get_buf_ptr() const noexcept {
    return _buf_ptr;
  }

  sycl::ext::oneapi::experimental::image_descriptor
  get_img_desc() const noexcept {
    return _img_desc;
  }

  size_t get_res_size_bytes() {
    return _res_size_bytes;
  }

  bool is_buffer_res() const {
    return _res_is_buffer;
  }

  void map_resource(sycl::queue q) {
    if (_res_is_buffer) {
      if (_buf_ptr) {
        throw std::runtime_error("Resource is already mapped!");
      }

      // NOTE: 'map_external_linear_memory' API is not yet merged by Codeplay
      /*
      _buf_ptr =
          sycl::ext::oneapi::experimental::map_external_linear_memory(
              get_interop_mem_handle(), _img_desc, q);
      */
      throw
        std::runtime_error(
            "Mapping a buffer resoure is not yet supported!");
    }
    else {
      if (_img_mem_wrapper_ptr) {
        throw std::runtime_error("Resource is already mapped!");
      }

      auto img_mem_handle =
          sycl::ext::oneapi::experimental::map_external_image_memory(
              get_interop_mem_handle(), _img_desc, q);
      
      _img_mem_wrapper_ptr =
          new image_mem_wrapper(get_img_desc(), img_mem_handle);
    }
  }

  image_mem_wrapper* get_mapped_mipmapped_array() {
    if(_res_is_buffer) {
      throw std::runtime_error(
          "Buffer resouce cannot be accessed as an array!");
    }
    if(!_img_mem_wrapper_ptr) {
      throw std::runtime_error("Resouce is not mapped!");
    }

    return _img_mem_wrapper_ptr;
  }

  image_mem_wrapper* get_sub_resource_mapped_array(
        unsigned int array_index, unsigned int mip_level) {
    if (_res_is_buffer) {
      throw std::runtime_error(
          "Buffer resouce cannot be accessed as an array!");
    }
    if (array_index != 0) {
      throw std::runtime_error(
          "Retrieving a particular layer from image array "
          "is not yet supported!");
    }
    if (array_index >= _img_desc.array_size) {
      throw std::runtime_error(
          "Array index requested exceeds total array size in resource!");
    }
    if (mip_level >= _img_desc.num_levels) {
        throw std::runtime_error(
          "Mip level requested exceeds total num of levels in resource!");
    }
    if (!_img_mem_wrapper_ptr) {
      throw std::runtime_error("Resouce is not mapped!");
    }

    return (mip_level == 0)?
          _img_mem_wrapper_ptr: 
          _img_mem_wrapper_ptr->get_mip_level(mip_level);
  }

  void unmap_resource(sycl::queue q) {
    if (_res_is_buffer) {
      if (!_buf_ptr) {
        throw std::runtime_error("Resource is not mapped!");
      }

      _buf_ptr = nullptr;

      throw std::runtime_error(
          "Unmapping a buffer resoure is not yet supported!");
    }
    else {
      if (!_img_mem_wrapper_ptr) {
        throw std::runtime_error("Resource is not mapped!");
      }

      delete _img_mem_wrapper_ptr;
      _img_mem_wrapper_ptr = nullptr;
    }
  }

#ifdef _WIN32
  // Helper function to query the properties of DX11 resource
  void query_res_info(ID3D11Resource* resource) {
    unsigned int res_width = 1;
    unsigned int res_height = 1;
    unsigned int res_depth = 1;
    unsigned int res_num_levels = 1;
    unsigned int res_arr_size = 1;

    sycl::image_channel_type img_ch_type =
        sycl::image_channel_type::unsigned_int8;

    image_channel channel;

    // Get the dimension info of DX11 resource
    D3D11_RESOURCE_DIMENSION dimension;
    resource->GetType(&dimension);

    switch (dimension) {
      case D3D11_RESOURCE_DIMENSION_BUFFER: {
        _res_is_buffer = true;
        
        D3D11_BUFFER_DESC desc;
        ((ID3D11Buffer*)resource)->GetDesc(&desc);

        res_width = desc.ByteWidth;

        _res_size_bytes = desc.ByteWidth;
        break;
      }
      case D3D11_RESOURCE_DIMENSION_TEXTURE1D: {
        D3D11_TEXTURE1D_DESC desc;
        ((ID3D11Texture1D*)resource)->GetDesc(&desc);

        res_width = desc.Width;
        res_arr_size = desc.ArraySize;
        res_num_levels = (desc.MipLevels? desc.MipLevels: 1);

        get_img_ch_info(desc.Format, channel);

        _res_size_bytes =
            res_width * res_arr_size * res_num_levels *
                channel.get_total_size();
            
        break;
      }
      case D3D11_RESOURCE_DIMENSION_TEXTURE2D: {
        D3D11_TEXTURE2D_DESC desc;
        ((ID3D11Texture2D*)resource)->GetDesc(&desc);

        res_width = desc.Width;
        res_height = desc.Height;
        res_arr_size = desc.ArraySize;
        res_num_levels = (desc.MipLevels? desc.MipLevels: 1);

        get_img_ch_info(desc.Format, channel);

        _res_size_bytes =
            res_width * res_height * res_arr_size * res_num_levels *
                channel.get_total_size();
        break;
      }
      case D3D11_RESOURCE_DIMENSION_TEXTURE3D: {
        D3D11_TEXTURE3D_DESC desc;
        ((ID3D11Texture3D*)resource)->GetDesc(&desc);

        res_width = desc.Width;
        res_height = desc.Height;
        res_depth = desc.Depth;
        res_num_levels = (desc.MipLevels? desc.MipLevels: 1);

        get_img_ch_info(desc.Format, channel);

        _res_size_bytes =
            res_width * res_height * res_depth * res_num_levels *
                channel.get_total_size();
        break;
      }
      default:
        throw std::runtime_error("Unsupported DX11 resource type!");
        break;
    }

    if (res_num_levels > 1 && res_arr_size > 1) {
      throw std::runtime_error(
          "Images with multiple mip and array levels are not supported!");
    }

    sycl::ext::oneapi::experimental::image_type img_type = 
        (res_num_levels > 1)?
            sycl::ext::oneapi::experimental::image_type::mipmap:
            sycl::ext::oneapi::experimental::image_type::standard;

    _img_desc = sycl::ext::oneapi::experimental::image_descriptor{
                    sycl::range<3>{res_width, res_height, res_depth},
#if (__SYCL_COMPILER_VERSION && __SYCL_COMPILER_VERSION < 20240527)
                    channel.get_channel_order(),
#else
                    channel.get_channel_num(),
#endif
                    channel.get_channel_type(), img_type, res_num_levels,
                    res_arr_size};
  }

  // Helper function to create shared handle for DXD11 resource
  HANDLE create_shared_handle(ID3D11Resource* resource) {
    HANDLE win_nt_handle = nullptr;

    IDXGIResource1* pDXGIResource1;
    resource->QueryInterface(__uuidof(IDXGIResource1), reinterpret_cast<void**>(&pDXGIResource1));

    if (_res_is_buffer) {
      pDXGIResource1->GetSharedHandle(&win_nt_handle);
    } else {
      pDXGIResource1->CreateSharedHandle(
          nullptr,
          DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE,
          ullptr,
          &win_nt_handle
      );
    }

    pDXGIResource1->Release();

    return win_nt_handle;
  }

  // Helper function to get the texture channel info of DX11 resource using DXGI
  // format
  void get_img_ch_info(DXGI_FORMAT format, image_channel &channel) {
    switch (format) {
      case DXGI_FORMAT_R16G16B16A16_FLOAT:
        channel = image_channel::create<sycl::half4>;
        break;
      case DXGI_FORMAT_R8G8B8A8_UNORM:
        channel = image_channel::create<sycl::uchar4>;
        break;
      case DXGI_FORMAT_R16G16_FLOAT:
        channel = image_channel::create<sycl::half2>;
        break;
      case DXGI_FORMAT_R32_FLOAT:
        channel = image_channel::create<sycl::float>;
        break;
      default:
        throw std::runtime_error("Unsupported DX11 resource format!");
        break;
    }
  }
#endif

private:
  ID3D11Resource *_d3d11_res;
  HANDLE _win_nt_handle = nullptr;
  int _reg_flags;

  bool _res_is_buffer = false;
  size_t _res_size_bytes = 0;

  void *_buf_ptr = nullptr;
  sycl::ext::oneapi::experimental::image_descriptor _img_desc;
  image_mem_wrapper *_img_mem_wrapper_ptr = nullptr;

  sycl::ext::oneapi::experimental::interop_mem_handle
    *_interop_mem_handle_ptr = nullptr;
};

namespace detail {
struct sampled_image_handle_compare {
  bool
  operator()(sycl::ext::oneapi::experimental::sampled_image_handle L,
             sycl::ext::oneapi::experimental::sampled_image_handle R) const {
    return L.raw_handle < R.raw_handle;
  }
};

inline std::pair<image_data, sampling_info> &get_img_info_map(
    const sycl::ext::oneapi::experimental::sampled_image_handle handle) {
  static std::map<sycl::ext::oneapi::experimental::sampled_image_handle,
                  std::pair<image_data, sampling_info>,
                  sampled_image_handle_compare>
      img_info_map;
  return img_info_map[handle];
}

inline image_mem_wrapper *&get_img_mem_map(
    const sycl::ext::oneapi::experimental::sampled_image_handle handle) {
  static std::map<sycl::ext::oneapi::experimental::sampled_image_handle,
                  image_mem_wrapper *, sampled_image_handle_compare>
      img_mem_map;
  return img_mem_map[handle];
}

static inline size_t
get_ele_size(const sycl::ext::oneapi::experimental::image_descriptor &decs) {
  size_t channel_size;
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
  return decs.num_channels * channel_size;
}

static inline sycl::event
dpct_memcpy(sycl::ext::oneapi::experimental::image_mem_handle src,
            const sycl::ext::oneapi::experimental::image_descriptor &desc_src,
            size_t w_offset_src, size_t h_offset_src, void *dest, size_t p,
            size_t w, size_t h, sycl::queue q) {
  const auto ele_size = get_ele_size(desc_src);
  const auto src_offset =
      sycl::range<3>(w_offset_src / ele_size, h_offset_src, 0);
  const auto dest_offset = sycl::range<3>(0, 0, 0);
  const auto dest_extend = sycl::range<3>(p / ele_size, 0, 0);
  const auto copy_extend = sycl::range<3>(w / ele_size, h, 0);
  return q.ext_oneapi_copy(src, src_offset, desc_src, dest, dest_offset,
                           dest_extend, copy_extend);
}

static inline std::vector<sycl::event> dpct_memcpy_to_host(
    sycl::ext::oneapi::experimental::image_mem_handle src,
    const sycl::ext::oneapi::experimental::image_descriptor &desc_src,
    size_t w_offset_src, size_t h_offset_src, void *dest_host_ptr, size_t s,
    sycl::queue q) {
  std::vector<sycl::event> event_list;
  const auto ele_size = get_ele_size(desc_src);
  const auto w = desc_src.width * ele_size;
  size_t offset_dest = 0;
  while (s - offset_dest > w - w_offset_src) {
    const auto src_offset =
        sycl::range<3>(w_offset_src / ele_size, h_offset_src, 0);
    const auto dest_offset = sycl::range<3>(offset_dest / ele_size, 0, 0);
    const auto dest_extend = sycl::range<3>(0, 0, 0);
    const auto copy_extend =
        sycl::range<3>((w - w_offset_src) / ele_size, 1, 0);
    event_list.push_back(q.ext_oneapi_copy(src, src_offset, desc_src,
                                           dest_host_ptr, dest_offset,
                                           dest_extend, copy_extend));
    offset_dest += w - w_offset_src;
    w_offset_src = 0;
    ++h_offset_src;
  }
  const auto src_offset =
      sycl::range<3>(w_offset_src / ele_size, h_offset_src, 0);
  const auto dest_offset = sycl::range<3>(offset_dest / ele_size, 0, 0);
  const auto dest_extend = sycl::range<3>(0, 0, 0);
  const auto copy_extend = sycl::range<3>((s - w_offset_src) / ele_size, 1, 0);
  event_list.push_back(q.ext_oneapi_copy(src, src_offset, desc_src,
                                         dest_host_ptr, dest_offset,
                                         dest_extend, copy_extend));
  return event_list;
}

static inline std::vector<sycl::event>
dpct_memcpy(sycl::ext::oneapi::experimental::image_mem_handle src,
            const sycl::ext::oneapi::experimental::image_descriptor &desc_src,
            size_t w_offset_src, size_t h_offset_src, void *dest, size_t s,
            sycl::queue q) {
  if (dpct::detail::get_pointer_attribute(q, dest) ==
      dpct::detail::pointer_access_attribute::device_only) {
    std::vector<sycl::event> event_list;
    dpct::detail::host_buffer buf(s, q, event_list);
    auto copy_events = dpct_memcpy_to_host(src, desc_src, w_offset_src,
                                           h_offset_src, buf.get_ptr(), s, q);
    event_list.push_back(dpct::detail::dpct_memcpy(
        q, dest, buf.get_ptr(), s, memcpy_direction::host_to_device,
        copy_events));
    return event_list;
  }
  return dpct_memcpy_to_host(src, desc_src, w_offset_src, h_offset_src, dest, s,
                             q);
}

static inline sycl::event
dpct_memcpy(const void *src,
            sycl::ext::oneapi::experimental::image_mem_handle dest,
            const sycl::ext::oneapi::experimental::image_descriptor &desc_dest,
            size_t w_offset_dest, size_t h_offset_dest, size_t p, size_t w,
            size_t h, sycl::queue q) {
  const auto ele_size = get_ele_size(desc_dest);
  const auto src_offset = sycl::range<3>(0, 0, 0);
  const auto src_extend = sycl::range<3>(p / ele_size, 0, 0);
  const auto dest_offset =
      sycl::range<3>(w_offset_dest / ele_size, h_offset_dest, 0);
  const auto copy_extend = sycl::range<3>(w / ele_size, h, 0);
  // TODO: Remove const_cast after refining the signature of ext_oneapi_copy.
  return q.ext_oneapi_copy(const_cast<void *>(src), src_offset, src_extend,
                           dest, dest_offset, desc_dest, copy_extend);
}

static inline std::vector<sycl::event> dpct_memcpy_from_host(
    const void *src_host_ptr,
    sycl::ext::oneapi::experimental::image_mem_handle dest,
    const sycl::ext::oneapi::experimental::image_descriptor &desc_dest,
    size_t w_offset_dest, size_t h_offset_dest, size_t s, sycl::queue q) {
  std::vector<sycl::event> event_list;
  const auto ele_size = get_ele_size(desc_dest);
  const auto w = desc_dest.width * ele_size;
  size_t offset_src = 0;
  while (s - offset_src >= w - w_offset_dest) {
    const auto src_offset = sycl::range<3>(offset_src / ele_size, 0, 0);
    const auto src_extend = sycl::range<3>(0, 0, 0);
    const auto dest_offset =
        sycl::range<3>(w_offset_dest / ele_size, h_offset_dest, 0);
    const auto copy_extend =
        sycl::range<3>((w - w_offset_dest) / ele_size, 1, 0);
    // TODO: Remove const_cast after refining the signature of ext_oneapi_copy.
    event_list.push_back(q.ext_oneapi_copy(
        const_cast<void *>(src_host_ptr), src_offset, src_extend, dest,
        dest_offset, desc_dest, copy_extend));
    offset_src += w - w_offset_dest;
    w_offset_dest = 0;
    ++h_offset_dest;
  }
  const auto src_offset = sycl::range<3>(offset_src / ele_size, 0, 0);
  const auto src_extend = sycl::range<3>(0, 0, 0);
  const auto dest_offset =
      sycl::range<3>(w_offset_dest / ele_size, h_offset_dest, 0);
  const auto copy_extend =
      sycl::range<3>((s - offset_src - w_offset_dest) / ele_size, 1, 0);
  // TODO: Remove const_cast after refining the signature of ext_oneapi_copy.
  event_list.push_back(q.ext_oneapi_copy(const_cast<void *>(src_host_ptr),
                                         src_offset, src_extend, dest,
                                         dest_offset, desc_dest, copy_extend));
  return event_list;
}

static inline std::vector<sycl::event> dpct_memcpy(
    const void *src, sycl::ext::oneapi::experimental::image_mem_handle dest,
    const sycl::ext::oneapi::experimental::image_descriptor &desc_dest,
    size_t w_offset_dest, size_t h_offset_dest, size_t s, sycl::queue q) {
  if (dpct::detail::get_pointer_attribute(q, src) ==
      dpct::detail::pointer_access_attribute::device_only) {
    std::vector<sycl::event> event_list;
    dpct::detail::host_buffer buf(s, q, event_list);
    event_list.push_back(dpct::detail::dpct_memcpy(
        q, buf.get_ptr(), src, s, memcpy_direction::device_to_host));
    auto copy_events = dpct_memcpy_from_host(
        buf.get_ptr(), dest, desc_dest, w_offset_dest, h_offset_dest, s, q);
    event_list.insert(event_list.end(), copy_events.begin(), copy_events.end());
    return event_list;
  }
  return dpct_memcpy_from_host(src, dest, desc_dest, w_offset_dest,
                               h_offset_dest, s, q);
}

static inline sycl::event
dpct_memcpy(const image_mem_wrapper *src, const sycl::id<3> &src_id,
            pitched_data &dest, const sycl::id<3> &dest_id,
            const sycl::range<3> &copy_extend, sycl::queue q) {
  const auto src_offset = sycl::range<3>(src_id[0], src_id[1], src_id[2]);
  const auto dest_offset = sycl::range<3>(dest_id[0], dest_id[1], dest_id[2]);
  const auto dest_extend = sycl::range<3>(dest.get_pitch(), dest.get_y(), 1);
  return q.ext_oneapi_copy(src->get_handle(), src_offset, src->get_desc(),
                           dest.get_data_ptr(), dest_offset, dest_extend,
                           copy_extend);
}

static inline sycl::event
dpct_memcpy(pitched_data src, const sycl::id<3> &src_id,
            image_mem_wrapper *dest, const sycl::id<3> &dest_id,
            const sycl::range<3> &copy_extend, sycl::queue q) {
  const auto src_offset = sycl::range<3>(src_id[0], src_id[1], src_id[2]);
  const auto src_extend = sycl::range<3>(src.get_pitch(), src.get_y(), 1);
  const auto dest_offset = sycl::range<3>(dest_id[0], dest_id[1], dest_id[2]);
  return q.ext_oneapi_copy(src.get_data_ptr(), src_offset, src_extend,
                           dest->get_handle(), dest_offset, dest->get_desc(),
                           copy_extend);
}
} // namespace detail

template <typename T>
bool check_duplicate_entries(int count, T **entries) {
  std::set<T *> uniqueEntries;

  for (size_t i = 0; i < count; ++i) {
    auto result = uniqueEntries.insert(entries[i]);
    if (!result.second) {
      return true;
    }
  }

  return false;
}

static inline void map_resources(int count, interop_mem_wrapper **handles,
    sycl::queue q = get_default_queue()) {
  if (check_duplicate_entries(count, handles)) {
    throw std::runtime_error(
        "Duplicate handle entries found during resource mapping!");
  }

  for (int i = 0; i < count; i++ ) {
    handles[i]->map_resource(q);
  }
}

static inline void get_mapped_pointer(
    void **ptr, size_t *size, interop_mem_wrapper *handle) {
  if (!handle->is_buffer_res()) {
    throw std::runtime_error(
        "Non buffer resouce cannot be accessed as a pointer!");
  }
  
  *ptr = handle->get_buf_ptr();
  if (!*ptr) {
      throw std::runtime_error("Resource is not mapped!");
  }

  *size = handle->get_res_size_bytes();
}

static inline void unmap_resources(int count, interop_mem_wrapper **handles,
    sycl::queue q = get_default_queue()) {
  if (check_duplicate_entries(count, handles) &&
      "Duplicate handle entries found during resource unmapping!");

  for (int i = 0; i < count; i++ ) {
    handles[i]->unmap_resource(q);
  }
}

/// Create bindless image according to image data and sampling info.
/// \param [in] data The image data used to create bindless image.
/// \param [in] info The image sampling info used to create bindless image.
/// \param [in] q The queue where the image creation be executed.
/// \returns The sampled image handle of created bindless image.
static inline sycl::ext::oneapi::experimental::sampled_image_handle
create_bindless_image(image_data data, sampling_info info,
                      sycl::queue q = get_default_queue()) {
  auto samp = sycl::ext::oneapi::experimental::bindless_image_sampler(
      info.get_addressing_mode(), info.get_coordinate_normalization_mode(),
      info.get_filtering_mode(), info.get_mipmap_filtering(),
      info.get_min_mipmap_level_clamp(), info.get_max_mipmap_level_clamp(),
      info.get_max_anisotropy());

  switch (data.get_data_type()) {
  case image_data_type::linear: {
    // linear memory only use sycl::filtering_mode::nearest.
    samp.filtering = sycl::filtering_mode::nearest;
    // TODO: Use pointer to create image when bindless image support.
    auto mem = new image_mem_wrapper(
        data.get_channel(), data.get_x() / data.get_channel().get_total_size());
    auto img = sycl::ext::oneapi::experimental::create_image(
        mem->get_handle(), samp, mem->get_desc(), q);
    detail::get_img_mem_map(img) = mem;
    auto ptr = data.get_data_ptr();
#ifdef DPCT_USM_LEVEL_NONE
    q.ext_oneapi_copy(get_buffer(ptr).get_host_access().get_pointer(),
                      mem->get_handle(), mem->get_desc())
        .wait();
#else
    q.ext_oneapi_copy(ptr, mem->get_handle(), mem->get_desc()).wait();
#endif
    detail::get_img_info_map(img) = {data, info};
    return img;
  }
  case image_data_type::pitch: {
#ifdef DPCT_USM_LEVEL_NONE
    auto mem =
        new image_mem_wrapper(data.get_channel(), data.get_x(), data.get_y());
    auto img = sycl::ext::oneapi::experimental::create_image(
        mem->get_handle(), samp, mem->get_desc(), q);
    detail::get_img_mem_map(img) = mem;
    q.ext_oneapi_copy(
         get_buffer(data.get_data_ptr()).get_host_access().get_pointer(),
         mem->get_handle(), mem->get_desc())
        .wait();
#else
    auto desc = sycl::ext::oneapi::experimental::image_descriptor(
        {data.get_x(), data.get_y()}, data.get_channel().get_channel_num(),
        data.get_channel_type());
    auto img = sycl::ext::oneapi::experimental::create_image(
        data.get_data_ptr(), data.get_pitch(), samp, desc, q);
#endif
    detail::get_img_info_map(img) = {data, info};
    return img;
  }
  case image_data_type::matrix: {
    const auto mem = static_cast<image_mem_wrapper *>(data.get_data_ptr());
    auto img = sycl::ext::oneapi::experimental::create_image(
        mem->get_handle(), samp, mem->get_desc(), q);
    detail::get_img_info_map(img) = {data, info};
    return img;
  }
  default:
    throw std::runtime_error(
        "Unsupported image_data_type in create_bindless_image!");
    break;
  }
  // Must not reach here.
  return sycl::ext::oneapi::experimental::sampled_image_handle();
}

/// Destroy bindless image.
/// \param [in] handle The bindless image should be destroyed.
/// \param [in] q The queue where the image destruction be executed.
static inline void destroy_bindless_image(
    sycl::ext::oneapi::experimental::sampled_image_handle handle,
    sycl::queue q = get_default_queue()) {
  auto &mem = detail::get_img_mem_map(handle);
  if (mem) {
    delete mem;
    mem = nullptr;
  }
  sycl::ext::oneapi::experimental::destroy_image_handle(handle, q);
}

/// Get the image data according to sampled image handle.
/// \param [in] handle The bindless image handle.
/// \returns The image data of sampled image.
static inline image_data
get_data(const sycl::ext::oneapi::experimental::sampled_image_handle handle) {
  return detail::get_img_info_map(handle).first;
}

/// Get the sampling info according to sampled image handle.
/// \param [in] handle The bindless image handle.
/// \returns The sampling info of sampled image.
static inline sampling_info get_sampling_info(
    const sycl::ext::oneapi::experimental::sampled_image_handle handle) {
  return detail::get_img_info_map(handle).second;
}

/// The base class of different template specialization bindless_image_wrapper
/// class.
class bindless_image_wrapper_base {
public:
  bindless_image_wrapper_base() {
    // Make sure that singleton class dev_mgr will destruct later than this.
    dev_mgr::instance();
  }

  /// Destroy bindless image wrapper.
  ~bindless_image_wrapper_base() {
    destroy_bindless_image(_img, get_default_queue());
  }

  /// Attach linear data to bindless image.
  /// \param [in] data The linear data used to create bindless image.
  /// \param [in] size The size of linear data used to create bindless image.
  /// \param [in] channel The image channel used to create bindless image.
  /// \param [in] q The queue where the image creation be executed.
  void attach(void *data, size_t size, const image_channel &channel,
              sycl::queue q = get_default_queue()) {
    detach(q);
    auto samp = sycl::ext::oneapi::experimental::bindless_image_sampler(
        _addressing_mode, _coordinate_normalization_mode, _filtering_mode);
    // TODO: Use pointer to create image when bindless image support.
    auto mem = new image_mem_wrapper(channel, size);
    _img = sycl::ext::oneapi::experimental::create_image(
        mem->get_handle(), samp, mem->get_desc(), q);
    detail::get_img_mem_map(_img) = mem;
    auto ptr = data;
#ifdef DPCT_USM_LEVEL_NONE
    q.ext_oneapi_copy(get_buffer(data).get_host_access().get_pointer(),
                      mem->get_handle(), mem->get_desc())
        .wait();
#else
    q.ext_oneapi_copy(ptr, mem->get_handle(), mem->get_desc()).wait();
#endif
  }

  /// Attach linear data to bindless image.
  /// \param [in] data The linear data used to create bindless image.
  /// \param [in] size The size of linear data used to create bindless image.
  /// \param [in] q The queue where the image creation be executed.
  void attach(void *data, size_t size, sycl::queue q = get_default_queue()) {
    attach(data, size, _channel, q);
  }

  /// Attach device_ptr data to bindless image.
  /// \param [in] desc The image_descriptor used to create bindless image.
  /// \param [in] ptr The data pointer used to create bindless image.
  /// \param [in] pitch The pitch of 2D data used to create bindless image.
  /// \param [in] q The queue where the image creation be executed.
  void attach(const sycl::ext::oneapi::experimental::image_descriptor *desc,
              device_ptr ptr, size_t pitch,
              sycl::queue q = get_default_queue()) {
    detach(q);
    auto samp = sycl::ext::oneapi::experimental::bindless_image_sampler(
        _addressing_mode, _coordinate_normalization_mode, _filtering_mode);
#ifdef DPCT_USM_LEVEL_NONE
    auto mem = new image_mem_wrapper(desc);
    _img = sycl::ext::oneapi::experimental::create_image(mem->get_handle(),
                                                         samp, *desc, q);
    detail::get_img_mem_map(_img) = mem;
    q.ext_oneapi_copy(get_buffer(ptr).get_host_access().get_pointer(),
                      mem->get_handle(), mem->get_desc())
        .wait();
#else
    _img = sycl::ext::oneapi::experimental::create_image(ptr, pitch, samp,
                                                         *desc, q);
#endif
  }

  /// Attach 2D data to bindless image.
  /// \param [in] data The 2D data used to create bindless image.
  /// \param [in] width The width of 2D data used to create bindless image.
  /// \param [in] height The height of 2D data used to create bindless image.
  /// \param [in] pitch The pitch of 2D data used to create bindless image.
  /// \param [in] channel The image channel used to create bindless image.
  /// \param [in] q The queue where the image creation be executed.
  void attach(void *data, size_t width, size_t height, size_t pitch,
              const image_channel &channel,
              sycl::queue q = get_default_queue()) {
    auto desc = sycl::ext::oneapi::experimental::image_descriptor(
        {width, height}, channel.get_channel_num(), channel.get_channel_type());
    attach(&desc, static_cast<device_ptr>(data), pitch, q);
  }

  /// Attach 2D data to bindless image.
  /// \param [in] data The 2D data used to create bindless image.
  /// \param [in] width The width of 2D data used to create bindless image.
  /// \param [in] height The height of 2D data used to create bindless image.
  /// \param [in] pitch The pitch of 2D data used to create bindless image.
  /// \param [in] q The queue where the image creation be executed.
  void attach(void *data, size_t width, size_t height, size_t pitch,
              sycl::queue q = get_default_queue()) {
    attach(data, width, height, pitch, _channel, q);
  }

  /// Attach image memory to bindless image.
  /// \param [in] mem The image memory used to create bindless image.
  /// \param [in] q The queue where the image creation be executed.
  void attach(image_mem_wrapper *mem, const image_channel &channel,
              sycl::queue q = get_default_queue()) {
    detach(q);
    auto samp = sycl::ext::oneapi::experimental::bindless_image_sampler(
        _addressing_mode, _coordinate_normalization_mode, _filtering_mode);
    _img = sycl::ext::oneapi::experimental::create_image(
        mem->get_handle(), samp, mem->get_desc(), q);
  }

  /// Attach image memory to bindless image.
  /// \param [in] mem The image memory used to create bindless image.
  /// \param [in] q The queue where the image creation be executed.
  void attach(image_mem_wrapper *mem, sycl::queue q = get_default_queue()) {
    attach(mem, _channel, q);
  }

  /// Detach bindless image data.
  /// \param [in] q The queue where the image destruction be executed.
  void detach(sycl::queue q = get_default_queue()) {
    destroy_bindless_image(_img, q);
  }

  /// Set image channel of bindless image.
  /// \param [in] channel The image channel used to set.
  void set_channel(image_channel channel) { _channel = channel; }
  /// Get image channel of bindless image.
  /// \returns The image channel of bindless image.
  image_channel get_channel() { return _channel; }

  /// Set channel size of image channel.
  /// \param [in] channel_num The channels number to set.
  /// \param [in] channel_size The size for each channel in bits.
  void set_channel_size(unsigned channel_num, unsigned channel_size) {
    return _channel.set_channel_size(channel_num, channel_size);
  }
  /// Get channel size of image channel.
  /// \returns The size for each channel in bits.
  unsigned get_channel_size() { return _channel.get_channel_size(); }

  /// Set image channel data type of image channel.
  /// \param [in] type The image channel data type to set.
  void set_channel_data_type(image_channel_data_type type) {
    _channel.set_channel_data_type(type);
  }
  /// Get image channel data type of image channel.
  /// \returns The image channel data type of image channel.
  image_channel_data_type get_channel_data_type() {
    return _channel.get_channel_data_type();
  }

  /// Set channel num of bindless image.
  /// \param [in] filtering_mode The channel num used to set.
  void set_channel_num(unsigned num) { return _channel.set_channel_num(num); }

  /// Set channel type of bindless image.
  /// \param [in] filtering_mode The channel type used to set.
  void set_channel_type(sycl::image_channel_type type) {
    return _channel.set_channel_type(type);
  }

  /// Set addressing mode of bindless image.
  /// \param [in] addressing_mode The addressing mode used to set.
  void set(sycl::addressing_mode addressing_mode) {
    _addressing_mode = addressing_mode;
  }
  /// Get addressing mode of bindless image.
  /// \returns The addressing mode of bindless image.
  sycl::addressing_mode get_addressing_mode() { return _addressing_mode; }

  /// Set coordinate normalization mode of bindless image.
  /// \param [in] coordinate_normalization_mode The coordinate normalization
  /// mode used to set.
  void set(sycl::coordinate_normalization_mode coordinate_normalization_mode) {
    _coordinate_normalization_mode = coordinate_normalization_mode;
  }
  /// Set coordinate normalization mode of bindless image.
  /// \param [in] is_normalized Determine whether the coordinate should be
  /// normalized.
  void set_coordinate_normalization_mode(int is_normalized) {
    _coordinate_normalization_mode =
        is_normalized ? sycl::coordinate_normalization_mode::normalized
                      : sycl::coordinate_normalization_mode::unnormalized;
  }
  /// Get coordinate normalization mode of bindless image.
  /// \returns The coordinate normalization mode of bindless image.
  bool is_coordinate_normalized() {
    return _coordinate_normalization_mode ==
           sycl::coordinate_normalization_mode::normalized;
  }

  /// Set filtering mode of bindless image.
  /// \param [in] filtering_mode The filtering mode used to set.
  void set(sycl::filtering_mode filtering_mode) {
    _filtering_mode = filtering_mode;
  }
  /// Get filtering mode of bindless image.
  /// \returns The filtering mode of bindless image.
  sycl::filtering_mode get_filtering_mode() { return _filtering_mode; }

  /// Get bindless image handle.
  /// \returns The sampled image handle of created bindless image.
  inline sycl::ext::oneapi::experimental::sampled_image_handle get_handle() {
    return _img;
  }

private:
  image_channel _channel;
  sycl::addressing_mode _addressing_mode = sycl::addressing_mode::clamp_to_edge;
  sycl::coordinate_normalization_mode _coordinate_normalization_mode =
      sycl::coordinate_normalization_mode::unnormalized;
  sycl::filtering_mode _filtering_mode = sycl::filtering_mode::nearest;
  sycl::ext::oneapi::experimental::sampled_image_handle _img{0};
};

/// The wrapper class of bindless image.
template <class T, int dimensions>
class bindless_image_wrapper : public bindless_image_wrapper_base {
public:
  /// Create bindless image wrapper according to template argument \p T.
  bindless_image_wrapper() { set_channel(image_channel::create<T>()); }
};

/// Asynchronously copies from the image memory to memory specified by a
/// pointer, The return of the function does NOT guarantee the copy is
/// completed.
/// \param [in] dest The destination memory address.
/// \param [in] p The pitch of destination memory.
/// \param [in] src The source image memory.
/// \param [in] w_offset_src The x offset of source image memory.
/// \param [in] h_offset_src The y offset of source image memory.
/// \param [in] w The width of matrix to be copied.
/// \param [in] h The height of matrix to be copied.
/// \param [in] q The queue to execute the copy task.
static inline void async_dpct_memcpy(void *dest, size_t p,
                                     const image_mem_wrapper *src,
                                     size_t w_offset_src, size_t h_offset_src,
                                     size_t w, size_t h,
                                     sycl::queue q = get_default_queue()) {
  detail::dpct_memcpy(src->get_handle(), src->get_desc(), w_offset_src,
                      h_offset_src, dest, p, w, h, q);
}

/// Synchronously copies from the image memory to memory specified by a
/// pointer, The function will return after the copy is completed.
/// \param [in] dest The destination memory address.
/// \param [in] p The pitch of destination memory.
/// \param [in] src The source image memory.
/// \param [in] w_offset_src The x offset of source image memory.
/// \param [in] h_offset_src The y offset of source image memory.
/// \param [in] w The width of matrix to be copied.
/// \param [in] h The height of matrix to be copied.
/// \param [in] q The queue to execute the copy task.
static inline void dpct_memcpy(void *dest, size_t p,
                               const image_mem_wrapper *src,
                               size_t w_offset_src, size_t h_offset_src,
                               size_t w, size_t h,
                               sycl::queue q = get_default_queue()) {
  detail::dpct_memcpy(src->get_handle(), src->get_desc(), w_offset_src,
                      h_offset_src, dest, p, w, h, q)
      .wait();
}

/// Asynchronously copies from the image memory to memory specified by a
/// pointer, The return of the function does NOT guarantee the copy is
/// completed.
/// \param [in] dest The destination memory address.
/// \param [in] src The source image memory.
/// \param [in] w_offset_src The x offset of source image memory.
/// \param [in] h_offset_src The y offset of source image memory.
/// \param [in] s The size to be copied.
/// \param [in] q The queue to execute the copy task.
static inline void async_dpct_memcpy(void *dest, const image_mem_wrapper *src,
                                     size_t w_offset_src, size_t h_offset_src,
                                     size_t s,
                                     sycl::queue q = get_default_queue()) {
  detail::dpct_memcpy(src->get_handle(), src->get_desc(), w_offset_src,
                      h_offset_src, dest, s, q);
}

/// Synchronously copies from the image memory to memory specified by a
/// pointer, The function will return after the copy is completed.
/// \param [in] dest The destination memory address.
/// \param [in] src The source image memory.
/// \param [in] w_offset_src The x offset of source image memory.
/// \param [in] h_offset_src The y offset of source image memory.
/// \param [in] s The size to be copied.
/// \param [in] q The queue to execute the copy task.
static inline void dpct_memcpy(void *dest, const image_mem_wrapper *src,
                               size_t w_offset_src, size_t h_offset_src,
                               size_t s, sycl::queue q = get_default_queue()) {
  sycl::event::wait(detail::dpct_memcpy(src->get_handle(), src->get_desc(),
                                        w_offset_src, h_offset_src, dest, s,
                                        q));
}

/// Asynchronously copies from memory specified by a pointer to the image
/// memory, The return of the function does NOT guarantee the copy is completed.
/// \param [in] dest The destination image memory.
/// \param [in] w_offset_dest The x offset of destination image memory.
/// \param [in] h_offset_dest The y offset of destination image memory.
/// \param [in] src The source memory address.
/// \param [in] p The pitch of source memory.
/// \param [in] w The width of matrix to be copied.
/// \param [in] h The height of matrix to be copied.
/// \param [in] q The queue to execute the copy task.
static inline void async_dpct_memcpy(image_mem_wrapper *dest,
                                     size_t w_offset_dest, size_t h_offset_dest,
                                     const void *src, size_t p, size_t w,
                                     size_t h,
                                     sycl::queue q = get_default_queue()) {
  detail::dpct_memcpy(src, dest->get_handle(), dest->get_desc(), w_offset_dest,
                      h_offset_dest, p, w, h, q);
}

/// Synchronously copies from memory specified by a pointer to the image
/// memory, The function will return after the copy is completed.
/// \param [in] dest The destination image memory.
/// \param [in] w_offset_dest The x offset of destination image memory.
/// \param [in] h_offset_dest The y offset of destination image memory.
/// \param [in] src The source memory address.
/// \param [in] p The pitch of source memory.
/// \param [in] w The width of matrix to be copied.
/// \param [in] h The height of matrix to be copied.
/// \param [in] q The queue to execute the copy task.
static inline void dpct_memcpy(image_mem_wrapper *dest, size_t w_offset_dest,
                               size_t h_offset_dest, const void *src, size_t p,
                               size_t w, size_t h,
                               sycl::queue q = get_default_queue()) {
  detail::dpct_memcpy(src, dest->get_handle(), dest->get_desc(), w_offset_dest,
                      h_offset_dest, p, w, h, q)
      .wait();
}

/// Asynchronously copies from memory specified by a pointer to the image
/// memory, The return of the function does NOT guarantee the copy is completed.
/// \param [in] dest The destination image memory.
/// \param [in] w_offset_dest The x offset of destination image memory.
/// \param [in] h_offset_dest The y offset of destination image memory.
/// \param [in] src The source memory address.
/// \param [in] s The size to be copied.
/// \param [in] q The queue to execute the copy task.
static inline void async_dpct_memcpy(image_mem_wrapper *dest,
                                     size_t w_offset_dest, size_t h_offset_dest,
                                     const void *src, size_t s,
                                     sycl::queue q = get_default_queue()) {
  detail::dpct_memcpy(src, dest->get_handle(), dest->get_desc(), w_offset_dest,
                      h_offset_dest, s, q);
}

/// Synchronously copies from memory specified by a pointer to the image
/// memory, The function will return after the copy is completed.
/// \param [in] dest The destination image memory.
/// \param [in] w_offset_dest The x offset of destination image memory.
/// \param [in] h_offset_dest The y offset of destination image memory.
/// \param [in] src The source memory address.
/// \param [in] s The size to be copied.
/// \param [in] q The queue to execute the copy task.
static inline void dpct_memcpy(image_mem_wrapper *dest, size_t w_offset_dest,
                               size_t h_offset_dest, const void *src, size_t s,
                               sycl::queue q = get_default_queue()) {
  sycl::event::wait(detail::dpct_memcpy(src, dest->get_handle(),
                                        dest->get_desc(), w_offset_dest,
                                        h_offset_dest, s, q));
}

/// Synchronously copies from image memory to the image memory, The function
/// will return after the copy is completed.
/// \param [in] dest The destination image memory.
/// \param [in] w_offset_dest The x offset of destination image memory.
/// \param [in] h_offset_dest The y offset of destination image memory.
/// \param [in] src The source image memory.
/// \param [in] w_offset_src The x offset of source image memory.
/// \param [in] h_offset_src The y offset of source image memory.
/// \param [in] w The width of matrix to be copied.
/// \param [in] h The height of matrix to be copied.
/// \param [in] q The queue to execute the copy task.
static inline void dpct_memcpy(image_mem_wrapper *dest, size_t w_offset_dest,
                               size_t h_offset_dest,
                               const image_mem_wrapper *src,
                               size_t w_offset_src, size_t h_offset_src,
                               size_t w, size_t h,
                               sycl::queue q = get_default_queue()) {
  auto temp = (void *)sycl::malloc_device(w * h, q);
  // TODO: Need change logic when sycl support image_mem to image_mem copy.
  dpct_memcpy(temp, w, src, w_offset_src, h_offset_src, w, h, q);
  dpct_memcpy(dest, w_offset_dest, h_offset_dest, temp, w, w, h, q);
  sycl::free(temp, q);
}

/// Synchronously copies from image memory to the image memory, The function
/// will return after the copy is completed.
/// \param [in] dest The destination image memory.
/// \param [in] w_offset_dest The x offset of destination image memory.
/// \param [in] h_offset_dest The y offset of destination image memory.
/// \param [in] src The source image memory.
/// \param [in] w_offset_src The x offset of source image memory.
/// \param [in] h_offset_src The y offset of source image memory.
/// \param [in] s The size to be copied.
/// \param [in] q The queue to execute the copy task.
static inline void dpct_memcpy(image_mem_wrapper *dest, size_t w_offset_dest,
                               size_t h_offset_dest,
                               const image_mem_wrapper *src,
                               size_t w_offset_src, size_t h_offset_src,
                               size_t s, sycl::queue q = get_default_queue()) {
  auto temp = (void *)sycl::malloc_device(s, q);
  // TODO: Need change logic when sycl support image_mem to image_mem copy.
  dpct_memcpy(temp, src, w_offset_src, h_offset_src, s, q);
  dpct_memcpy(dest, w_offset_dest, h_offset_dest, temp, s, q);
  sycl::free(temp, q);
}

using image_mem_wrapper_ptr = image_mem_wrapper *;
using interop_mem_wrapper_ptr = interop_mem_wrapper *;
using bindless_image_wrapper_base_p = bindless_image_wrapper_base *;

#endif

} // namespace experimental
} // namespace dpct

#endif // !__DPCT_BINDLESS_IMAGE_HPP__
