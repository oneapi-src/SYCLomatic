//==---- codepin.hpp -------------------------*- C++ -*------------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===-------------------------------------------------------------------===//
#ifndef __DPCT_CODEPIN_HPP__
#define __DPCT_CODEPIN_HPP__

#include "serialization/basic.hpp"
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>
namespace dpct {
namespace experimental {

namespace detail {

class Logger {
public:
  Logger(const std::string &dump_file) : dst_output(dump_file) {
    opf.open(dst_output);
    ss.print_left_bracket();
  }

  ~Logger() {
    this->get_stringstream().remove_last_comma_in_stream();
    ss.print_right_bracket();
    opf << ss.str();
    opf.close();
  }

  dpct::experimental::detail::json_stringstream &get_stringstream() {
    return this->ss;
  }

private:
  std::string dst_output;
  std::ofstream opf;
  dpct::experimental::detail::json_stringstream ss;
};

inline static std::unordered_set<void *> ptr_unique;
inline static std::map<std::string, int> api_index;
inline static std::string data_file = "app_runtime_data_record.json";
inline static Logger log(data_file);

inline std::map<void *, uint32_t> &get_ptr_size_map() {
  static std::map<void *, uint32_t> ptr_size_map;
  return ptr_size_map;
}

inline uint32_t get_ptr_size_in_bytes(void *ptr) {
  const std::map<void *, uint32_t> &ptr_size_map = get_ptr_size_map();
  const auto &it = ptr_size_map.find(ptr);
  return (it != ptr_size_map.end()) ? it->second : 0;
}

inline bool is_dev_ptr(void *p) {
#ifdef __NVCC__
  cudaPointerAttributes attr;
  cudaPointerGetAttributes(&attr, p);
  if (attr.type == cudaMemoryTypeDevice)
    return true;
  return false;
#else
  dpct::pointer_attributes attributes;
  attributes.init(p);
  if (attributes.get_device_pointer() != nullptr)
    return true;
  return false;
#endif
}

template <class T>
class DataSer<T, typename std::enable_if<std::is_pointer<T>::value>::type> {
public:
  static void dump(dpct::experimental::detail::json_stringstream &ss, T value,
                   dpct::experimental::StreamType stream) {
    if (ptr_unique.find(value) != ptr_unique.end()) {
      return;
    }
    ptr_unique.insert(value);
    ss.print_type_begin("Pointer");
    int size = get_ptr_size_in_bytes(value);
    size = size == 0 ? 1 : size / sizeof(*value);
    using PointedType =
        std::remove_reference_t<std::remove_cv_t<std::remove_pointer_t<T>>>;
    if (is_dev_ptr(value)) {
      PointedType *h_data = new PointedType[size];
#ifdef __NVCC__
      cudaMemcpyAsync(h_data, value, size * sizeof(PointedType),
                      cudaMemcpyDeviceToHost, stream);
      cudaStreamSynchronize(stream);
#else
      stream->memcpy(h_data, value, size * sizeof(PointedType)).wait();
#endif
      for (int i = 0; i < size; ++i) {
        dpct::experimental::detail::DataSer<PointedType>::dump(
            ss, *(h_data + i), stream);
        if (i != size - 1)
          ss.print_comma();
      }
      delete[] h_data;
    } else {
      for (int i = 0; i < size; ++i) {
        dpct::experimental::detail::DataSer<PointedType>::dump(ss, *(value + i),
                                                               stream);
        if (i != size - 1)
          ss.print_comma();
      }
    }

    ss.print_type_end();
  }
};

template <class T>
class DataSer<T, typename std::enable_if<std::is_array<T>::value>::type> {
public:
  static void dump(dpct::experimental::detail::json_stringstream &ss, T value,
                   dpct::experimental::StreamType stream) {
    ss.print_type_begin("Array");
    for (auto tmp : value) {
      dpct::experimental::detail::DataSer<std::remove_extent_t<T>>::dump(
          ss, tmp, stream);
      ss.print_comma();
    }
    ss.print_type_end();
  }
};

inline void serialize_var(dpct::experimental::detail::json_stringstream &ss,
                          dpct::experimental::StreamType stream) {
  ;
}

template <class T, class... Args>
void serialize_var(dpct::experimental::detail::json_stringstream &ss,
                   dpct::experimental::StreamType stream,
                   const std::string &var_name, T var, Args... args) {
  ss.print_dict_item_key(var_name);
  ptr_unique.clear();
  dpct::experimental::detail::DataSer<T>::dump(ss, var, stream);
  ss.print_comma();
  serialize_var(ss, stream, args...);
}

template <class... Args>
void gen_log_API_CP(const std::string &api_name,
                    dpct::experimental::StreamType stream, Args... args) {
  if (api_index.find(api_name) == api_index.end()) {
    api_index[api_name] = 0;
  } else {
    api_index[api_name]++;
  }
  std::string new_api_name =
      api_name + ":" + std::to_string(api_index[api_name]);
  log.get_stringstream().print_ID_checkpoint_begin(new_api_name);
  serialize_var(log.get_stringstream(), stream, args...);
  log.get_stringstream().remove_last_comma_in_stream();
  log.get_stringstream().print_ID_checkpoint_end();
}
} // namespace detail

#ifdef __NVCC__
inline void synchronize(cudaStream_t stream) { cudaStreamSynchronize(stream); }
#else
inline void synchronize(sycl::queue *q) { q->wait(); }
#endif

/// Generate API check point prolog.
/// \param api_name The UID of the function call.
/// \param queue The sycl queue to synchronize the command execution.
/// \param args The var name string and variable value pair list.
template <class... Args>
void gen_prolog_API_CP(const std::string &api_name,
                       dpct::experimental::StreamType queue, Args... args) {
  synchronize(queue);
  dpct::experimental::detail::gen_log_API_CP(api_name, queue, args...);
}

/// Generate API check point epilog.
/// \param api_name The UID of the function call.
/// \param stream The sycl queue to synchronize the command execution.
/// \param args The var name string and variable value pair list.
template <class... Args>
void gen_epilog_API_CP(const std::string &api_name,
                       dpct::experimental::StreamType queue, Args... args) {
  gen_prolog_API_CP(api_name, queue, args...);
}

inline std::map<void *, uint32_t> &get_ptr_size_map() {
  return dpct::experimental::detail::get_ptr_size_map();
}

} // namespace experimental
} // namespace dpct
#endif // End of __DPCT_CODEPIN_HPP__
