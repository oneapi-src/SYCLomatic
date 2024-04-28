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
#include <chrono>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>
namespace dpct {
namespace experimental {

namespace detail {
inline static std::unordered_set<void *> ptr_unique;
inline static std::map<std::string, int> api_index;
inline static std::string data_file = "app_runtime_data_record.json";
class logger {
public:
  logger(const std::string &dump_file)
      : opf(dump_file), json_ss(opf), arr(json_ss) {}
  ~logger() {}

  dpct::experimental::detail::json_stringstream &get_stringstream() {
    return this->json_ss;
  }

  template <class... Args>
  void print_CP(std::string &new_api_name, dpct::experimental::queue_t queue,
                Args... args) {
    ptr_unique.clear();
    auto obj = arr.object();
    obj.key("ID");
    obj.value(new_api_name);
    obj.key("CheckPoint");
    auto cp_obj =
        obj.value<dpct::experimental::detail::json_stringstream::json_obj>();
    print_args(cp_obj, queue, args...);
  }

  template <class First, class... RestArgs>
  void print_args(json_stringstream::json_obj &obj,
                  dpct::experimental::queue_t queue, std::string_view arg_name,
                  First &arg) {
    obj.key(arg_name);
    auto type_obj =
        obj.value<dpct::experimental::detail::json_stringstream::json_obj>();
    dpct::experimental::detail::data_ser<First>::print_type_name(type_obj);
    type_obj.key("Data");
    dpct::experimental::detail::data_ser<First>::dump(json_ss, arg, queue);
  }

  template <class First, class... RestArgs>
  void print_args(json_stringstream::json_obj &obj,
                  dpct::experimental::queue_t queue, std::string_view arg_name,
                  First &arg, RestArgs... args) {
    obj.key(arg_name);
    {
      auto type_obj =
          obj.value<dpct::experimental::detail::json_stringstream::json_obj>();
      dpct::experimental::detail::data_ser<First>::print_type_name(type_obj);
      type_obj.key("Data");
      dpct::experimental::detail::data_ser<First>::dump(json_ss, arg, queue);
    }
    print_args(obj, queue, args...);
  }

private:
  std::ofstream opf;
  dpct::experimental::detail::json_stringstream json_ss;
  dpct::experimental::detail::json_stringstream::json_array arr;
};

#ifdef __NVCC__
inline static std::string data_file_prefix = "CodePin_CUDA_";
#else
inline static std::string data_file_prefix = "CodePin_SYCL_";
#endif

std::string get_data_file_name(std::string_view data_file_prefix) {
  auto now = std::chrono::system_clock::now();
  std::time_t now_time = std::chrono::system_clock::to_time_t(now);
  std::tm *now_tm = std::localtime(&now_time);
  std::stringstream strs;
  strs << data_file_prefix << std::put_time(now_tm, "%Y-%m-%d_%H-%M-%S")<<".json";
  return strs.str();
}

inline static logger log(get_data_file_name(data_file_prefix));

inline std::map<void *, uint32_t> &get_ptr_size_map() {
  static std::map<void *, uint32_t> ptr_size_map;
  return ptr_size_map;
}

inline uint32_t get_ptr_size_in_bytes(void *ptr) {
  const std::map<void *, uint32_t> &ptr_size_map = get_ptr_size_map();
  const auto &it = ptr_size_map.find(ptr);
  return (it != ptr_size_map.end()) ? it->second : 0;
}

template <class T>
class data_ser<T*, void> {
public:
  static void dump(dpct::experimental::detail::json_stringstream &ss, T* value,
                   dpct::experimental::queue_t queue) {
    if (ptr_unique.find(value) != ptr_unique.end()) {
      return;
    }
    ptr_unique.insert(value);
    int size = get_ptr_size_in_bytes(value);
    size = size == 0 ? 1 : size / sizeof(*value);
    using PointeeType =
        std::remove_cv_t<std::remove_pointer_t<T>>;
    PointeeType *dump_addr = value;
    bool is_dev = is_dev_ptr(value);
    if (is_dev) {
      PointeeType *h_data = new PointeeType[size];
#ifdef __NVCC__
      cudaMemcpyAsync(h_data, value, size * sizeof(PointeeType),
                      cudaMemcpyDeviceToHost, queue);
      cudaStreamSynchronize(queue);
#else
      queue->memcpy(h_data, value, size * sizeof(PointeeType)).wait();
#endif
      dump_addr = h_data;    
    }
    auto arr = ss.array();
    for (int i = 0; i < size; ++i) {
      auto obj = arr.object();
      dpct::experimental::detail::data_ser<PointeeType>::print_type_name(obj);
      obj.key("Data");
      dpct::experimental::detail::data_ser<PointeeType>::dump(
          ss, *(dump_addr + i), queue);
    }
    if(is_dev)
      delete[] dump_addr;
  }
  static void print_type_name(
      dpct::experimental::detail::json_stringstream::json_obj &obj) {
    obj.key("Type");
    obj.value("Pointer");
  }
};

template <class T>
class data_ser<T, typename std::enable_if<std::is_array<T>::value>::type> {
public:
  static void dump(dpct::experimental::detail::json_stringstream &ss, T value,
                   dpct::experimental::queue_t queue) {
    auto arr = ss.array();
    size_t size = sizeof(T) / sizeof(value[0]);
    for (size_t i = 0; i < size; i++) {
      auto obj = arr.object();
      dpct::experimental::detail::data_ser<
          std::remove_extent_t<T>>::print_type_name(obj);
      obj.key("Data");
      dpct::experimental::detail::data_ser<std::remove_extent_t<T>>::dump(
          ss, value[i], queue);
    }
  }
  static void print_type_name(
      dpct::experimental::detail::json_stringstream::json_obj &obj) {
    obj.key("Type");
    obj.value("Array");
  }
};

template <class... Args>
void gen_log_API_CP(const std::string &api_name,
                    dpct::experimental::queue_t queue, Args... args) {
  if (api_index.find(api_name) == api_index.end()) {
    api_index[api_name] = 0;
  } else {
    api_index[api_name]++;
  }
  std::string new_api_name =
      api_name + ":" + std::to_string(api_index[api_name]);
  log.print_CP(new_api_name, queue, args...);
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
                       dpct::experimental::queue_t queue, Args&&... args) {
  synchronize(queue);
  size_t free_byte, total_byte;
#ifdef __NVCC__
  cudaMemGetInfo(&free_byte, &total_byte);
  cudaEvent_t event;
  cudaEventCreate(&event);
  cudaEventRecord(event, queue);
#else
  dpct::get_current_device().get_memory_info(free_mem, total_mem);
#endif

  dpct::experimental::detail::gen_log_API_CP(api_name, queue,
                                             args...);
}

/// Generate API check point epilog.
/// \param api_name The UID of the function call.
/// \param stream The sycl queue to synchronize the command execution.
/// \param args The var name string and variable value pair list.
template <class... Args>
void gen_epilog_API_CP(const std::string &api_name,
                       dpct::experimental::queue_t queue, Args&&... args) {
  size_t free_byte, total_byte;
#ifdef __NVCC__
  cudaMemGetInfo(&free_byte, &total_byte);
  cudaEvent_t event;
  cudaEventCreate(&event);
  cudaEventRecord(event, queue);
#else
  dpct::get_current_device().get_memory_info(free_mem, total_mem);
#endif
  gen_prolog_API_CP(api_name, queue, args...);
}

inline std::map<void *, uint32_t> &get_ptr_size_map() {
  return dpct::experimental::detail::get_ptr_size_map();
}

} // namespace experimental
} // namespace dpct
#endif // End of __DPCT_CODEPIN_HPP__
