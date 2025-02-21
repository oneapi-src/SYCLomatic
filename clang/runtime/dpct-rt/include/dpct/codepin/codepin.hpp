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
#include <ctime>
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
#include <iostream>
#include <stdlib.h>
#include <string.h>

// Random seed for data sampling.
#ifndef CODEPIN_RAND_SEED
#define CODEPIN_RAND_SEED 0
#endif

// Data size threshold to trigger data sampling.
// array/pointer size larger than the threshold will be sampled.
#ifndef CODEPIN_SAMPLING_THRESHOLD
#define CODEPIN_SAMPLING_THRESHOLD 20
#endif

// Sampling percent, interval: [0, 100]
// 0: No data will be logged.  
// 100: all data will be logged.
#ifndef CODEPIN_SAMPLING_PERCENT
#define CODEPIN_SAMPLING_PERCENT 1
#endif

#define CODEPIN_TO_STR(x) CODEPIN_STR(x)
#define CODEPIN_STR(x) #x
#pragma message(                                                               \
    "CodePin data sampling feature is enabled for data dump. As follow list 3 configs for data sampling:")
#pragma message("CODEPIN_RAND_SEED: " CODEPIN_TO_STR(CODEPIN_RAND_SEED))
#pragma message("CODEPIN_SAMPLING_THRESHOLD: " CODEPIN_TO_STR(                \
    CODEPIN_SAMPLING_THRESHOLD))
#pragma message("CODEPIN_SAMPLING_PERCENT: " CODEPIN_TO_STR(CODEPIN_SAMPLING_PERCENT))
#pragma message(                                                               \
    "Define the macros in the build command to change sampling configs. Also refer to codepin.hpp for definitions and default value of the macros.")

namespace dpct {
namespace experimental {
namespace codepin {

inline static std::map<std::string, int> api_index;
inline static std::map<std::string, event_t> event_map;
inline static bool rand_seed_setup = false;
namespace detail {

inline std::map<void *, size_t> &get_ptr_size_map() {
  static std::map<void *, size_t> ptr_size_map;
  return ptr_size_map;
}

template <class T>
size_t get_size_of_type(T &t) {
  if constexpr (std::is_pointer_v<T>) {
    return get_ptr_size_map()[(void *)t];
  }
  return sizeof(T);
}

inline size_t &get_bin_offset() {
  static size_t bin_offset = 0;
  return bin_offset;
}
inline void read_until_null_terminator(std::ifstream &is, std::string &output) {
    char ch;

    while (is.get(ch)) {
        std::cout << "ch is " << ch << std::endl;
        if (ch == '\0') {
            break;
        }
        output += ch;
    }
}

// void read_length_of_tlv(std::ifstream &is, size_t &length) {

//   is.read(reinterpret_cast<char *>(&length), sizeof(size_t));
// }
inline size_t get_ptr_size_in_bytes(void *ptr) {
  const std::map<void *, size_t> &ptr_size_map = get_ptr_size_map();
  const auto &it = ptr_size_map.find(ptr);
  return (it != ptr_size_map.end()) ? it->second : 0;
}

inline static std::unordered_set<void *> ptr_unique;

class logger {
public:
  logger(const std::string &dump_file, const std::string &dump_bin)
      : ofs_json(dump_file), ofs_bin(dump_bin), ifs_bin(dump_bin), json_ss(ofs_json), arr(json_ss) {
    auto top_obj = arr.object();
    top_obj.key("CodePin Random Seed");
    top_obj.value(CODEPIN_RAND_SEED);
    top_obj.key("CodePin Sampling Threshold");
    top_obj.value(CODEPIN_SAMPLING_THRESHOLD);
    top_obj.key("CodePin Sampling Percent");
    top_obj.value(CODEPIN_SAMPLING_PERCENT);
  }
  ~logger() {}

  detail::json_stringstream &get_stringstream() {
    return this->json_ss;
  }

  template <class... Args>
  void print_CP(const std::string &cp_id, std::string device_name,
                size_t free_byte, size_t total_byte, float elapse_time,
                queue_t queue, Args... args) {
    ptr_unique.clear();
    auto obj = arr.object();
    obj.key("ID");
    obj.value(cp_id);
    obj.key("Device Name");
    obj.value(device_name);
    obj.key("Device ID");
#ifdef __NVCC__
    int device_id;
    cudaGetDevice(&device_id);
    obj.value(device_id);
#else
    obj.value((int)dpct::get_current_device_id());
#endif
    obj.key("Stream Address");
    obj.value((void *)queue);
    obj.key("Free Device Memory");
    obj.value(free_byte);
    obj.key("Total Device Memory");
    obj.value(total_byte);
    obj.key("Elapse Time(ms)");
    obj.value(elapse_time);
    obj.key("CheckPoint");
    auto cp_obj =
        obj.value<detail::json_stringstream::json_obj>();
    size_t &offset = get_bin_offset();
    size_t old_offset = bin_offset;
    print_args(cp_obj, ofs_bin, queue, 0, args...);
    json_ss.flush();
    ofs_bin.flush();
    // read_value(ifs_bin, old_offset, args...);
    // offset = bin_offset;
  }


  void print_args(json_stringstream::json_obj &obj, std::ofstream &ofst, queue_t queue,
                  int index = 0) {}
  template <class First, class... RestArgs>
  void print_args(json_stringstream::json_obj &obj, std::ofstream &ofst, queue_t queue, int index,
                  std::string_view arg_name, First &arg, RestArgs... args) {
    obj.key(arg_name);
    {
      auto type_obj =
          obj.value<detail::json_stringstream::json_obj>();
      detail::data_ser<First>::print_type_name(type_obj);
      type_obj.key("Address");
      print_address(type_obj, arg);
      type_obj.key("Index");
      type_obj.value(index);
      // type_obj.key("Length");
      // type_obj.value(length);
      type_obj.key("Offset");
      type_obj.value(static_cast<size_t>(ofst.tellp()));
      std::cout << "CCCCVVVV \n";
      std::cout << is_expand_to_dump<std::remove_pointer_t<First>>() << std::endl;
      if (std::is_arithmetic_v<First> || is_expand_to_dump<std::remove_pointer_t<First>>()) {
      // if (std::is_arithmetic_v<First> || (std::is_pointer_v<First> && std::is_arithmetic_v<std::remove_pointer_t<First>>)) {
        detail::data_ser<First>::dump(json_ss, ofst, arg, queue);
      } else {
        type_obj.key("Data");
        detail::data_ser<First>::dump(json_ss, ofst, arg, queue);
      }
  
    }
    print_args(obj, ofst, queue, index + 1, args...);
  }

  template <class ArgT>
  void print_address(json_stringstream::json_obj &obj, ArgT arg) {
    if constexpr (std::is_pointer<ArgT>::value) {
      obj.value((void *)arg);
    } else {
      obj.value((void *)&arg);
    }
  }

  static void read_value(std::ifstream &, size_t offset) {}

  template <class First, class... RestArgs>
  static void read_value(std::ifstream &is, size_t offset, std::string_view arg_name,
                         First &arg, RestArgs... args) {
    if (is) {
      unsigned length = 0;
      if (std::is_pointer_v<First>) { // How to solve the two level pointer?
        std::cout << "Value is22222 " << std::endl;

        using PointeeType = std::remove_cv_t<std::remove_pointer_t<First>>;
        std::cout << "Va " << typeid(PointeeType).name() << std::endl;
        length = read_tlv_from_ifstream<PointeeType>(is, offset);

      } else {
        length = read_tlv_from_ifstream<First>(is, offset);
      }
      std::cout << "RRR VVVV Length is " << length << std::endl;
      offset += length;
      read_value(is, offset, args...);
    }
  }
  template <class T>
  static size_t read_tlv_from_ifstream(std::ifstream &is, size_t start_offset) {
    if (is) {
      is.seekg(start_offset);
    std::cout << "MMMMMM " << start_offset << std::endl;
    std::cout << typeid(T).name() << std::endl;
      std::string type = "";
      read_until_null_terminator(is, type);
      size_t length;
      is.read(reinterpret_cast<char *>(&length), sizeof(size_t));
      std::cout << "aaaa " << length << std::endl;
      bool is_pointer = false;
      bool is_array = false;
      size_t type_len = 0;
      size_t array_size = 0;
      std::string type_name = "";

      for (size_t i = 0; i < type.size(); i++) {
        std::cout << "vvvv " << type[i] << std::endl;
        if (i == 0 && type[i] == 'P') {
          is_pointer = true;
          type_name += type[i];
          continue;
        }
        if (type[i] == '\0') {
          break;
        }
        if (type[i] == '[') {
          is_array = true;
          continue;
        } else if (type[i] == ']') {
          is_array = false;
          break;
        }
        if (is_array) {
          std::cout << "type is i " << type[i] << std::endl;
          std::cout << "array size " << array_size << std::endl;
          array_size = array_size * 10 + (type[i] - '0');
          type_name += type[i];
          continue;
        }
        type_name += type[i];
      }
      std::cout << "Type name " << type_name << "  size " << array_size
                << std::endl;
      std::cout << "Length " << length << std::endl;
      T *data = new T[length / sizeof(T)];
      is.read(reinterpret_cast<char *>(&data[0]), length);
      // for (int i = 0; i < length / sizeof(T); i++) {
      //   std::cout << "Daxxxta " << data[i] << std::endl;
      // }
      return type_name.length() + 1 + sizeof(size_t) + length;
    }
  }

private:
  std::ofstream ofs_json;
  std::ofstream ofs_bin;
  std::ifstream ifs_bin;
  detail::json_stringstream json_ss;
  detail::json_stringstream::json_array arr;
  size_t bin_offset = 0;
};




#ifdef __NVCC__
inline std::string data_file_prefix = "CodePin_CUDA_";
#else
inline std::string data_file_prefix = "CodePin_SYCL_";
#endif

inline std::string get_formatted_time() {
    static std::string formatted_time;
    if (formatted_time.empty()) {
        std::time_t now_time = std::time(nullptr);
        std::tm* now_tm = std::localtime(&now_time);
        std::ostringstream oss;
        oss << std::put_time(now_tm, "%Y-%m-%d_%H-%M-%S");
        formatted_time = oss.str();
    }
    return formatted_time;
}
inline std::string get_data_file_name(const std::string &data_file_prefix) {
  std::string prefix = data_file_prefix;
  return prefix + get_formatted_time();
}
inline std::string get_json_file_name(const std::string &data_file_prefix) {
  std::stringstream strs;
  strs << get_data_file_name(data_file_prefix) << ".json";
  return strs.str();
}

inline std::string get_bin_file_name(const std::string &data_file_prefix) {
  std::stringstream strs;
  strs << get_data_file_name(data_file_prefix) << ".bin";
  return strs.str();
}


inline logger log(get_json_file_name(data_file_prefix), get_bin_file_name(data_file_prefix));

template <class T>
class data_ser<T*, void> {
public:
  static size_t dump(json_stringstream &ss, std::ofstream &ofst, T* value,
                   queue_t queue, bool top_run = true) {

    size_t length = 0;
    using PointeeType = std::remove_cv_t<std::remove_pointer_t<T>>;
    PointeeType *non_const_value = const_cast<PointeeType *>(value);
    if (ptr_unique.find(non_const_value) != ptr_unique.end()) {
      return 0;
    }
    ptr_unique.insert(non_const_value);
    size_t ptr_size = get_ptr_size_in_bytes(non_const_value);
    int size = ptr_size == 0 ? 1 : ptr_size / sizeof(*value);
    PointeeType *dump_addr = non_const_value;
    bool is_dev = is_dev_ptr(non_const_value);
    if (is_dev) {
      PointeeType *h_data = new PointeeType[size];
#ifdef __NVCC__
      cudaMemcpyAsync(h_data, value, size * sizeof(PointeeType),
                      cudaMemcpyDeviceToHost, queue);
      cudaStreamSynchronize(queue);
#else
      queue->memcpy((void *)h_data, (void *)value, size * sizeof(PointeeType))
          .wait();
#endif
      dump_addr = h_data;
    }


    std::string tag = get_demangle_type_name<PointeeType>(true);
    // ofst.write(&tag[0], tag.length()+1);
    std::cout << "tag " << tag << std::endl;
    std::cout << "ptr SIZE " << ptr_size << std::endl;
    // ofst.write(reinterpret_cast<char *>(&ptr_size), sizeof(size_t));
    // ofst.flush();
    // length += tag.length() + 1 + sizeof(size_t) + sizeof(PointeeType) * size;

    if (std::is_arithmetic_v<PointeeType> || is_expand_to_dump<PointeeType>()) {
      for (int i = 0; i < size; ++i) {
        detail::data_ser<PointeeType>::dump(ss, ofst, *(dump_addr + i), queue,
                                            false);
      }
    } else {
      auto arr = ss.array();
      for (int i = 0; i < size; ++i) {
        auto obj = arr.object();
        std::string key = "Mem" + std::to_string(i);
        obj.key(key);
        detail::data_ser<PointeeType>::dump(ss, ofst, *(dump_addr + i), queue,
                                            false);
      }
    }

    if(is_dev)
      delete[] dump_addr;
    return length;
  }
  static void print_type_name(
      detail::json_stringstream::json_obj &obj) {
    obj.key("Type");
    obj.value("P" + std::to_string(strlen(typeid(T).name())) + typeid(T).name()); //+ std::to_string(size));
  }
};

template <class T>
class data_ser<T, typename std::enable_if<std::is_array<T>::value>::type> {
public:
  static size_t dump(detail::json_stringstream &ss, std::ofstream &ofst, T value,
                   queue_t queue, bool top_run = true) {
    size_t length = 0;
    // auto arr = ss.array();
    auto obj = ss.object();
    size_t size = sizeof(T) / sizeof(value[0]);
    for (size_t i = 0; i < size; ++i) {
      if (size > CODEPIN_SAMPLING_THRESHOLD && i != 0) {
        float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        if (r > (float)CODEPIN_SAMPLING_PERCENT/(float)100)
          continue;
      }
      // auto obj = arr.object();
      
      detail::data_ser<
          std::remove_extent_t<T>>::print_type_name(obj);
      // obj.key("Data222");
      obj.key("Offset");
      obj.value(static_cast<size_t>(ofst.tellp()));
      detail::data_ser<std::remove_extent_t<T>>::dump(ss, ofst,
           value[i], queue, true);
    }
  return length;
  }
  static void print_type_name(
      detail::json_stringstream::json_obj &obj) {
    obj.key("Type");
    obj.value("Array");
  }
};

template <class... Args>
void gen_log_API_CP(const std::string &cp_id, std::string device_name,
                    size_t free_byte, size_t total_byte, float elapse_time,
                    queue_t queue, Args... args) {
  if (!rand_seed_setup) {
    srand(CODEPIN_RAND_SEED);
    rand_seed_setup = true;
    std::cout << "CodePin data sampling is enabled for data dump. As follow list 3 "
                 "configs for data sampling:"
              << std::endl;
    std::cout << "CODEPIN_RAND_SEED: " << CODEPIN_RAND_SEED << std::endl;
    std::cout << "CODEPIN_SAMPLING_THRESHOLD: " << CODEPIN_SAMPLING_THRESHOLD
              << std::endl;
    std::cout << "CODEPIN_SAMPLING_PERCENT: " << CODEPIN_SAMPLING_PERCENT
              << std::endl;
  }
  log.print_CP(cp_id, device_name, free_byte, total_byte, elapse_time, queue,
               args...);
}
} // namespace detail

#ifdef __NVCC__
inline void synchronize(cudaStream_t stream) { cudaStreamSynchronize(stream); }
#else
inline void synchronize(sycl::queue *q) { q->wait(); }
#endif

/// Generate API check point prolog.
/// \param cp_id The UID of the function call.
/// \param queue The sycl queue to synchronize the command execution.
/// \param args The var name string and variable value pair list.
template <class... Args>
void gen_prolog_API_CP(const std::string &cp_id,
                       queue_t queue, Args&&... args) {
  synchronize(queue);
  std::string prolog_tag = cp_id + ":" + "prolog";
  if (api_index.find(cp_id) == api_index.end()) {
    api_index[cp_id] = 0;
  } else {
    api_index[cp_id]++;
  }
  std::string event_id =
      cp_id + ":" + std::to_string(api_index[cp_id]);
  size_t free_byte, total_byte;
#ifdef __NVCC__
  int device;
  cudaGetDevice(&device);  
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device);
  std::string device_name(deviceProp.name);
  cudaMemGetInfo(&free_byte, &total_byte);
  cudaEvent_t event;
  cudaEventCreate(&event);
  cudaEventRecord(event, queue);
  event_map[event_id] = event;
#else
  dpct::get_current_device().get_memory_info(free_byte, total_byte);
  std::string device_name = dpct::get_current_device().get_info<sycl::info::device::name>();
#ifdef DPCT_PROFILING_ENABLED
  sycl::event event = queue->ext_oneapi_submit_barrier();
  event_map[event_id] = event;
#endif //DPCT_PROFILING_ENABLED
#endif

  detail::gen_log_API_CP(prolog_tag, device_name, free_byte, total_byte, 0.0f,
                         queue, args...);
}

/// Generate API check point epilog.
/// \param cp_id The UID of the function call.
/// \param stream The sycl queue to synchronize the command execution.
/// \param args The var name string and variable value pair list.
template <class... Args>
void gen_epilog_API_CP(const std::string &cp_id,
                       queue_t queue, Args&&... args) {
  synchronize(queue);
  std::string epilog_tag = cp_id + ":" + "epilog";
  std::string event_id =
      cp_id + ":" + std::to_string(api_index[cp_id]);
  size_t free_byte, total_byte;
  float kernel_elapsed_time = 0.0f;
#ifdef __NVCC__
  int device;
  cudaGetDevice(&device);  
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device);
  std::string device_name(deviceProp.name);
  cudaMemGetInfo(&free_byte, &total_byte);
  cudaEvent_t event;
  cudaEventCreate(&event);
  cudaEventRecord(event, queue);
  auto pre_event = event_map[event_id];
  event_map.erase(event_id);
  cudaEventSynchronize(event);
  cudaEventElapsedTime(&kernel_elapsed_time, pre_event, event);
#else
#ifdef DPCT_PROFILING_ENABLED
  sycl::event event = queue->ext_oneapi_submit_barrier();
  auto pre_event = event_map[event_id];
  event_map.erase(event_id);
  event.wait_and_throw();
  kernel_elapsed_time =
      (event.get_profiling_info<sycl::info::event_profiling::command_end>() -
       pre_event
           .get_profiling_info<sycl::info::event_profiling::command_start>()) /
      1000000.0f;
#endif //DPCT_PROFILING_ENABLED
  std::string device_name = dpct::get_current_device().get_info<sycl::info::device::name>();
  dpct::get_current_device().get_memory_info(free_byte, total_byte);
#endif
  detail::gen_log_API_CP(epilog_tag, device_name, free_byte, total_byte,
                         kernel_elapsed_time, queue, args...);
}

inline std::map<void *, size_t> &get_ptr_size_map() {
  return detail::get_ptr_size_map();
}

inline void set_ptr_size_map(void *ptr, size_t size) {
  if (get_ptr_size_map().find(ptr) != get_ptr_size_map().end()) {
    if (get_ptr_size_map()[ptr] < size) {
      get_ptr_size_map()[ptr] = size;
    }
    return;
  }
  get_ptr_size_map()[ptr] = size;
}
} // namespace codepin
} // namespace experimental
} // namespace dpct

namespace dpctexp = dpct::experimental;


#endif // End of __DPCT_CODEPIN_HPP__
