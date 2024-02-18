//==---- dispatcher.hpp -------------------------------*- C++
//-*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===--------------------------------------------------------------------------===//
#ifndef __DPCT_CODEPIN_DISPATCHER_HPP__
#define __DPCT_CODEPIN_DISPATCHER_HPP__

#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef __NVCC__
#include <cuda_runtime.h>
#else
#include <dpct/dpct.hpp>
#include <sycl/sycl.hpp>
#endif

namespace dpct {
namespace experimental {
namespace detail {

inline static std::map<std::string, int> api_index;
inline static std::vector<std::string> dump_json;
inline static std::string dump_file = "dump_log.json";

class Logger {
public:
  Logger(const std::string &dumpFile)
      : dump_file(dumpFile), ipf(dumpFile, std::ios::in) {
    if (ipf.is_open()) {
      std::getline(ipf, data);
      ipf.close();
    }
  }

  ~Logger() {
    opf.open(dump_file);
    std::string json = std::accumulate(
        dump_json.begin(), dump_json.end(), std::string("{"),
        [](std::string acc, std::string val) { return acc + val + ','; });
    if (!json.empty()) {
      json.pop_back();
    }
    json += "}\n";
    opf << json;
    if (!opf.is_open()) {
      opf.close();
    }
  }
  const std::string &get_data() { return data; }

private:
  std::string dump_file;
  std::ifstream ipf;
  std::ofstream opf;
  std::string data;
};

static Logger log(dump_file);

inline std::map<void *, uint32_t> &get_ptr_size_map() {
  static std::map<void *, uint32_t> ptr_size_map;
  return ptr_size_map;
}

inline uint32_t get_pointer_size_in_bytes_from_map(void *ptr) {
  const std::map<void *, uint32_t> &ptr_size_map = get_ptr_size_map();
  const auto &it = ptr_size_map.find(ptr);
  return (it != ptr_size_map.end()) ? it->second : 0;
}

// Add faker dump for pass lit test
#ifdef __NVCC__
void dump(std::string &log, const int3 &value) {
  // plcaeholder
}

void dump(std::string &log, const float3 &value) {
  // plcaeholder
}
#else
void dump(std::string &log, const sycl::int3 &value) {
  // plcaeholder
}

void dump(std::string &log, const sycl::float3 &value) {
  // plcaeholder
}
#endif

void dump(std::string &log, const char *value) {
  // plcaeholder
  log = log + '\t' + std::string(value);
}

void dump(std::string &log, const std::string &value) {
  // plcaeholder
  log = log + '\t' + std::string(value);
}

template <class T>
typename std::enable_if<std::is_arithmetic<T>::value>::type
dump(std::string &log, const T &value) {
  // plcaeholder
  log = log + '\t' + std::to_string(value);
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
#ifdef __NVCC__
void dispatchOutputCodePin(std::string &log, T a, cudaStream_t stream) {
#else
void dispatchOutputCodePin(std::string &log, T a, dpct::queue_ptr stream) {
#endif
  if constexpr (std::is_pointer_v<T>) {
    int size = get_pointer_size_in_bytes_from_map(a);
    size = size == 0 ? 1 : size / sizeof(*a);
    if (is_dev_ptr(a)) {
      using PointedType =
          std::remove_reference_t<std::remove_cv_t<std::remove_pointer_t<T>>>;
      PointedType *hData = new PointedType[size];
#ifdef __NVCC__
      cudaMemcpyAsync(hData, a, size * sizeof(PointedType),
                      cudaMemcpyDeviceToHost, stream);
      cudaStreamSynchronize(stream);
#else
      stream->memcpy(hData, a, size * sizeof(PointedType)).wait();
#endif
      for (int i = 0; i < size; ++i) {
        dispatchOutputCodePin(log, *(hData + i), stream);
      }
      delete[] hData;
    } else {
      for (int i = 0; i < size; ++i) {
        dispatchOutputCodePin(log, *(a + i), stream);
      }
    }

  } else if constexpr (std::is_array_v<T>) {
    for (auto tmp : a) {
      dispatchOutputCodePin(log, tmp, stream);
    }
  } else {
    dump(log, a);
  }
}

#ifdef __NVCC__
inline void process_var(std::string log, cudaStream_t stream) { ; }
#else
inline void process_var(std::string log, dpct::queue_ptr stream) { ; }
#endif

template <class T, class... Args>
#ifdef __NVCC__
void process_var(std::string log, cudaStream_t stream,
                 const std::string &varName, T var, Args... args) {
#else
void process_var(std::string log, dpct::queue_ptr stream,
                 const std::string &varName, T var, Args... args) {
#endif
  log += "\"" + varName + "\":\"";
  dispatchOutputCodePin(log, var, stream);
  process_var(log, stream, args...);
}

inline void dump_data(const std::string &name, const std::string &data) {
  std::string data_str = "\"" + name + "\" : " + "{" + data + "}";
  dump_json.push_back(data_str);
}

template <class... Args>
#ifdef __NVCC__
void gen_log_API_CP(const std::string &api_name, cudaStream_t stream,
                    Args... args) {
#else
void gen_log_API_CP(const std::string &api_name, dpct::queue_ptr stream,
                    Args... args) {
#endif
  if (api_index.find(api_name) == api_index.end()) {
    api_index[api_name] = 0;
  } else {
    api_index[api_name]++;
  }
  std::string new_api_name =
      api_name + ":" + std::to_string(api_index[api_name]);
  std::string log;
  process_var(log, stream, args...);
  if (log.back() == ',')
    log.pop_back(); // Pop last ',' character
  dump_data(new_api_name, log);
}

} // namespace detail
} // namespace experimental
} // namespace dpct

#endif