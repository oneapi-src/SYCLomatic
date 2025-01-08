//==----------- basic.hpp ----------------------------*-C++-*-------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===--------------------------------------------------------------------===//
#ifndef __DPCT_CODEPIN_SER_BASIC_HPP__
#define __DPCT_CODEPIN_SER_BASIC_HPP__

#if defined(__linux__)
#include <cxxabi.h>
#endif
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string.h>
#ifdef __NVCC__
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#else
#include <dpct/dpct.hpp>
#include <sycl/sycl.hpp>
#endif

namespace dpct {
namespace experimental {
namespace codepin {

#ifdef __NVCC__
typedef cudaStream_t queue_t;
#else
typedef dpct::queue_ptr queue_t;
#endif

#ifdef __NVCC__
typedef cudaEvent_t event_t;
#else
typedef sycl::event event_t;
#endif

namespace detail {

template <typename T> inline std::string demangle_name() {
  return typeid(T).name();
}

#ifdef __NVCC__
template <> inline std::string demangle_name<__half>() { return "fp16"; }
template <> inline std::string demangle_name<__nv_bfloat16>() { return "bf16"; }
#else
template <> inline std::string demangle_name<sycl::half>() { return "fp16"; }
template <> inline std::string demangle_name<sycl::ext::oneapi::bfloat16>() {
  return "bf16";
}
#endif

template <class T>
std::string get_demangle_type_name(bool is_pointer = false) {
  std::string type = "";
  if (is_pointer) {
    type += "P";
  }
  std::string demangle_type =  std::string(demangle_name<T>());
  type = type + std::to_string(demangle_type.size()) + demangle_type;
  return type;
}

   // If pointer, how many?  // size + type.
      // len 1: if build in type, then generate the build array size. 2: if user defined type, then generate all the needed size;
      // Value: if build in type, then generate the value. 2: if user defined type, then need to generate value recursively
      // When the write is build in type, like the vector add.
  // template <class T, typename std::enable_if<std::is_arithmetic<T>::value>::type>
  template <class T>
  static size_t write_tlv_to_file(std::ofstream &ofst, T*data, size_t size) {
    std::string tag = get_demangle_type_name<T>(true);
    ofst.write(&tag[0], tag.length() + 1);
    ofst.write(reinterpret_cast<char *>(&size), sizeof(size_t));
    ofst.write(reinterpret_cast<char *>(data), size);
    ofst.flush();
    return size;
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


class json_stringstream {
  public:
  json_stringstream(std::ofstream &ofst) : os(ofst) {
    if (!ofst.is_open()) {
      throw std::runtime_error("Error while openning file: ");
    }
  }
  void flush() {
    os.flush();    
  }
private:
  std::string indent;
  const size_t tab_length = 2;
  std::ofstream &os;

#if defined(__linux__)
  const std::string newline = "\n";
#elif defined(_WIN64)
  const std::string newline = "\r\n";
#else
#error Only support windows and Linux.
#endif


public:
  class json_obj {
    bool isFirst = true;
    json_stringstream &js;
  private:
    friend class json_stringstream;
    json_obj(json_stringstream &json_ss) : js(json_ss) {
      js.os << "{" << js.newline;
      js.indent.append(js.tab_length, ' ');
      js.os << js.indent;
    }

  public:
    template<class T> T value();

    void key(std::string_view key) {
      if (!isFirst){
        js.os << "," << js.newline << js.indent;
      } else {
        isFirst = false;
      }
      js.os << "\"" << key << "\": ";
    };

    void value(std::string_view value) { js.os << "\"" << value << "\""; };
    void value(float value) { js.os << "\"" << value << "\""; };
    void value(size_t value) { js.os << "\"" << value << "\""; };
    void value(int value) { js.os << "\"" << value << "\""; };
    void value(void *value) { js.os << "\"" << value << "\""; };
    ~json_obj() {
      js.indent.resize(js.indent.size() - js.tab_length);
      js.os << js.newline;
      js.os << js.indent;
      js.os << "}";
    }
  };

  class json_array {
    bool isFirst = true;
    json_stringstream &js;
  public:
    json_array(json_stringstream &json_ss) : js(json_ss) {
      if(!(js.os))
        return;
      js.os << "[" << js.newline;
      js.indent.append(js.tab_length, ' ');
      js.os << js.indent;
    }

    json_obj object() {
      if(!isFirst){
        js.os << "," << js.newline << js.indent;
      } else {
        isFirst = false;
      }
      return json_obj(js);      
    }

    template<class MemberT>
    void member(const MemberT &t) {
      if(!isFirst){
        js.os << "," << js.newline << js.indent;
      } else {
        isFirst = false;
      }
      js.os << t;    
    }

    ~json_array() {
      js.indent.resize(js.indent.size() - js.tab_length);
      js.os << js.newline;
      js.os << js.indent;
      js.os << "]";
    }
  };

  template <typename T, typename = std::enable_if_t<
                            !std::is_same_v<const char *, std::decay_t<T>>>>
  json_stringstream &operator<<(T &&value) {
    os << std::forward<T>(value);
    return *this;
  }

  json_obj object(){
    return json_obj(*this);
  }
  json_array array(){
    return json_array(*this);
  }
};

  static void print_key_value_pair(json_stringstream::json_obj &obj,
                                   std::string_view key,
                                   std::string_view value) {
    obj.key(key);
    obj.value(value);
  }
template <>
inline json_stringstream::json_obj
json_stringstream::json_obj::value<json_stringstream::json_obj>() {
  return js.object();
}


template <class T, class T2 = void> class data_ser {

public:
  static size_t dump(json_stringstream &ss, std::ofstream &ofst, T value, queue_t queue) {
    // auto obj = ss.object();
    // obj.key("Data");
    // obj.value("CODEPIN:ERROR:1: Unable to find the corresponding serialization "
    //           "function.");
  }
  static void print_type_name(json_stringstream::json_obj &obj) {
    print_key_value_pair(obj, "Type", std::string(demangle_name<T>()));
  }

};

template <class T>
class data_ser<T, typename std::enable_if<std::is_arithmetic<T>::value>::type> {
public:
  static size_t dump(json_stringstream &ss, std::ofstream &ofst, const T &value, queue_t queue) {
    size_t size = sizeof(T);
    ofst.write(get_demangle_type_name<T>().c_str(), get_demangle_type_name<T>().length() + 1);
    ofst.write(reinterpret_cast<char*>(&size), sizeof(size_t));
    ofst.write(reinterpret_cast<const char*>(&value), sizeof(T));
    return size;
  }
  static void print_type_name(json_stringstream::json_obj &obj) {
      obj.key("Type");
      obj.value(std::to_string(strlen(typeid(T).name())) + std::string(demangle_name<T>()));
  }
  static void read(std::ifstream &ifst,  size_t size) {

  }
};

#ifdef __NVCC__
template <> class data_ser<__half> {
public:
  static size_t dump(json_stringstream &ss, std::ofstream &ofst, const __half &value, queue_t queue) {
    float f = __half2float(value);
    // auto arr = ss.array();
    // arr.member<float>(value);
  }
  static void print_type_name(json_stringstream::json_obj &obj) {
    obj.key("Type");
    obj.value(std::string("" + demangle_name<__half>().size() + demangle_name<__half>()));
  }
};
template <> class data_ser<__nv_bfloat16> {
public:
  static size_t dump(json_stringstream &ss, std::ofstream &ofst, const __nv_bfloat16 &value,
                   queue_t queue) {
    float f = __bfloat162float(value);
    // auto arr = ss.array();
    // arr.member<float>(value);
  }
  static void print_type_name(json_stringstream::json_obj &obj) {
    obj.key("Type");
    obj.value(std::string(demangle_name<__nv_bfloat16>()));
  }
};
#else
template <typename T>
class data_ser<T,
               typename std::enable_if<
                   std::is_same<T, sycl::half>::value ||
                   std::is_same<T, sycl::ext::oneapi::bfloat16>::value>::type> {
public:
  static size_t dump(json_stringstream &ss, std::ofstream &ofst, const T &value, queue_t queue) {
    // auto arr = ss.array();
    // arr.member<T>(value);
  }
  static void print_type_name(json_stringstream::json_obj &obj) {
    obj.key("Type");
    obj.value(std::string(demangle_name<T>()));
  }

};
#endif

#ifdef __NVCC__
template <> class data_ser<int3> {
public:
  static size_t dump(json_stringstream &ss, std::ofstream &ofst,
                     const int3 *value, size_t size, queue_t queue) {
    size_t items = size / sizeof(int3);
    size_t total_len = items * sizeof(int3);
    std::string tag = get_demangle_type_name<int3>(true);
    ofst.write(&tag[0], tag.length() + 1);
    ofst.write(reinterpret_cast<char *>(&total_len), sizeof(size_t));
    for (size_t i = 0; i < items; i++) {
      ofst.write(reinterpret_cast<const char *>(&value[i].x), sizeof(int));
      ofst.write(reinterpret_cast<const char *>(&value[i].y), sizeof(int));
      ofst.write(reinterpret_cast<const char *>(&value[i].z), sizeof(int));
    }
    ofst.flush();
    return total_len;
  }
  static void print_type_name(json_stringstream::json_obj &obj){
    obj.key("Type");
    obj.value("int3");
  }
};

template <> class data_ser<float3> {
public:
  static size_t dump(json_stringstream &ss, std::ofstream &ofst,
                     const float3 *value, size_t size, queue_t queue) {
    size_t items = size / sizeof(float3);
    size_t total_len = items * sizeof(float3);
    std::string tag = get_demangle_type_name<float3>(true);
    ofst.write(&tag[0], tag.length() + 1);
    ofst.write(reinterpret_cast<char *>(&total_len), sizeof(size_t));
    for (size_t i = 0; i < items; i++) {
      ofst.write(reinterpret_cast<const char *>(&value[i].x()), sizeof(float));
      ofst.write(reinterpret_cast<const char *>(&value[i].y()), sizeof(float));
      ofst.write(reinterpret_cast<const char *>(&value[i].z()), sizeof(float));
    }
    ofst.flush();
    return total_len;
  }
  static void print_type_name(json_stringstream::json_obj &obj){
    obj.key("Type");
    obj.value("float3");
  }
};

#else
template <> class data_ser<sycl::int3> {
public:
  static size_t dump(json_stringstream &ss, std::ofstream &ofst, const sycl::int3 *value,
                   size_t size, queue_t queue) {
    size_t items = size/sizeof(sycl::int3);
    size_t total_len = items * sizeof(int) * 3;
    std::string tag = get_demangle_type_name<sycl::int3>(true);
    ofst.write(&tag[0], tag.length() + 1);
    ofst.write(reinterpret_cast<char *>(&total_len), sizeof(size_t));
    for (size_t i = 0; i < items; i++) {
      ofst.write(reinterpret_cast<const char *>(&value[i].x()), sizeof(int));
      ofst.write(reinterpret_cast<const char *>(&value[i].y()), sizeof(int));
      ofst.write(reinterpret_cast<const char *>(&value[i].z()), sizeof(int));
    }
    ofst.flush();
    return total_len;
  }
  static void read(std::ifstream &ifst, size_t size) {
    size_t items = size/(sizeof(int) * 3);
    sycl::int3 *data = new sycl::int3[items];
    for (size_t i = 0; i < items; i++) {
      ifst.read(reinterpret_cast<char *>(&data[i].x()), sizeof(int));
      ifst.read(reinterpret_cast<char *>(&data[i].y()), sizeof(int));
      ifst.read(reinterpret_cast<char *>(&data[i].z()), sizeof(int));
      std::cout << "Read " << data[i].x() << " " << data[i].y() << " " << data[i].z() << std::endl;
    }
  }
  static void print_type_name(json_stringstream::json_obj &obj){
    obj.key("Type");
    obj.value("sycl::int3");
  }
};

template <> class data_ser<sycl::float3> {
public:
  static size_t dump(json_stringstream &ss, std::ofstream &ofst, const sycl::float3 *value, size_t size,
                   queue_t queue) {
    size_t items = size / sizeof(sycl::float3);
    size_t total_len = items * sizeof(float) * 3;
    std::string tag = get_demangle_type_name<sycl::float3>(true);
    ofst.write(&tag[0], tag.length() + 1);
    ofst.write(reinterpret_cast<char *>(&total_len), sizeof(size_t));
    for (size_t i = 0; i < items; i++) {
      ofst.write(reinterpret_cast<const char *>(&value[i].x()), sizeof(float));
      ofst.write(reinterpret_cast<const char *>(&value[i].y()), sizeof(float));
      ofst.write(reinterpret_cast<const char *>(&value[i].z()), sizeof(float));
    }
    ofst.flush();
    return total_len;
  }
  static void print_type_name(json_stringstream::json_obj &obj){
    obj.key("Type");
    obj.value("sycl::float3");
  }
};
#endif

template <> class data_ser<char *> {
public:
  static size_t dump(json_stringstream &ss, std::ofstream &ofst, const char *value,
                   queue_t queue) {
    // auto obj = ss.object();
    // obj.key("Data");
    const char *dump_addr = value;
    bool is_dev = is_dev_ptr((void*)value);
    if (is_dev) {
      const char *h_data = new char[strlen(value)];
#ifdef __NVCC__
      cudaMemcpyAsync((void *)h_data, (void *)value,
                      strlen(value) * sizeof(char), cudaMemcpyDeviceToHost,
                      queue);
      cudaStreamSynchronize(queue);
#else
      queue->memcpy((void *)h_data, (void *)value, strlen(value) * sizeof(char))
          .wait();
#endif
      dump_addr = h_data;    
    }
    // obj.value(std::string(dump_addr));
  }
  static void print_type_name(json_stringstream::json_obj &obj){
    obj.key("Type");
    obj.value("char *");
  }
};

template <> class data_ser<std::string> {
public:
  static size_t dump(json_stringstream &ss, std::ofstream &ofst, const std::string &value,
                   queue_t queue) {
    // auto obj = ss.object();
    // obj.key("Data");
    // obj.value(value);
  }
  static void print_type_name(json_stringstream::json_obj &obj){
    obj.key("Type");
    obj.value("std::string");
  }
};

} // namespace detail
} // namespace codepin
} // namespace experimental
} // namespace dpct

#endif
