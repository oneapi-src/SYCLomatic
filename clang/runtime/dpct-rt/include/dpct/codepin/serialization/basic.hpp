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
#include <iostream>
#include <sstream>
#include <string>
#ifdef __NVCC__
#include <cuda_runtime.h>
#else
#include <dpct/dpct.hpp>
#include <sycl/sycl.hpp>
#endif

namespace dpct {
namespace experimental {

#ifdef __NVCC__
typedef cudaStream_t StreamType;
#else
typedef dpct::queue_ptr StreamType;
#endif

namespace detail {

class json_stringstream {
private:
  int indent = 0;
  const size_t tab_length = 2;
  size_t get_indent_size() { return indent * tab_length; }
  std::string get_indent_str() { return std::string(get_indent_size(), ' '); };
  std::stringstream ss;

public:
  /// This function is used to remove the last character from the stringstream.
  /// When outputting JSON, commas are typically used to separate key-value
  /// pairs. However, the last key-value pair does not require a trailing comma.
  /// Therefore, after completing the output of the last key-value pair, this
  /// function is called to remove the last comma.
  void remove_last_comma_in_stream() {
    std::string str = ss.str();
    std::size_t last_comma = str.rfind(',');

    if (last_comma != std::string::npos) {
      str.erase(last_comma,
                2 + get_indent_size()); // Remove the ",\n" and indent space.
    }
    ss.str("");
    ss << str;
  }

  void print_left_brace() { *this << "{"; }
  void print_right_brace() { *this << "}"; }
  void print_left_bracket() { *this << "["; }
  void print_right_bracket() { *this << "]"; }
  void print_comma() { *this << ","; }
  void print_type_begin(std::string type) {
    *this << "{\"Type\":\"" << type << "\" ,\"Data\":[";
  }
  void print_type_end() { *this << "]}"; }

  void print_dict_item_key(std::string key) { *this << "\"" << key << "\":"; }
  void print_ID_checkpoint_begin(std::string ID) {
    *this << "{\"ID\":" << ID;
    *this << ",\"CheckPoint\":{";
  }
  void print_ID_checkpoint_end() { *this << "}},"; }
  void print_data_mem_begin(std::string mem_name) {
    *this << "{\"" << mem_name << "\":";
  }
  void print_data_mem_end() { *this << "}"; }
  template <typename T> void print_type_data(std::string type, T data) {
    print_type_begin(type);
    *this << data;
    print_type_end();
  }

  json_stringstream &operator<<(const std::string &s) {
    std::string ret = "";
    for (char c : s) {
      switch (c) {
      case '[':
      case '{':
        ret += c;
        ret += '\n';
        indent++;
        ret += get_indent_str();
        break;
      case '}':
      case ']':
        ret += '\n';
        indent--;
        ret += get_indent_str();
        ret += c;
        break;
      case ',':
        ret += c;
        ret += '\n';
        ret += get_indent_str();
        break;
      case ':':
        ret += c;
        ret += ' ';
        break;
      default:
        ret += c;
        break;
      }
    }
    ss << ret;
    return *this;
  }
  template <typename T, typename = std::enable_if_t<
                            !std::is_same_v<const char *, std::decay_t<T>>>>
  json_stringstream &operator<<(T &&value) {
    ss << std::forward<T>(value);
    return *this;
  }
  auto str() { return ss.str(); }
};

template <typename T> std::string demangle_name() {
  std::string ret_str = "";
#if defined(__linux__)
  int s;
  auto mangle_name = typeid(T).name();
  auto demangle_name = abi::__cxa_demangle(mangle_name, NULL, NULL, &s);
  if (s != 0) {
    ret_str = "CODEPIN:ERROR:0: Unable to demangle symbol " +
              std::string(mangle_name) + ".";
  } else {
    ret_str = demangle_name;
    std::free(demangle_name);
  }
#else
  ret_str = typeid(T).name();
#endif
  return ret_str;
}
template <class T, class T2 = void> class DataSer {
public:
  static void dump(json_stringstream &ss, T value,
                   dpct::experimental::StreamType stream) {
    ss.print_type_data(
        std::string(demangle_name<T>()),
        "CODEPIN:ERROR:1: Unable to find the corresponding serialization "
        "function.");
  }
};

template <class T>
class DataSer<T, typename std::enable_if<std::is_arithmetic<T>::value>::type> {
public:
  static void dump(json_stringstream &ss, const T &value,
                   dpct::experimental::StreamType stream) {
    ss.print_type_data(std::string(demangle_name<T>()), value);
  }
};

#ifdef __NVCC__
template <> class DataSer<int3> {
public:
  static void dump(json_stringstream &ss, const int3 &value,
                   dpct::experimental::StreamType stream) {
    ss.print_type_begin("int3");

    ss.print_data_mem_begin("x");
    dpct::experimental::detail::DataSer<int>::dump(ss, value.x, stream);
    ss.print_data_mem_end();
    ss.print_comma();

    ss.print_data_mem_begin("y");
    dpct::experimental::detail::DataSer<int>::dump(ss, value.y, stream);
    ss.print_data_mem_end();
    ss.print_comma();

    ss.print_data_mem_begin("z");
    dpct::experimental::detail::DataSer<int>::dump(ss, value.z, stream);
    ss.print_data_mem_end();

    ss.print_type_end();
  }
};

template <> class DataSer<float3> {
public:
  static void dump(json_stringstream &ss, const float3 &value,
                   dpct::experimental::StreamType stream) {
    ss.print_type_begin("float3");

    ss.print_data_mem_begin("x");
    dpct::experimental::detail::DataSer<int>::dump(ss, value.x, stream);
    ss.print_data_mem_end();
    ss.print_comma();

    ss.print_data_mem_begin("y");
    dpct::experimental::detail::DataSer<int>::dump(ss, value.y, stream);
    ss.print_data_mem_end();
    ss.print_comma();

    ss.print_data_mem_begin("z");
    dpct::experimental::detail::DataSer<int>::dump(ss, value.z, stream);
    ss.print_data_mem_end();

    ss.print_type_end();
  }
};

#else
template <> class DataSer<sycl::int3> {
public:
  static void dump(json_stringstream &ss, const sycl::int3 &value,
                   dpct::experimental::StreamType stream) {
    ss.print_type_begin("sycl::int3");

    ss.print_data_mem_begin("x");
    dpct::experimental::detail::DataSer<int>::dump(ss, value.x(), stream);
    ss.print_data_mem_end();
    ss.print_comma();

    ss.print_data_mem_begin("y");
    dpct::experimental::detail::DataSer<int>::dump(ss, value.y(), stream);
    ss.print_data_mem_end();
    ss.print_comma();

    ss.print_data_mem_begin("z");
    dpct::experimental::detail::DataSer<int>::dump(ss, value.z(), stream);
    ss.print_data_mem_end();

    ss.print_type_end();
  }
};

template <> class DataSer<sycl::float3> {
public:
  static void dump(json_stringstream &ss, const sycl::float3 &value,
                   dpct::experimental::StreamType stream) {

    ss.print_type_begin("sycl::float3");

    ss.print_data_mem_begin("x");
    dpct::experimental::detail::DataSer<int>::dump(ss, value.x(), stream);
    ss.print_data_mem_end();
    ss.print_comma();

    ss.print_data_mem_begin("y");
    dpct::experimental::detail::DataSer<int>::dump(ss, value.y(), stream);
    ss.print_data_mem_end();
    ss.print_comma();

    ss.print_data_mem_begin("z");
    dpct::experimental::detail::DataSer<int>::dump(ss, value.z(), stream);
    ss.print_data_mem_end();

    ss.print_type_end();
  }
};
#endif

template <> class DataSer<char *> {
public:
  static void dump(json_stringstream &ss, const char *value,
                   dpct::experimental::StreamType stream) {
    ss.print_type_data("char *", std::string(value));
  }
};

template <> class DataSer<std::string> {
public:
  static void dump(json_stringstream &ss, const std::string &value,
                   dpct::experimental::StreamType stream) {
    ss.print_type_data("char *", value);
  }
};

} // namespace detail
} // namespace experimental
} // namespace dpct

#endif
