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

#define error_exit(msg)                                                        \
  {                                                                            \
    std::cerr << "Failed at:" << __FILE__ << "\nLine number is : " << __LINE__ \
              << "\n" msg << std::endl;                                        \
    std::exit(-1);                                                             \
  }

static std::string json_key = "";
class json_format {
private:
  std::string json_str;
  const char *begin = nullptr;
  const char *cur_p = nullptr;
  const char *end = nullptr;
  unsigned ident_size = 0;
  bool generate_formatted_json(std::string &formatted_json);

public:
  json_format(const std::string &json)
      : json_str(json), begin(json_str.c_str()), cur_p(json_str.c_str()),
        end(json_str.c_str() + json_str.size()) {}
  std::string get_formatted_json() {
    std::string formatted_json = "";
    generate_formatted_json(formatted_json);
    return formatted_json;
  }
  bool parse_str(std::string &ret);
  bool parse_num(char first, std::string &out);
  void ignore_space() {
    while (cur_p != end && (*cur_p == ' ' || *cur_p == '\t' || *cur_p == '\r' ||
                            *cur_p == '\n'))
      cur_p++;
  }
  char next() { return cur_p != end ? *cur_p++ : 0; }
  char peek() { return cur_p != end ? *cur_p : 0; }
  std::string get_indented_str() { return std::string(ident_size * 2, ' '); }
};

inline bool json_format::parse_str(std::string &str) {
  while (peek() != '"') {
    if (cur_p == end) {
      return false;
    }
    str += next();
  }
  next();
  return true;
}

inline static bool is_number(char c) {
  return (c == '0') || (c == '1') || (c == '2') || (c == '3') || (c == '4') ||
         (c == '5') || (c == '6') || (c == '7') || (c == '8') || (c == '9') ||
         (c == '-') || (c == '+') || (c == '.') || (c == 'e') || (c == 'E');
}

inline bool json_format::parse_num(char first, std::string &out) {
  out += first;
  while (is_number(peek())) {
    out += next();
  }
  try {
    size_t pos;
    std::stod(out, &pos);
    return pos == out.length();
  } catch (const std::invalid_argument &ia) {
    error_exit("[JSON format]: Parsing number value failed. Value is " + out);
  } catch (const std::out_of_range &oor) {
    error_exit("[JSON format]: Parsing number value failed. Value is " + out);
  }
}
inline bool json_format::generate_formatted_json(std::string &formatted_json) {
  ignore_space();
  char c = next();

  switch (c) {
  case '[': {
    ident_size++;
    formatted_json += "[\n";
    for (;;) {
      ignore_space();
      formatted_json += get_indented_str();
      generate_formatted_json(formatted_json);
      ignore_space();
      switch (next()) {
      case ',':
        formatted_json += ",\n";
        continue;
      case ']':
        ident_size--;
        formatted_json += std::string("\n") + get_indented_str() + ']';
        return true;
      default:
        error_exit("[JSON format]: Parsing JSON value error. The key is " +
                   json_key);
      }
    }
  } break;

  case '{': {
    ident_size++;
    formatted_json += "{\n";
    for (;;) {
      ignore_space();
      std::string key = "";
      if (peek() == '"') {
        formatted_json += get_indented_str();
        if (!generate_formatted_json(key)) {
          error_exit("[JSON format]: key value of a JSON need to be wrapped in "
                     "\". Please check the JSON file format.");
        } else {
          json_key = key;
          formatted_json += key;
        }
      }
      ignore_space();
      if (next() == ':') {
        formatted_json += " : ";
        if (!generate_formatted_json(formatted_json)) {
          error_exit("[JSON format]: Can not parse value, the JSON key is " +
                     key + ".\n");
        }
      }
      ignore_space();
      switch (next()) {
      case ',': {
        formatted_json += ",\n";
        continue;
      }
      case '}': {
        ident_size--;
        formatted_json += std::string("\n") + get_indented_str() + '}';
        return true;
      }
      default:
        error_exit("[JSON format]: The " + json_key +
                   " value pair should be end with '}' or ','.\n")
      }
    }
  } break;
  case '"': {
    std::string str = "";
    if (!parse_str(str)) {
      error_exit("[JSON format]: The Json is invalid after " + formatted_json +
                 "\n");
    }
    formatted_json += "\"" + str + "\"";
    return true;
  }
  case 't':
    if (next() == 'r' && next() == 'u' && next() == 'e') {
      formatted_json += "true";
      return true;
    }
    error_exit("[JSON format]: The bool value of " + json_key +
               " should be \"true\", please check "
               "the spelling.");
  case 'f':
    if (next() == 'a' && next() == 'l' && next() == 's' && next() == 'e') {
      formatted_json += "false";
      return true;
    }
    error_exit("[JSON format]: The bool value of " + json_key +
               " should be \"false\", please "
               "check the spelling.");
  default:
    if (is_number(c)) {
      std::string value = "";
      parse_num(c, value);
      formatted_json += value;
      return true;
    }
    error_exit("[JSON format]: Unkown JSON type, the last key is " + json_key +
               ". Please check the format JSON "
               "format.\n");
  }
}

template <typename T> void demangle_name(std::ostream &ss) {
#if defined(__linux__)
  int s;
  auto mangle_name = typeid(T).name();
  auto demangle_name = abi::__cxa_demangle(mangle_name, NULL, NULL, &s);
  if (s != 0) {
    ss << "CODEPIN:ERROR:0: Unable to demangle symbol " << mangle_name << ".";
  } else {
    ss << demangle_name;
    std::free(demangle_name);
  }
#else
  ss << typeid(T).name();
#endif
}

template <class T, class T2 = void> class DataSer {
public:
  static void dump(std::ostream &ss, T value,
                   dpct::experimental::StreamType stream) {
    ss << "{\"Type\":\"";
    demangle_name<T>(ss);
    ss << "\",\"Data\":[";
    ss << "CODEPIN:ERROR:1: Unable to find the corresponding serialization "
          "function.";
    ss << "]}";
  }
};

template <class T>
class DataSer<T, typename std::enable_if<std::is_arithmetic<T>::value>::type> {
public:
  static void dump(std::ostream &ss, const T &value,
                   dpct::experimental::StreamType stream) {
    ss << "{\"Type\":\"";
    demangle_name<T>(ss);
    ss << "\",\"Data\":[" << value << "]}";
  }
};

#ifdef __NVCC__
template <> class DataSer<int3> {
public:
  static void dump(std::ostream &ss, const int3 &value,
                   dpct::experimental::StreamType stream) {
    ss << "{\"Type\":\"int3\",\"Data\":[";
    ss << "{\"x\":";
    dpct::experimental::detail::DataSer<int>::dump(ss, value.x, stream);
    ss << "},";
    ss << "{\"y\":";
    dpct::experimental::detail::DataSer<int>::dump(ss, value.y, stream);
    ss << "},";
    ss << "{\"z\":";
    dpct::experimental::detail::DataSer<int>::dump(ss, value.z, stream);
    ss << "}";
    ss << "]}";
  }
};

template <> class DataSer<float3> {
public:
  static void dump(std::ostream &ss, const float3 &value,
                   dpct::experimental::StreamType stream) {
    ss << "{\"Type\":\"float3\",\"Data\":[";
    ss << "{\"x\":";
    dpct::experimental::detail::DataSer<float>::dump(ss, value.x, stream);
    ss << "},";
    ss << "{\"y\":";
    dpct::experimental::detail::DataSer<float>::dump(ss, value.y, stream);
    ss << "},";
    ss << "{\"z\":";
    dpct::experimental::detail::DataSer<float>::dump(ss, value.z, stream);
    ss << "}";
    ss << "]}";
  }
};

#else
template <> class DataSer<sycl::int3> {
public:
  static void dump(std::ostream &ss, const sycl::int3 &value,
                   dpct::experimental::StreamType stream) {
    ss << "{\"Type\":\"sycl::int3\",\"Data\":[";
    ss << "{\"x\":";
    dpct::experimental::detail::DataSer<int>::dump(ss, value.x(), stream);
    ss << "},";
    ss << "{\"y\":";
    dpct::experimental::detail::DataSer<int>::dump(ss, value.y(), stream);
    ss << "},";
    ss << "{\"z\":";
    dpct::experimental::detail::DataSer<int>::dump(ss, value.z(), stream);
    ss << "}";
    ss << "]}";
  }
};

template <> class DataSer<sycl::float3> {
public:
  static void dump(std::ostream &ss, const sycl::float3 &value,
                   dpct::experimental::StreamType stream) {
    ss << "{\"Type\":\"sycl::float3\",\"Data\":[";
    ss << "{\"x\":";
    dpct::experimental::detail::DataSer<float>::dump(ss, value.x(), stream);
    ss << "},";
    ss << "{\"y\":";
    dpct::experimental::detail::DataSer<float>::dump(ss, value.y(), stream);
    ss << "},";
    ss << "{\"z\":";
    dpct::experimental::detail::DataSer<float>::dump(ss, value.z(), stream);
    ss << "}";
    ss << "]}";
  }
};
#endif

template <> class DataSer<char *> {
public:
  static void dump(std::ostream &ss, const char *value,
                   dpct::experimental::StreamType stream) {
    ss << "{\"Type\":\"char *\",\"Data\":[";
    ss << std::string(value);
    ss << "]}";
  }
};

template <> class DataSer<std::string> {
public:
  static void dump(std::ostream &ss, const std::string &value,
                   dpct::experimental::StreamType stream) {
    ss << "{\"Type\":\"char *\",\"Data\":[";
    ss << value;
    ss << "]}";
  }
};

} // namespace detail
} // namespace experimental
} // namespace dpct

#endif
