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

#define emit_error_msg(msg)                                                    \
  {                                                                            \
    std::cerr << "Failed at:" << __FILE__ << "\nLine number is : " << __LINE__ \
              << "\n" msg << std::endl;                                        \
  }

// The class json has 2 public member functions to format json by
// get_formmat_json function and validate input json by is_valid function.
class json {
private:
  std::string original_json;
  std::string formatted_json = "";
  std::string cur_key = "";
  const char *begin = nullptr;
  const char *cur_p = nullptr;
  const char *end = nullptr;
  unsigned indent = 0;
  static const unsigned tab_length = 2;

  bool format();
  bool parse_str(std::string &ret);
  bool parse_number(char first, std::string &out);
  bool is_number(char c);
  std::string inserted_indent() {
    return std::string(indent * tab_length, ' ');
  }
  void ignore_space() {
    while (cur_p != end && (*cur_p == ' ' || *cur_p == '\t' || *cur_p == '\r' ||
                            *cur_p == '\n'))
      cur_p++;
  }
  char next() { return cur_p != end ? *cur_p++ : 0; }
  char peek() { return cur_p != end ? *cur_p : 0; }

public:
  json(const std::string &json)
      : original_json(json), begin(original_json.c_str()),
        cur_p(original_json.c_str()),
        end(original_json.c_str() + original_json.size()) {}
  std::string get_formatted_json() {
    if (formatted_json.empty()) {
      format();
    }
    return formatted_json;
  }
  bool is_valid() { return format(); }
};

inline bool json::parse_str(std::string &str) {
  char prev = peek();
  while (peek() != '"' || (peek() == '"' && prev == '\\')) {
    prev = peek();
    if (cur_p == end) {
      return false;
    }
    str += next();
  }
  next();
  return true;
}
inline bool json::is_number(char c) {
  return (c == '0') || (c == '1') || (c == '2') || (c == '3') || (c == '4') ||
         (c == '5') || (c == '6') || (c == '7') || (c == '8') || (c == '9') ||
         (c == '-') || (c == '+') || (c == '.') || (c == 'e') || (c == 'E');
}

// Parse the char value one by one to generate the number string.
inline bool json::parse_number(char c, std::string &number) {
  number += c;
  while (is_number(peek())) {
    number += next();
  }
  try {
    size_t pos;
    std::stod(number, &pos);
    return pos == number.length();
  } catch (const std::invalid_argument &ia) {
    emit_error_msg("[JSON formatter]: Parsing number value failed. Value is " +
                   number);
    return false;
  } catch (const std::out_of_range &oor) {
    emit_error_msg("[JSON formatter]: Parsing number value failed. Value is " +
                   number);
    return false;
  }
}
inline bool json::format() {
  ignore_space();
  char c = next();
  switch (c) {
  case '[': {
    indent++;
    formatted_json += "[\n";
    for (;;) {
      ignore_space();
      formatted_json += inserted_indent();
      if (!format())
        return false;
      ignore_space();
      switch (next()) {
      case ',':
        formatted_json += ",\n";
        continue;
      case ']':
        indent--;
        formatted_json += std::string("\n") + inserted_indent() + ']';
        return true;
      default:
        emit_error_msg(
            "[JSON formatter]: Parsing JSON value error. The key is " +
            cur_key);
        return false;
      }
    }
  } break;

  case '{': {
    indent++;
    formatted_json += "{\n";
    for (;;) {
      ignore_space();
      if (peek() == '"') {
        std::string key = "";
        next();
        formatted_json += inserted_indent();
        if (!parse_str(key)) {
          emit_error_msg(
              "[JSON formatter]: key value of a JSON need to be wrapped in "
              "\". Please check the JSON file format.");
          return false;
        } else {
          cur_key = "\"" + key + "\"";
          formatted_json += cur_key;
        }
      }
      ignore_space();
      if (next() == ':') {
        formatted_json += " : ";
        if (!format()) {
          emit_error_msg(
              "[JSON formatter]: Can not parse value, the JSON key is " +
              cur_key + ".\n");
          return false;
        }
      }
      ignore_space();
      switch (next()) {
      case ',': {
        formatted_json += ",\n";
        continue;
      }
      case '}': {
        indent--;
        formatted_json += std::string("\n") + inserted_indent() + '}';
        return true;
      }
      default:
        emit_error_msg("[JSON formatter]: The " + cur_key +
                       " value pair should be end with '}' or ','.\n");
        return false;
      }
    }
  } break;
  case '"': {
    std::string str = "";
    if (!parse_str(str)) {
      emit_error_msg("[JSON formatter]: The Json is invalid after " +
                     formatted_json + "\n");
      return false;
    }
    formatted_json += "\"" + str + "\"";
    return true;
  }
  case 't':
    if (next() == 'r' && next() == 'u' && next() == 'e') {
      formatted_json += "true";
      return true;
    }
    emit_error_msg("[JSON formatter]: The bool value of " + cur_key +
                   " should be \"true\", please check "
                   "the spelling.");
    return false;
  case 'f':
    if (next() == 'a' && next() == 'l' && next() == 's' && next() == 'e') {
      formatted_json += "false";
      return true;
    }
    emit_error_msg("[JSON formatter]: The bool value of " + cur_key +
                   " should be \"false\", please "
                   "check the spelling.");
    return false;
  default:
    if (is_number(c)) { // When the value is not string, bool value, dict and
                        // array. Then it should be literal number
      std::string num = "";
      parse_number(c, num);
      formatted_json += num;
      return true;
    }
    emit_error_msg("[JSON formatter]: Unkown JSON type, the last key is " +
                   cur_key +
                   ". Please check the format JSON "
                   "format.\n");
    return false;
  }
  return true;
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
