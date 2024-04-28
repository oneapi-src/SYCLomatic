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
#include <fstream>
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
typedef cudaStream_t queue_t;
#else
typedef dpct::queue_ptr queue_t;
#endif

namespace detail {

class json_stringstream {
  public:
  json_stringstream(std::ofstream &ofst) : os(ofst) {
    if (!ofst.is_open()) {
      throw std::runtime_error("Error while openning file: ");
    }
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
      }
      js.os << "\"" << key << "\": ";
      isFirst = false;
    };

    void value(std::string_view value) { js.os << "\"" << value << "\""; };
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
      }
      isFirst = false;
      return json_obj(js);      
    }

    template<class MemberT>
    void member(const MemberT &t) {
      if(!isFirst){
        js.os << "," << js.newline << js.indent;
      }
      isFirst = false;
      js.os << t;    
    }

    ~json_array() {
      js.indent.resize(js.indent.size() - js.tab_length);
      js.os << js.newline;
      js.os << js.indent;
      js.os << "]";
    }
  };

  // void print_left_brace() { *this << "{"; }
  // void print_right_brace() { *this << "}"; }
  // void print_left_bracket() { *this << "["; }
  // void print_right_bracket() { *this << "]"; }
  // void print_comma() { *this << ","; }
  // void print_type_begin(std::string type) {
  //   *this << "{\"Type\":\"" << type << "\" ,\"Data\":[";
  // }
  // void print_type_end() { *this << "]}"; }

  // void print_dict_item_key(std::string key) { *this << "\"" << key << "\":"; }
  // void print_ID_checkpoint_begin(std::string ID) {
  //   *this << "{\"ID\":"
  //         << "\"" << ID << "\"";
  //   *this << ",\"CheckPoint\":{";
  // }
  // void print_ID_checkpoint_end() { *this << "}},"; }
  // void print_data_mem_begin(std::string mem_name) {
  //   *this << "{\"" << mem_name << "\":";
  // }
  // void print_data_mem_end() { *this << "}"; }
  // template <typename T> void print_type_data(std::string type, T data) {
  //   print_type_begin(type);
  //   *this << data;
  //   print_type_end();
  // }

  // json_stringstream &operator<<(const std::string &s) {
  //   std::string ret = "";
  //   for (char c : s) {
  //     switch (c) {
  //     case '[':
  //     case '{':
  //       ret += c;
  //       ret += '\n';
  //       indent++;
  //       ret += get_indent_str();
  //       break;
  //     case '}':
  //     case ']':
  //       ret += '\n';
  //       indent--;
  //       ret += get_indent_str();
  //       ret += c;
  //       break;
  //     case ',':
  //       ret += c;
  //       ret += '\n';
  //       ret += get_indent_str();
  //       break;
  //     case ':':
  //       ret += c;
  //       ret += ' ';
  //       break;
  //     default:
  //       ret += c;
  //       break;
  //     }
  //   }
  //   ss << ret;
  //   return *this;
  // }
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

template <>
inline json_stringstream::json_obj
json_stringstream::json_obj::value<json_stringstream::json_obj>() {
  return js.object();
}

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
template <class T, class T2 = void> class data_ser {
public:
  static void dump(json_stringstream &ss, T value,
                   dpct::experimental::queue_t stream) {
    // ss.print_type_data(
    //     std::string(demangle_name<T>()),
    //     "CODEPIN:ERROR:1: Unable to find the corresponding serialization "
    //     "function.");
  }
  static void print_type_name(json_stringstream::json_obj &obj) {
    obj.key("Type");
    obj.value(std::string(demangle_name<T>()));
  }
};

template <class T>
class data_ser<T, typename std::enable_if<std::is_arithmetic<T>::value>::type> {
public:
  static void dump(json_stringstream &ss, const T &value,
                   dpct::experimental::queue_t stream) {
    auto arr = ss.array();
    arr.member<T>(value);
  }
  static void print_type_name(json_stringstream::json_obj &obj){
    obj.key("Type");
    obj.value(std::string(demangle_name<T>()));
  }
};

#ifdef __NVCC__
template <> class data_ser<int3> {
public:
  static void dump(json_stringstream &ss, const int3 &value,
                   dpct::experimental::queue_t queue) {
    auto arr = ss.array();
    {
      auto obj_x = arr.object();
      obj_x.key("x");
      auto value_x =
          obj_x
              .value<dpct::experimental::detail::json_stringstream::json_obj>();
      dpct::experimental::detail::data_ser<int>::print_type_name(value_x);
      value_x.key("Data");
      dpct::experimental::detail::data_ser<int>::dump(ss, value.x, queue);
    }
    {
      auto obj_y = arr.object();
      obj_y.key("y");
      auto value_y =
          obj_y
              .value<dpct::experimental::detail::json_stringstream::json_obj>();
      dpct::experimental::detail::data_ser<int>::print_type_name(value_y);
      value_y.key("Data");
      dpct::experimental::detail::data_ser<int>::dump(ss, value.y, queue);
    }
    {
      auto obj_z = arr.object();
      obj_z.key("z");
      auto value_z =
          obj_z
              .value<dpct::experimental::detail::json_stringstream::json_obj>();
      dpct::experimental::detail::data_ser<int>::print_type_name(value_z);
      value_z.key("Data");
      dpct::experimental::detail::data_ser<int>::dump(ss, value.z, queue);
    }
  }
  static void print_type_name(json_stringstream::json_obj &obj){
    obj.key("Type");
    obj.value("int3");
  }
};

// template <> class data_ser<float3> {
// public:
//   static void dump(json_stringstream &ss, const float3 &value,
//                    dpct::experimental::queue_t queue) {
//     ss.print_type_begin("float3");

//     ss.print_data_mem_begin("x");
//     dpct::experimental::detail::data_ser<float3>::dump(ss, value.x, queue);
//     ss.print_data_mem_end();
//     ss.print_comma();

//     ss.print_data_mem_begin("y");
//     dpct::experimental::detail::data_ser<float3>::dump(ss, value.y, queue);
//     ss.print_data_mem_end();
//     ss.print_comma();

//     ss.print_data_mem_begin("z");
//     dpct::experimental::detail::data_ser<float3>::dump(ss, value.z, queue);
//     ss.print_data_mem_end();

//     ss.print_type_end();
//   }
// };

#else
template <> class data_ser<sycl::int3> {
public:
  static void dump(json_stringstream &ss, const sycl::int3 &value,
                   dpct::experimental::queue_t queue) {
    // ss.print_type_begin("sycl::int3");

    // ss.print_data_mem_begin("x");
    // dpct::experimental::detail::data_ser<int>::dump(ss, value.x(), queue);
    // ss.print_data_mem_end();
    // ss.print_comma();

    // ss.print_data_mem_begin("y");
    // dpct::experimental::detail::data_ser<int>::dump(ss, value.y(), queue);
    // ss.print_data_mem_end();
    // ss.print_comma();

    // ss.print_data_mem_begin("z");
    // dpct::experimental::detail::data_ser<int>::dump(ss, value.z(), queue);
    // ss.print_data_mem_end();

    // ss.print_type_end();
    auto arr = ss.array();

  }
  static void print_type_name(json_stringstream::json_obj &obj){
    obj.key("Type");
    obj.value("sycl::int3");
  }
};

// template <> class data_ser<sycl::float3> {
// public:
//   static void dump(json_stringstream &ss, const sycl::float3 &value,
//                    dpct::experimental::queue_t queue) {

//     ss.print_type_begin("sycl::float3");

//     ss.print_data_mem_begin("x");
//     dpct::experimental::detail::data_ser<float3>::dump(ss, value.x(), queue);
//     ss.print_data_mem_end();
//     ss.print_comma();

//     ss.print_data_mem_begin("y");
//     dpct::experimental::detail::data_ser<float3>::dump(ss, value.y(), queue);
//     ss.print_data_mem_end();
//     ss.print_comma();

//     ss.print_data_mem_begin("z");
//     dpct::experimental::detail::data_ser<float3>::dump(ss, value.z(), queue);
//     ss.print_data_mem_end();

//     ss.print_type_end();
//   }
// };
#endif

template <> class data_ser<char *> {
public:
  static void dump(json_stringstream &ss, const char *value,
                   dpct::experimental::queue_t queue) {
    //ss.print_type_data("char *", std::string(value));
  }
};

template <> class data_ser<std::string> {
public:
  static void dump(json_stringstream &ss, const std::string &value,
                   dpct::experimental::queue_t queue) {
    //ss.print_type_data("char *", value);
  }
};

} // namespace detail
} // namespace experimental
} // namespace dpct

#endif
