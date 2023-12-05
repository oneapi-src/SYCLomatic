#ifndef __DPCT_SCHEMA_HELPER__
#define __DPCT_SCHEMA_HELPER__
#include "debug_helper.hpp"
#include "json.hpp"
#include <algorithm>
#include <assert.h>
#include <cctype>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
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

class Schema;

static std::map<std::string, std::shared_ptr<Schema>> schema_map;
static std::map<std::string, size_t> schema_size;
size_t get_size_of_schema(const std::string &schema) {
  if (schema_size.find(schema) == schema_size.end()) {
    schema_size[schema] = 0; // '0' indicates that the size will match the type
                             // size specified in the schema string
  }
  return schema_size[schema];
}
void set_size_of_schema(const std::string &schema, size_t size) {
  schema_size[schema] = size;
}
enum class ValType { SCALAR, POINTER, ARRAY, POINTERTOPOINTER };
enum class MemLoc { NONE, HOST, DEVICE };
enum class schema_type {
  TYPE,  // class or struct type.
  DATA,  // alloc data type.
  MEMBER // data member in class or struct.
};

inline std::string to_upper(std::string str) {
  std::transform(str.begin(), str.end(), str.begin(),
                 [](unsigned char c) { return std::toupper(c); });
  return str;
}

inline ValType getValType(const std::string &str) {
  std::string to_str = to_upper(str);
  if (to_str == "SCALAR") {
    return ValType::SCALAR;
  } else if (to_str == "POINTER") {
    return ValType::POINTER;
  } else if (to_str == "ARRAY") {
    return ValType::ARRAY;
  }
  error_exit("The value type:" + str + " is unkonwn");
}

inline MemLoc getMemLoc(const std::string &str) {
  std::string to_str = to_upper(str);
  if (to_str == "NONE") {
    return MemLoc::NONE;
  } else if (to_str == "HOST") {
    return MemLoc::HOST;
  } else if (to_str == "DEVICE") {
    return MemLoc::DEVICE;
  }
  error_exit("The memory location:" + str + " is unkonwn");
}

inline schema_type get_schema_type(const std::string &str) {
  std::string to_str = to_upper(str);
  if (to_str == "TYPE") {
    return schema_type::TYPE;
  } else if (to_str == "DATA") {
    return schema_type::DATA;
  } else if (to_str == "MEMBER") {
    return schema_type::MEMBER;
  }
  error_exit("The schmea type:" + str + " is unkonwn");
}
class Schema {
public:
  // Construct the class type.
  Schema(const std::string &TypeName, schema_type SchemaTy, size_t FieldNum,
         bool IsVirtual, size_t TypeSize, const std::string &FilePath)
      : TypeName(TypeName), SchemaTy(SchemaTy), FieldNum(FieldNum),
        IsVirtual(IsVirtual), TypeSize(TypeSize), FilePath(FilePath) {}

  // Construct the data member.
  Schema(const std::string &VarName, const std::string &TypeName,
         size_t TypeSize, bool IsBasicType, ValType ValTy, size_t ValSize,
         size_t Offset, MemLoc Location)
      : VarName(VarName), TypeName(TypeName), TypeSize(TypeSize),
        IsBasicType(IsBasicType), ValTy(ValTy), ValSize(ValSize),
        Offset(Offset), Location(Location) {}

  // Construct the data member in class.
  Schema(const std::string &VarName, const std::string &TypeName,
         size_t TypeSize, schema_type SchemaTy, bool IsBasicType, ValType ValTy,
         size_t ValSize, MemLoc Location)
      : TypeName(TypeName), TypeSize(TypeSize), VarName(VarName),
        SchemaTy(SchemaTy), IsBasicType(IsBasicType), ValTy(ValTy),
        ValSize(ValSize), Location(Location) {}

  void add_data_schema(std::shared_ptr<Schema> &data) {
    Members.push_back(data);
  }

  ValType get_val_type() { return ValTy; }
  size_t get_val_size() { return ValSize; }
  size_t get_type_size() { return TypeSize; }
  bool is_basic_type() { return IsBasicType; }
  const std::string &get_type_name() { return TypeName; }
  bool is_virtual_type() { return IsVirtual; }
  size_t get_feild_num() { return FieldNum; }
  std::vector<std::shared_ptr<Schema>> &get_type_member() { return Members; }
  size_t get_offset() { return Offset; }
  const std::string &get_var_name() { return VarName; }

private:
  // Common Part
  std::string TypeName = "";                // namespace + class type + name;
  schema_type SchemaTy = schema_type::TYPE; // Data or Type
  std::string VarName = "";
  bool IsBasicType = true;
  std::string FilePath;
  /* Class Type only */
  bool IsVirtual = false;
  size_t FieldNum = 0;
  size_t TypeSize = 0;
  std::vector<std::shared_ptr<Schema>> Members;
  /* Data member only */
  ValType ValTy = ValType::SCALAR;
  size_t Offset = 0;
  size_t ValSize = 0;
  MemLoc Location = MemLoc::NONE;
};

// data, test_namespace::A + varName,
inline std::pair<std::string, std::shared_ptr<Schema>>
create_schema_struct(const std::string &TypeName, schema_type SchemaTy,
                     size_t FieldNum, bool IsVirtual, size_t TypeSize,
                     const std::string &FilePath) {
  return std::pair<std::string, std::shared_ptr<Schema>>(
      TypeName, std::make_shared<Schema>(TypeName, SchemaTy, FieldNum,
                                         IsVirtual, TypeSize, FilePath));
}

inline void create_schema_member(std::shared_ptr<Schema> TypeSchemaStruct,
                                 const std::string &VarName,
                                 const std::string &TypeName, size_t type_size,
                                 bool IsBasicType, ValType ValTy,
                                 size_t ValSize, size_t Offset,
                                 MemLoc Location) {
  std::shared_ptr<Schema> DataMember =
      std::make_shared<Schema>(VarName, TypeName, type_size, IsBasicType, ValTy,
                               ValSize, Offset, Location);
  TypeSchemaStruct->add_data_schema(DataMember);
}

inline std::pair<std::string, std::shared_ptr<Schema>>
create_schema_var(const std::string &VarName, const std::string &TypeName,
                  size_t type_size, schema_type SchemaTy, bool IsBasicType,
                  ValType ValTy, size_t ValSize, MemLoc Location) {
  return std::pair<std::string, std::shared_ptr<Schema>>(
      TypeName,
      std::make_shared<Schema>(VarName, TypeName, type_size, SchemaTy,
                               IsBasicType, ValTy, ValSize, Location));
}

inline std::shared_ptr<Schema> gen_type_schema(dpct_json::value &v) {
  auto obj = v.get_value<dpct_json::object>();
  std::string schema_name = obj.get("TypeName").get_value<std::string>();
  std::string schema_type_name =
      obj.get("SchemaType").get_value<std::string>();
  int field_num = obj.get("FieldNum").get_value<int>();
  size_t type_size = obj.get("TypeSize").get_value<int>();
  bool is_virtual = obj.get("IsVirtual").get_value<bool>();
  std::string file_path = obj.get("FilePath").get_value<std::string>();
  std::pair<std::string, std::shared_ptr<Schema>> schema_struct_pair =
      create_schema_struct(schema_name, get_schema_type(schema_type_name),
                           field_num, is_virtual, type_size, file_path);
  std::shared_ptr<Schema> struct_schema = schema_struct_pair.second;
  schema_map[schema_struct_pair.first] = struct_schema;

  if (field_num != 0) {
    dpct_json::array mem_arr = obj["Members"].get_value<dpct_json::array>();
    for (int i = 0; i < field_num; i++) {
      dpct_json::object data_obj = mem_arr[i].get_value<dpct_json::object>();
      std::string var_name = data_obj.get("VarName").get_value<std::string>();
      std::string type_name = data_obj.get("TypeName").get_value<std::string>();
      size_t type_size = obj.get("TypeSize").get_value<int>();
      bool is_basic_type = data_obj.get("IsBasicType").get_value<bool>();
      std::string val_type = data_obj.get("ValType").get_value<std::string>();
      size_t val_size = data_obj.get("ValSize").get_value<int>();
      size_t offset = data_obj.get("Offset").get_value<int>();
      std::string location = data_obj.get("Location").get_value<std::string>();
      create_schema_member(struct_schema, var_name, type_name, type_size,
                           is_basic_type, getValType(val_type), val_size,
                           offset, getMemLoc(location));
    }
  }
  return struct_schema;
}

inline std::shared_ptr<Schema> gen_data_schema(dpct_json::value &v) {
  dpct_json::object data_obj = v.get_value<dpct_json::object>();
  std::string var_name = data_obj.get("VarName").get_value<std::string>();
  std::string type_name = data_obj.get("TypeName").get_value<std::string>();
  const schema_type local_type =
      get_schema_type(data_obj.get("SchemaType").get_value<std::string>());
  bool is_basic_type = data_obj.get("IsBasicType").get_value<bool>();
  size_t type_size = data_obj.get("TypeSize").get_value<int>();
  std::string val_type = data_obj.get("ValType").get_value<std::string>();
  size_t val_size = data_obj.get("ValSize").get_value<int>();
  std::string location = data_obj.get("Location").get_value<std::string>();

  std::pair<std::string, std::shared_ptr<Schema>> schema_data_pair =
      create_schema_var(var_name, type_name, type_size, local_type,
                        is_basic_type, getValType(val_type), val_size,
                        getMemLoc(location));
  return schema_data_pair.second;
}

inline std::shared_ptr<Schema> gen_schema(dpct_json::value &value) {
  auto json_obj = value.get_value<dpct_json::object>();
  const schema_type schema_type =
      get_schema_type(json_obj.get("SchemaType").get_value<std::string>());
  if (schema_type::TYPE == schema_type) {
    return gen_type_schema(value);
  } else if (schema_type::DATA == schema_type) {
    return gen_data_schema(value);
  }
  return nullptr;
}

inline bool dpct_json::parse(const std::string &json, dpct_json::value &v) {
  dpct_json::json_parser parse(json);
  if (parse.parse_value(v))
    return true;
  error_exit("The Pass file not success.");
}

inline std::shared_ptr<Schema> parse_var_schema_str(const std::string &str) {
  dpct_json::value v(nullptr);
  dpct_json::parse(str, v);
  if (v.real_type == dpct_json::value::object_t) {
    return gen_schema(v);
  }
  return nullptr;
}

inline void parse_type_schema_str(const std::string &str) {
  dpct_json::value v(nullptr);
  dpct_json::parse(str, v);
  if (v.real_type == dpct_json::value::array_t) {
    dpct_json::array arr = v.get_value<dpct_json::array>();
    for (auto iter = arr.begin(); iter != arr.end(); iter++) {
      dpct_json::value &cur_val = *iter;
      if (cur_val.real_type ==
          dpct_json::value::object_t) {
        std::shared_ptr<Schema> type_schema = gen_schema(cur_val);
        if (type_schema != nullptr) {
          schema_map[type_schema->get_type_name()] = type_schema;
        }
      }
    }
    return;
  }
  error_exit("The type schema must be the array type.")
}

inline void get_data_as_hex(const void *data, size_t data_size,
                            std::string &hex_str) {

  std::stringstream ss("");
  const char *byte_data = static_cast<const char *>(data);
  for (size_t i = 0; i < data_size; i++) {
    ss << std::hex << std::setw(2) << std::setfill('0')
       << static_cast<unsigned int>(static_cast<unsigned char>(byte_data[i]));
    ss << ", ";
    if (i % 30 == 0 && i != 0) {
      ss << std::endl;
    }
  }
  hex_str = ss.str();
  hex_str.erase(hex_str.rfind(','));
}

void copy_mem_to_device(void *dst, void *src, size_t size) {
// To do: To enable clang++ compiler supported CUDA code.
#ifdef __NVCC__
  cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
#else
  dpct::get_default_queue().memcpy(dst, src, size).wait();
#endif
}

inline bool is_device_point(void *p) {
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

inline void get_val_from_addr(std::string &value, std::shared_ptr<Schema> schema, void *addr,
                                     size_t size) {
  void *h_addr = addr;
  if (is_device_point(addr)) {
    h_addr = malloc(size);
    copy_mem_to_device(h_addr, addr, size);
  }
  if (schema->is_basic_type()) {
    value += schema->get_var_name() + ": ";
    std::string hex_str = "";
    get_data_as_hex(h_addr, size, hex_str);
    value += hex_str + "\n";
    if (is_device_point(addr))
      free(h_addr);
    return;
  }
  std::shared_ptr<Schema> type_schema = schema_map[schema->get_type_name()];

  std::vector<std::shared_ptr<Schema>> type_members =
      type_schema->get_type_member();
  value += "The data of " + schema->get_var_name() + " is:\n";
  for (auto member : type_members) {
    value += member->get_var_name() + ": ";
    std::string hex_str = "";
    char *addr_with_offset = (char *)h_addr + member->get_offset();
    if (member->is_basic_type()) {
      get_data_as_hex((void *)addr_with_offset, member->get_val_size(),
                      hex_str);
    } else {
      get_val_from_addr(hex_str, member, (void *)addr_with_offset, size);
    }
    value += hex_str + " \n";
  }
  if (is_device_point(addr))
      free(h_addr);
}

inline void process_var(std::string &log) { log = ""; }

template <class... Args>
void process_var(std::string &log, const std::string &schema_str, long value, size_t size,
                        Args... args) {
  std::shared_ptr<Schema> schema = parse_var_schema_str(schema_str);
  if (schema == nullptr) {
    error_exit("Cannot parse the variable schema, please double check.\n");
  }
  if (size == 0) {
    size = schema->get_type_size();
  }
  set_size_of_schema(schema_str, size);
  switch (schema->get_val_type()) {
  case ValType::SCALAR:
    get_val_from_addr(log, schema, (void *)&value, size);
    break;
  case ValType::ARRAY:
  case ValType::POINTER:
    get_val_from_addr(log, schema, (void *)value, size);
    break;
  case ValType::POINTERTOPOINTER:
    get_val_from_addr(log, schema, *(void **)value, size);
    break;
  };
  std::string ret;
  process_var(ret, args...);
  log += ret;
}

static std::map<std::string, int> api_index;

template <class... Args>
void gen_log_API_CP(const std::string &api_name, Args... args) {
  if (api_index.find(api_name) == api_index.end()) {
    api_index[api_name] = 0;
  } else {
    api_index[api_name]++;
  }
  std::string new_api_name = api_name + std::to_string(api_index[api_name]);
  std::string log;
  process_var(log, args...);
  std::cout << "API name: " << new_api_name << " \nData Memory: \n"
            << log << std::endl;
}

} // namespace experimental
} // namespace dpct
#endif // End of __DPCT_SCHEMA_HELPER__
