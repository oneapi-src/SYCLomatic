#include <iostream>
#include "../../debug_helper.hpp"
static std::string type_schema_array=
"[ {"
    "\"FieldNum\": 2,"
    "\"FilePath\": \"Invalid Path\","
      "\"IsVirtual\": false,"
      "\"Members\": ["
      "{"
      "\"IsBasicType\": true,"
      "\"Location\": \"None\","
      "\"VarName\": \"a\","
      "\"Offset\": 0,"
      "\"TypeName\": \"int\","
      "\"ValType\": \"Scalar\","
      "\"ValSize\": 4,"
      "\"TypeSize\": 4"
      "},"
      "{"
      "\"IsBasicType\": true,"
      "\"Location\": \"None\","
      "\"VarName\": \"b\","
      "\"Offset\": 4,"
      "\"TypeName\": \"int\","
      "\"ValType\": \"Scalar\","
      "\"ValSize\": 4,"
      "\"TypeSize\": 4"
      "}"
      "],"
      "\"SchemaType\": \"TYPE\","
      "\"TypeName\": \"TestClass\","
      "\"TypeSize\": 8"
      "}"
      "]";

class Init {
public:
  Init() {
    dpct::experimental::parse_type_schema_str(type_schema_array);
  }
};
static Init init;

#define TYPE_SHCEMA_002                                                        \
  " {  \"VarName\": \"h_A\","                                                  \
  "\"TypeName\": \"TestClass\","                                                   \
  "\"TypeSize\": 8,"                                                           \
  "\"SchemaType\": \"DATA\","                                                  \
  "\"IsBasicType\": false,"                                                     \
  "\"ValType\": \"pointer\","                                                  \
  "\"ValSize\": 8,"                                                            \
  "\"Location\": \"None\""                                                     \
  "}"

#define TYPE_SHCEMA_003                                                        \
  " {  \"VarName\": \"h_B\","                                                  \
  "\"TypeName\": \"TestClass\","                                                   \
  "\"TypeSize\": 8,"                                                           \
  "\"SchemaType\": \"DATA\","                                                  \
  "\"IsBasicType\": false,"                                                     \
  "\"ValType\": \"pointer\","                                                  \
  "\"ValSize\": 8,"                                                            \
  "\"Location\": \"None\""                                                     \
  "}"

#define TYPE_SHCEMA_004                                                        \
  " {  \"VarName\": \"h_C\","                                                  \
  "\"TypeName\": \"TestClass\","                                                   \
  "\"TypeSize\": 8,"                                                           \
  "\"SchemaType\": \"DATA\","                                                  \
  "\"IsBasicType\": false,"                                                     \
  "\"ValType\": \"pointer\","                                                  \
  "\"ValSize\": 8,"                                                            \
  "\"Location\": \"None\""                                                     \
  "}"

#define TYPE_SHCEMA_005                                                       \
  " {  \"VarName\": \"d_A\","                                                  \
  "\"TypeName\": \"TestClass\","                                                   \
  "\"TypeSize\": 8,"                                                           \
  "\"SchemaType\": \"DATA\","                                                  \
  "\"IsBasicType\": false,"                                                     \
  "\"ValType\": \"pointer\","                                                  \
  "\"ValSize\": 8,"                                                            \
  "\"Location\": \"None\""                                                     \
  "}"

#define TYPE_SHCEMA_006                                                        \
  " {  \"VarName\": \"d_B\","                                                  \
  "\"TypeName\": \"TestClass\","                                               \
  "\"TypeSize\": 8,"                                                           \
  "\"SchemaType\": \"DATA\","                                                  \
  "\"IsBasicType\": false,"                                                     \
  "\"ValType\": \"pointer\","                                                  \
  "\"ValSize\": 8,"                                                            \
  "\"Location\": \"None\""                                                     \
  "}"

#define TYPE_SHCEMA_007                                                        \
  " {  \"VarName\": \"d_C\","                                                  \
  "\"TypeName\": \"TestClass\","                                                   \
  "\"TypeSize\": 8,"                                                           \
  "\"SchemaType\": \"DATA\","                                                  \
  "\"IsBasicType\": false,"                                                     \
  "\"ValType\": \"pointer\","                                                  \
  "\"ValSize\": 8,"                                                            \
  "\"Location\": \"None\""                                                     \
  "}"

#define TYPE_SHCEMA_008                                                        \
  " {  \"VarName\": \"numElements_schema\","                                   \
  "\"TypeName\": \"int\","                                                     \
  "\"TypeSize\": 4,"                                                           \
  "\"SchemaType\": \"DATA\","                                                  \
  "\"IsBasicType\": true,"                                                    \
  "\"ValType\": \"scalar\","                                                   \
  "\"ValSize\": 4,"                                                            \
  "\"Location\": \"None\""                                                     \
  "}"
