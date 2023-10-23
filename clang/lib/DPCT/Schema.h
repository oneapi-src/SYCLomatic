//===--------------------------- Schema.h ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef DPCT_SCHEMA_H
#define DPCT_SCHEMA_H

#include "AnalysisInfo.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <clang/AST/ASTContext.h>
#include <clang/AST/Decl.h>
#include <clang/AST/DeclCXX.h>
#include <clang/Basic/LLVM.h>
#include <iostream>
#include <string>
#include <vector>

namespace clang {
namespace dpct {

enum ValType : int {
  ScalarValue = 0,
  ArrayValue,
  Pointer,
  PointerOfPointer,
};

inline std::ostream &operator<<(std::ostream &strm, ValType tt) {
  const std::string VTStr[] = {"ScalarValue", "ArrayValue", "Pointer",
                               "PointerOfPointer"};
  return strm << VTStr[tt];
}

inline std::string getValTypeStr(ValType tt) {
  const std::string VTStr[] = {"ScalarValue", "ArrayValue", "Pointer",
                               "PointerOfPointer"};
  return VTStr[tt];
}

struct FieldSchema {
  std::string FieldName;
  ValType ValTypeOfField;
  std::string FieldType;
  bool IsBasicType = false;
  int64_t ValSize = 0;
  int64_t Offset = 0;
  std::string Location = "None";
};

struct TypeSchema {
  std::string TypeName;
  int FieldNum = 0;
  int64_t TypeSize = 0;
  bool IsVirtual = false;
  std::string FileName;
  std::vector<FieldSchema> Members;
};

struct VarSchema {
  std::string VarName;
  ValType ValTypeOfVar;
  std::string FileName;
  std::string VarType;
  bool IsBasicType = false;
  int64_t VarSize = 0;
  std::string Location = "None";
};

std::string getFilePathFromDecl(const Decl *D, const SourceManager &SM);

inline void printSchema(const FieldSchema &FS) {
  std::cout << "FieldName: " << FS.FieldName << '\n';
  std::cout << "FieldType: " << FS.FieldType << '\n';
  std::cout << "Memory Size Of Field: " << FS.ValSize << '\n';
  std::cout << "FieldOffset: " << FS.Offset << '\n';
  std::cout << "IsBasicType: " << std::boolalpha << FS.IsBasicType << '\n';
  std::cout << "ValTypeOfField: " << FS.ValTypeOfField << '\n';
  std::cout << std::endl;
}

inline void printSchema(const TypeSchema &TS) {
  std::cout << "TypeName: " << TS.TypeName << '\n';
  std::cout << "FieldNum: " << TS.FieldNum << '\n';
  std::cout << "TypeSize" << TS.TypeSize << '\n';
  std::cout << "IsBuiltInType: " << std::boolalpha << TS.IsVirtual << '\n';
  std::cout << "File name: " << TS.FileName << '\n';
  std::cout << "\n";
  for (auto it : TS.Members) {
    printSchema(it);
  }
  std::cout << "----------------------------------------------------\n\n";
}

inline void printSchema(const VarSchema &VS) {
  std::cout << "VarName: " << VS.VarName << '\n';
  std::cout << "VarType: " << VS.VarType << '\n';
  std::cout << "VarSize: " << VS.VarSize << '\n';
  std::cout << "File name: " << VS.FileName << '\n';
  std::cout << "IsBasicType: " << std::boolalpha << VS.IsBasicType << '\n';
  std::cout << "ValTypeOfVar: " << VS.ValTypeOfVar << '\n';
  std::cout << std::endl;
}

ValType getValType(const clang::QualType &QT);

FieldSchema constructFieldSchema(const clang::FieldDecl *FD,
                                 std::string ClassTypeName);

inline FieldSchema constructFieldSchema(const clang::FieldDecl *FD);

void DFSBaseClass(clang::CXXRecordDecl *RD, TypeSchema &TS);

TypeSchema constructTypeSchema(const clang::RecordType *RT);

void registerTypeSchema(const clang::QualType QT);

VarSchema constructVarSchema(const clang::DeclRefExpr *DRE);

extern std::map<std::string, TypeSchema> TypeSchemaMap;

llvm::json::Array
serializeSchemaMapToJson(const std::map<std::string, TypeSchema> &TSMap);

llvm::json::Object serializeSchemaToJson(const TypeSchema &TS);

void serializeSchemaMapToFile(const std::map<std::string, TypeSchema> &TSMap,
                              const std::string &FilePath);

} // namespace dpct
} // namespace clang

#endif
