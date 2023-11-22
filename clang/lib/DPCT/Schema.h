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
  int64_t TypeSize = 0;
  int64_t Offset = 0;
  std::string Location = "None";
  FieldSchema(const std::string &name = "", ValType VT = ScalarValue,
              const std::string &FT = "", bool IBT = false, int64_t VS = 0,
              int64_t OS = 0, const std::string &loc = "None")
      : FieldName(name), ValTypeOfField(VT), FieldType(FT), IsBasicType(IBT),
        ValSize(VS), Offset(OS), Location(loc) {}
};

struct TypeSchema {
  std::string TypeName;
  int FieldNum = 0;
  int64_t TypeSize = 0;
  bool IsVirtual = false;
  std::string FileName;
  std::vector<FieldSchema> Members;
  TypeSchema(const std::string &name = "", int FN = 0, int64_t TS = 0,
             bool IV = false, const std::string &FP = "",
             std::vector<FieldSchema> mem = std::vector<FieldSchema>())
      : TypeName(name), FieldNum(FN), TypeSize(TS), IsVirtual(IV), FileName(FP),
        Members(mem) {}
};

struct VarSchema {
  std::string VarName;
  ValType ValTypeOfVar;
  std::string FileName;
  std::string VarType;
  bool IsBasicType = false;
  int64_t TypeSize = 0;
  int64_t VarSize = 0;
  std::string Location = "None";
  VarSchema(const std::string &name = "", ValType VT = ScalarValue,
            const std::string &FP = "", const std::string &FT = "",
            bool IBT = false, int64_t VS = 0, int64_t OS = 0,
            const std::string &loc = "None")
      : VarName(name), ValTypeOfVar(VT), FileName(FP), VarType(FT),
        IsBasicType(IBT), VarSize(VS), Location(loc) {}
};

std::string getFilePathFromDecl(const Decl *D, const SourceManager &SM);

ValType getValType(const clang::QualType &QT);

FieldSchema constructFieldSchema(const clang::FieldDecl *FD,
                                 std::string ClassTypeName);

inline FieldSchema constructFieldSchema(const clang::FieldDecl *FD);

void DFSBaseClass(clang::CXXRecordDecl *RD, TypeSchema &TS);

TypeSchema constructTypeSchema(const clang::RecordType *RT);

TypeSchema registerTypeSchema(const clang::QualType &QT);

VarSchema constructVarSchema(const clang::DeclRefExpr *DRE);

extern std::map<std::string, TypeSchema> TypeSchemaMap;

llvm::json::Array
serializeSchemaToJsonArray(const std::map<std::string, TypeSchema> &TSMap);

llvm::json::Array
serializeSchemaToJsonArray(const std::vector<TypeSchema> &TSVec);

llvm::json::Object serializeTypeSchemaToJson(const TypeSchema &TS);

llvm::json::Object serializeVarSchemaToJson(const VarSchema &VS);

void serializeJsonArrayToFile(llvm::json::Array &&Arr,
                              const std::string &FilePath);

std::vector<TypeSchema> getRelatedTypeSchema(const clang::QualType QT);

void setTypeSchemaMap();

inline std::string jsonToString(llvm::json::Array Arr){
  std::string Str;
  llvm::raw_string_ostream  OS(Str);
  llvm::json::OStream(OS).value(std::move(Arr));
  return OS.str();
}

inline std::string jsonToString(llvm::json::Object Obj){
  std::string Str;
  llvm::raw_string_ostream  OS(Str);
  llvm::json::OStream(OS).value(std::move(Obj));
  return OS.str();
}
} // namespace dpct
} // namespace clang

#endif
