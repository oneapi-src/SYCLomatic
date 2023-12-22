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

inline const char *getValTypeStr(ValType tt) {
  const char *VTStr[] = {"Scalar", "Array", "Pointer", "PointerOfPointer"};
  return VTStr[tt];
}

inline std::ostream &operator<<(std::ostream &strm, ValType tt) {
  return strm << getValTypeStr(tt);
}

struct FieldSchema {
  ValType ValTypeOfField;
  bool IsBasicType = false;
  int64_t ValSize = 0;
  int64_t TypeSize = 0;
  int64_t Offset = 0;
  std::string FieldName;
  std::string FieldType;
  std::string Location = "None";
  FieldSchema(const std::string &name = "", ValType VT = ScalarValue,
              const std::string &FT = "", bool IBT = false, int64_t VS = 0,
              int64_t TS = 0, int64_t OS = 0, const std::string &loc = "None")
      : ValTypeOfField(VT), IsBasicType(IBT), ValSize(VS), TypeSize(TS),
        Offset(OS), FieldName(name), FieldType(FT), Location(loc) {}
};

struct TypeSchema {
  bool IsVirtual = false;
  int FieldNum = 0;
  int TypeAlign = 1;
  int64_t TypeSize = 0;
  std::string TypeName;
  std::string FileName;
  std::vector<FieldSchema> Members;
  TypeSchema(const std::string &name = "", int FN = 0, int TA = 1,
             int64_t TS = 0, bool IV = false, const std::string &FP = "",
             std::vector<FieldSchema> mem = std::vector<FieldSchema>())
      : IsVirtual(IV), FieldNum(FN), TypeAlign(TA), TypeSize(TS),
        TypeName(name), FileName(FP), Members(mem) {}
};

struct VarSchema {
  bool IsBasicType = false;
  ValType ValTypeOfVar;
  int64_t TypeSize = 0;
  int64_t VarSize = 0;
  std::string VarName;
  std::string FileName;
  std::string VarType;
  std::string Location = "None";
  VarSchema(const std::string &name = "", ValType VT = ScalarValue,
            const std::string &FP = "", const std::string &FT = "",
            bool IBT = false, int64_t VS = 0, int64_t OS = 0,
            const std::string &loc = "None")
      : IsBasicType(IBT), ValTypeOfVar(VT), VarSize(VS), VarName(name),
        FileName(FP), VarType(FT), Location(loc) {}
};

std::string getFilePathFromDecl(const Decl *D, const SourceManager &SM);

ValType getValType(const clang::QualType &QT);

TypeSchema constructTypeSchema(const clang::RecordType *RT);

TypeSchema registerTypeSchema(const clang::QualType &QT);

TypeSchema registerSYCLTypeSchema(const clang::QualType &QT);

VarSchema constructVarSchema(const clang::DeclRefExpr *DRE);

VarSchema constructSyclVarSchema(const VarSchema &CVS);

extern std::map<std::string, TypeSchema> CTypeSchemaMap;

extern std::map<std::string, TypeSchema> STypeSchemaMap;

llvm::json::Array
serializeSchemaToJsonArray(const std::map<std::string, TypeSchema> &TSMap);

llvm::json::Array
serializeSchemaToJsonArray(const std::vector<TypeSchema> &TSVec);

llvm::json::Object serializeTypeSchemaToJson(const TypeSchema &TS);

llvm::json::Object serializeVarSchemaToJson(const VarSchema &VS);

std::vector<TypeSchema>
getRelatedTypeSchema(const std::string &TypeName,
                     const std::map<std::string, TypeSchema> &TypeSchemaMap);

void setSTypeSchemaMap();

inline std::string jsonToString(llvm::json::Array Arr) {
  std::string Str;
  llvm::raw_string_ostream OS(Str);
  llvm::json::OStream(OS).value(std::move(Arr));
  std::size_t found = Str.rfind("\"");
  while (found != std::string::npos) {
    Str.insert(found, "\\");
    found = Str.rfind("\"", found);
  }
  return Str;
}

inline std::string jsonToString(llvm::json::Object Obj) {
  std::string Str;
  llvm::raw_string_ostream OS(Str);
  llvm::json::OStream(OS).value(std::move(Obj));
  std::size_t found = Str.rfind("\"");
  while (found != std::string::npos) {
    Str.insert(found, "\\");
    found = Str.rfind("\"", found);
  }
  return Str;
}
} // namespace dpct
} // namespace clang

#endif
