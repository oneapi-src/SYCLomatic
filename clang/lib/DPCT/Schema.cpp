//===--------------------------- Schema.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
#include "Schema.h"
#include <cstddef>
#include <deque>

namespace clang {
namespace dpct {
// Key: CUDA type name
// Value: The corresponding schema of the type
std::map<std::string, TypeSchema> CTypeSchemaMap;
// Key: SYCL type name
// Value: The corresponding schema of the type
std::map<std::string, TypeSchema> STypeSchemaMap;
} // namespace dpct
} // namespace clang

using namespace clang;
using namespace clang::dpct;

QualType getDerefType(const QualType &QT) {
  if (QT->isPointerType()) {
    return getDerefType(QT->getPointeeType());
  } else if (QT->isArrayType()) {
    return getDerefType(dyn_cast<ArrayType>(QT.getTypePtr())->getElementType());
  }
  return QT;
}

std::string dpct::getFilePathFromDecl(const Decl *D, const SourceManager &SM) {
  SourceLocation Loc = D->getLocation();
  if (Loc.isValid()) {
    const FileEntry *fileEntry = SM.getFileEntryForID(SM.getFileID(Loc));
    if (fileEntry) {
      return fileEntry->tryGetRealPathName().str();
    }
  }
  return "Invalid Path";
}

ValType dpct::getValType(const clang::QualType &QT) {
  if (QT->isArrayType()) {
    return ValType::ArrayValue;
  } else if (QT->isPointerType()) {
    if (QT->getPointeeType()->isPointerType()) {
      return ValType::PointerOfPointer;
    }
    return ValType::Pointer;
  } else {
    return ValType::ScalarValue;
  }
}

FieldSchema constructFieldSchema(const clang::FieldDecl *FD) {
  clang::ASTContext &AstContext = dpct::DpctGlobalInfo::getContext();
  FieldSchema FS;
  const clang::QualType FT = FD->getType().getCanonicalType();
  const clang::QualType OriginT = getDerefType(FT);
  FS.FieldName = FD->getNameAsString();
  FS.FieldType = DpctGlobalInfo::getTypeName(OriginT);
  FS.IsBasicType = OriginT->isBuiltinType();
  FS.ValTypeOfField = getValType(FT);
  FS.ValSize = AstContext.getTypeSize(FT) / CHAR_BIT;
  FS.Offset = AstContext.getFieldOffset(FD) / CHAR_BIT;
  FS.TypeSize = AstContext.getTypeSize(OriginT) / CHAR_BIT;
  if (auto It = CTypeSchemaMap.find(DpctGlobalInfo::getTypeName(OriginT));
      It != CTypeSchemaMap.end() && !FS.IsBasicType)
    registerCUDATypeSchema(OriginT);
  return FS;
}

uint32_t getFieldTypeAlign(const FieldSchema &FS) {
  if (FS.ValTypeOfField == ValType::Pointer ||
      FS.ValTypeOfField == ValType::PointerOfPointer) {
    return sizeof(std::ptrdiff_t);
  } else if (FS.IsBasicType) {
    return FS.TypeSize;
  } else {
    return STypeSchemaMap[FS.FieldType].TypeAlign;
  }
}

int ceilDevide(int num, int base) { return ((num + base - 1) / base) * base; }

void registerBaseClassForCUDA(clang::CXXRecordDecl *RD, TypeSchema &TS) {
  if (RD == nullptr)
    return;
  int BaseOffset = 0;
  for (const auto &base : RD->bases()) {
    if (base.isVirtual())
      TS.IsVirtual = true;
    clang::CXXRecordDecl *BCD = base.getType()->getAsCXXRecordDecl();
    if (BCD) {
      auto TsTmp = registerCUDATypeSchema(base.getType());
      assert(TsTmp.TypeAlign != 0 && "The type alignment should not be 0.");
      BaseOffset = ceilDevide(BaseOffset, TsTmp.TypeAlign);
      for (auto &mem : TsTmp.Members) {
        mem.Offset += BaseOffset;
        mem.FieldName = base.getType().getAsString() + "::" + mem.FieldName;
        TS.Members.emplace_back(mem);
      }
      BaseOffset += TsTmp.TypeSize;
    }
  }
  return;
}

int registerBaseClassForSYCL(clang::CXXRecordDecl *RD, TypeSchema &TS) {
  int BaseOffset = 0;
  if (RD == nullptr)
    return BaseOffset;
  for (const auto &base : RD->bases()) {
    if (base.isVirtual())
      TS.IsVirtual = true;
    clang::CXXRecordDecl *BCD = base.getType()->getAsCXXRecordDecl();
    if (BCD) {
      auto TsTmp = registerSYCLTypeSchema(base.getType());
      BaseOffset = ceilDevide(BaseOffset, TsTmp.TypeAlign);
      for (auto mem : TsTmp.Members) {
        mem.Offset += BaseOffset;
        mem.FieldName = base.getType().getAsString() + "::" + mem.FieldName;
        TS.Members.emplace_back(mem);
        if (TsTmp.TypeAlign > TS.TypeAlign)
          TS.TypeAlign = TsTmp.TypeAlign;
      }
      BaseOffset += TsTmp.TypeSize;
    }
  }
  return BaseOffset;
}

FieldSchema convertCFieldSchemaToSFieldSChema(const FieldSchema &CFS) {
  FieldSchema SFS = CFS;
  if (MapNames::TypeNamesMap.find(CFS.FieldType) !=
      MapNames::TypeNamesMap.end()) {
    SFS.FieldType = MapNames::TypeNamesMap[CFS.FieldType]->NewName;
  }
  if (STypeSchemaMap.find(SFS.FieldType) != STypeSchemaMap.end()) {
    SFS.TypeSize = STypeSchemaMap[SFS.FieldType].TypeSize;
    if (SFS.ValTypeOfField == ValType::ArrayValue) {
      SFS.ValSize = CFS.ValSize / CFS.TypeSize * SFS.TypeSize;
    } else if (SFS.ValTypeOfField == ValType::ScalarValue) {
      SFS.ValSize = CFS.TypeSize;
    }
  }
  return SFS;
}

TypeSchema dpct::registerSYCLTypeSchema(const clang::QualType &QT) {
  const std::string &KeyStr = DpctGlobalInfo::getTypeName(QT);
  if (auto It = STypeSchemaMap.find(KeyStr); It != STypeSchemaMap.end())
    return It->second;
  if (auto It = MapNames::TypeNamesMap.find(KeyStr);
      It != MapNames::TypeNamesMap.end())
    return STypeSchemaMap[It->second->NewName];
  TypeSchema TS;
  TS.TypeName = KeyStr;
  if (const clang::RecordType *RT = QT->getAs<clang::RecordType>()) {
    TS.FileName =
        getFilePathFromDecl(RT->getDecl(), DpctGlobalInfo::getSourceManager());
    int OffSet = 0;
    if (isa<clang::CXXRecordDecl>(RT->getDecl())) {
      clang::CXXRecordDecl *RD = RT->getAsCXXRecordDecl();
      // visit base class
       OffSet = registerBaseClassForSYCL(RD, TS);
    }
    
    for (const clang::FieldDecl *field : RT->getDecl()->fields()) {
      auto FsTmp =
          convertCFieldSchemaToSFieldSChema(constructFieldSchema(field));
      // Update Field Offset
      int CurAlign = getFieldTypeAlign(FsTmp);
      FsTmp.Offset = ceilDevide(OffSet, CurAlign);
      OffSet = FsTmp.Offset + FsTmp.ValSize;
      TS.Members.push_back(FsTmp);
      TS.TypeAlign = std::max(TS.TypeAlign, CurAlign);
    }
    TS.FieldNum = TS.Members.size();
    TS.TypeSize =
        TS.FieldNum == 0
            ? 0
            : ceilDevide((TS.Members.back().Offset + TS.Members.back().ValSize),
                         TS.TypeAlign);
  }
  STypeSchemaMap[KeyStr] = TS;
  return TS;
}

TypeSchema dpct::registerCUDATypeSchema(const clang::QualType &QT) {
  if (QT.isNull()||QT->isDependentType())
    return TypeSchema();
  const std::string &KeyStr = DpctGlobalInfo::getTypeName(QT);
  if (auto It = CTypeSchemaMap.find(KeyStr); It != CTypeSchemaMap.end())
    return It->second;
  clang::ASTContext &AstContext = dpct::DpctGlobalInfo::getContext();
  TypeSchema TS;
  TS.TypeName = KeyStr;
  TS.TypeSize = AstContext.getTypeSize(QT) / CHAR_BIT;
  if (const clang::RecordType *RT = QT->getAs<clang::RecordType>()) {
    TS.FileName =
        getFilePathFromDecl(RT->getDecl(), DpctGlobalInfo::getSourceManager());
    if (isa<clang::CXXRecordDecl>(RT->getDecl())) {
      clang::CXXRecordDecl *RD = RT->getAsCXXRecordDecl();
      // visit base class
      registerBaseClassForCUDA(RD, TS);
    }
    for (const clang::FieldDecl *field : RT->getDecl()->fields()) {
      TS.Members.push_back(constructFieldSchema(field));
    }
    TS.FieldNum = TS.Members.size();
    TS.TypeAlign = AstContext.getTypeAlign(QT) / CHAR_BIT;
  }
  CTypeSchemaMap[KeyStr] = TS;
  registerSYCLTypeSchema(QT);
  return TS;
}

VarSchema dpct::constructCUDAVarSchema(const clang::DeclRefExpr *DRE) {
  clang::ASTContext &AstContext = dpct::DpctGlobalInfo::getContext();
  VarSchema VA;
  VA.VarName = DRE->getNameInfo().getAsString();
  VA.IsBasicType =
      getDerefType(DRE->getType().getCanonicalType())->isBuiltinType();
  VA.ValTypeOfVar = getValType(DRE->getType());
  VA.FileName =
      getFilePathFromDecl(DRE->getDecl(), DpctGlobalInfo::getSourceManager());
  VA.VarSize = AstContext.getTypeSize(DRE->getType()) / CHAR_BIT;
  VA.VarType = DpctGlobalInfo::getTypeName(getDerefType(DRE->getType()));
  VA.TypeSize = AstContext.getTypeSize(getDerefType(DRE->getType())) / CHAR_BIT;
  if (!VA.IsBasicType) {
    registerCUDATypeSchema(getDerefType(DRE->getType()));
  }
  return VA;
}

VarSchema dpct::constructSyclVarSchema(const VarSchema &CVS) {
  VarSchema VA = CVS;
  // Is SYCL type
  if (MapNames::TypeNamesMap.find(CVS.VarType) !=
      MapNames::TypeNamesMap.end()) {
    VA.VarType = MapNames::TypeNamesMap[CVS.VarType]->NewName;
    // Can find SYCL Schema
    if (STypeSchemaMap.find(VA.VarType) != STypeSchemaMap.end()) {
      VA.TypeSize =
          STypeSchemaMap[MapNames::TypeNamesMap[CVS.VarType]->NewName].TypeSize;
      if (CVS.ValTypeOfVar == ValType::ArrayValue) {
        VA.VarSize = CVS.VarSize / CVS.TypeSize * VA.TypeSize;
      } else if (CVS.ValTypeOfVar == ValType::ScalarValue) {
        VA.VarSize = VA.TypeSize;
      }
    }
  }
  return VA;
}

llvm::json::Array dpct::serializeSchemaToJsonArray(
    const std::map<std::string, TypeSchema> &TSMap) {
  llvm::json::Array JArray;
  for (auto &it : TSMap) {
    JArray.push_back(serializeTypeSchemaToJson(it.second));
  }
  return JArray;
}

llvm::json::Object serializeFieldSchemaToJson(const FieldSchema &FS) {
  llvm::json::Object JObj;
  JObj.try_emplace("VarName", FS.FieldName);
  JObj.try_emplace("TypeName", FS.FieldType);
  JObj.try_emplace("IsBasicType", FS.IsBasicType);
  JObj.try_emplace("ValType", getValTypeStr(FS.ValTypeOfField));
  JObj.try_emplace("ValSize", FS.ValSize);
  JObj.try_emplace("Offset", FS.Offset);
  JObj.try_emplace("Location", FS.Location);
  JObj.try_emplace("TypeSize", FS.TypeSize);
  return JObj;
}

llvm::json::Object dpct::serializeTypeSchemaToJson(const TypeSchema &TS) {
  llvm::json::Object JObj;
  JObj.try_emplace("SchemaType", "TYPE");
  JObj.try_emplace("TypeName", TS.TypeName);
  JObj.try_emplace("FieldNum", TS.FieldNum);
  JObj.try_emplace("IsVirtual", TS.IsVirtual);
  JObj.try_emplace("TypeSize", TS.TypeSize);
  JObj.try_emplace("FilePath", TS.FileName);
  JObj.try_emplace("TypeAlign", TS.TypeAlign);
  if (!TS.Members.empty()) {
    llvm::json::Array JArray;
    for (const auto &it : TS.Members) {
      JArray.push_back(serializeFieldSchemaToJson(it));
    }
    JObj.try_emplace("Members", std::move(JArray));
  }

  return JObj;
}

llvm::json::Object dpct::serializeVarSchemaToJson(const VarSchema &VS) {
  llvm::json::Object JObj;
  JObj.try_emplace("SchemaType", "DATA");
  JObj.try_emplace("VarName", VS.VarName);
  JObj.try_emplace("TypeName", VS.VarType);
  JObj.try_emplace("IsBasicType", VS.IsBasicType);
  JObj.try_emplace("TypeSize", VS.TypeSize);
  JObj.try_emplace("ValSize", VS.VarSize);
  JObj.try_emplace("ValType", getValTypeStr(VS.ValTypeOfVar));
  JObj.try_emplace("FilePath", VS.FileName);
  JObj.try_emplace("Location", VS.Location);
  return JObj;
}

llvm::json::Array
dpct::serializeSchemaToJsonArray(const std::vector<TypeSchema> &SVec) {
  llvm::json::Array JArray;
  for (auto &it : SVec) {
    JArray.push_back(serializeTypeSchemaToJson(it));
  }
  return JArray;
}
