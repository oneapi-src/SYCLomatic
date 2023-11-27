//===--------------------------- Schema.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
#include "Schema.h"
#include <deque>

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
  if (!FS.IsBasicType)
    registerTypeSchema(OriginT);
  return FS;
}

void DFSBaseClass(clang::CXXRecordDecl *RD, TypeSchema &TS, int BaseOffset) {
  if (RD == nullptr)
    return;
  for (const auto &base : RD->bases()) {
    if (base.isVirtual())
      TS.IsVirtual = true;
    clang::CXXRecordDecl *BCD = base.getType()->getAsCXXRecordDecl();
    if (BCD) {
      auto TsTmp = registerTypeSchema(base.getType());
      for (auto mem : TsTmp.Members) {
        mem.Offset += BaseOffset;
        mem.FieldName = base.getType().getAsString() + "::" + mem.FieldName;
        TS.Members.emplace_back(mem);
      }
      BaseOffset += TsTmp.TypeSize;
    }
  }
  return;
}

int DFSBaseClassForSYCL(clang::CXXRecordDecl *RD, TypeSchema &TS,
                        int BaseOffset) {
  if (RD == nullptr)
    return BaseOffset;
  for (const auto &base : RD->bases()) {
    if (base.isVirtual())
      TS.IsVirtual = true;
    clang::CXXRecordDecl *BCD = base.getType()->getAsCXXRecordDecl();
    if (BCD) {
      auto TsTmp = registerSYCLTypeSchema(base.getType());
      for (auto mem : TsTmp.Members) {
        mem.Offset += BaseOffset;
        mem.FieldName = base.getType().getAsString() + "::" + mem.FieldName;
        TS.Members.emplace_back(mem);
      }
      BaseOffset += TsTmp.TypeSize;
    }
  }
  return BaseOffset;
}

FieldSchema convertCFieldSchemaToSFieldSChema(const FieldSchema &CFS,
                                              int BaseOF, int OffSet) {
  FieldSchema SFS = CFS;
  if (MapNames::TypeNamesMap.find(CFS.FieldType) !=
      MapNames::TypeNamesMap.end()) {
    SFS.FieldType = MapNames::TypeNamesMap[CFS.FieldType]->NewName;
  }
  if (STypeSchemaMap.find(SFS.FieldType) != STypeSchemaMap.end()) {
    SFS.TypeSize = STypeSchemaMap[SFS.FieldType].TypeSize;
    SFS.Offset = CFS.Offset - BaseOF + OffSet;
    if (SFS.ValTypeOfField == ValType::ArrayValue) {
      SFS.ValSize = CFS.ValSize / CFS.TypeSize * SFS.TypeSize;
    } else if (SFS.ValTypeOfField == ValType::ScalarValue) {
      SFS.ValSize = CFS.TypeSize;
    }
  }
  return SFS;
}

TypeSchema dpct::registerSYCLTypeSchema(const clang::QualType &QT, int OffSet) {
  const std::string &KetStr = DpctGlobalInfo::getTypeName(QT);
  if (STypeSchemaMap.find(KetStr) != STypeSchemaMap.end())
    return STypeSchemaMap[KetStr];
  if (MapNames::TypeNamesMap.find(KetStr) != MapNames::TypeNamesMap.end())
    return STypeSchemaMap[MapNames::TypeNamesMap[KetStr]->NewName];
  TypeSchema TS;
  TS.TypeName = KetStr;
  clang::ASTContext &AstContext = dpct::DpctGlobalInfo::getContext();
  if (const clang::RecordType *RT = QT->getAs<clang::RecordType>()) {
    TS.FileName =
        getFilePathFromDecl(RT->getDecl(), DpctGlobalInfo::getSourceManager());
    if (isa<clang::CXXRecordDecl>(RT->getDecl())) {
      clang::CXXRecordDecl *RD = RT->getAsCXXRecordDecl();
      // visit base class
      OffSet = DFSBaseClassForSYCL(RD, TS, OffSet);
    }
    int BaseOF = RT->getDecl()->field_empty()
                     ? 0
                     : AstContext.getFieldOffset(*RT->getDecl()->field_begin());
    for (const clang::FieldDecl *field : RT->getDecl()->fields()) {
      TS.Members.push_back(convertCFieldSchemaToSFieldSChema(
          constructFieldSchema(field), BaseOF, OffSet));
    }
    TS.FieldNum = TS.Members.size();
    TS.TypeSize = TS.FieldNum == 0
                      ? 0
                      : (TS.Members.back().Offset + TS.Members.back().ValSize);
  }
  STypeSchemaMap[KetStr] = TS;
  return TS;
}

TypeSchema dpct::registerTypeSchema(const clang::QualType &QT, int OffSet) {
  if (QT.isNull())
    return TypeSchema();
  const std::string &KetStr = DpctGlobalInfo::getTypeName(QT);
  if (CTypeSchemaMap.find(KetStr) != CTypeSchemaMap.end())
    return CTypeSchemaMap[KetStr];
  clang::ASTContext &AstContext = dpct::DpctGlobalInfo::getContext();
  TypeSchema TS;
  TS.TypeName = KetStr;
  TS.TypeSize = AstContext.getTypeSize(QT) / CHAR_BIT;
  if (const clang::RecordType *RT = QT->getAs<clang::RecordType>()) {
    TS.FileName =
        getFilePathFromDecl(RT->getDecl(), DpctGlobalInfo::getSourceManager());
    if (isa<clang::CXXRecordDecl>(RT->getDecl())) {
      clang::CXXRecordDecl *RD = RT->getAsCXXRecordDecl();
      // visit base class
      DFSBaseClass(RD, TS, OffSet);
    }
    for (const clang::FieldDecl *field : RT->getDecl()->fields()) {
      TS.Members.push_back(constructFieldSchema(field));
    }
    TS.FieldNum = TS.Members.size();
  }
  CTypeSchemaMap[KetStr] = TS;
  registerSYCLTypeSchema(QT);
  return TS;
}

VarSchema dpct::constructVarSchema(const clang::DeclRefExpr *DRE) {
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
    registerTypeSchema(getDerefType(DRE->getType()));
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

std::map<std::string, TypeSchema> clang::dpct::CTypeSchemaMap;

std::map<std::string, TypeSchema> clang::dpct::STypeSchemaMap;

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

// For test
//  we need remove this function
void dpct::serializeJsonArrayToFile(llvm::json::Array &&Arr,
                                    const std::string &FilePath) {
  std::error_code EC;
  llvm::raw_fd_ostream FOut(FilePath, EC);
  if (!EC) {
    llvm::json::OStream(FOut, 0).value(std::move(Arr));
    // FOut << llvm::json:: ;
    FOut.close();
  } else {
    llvm::errs() << "Error opening file: " << EC.message() << "\n";
    return;
  }
}

std::vector<TypeSchema> dpct::getRelatedTypeSchema(
    const std::string &TypeName,
    const std::map<std::string, TypeSchema> &TypeSchemaMap) {
  if (TypeSchemaMap.find(TypeName) == TypeSchemaMap.end())
    return std::vector<TypeSchema>();
  TypeSchema TS = TypeSchemaMap.find(TypeName)->second;
  std::deque<TypeSchema> Q;
  std::vector<TypeSchema> Res;
  Q.push_back(TS);
  while (!Q.empty()) {
    TS = Q.front();
    Res.push_back(TS);
    Q.pop_front();
    for (auto &it : TS.Members) {
      if (TypeSchemaMap.find(it.FieldType) != TypeSchemaMap.end())
        Q.push_back(TypeSchemaMap.find(it.FieldType)->second);
    }
  }
  return Res;
}

llvm::json::Array
dpct::serializeSchemaToJsonArray(const std::vector<TypeSchema> &SVec) {
  llvm::json::Array JArray;
  for (auto &it : SVec) {
    JArray.push_back(serializeTypeSchemaToJson(it));
  }
  return JArray;
}
