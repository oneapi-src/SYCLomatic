//===--------------------------- Schema.cpp --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "Schema.h"

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

inline FieldSchema dpct::constructFieldSchema(const clang::FieldDecl *FD) {
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
  if (!FS.IsBasicType)
    registerTypeSchema(OriginT);
  return FS;
}

inline FieldSchema dpct::constructFieldSchema(const clang::FieldDecl *FD,
                                              std::string ClassTypeName) {
  FieldSchema FS = constructFieldSchema(FD);
  FS.FieldName = ClassTypeName + "::" + FD->getNameAsString();
  return FS;
}

void dpct::DFSBaseClass(clang::CXXRecordDecl *RD, TypeSchema &TS) {
  if (RD == nullptr)
    return;
  for (auto base : RD->bases()) {
    clang::CXXRecordDecl *BCD = base.getType()->getAsCXXRecordDecl();
    if (base.isVirtual())
      TS.IsVirtual = true;
    if (BCD) {
      for (const clang::FieldDecl *field : BCD->fields()) {
        TS.Members.push_back(
            constructFieldSchema(field, BCD->getNameAsString()));
      }
      DFSBaseClass(BCD, TS);
    }
  }
  return;
}

void dpct::registerTypeSchema(const clang::QualType QT) {
  if (QT.isNull())
    return;
  const std::string &KetStr = DpctGlobalInfo::getTypeName(QT);
  if (TypeSchemaMap.find(KetStr) != TypeSchemaMap.end())
    return;
  clang::ASTContext &AstContext = dpct::DpctGlobalInfo::getContext();
  if (const clang::RecordType *RT = QT->getAs<clang::RecordType>()) {
    TypeSchema TS;
    TS.TypeName = KetStr;
    TS.TypeSize = AstContext.getTypeSize(RT) / CHAR_BIT;
    TS.FileName =
        getFilePathFromDecl(RT->getDecl(), DpctGlobalInfo::getSourceManager());
    for (const clang::FieldDecl *field : RT->getDecl()->fields()) {
      TS.Members.push_back(constructFieldSchema(field));
    }
    if (isa<clang::CXXRecordDecl>(RT->getDecl())) {
      clang::CXXRecordDecl *RD = RT->getAsCXXRecordDecl();
      // visit base class
      DFSBaseClass(RD, TS);
    }
    TS.FieldNum = TS.Members.size();
    TypeSchemaMap[KetStr] = TS;
  }
  return;
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
  if (!VA.IsBasicType) {
    registerTypeSchema(getDerefType(DRE->getType()));
  }
  return VA;
}

std::map<std::string, TypeSchema> clang::dpct::TypeSchemaMap;

llvm::json::Array
dpct::serializeSchemaMapToJson(const std::map<std::string, TypeSchema> &TSMap) {
  llvm::json::Array JArray;
  for (auto &it : TSMap) {
    JArray.push_back(serializeSchemaToJson(it.second));
  }
  return JArray;
}

llvm::json::Object serializeFieldSchemaToJson(const FieldSchema &FS) {
  llvm::json::Object JObj;
  JObj.try_emplace("Name", FS.FieldName);
  JObj.try_emplace("Type", FS.FieldType);
  JObj.try_emplace("IsBasicType", FS.IsBasicType);
  JObj.try_emplace("ValType", getValTypeStr(FS.ValTypeOfField));
  JObj.try_emplace("VarSize", FS.ValSize);
  JObj.try_emplace("Offset", FS.Offset);
  JObj.try_emplace("Location", FS.Location);
  return JObj;
}

llvm::json::Object dpct::serializeSchemaToJson(const TypeSchema &TS) {
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


//For test
// we need remove this function
void dpct::serializeSchemaMapToFile(
    const std::map<std::string, TypeSchema> &TSMap,
    const std::string &FilePath) {
  std::error_code EC;
  llvm::raw_fd_ostream FOut(FilePath, EC);
  if (!EC) {
    llvm::json::OStream(FOut, 0).value(serializeSchemaMapToJson(TypeSchemaMap));
    // FOut << llvm::json:: ;
    FOut.close();
  } else {
    llvm::errs() << "Error opening file: " << EC.message() << "\n";
    return;
  }
}
