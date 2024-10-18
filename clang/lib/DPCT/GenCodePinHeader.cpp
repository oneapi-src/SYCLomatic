//===---GenCodePinHeader.cpp -----------------------------*- C++ -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#include "GenCodePinHeader.h"
#include "ASTTraversal.h"
#include "AnalysisInfo.h"
#include "Diagnostics.h"
#include "MapNames.h"
#include "Statics.h"
#include "clang/AST/Expr.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ConvertUTF.h"

namespace clang {
namespace dpct {

using namespace clang::ast_matchers;

void GenCodePinHeaderRule::registerMatcher(ast_matchers::MatchFinder &MF) {
  if (DpctGlobalInfo::isCodePinEnabled()) {
    MF.addMatcher(cudaKernelCallExpr().bind("kernelCall"), this);
  }
}

std::string GenCodePinHeaderRule::getCodePinTypeHashKey(QualType T) {
  if (const RecordType *RT = T->getAs<RecordType>()) {
    if (const CXXRecordDecl *RD = dyn_cast<CXXRecordDecl>(RT->getDecl())) {
      RD = RD->getDefinition();
      if (!RD) {
        return std::string();
      }
      if (const ClassTemplateSpecializationDecl *Spec =
              dyn_cast<ClassTemplateSpecializationDecl>(RD)) {
        auto P = Spec->getInstantiatedFrom();
        if (!P.isNull() &&
            Spec->getInstantiatedFrom().is<ClassTemplateDecl *>()) {
          const ClassTemplateDecl *OriginalTemplate =
              Spec->getInstantiatedFrom().get<ClassTemplateDecl *>();
          const CXXRecordDecl *CRD = OriginalTemplate->getTemplatedDecl();
          return getStrFromLoc(CRD->getBeginLoc());
        }
      }
      return getStrFromLoc(RD->getBeginLoc());
    }
  }
  return std::string();
}

void GenCodePinHeaderRule::processTemplateTypeForCodePin(
    const ClassTemplateSpecializationDecl *Spec, VarInfoForCodePin &VI,
    std::vector<QualType> &MembersType, bool IsBaseMember,
    clang::PrintingPolicy &PrintPolicy) {
  if (!IsBaseMember) {
    VI.TemplateFlag = true;
  }
  auto P = Spec->getInstantiatedFrom();
  if (!P.isNull() && Spec->getInstantiatedFrom().is<ClassTemplateDecl *>()) {
    const ClassTemplateDecl *OriginalTemplate =
        Spec->getInstantiatedFrom().get<ClassTemplateDecl *>();

    const CXXRecordDecl *CRD = OriginalTemplate->getTemplatedDecl();
    std::string HashKey = getStrFromLoc(CRD->getBeginLoc());
    if (VI.HashKey.empty()) {
      VI.HashKey = HashKey;
    }
    auto &TemplateVec = DpctGlobalInfo::getCodePinTemplateTypeInfoVec();
    auto Iter = std::find_if(
        TemplateVec.begin(), TemplateVec.end(),
        [&HashKey](const auto &pair) { return pair.first == HashKey; });
    if (Iter == TemplateVec.end()) {
      VarInfoForCodePin TemplateVI;
      if (const NamespaceDecl *ND =
              dyn_cast<NamespaceDecl>(CRD->getDeclContext())) {
        getNameSpace(ND, TemplateVI.Namespaces);
      }
      TemplateVI.IsValid = true;
      TemplateVI.HashKey = HashKey;
      TemplateVI.VarRecordType = getRecordTypeStr(CRD);
      TemplateVI.VarNameWithoutScopeAndTemplateArgs = CRD->getName().str();
      TemplateVI.VarName = TemplateVI.VarNameWithoutScopeAndTemplateArgs;
      TemplateParameterList *Params = OriginalTemplate->getTemplateParameters();
      for (NamedDecl *Param : *Params) {
        if (TemplateTypeParmDecl *TypeParam =
                dyn_cast<TemplateTypeParmDecl>(Param)) {
          TemplateVI.TemplateArgs.push_back("typename " +
                                            TypeParam->getName().str());
        } else if (NonTypeTemplateParmDecl *NonTypeParam =
                       dyn_cast<NonTypeTemplateParmDecl>(Param)) {
          TemplateVI.TemplateArgs.push_back(
              NonTypeParam->getType().getAsString(PrintPolicy) + " " +
              NonTypeParam->getName().str());
        }
      }
      for (const auto &Base : CRD->bases()) {
        QualType BaseType = Base.getType();
        processCodePinTypeMemberOrBase(BaseType, "", TemplateVI, MembersType,
                                       true, false, IsBaseMember, PrintPolicy);
      }

      for (auto F : CRD->fields()) {
        processCodePinTypeMemberOrBase(F->getType(), F->getName().str(),
                                       TemplateVI, MembersType, false, false,
                                       IsBaseMember, PrintPolicy);
      }
      TemplateVec.push_back({HashKey, TemplateVI});
    }
  }
}

void GenCodePinHeaderRule::saveCodePinTypeDeps(
    std::string &Key, QualType &DepT,
    std::vector<std::pair<std::string, std::vector<std::string>>> &DepsVec,
    bool IsDumpFunc, clang::PrintingPolicy &PrintPolicy) {
  auto Iter =
      std::find_if(DepsVec.begin(), DepsVec.end(),
                   [&Key](const auto &pair) { return pair.first == Key; });
  std::string DepHashKey;
  if (IsDumpFunc) {
    DepHashKey = DepT.getAsString(PrintPolicy);
  } else {
    DepHashKey = getCodePinTypeHashKey(DepT);
  }
  if (DepHashKey.empty()) {
    return;
  }
  if (Iter != DepsVec.end()) {
    auto &V = Iter->second;
    auto SIter = std::find_if(V.begin(), V.end(), [&DepHashKey](const auto &E) {
      return E == DepHashKey;
    });
    if (SIter == V.end()) {
      V.push_back(DepHashKey);
    }
  } else {
    DepsVec.push_back({Key, {DepHashKey}});
  }
}

void GenCodePinHeaderRule::processCodePinTypeMemberOrBase(
    QualType MT, std::string Name, VarInfoForCodePin &VarInfo,
    std::vector<QualType> &MembersType, bool IsBase, bool IsBFS,
    bool IsBaseMember, clang::PrintingPolicy &PrintPolicy) {
  MemberOrBaseInfoForCodePin MemberOrBaseInfo;
  MT = MT.getUnqualifiedType();
  QualType NextMT = MT;

  while (NextMT->isPointerType()) {
    MemberOrBaseInfo.PointerDepth++;
    NextMT = NextMT->getPointeeType();
  }
  if (NextMT->isArrayType()) {
    const Type *Tptr = NextMT.getTypePtr();
    while (auto *AT = Tptr->getAsArrayTypeUnsafe()) {
      if (auto *CAT = dyn_cast<clang::ConstantArrayType>(AT)) {
        MemberOrBaseInfo.Dims.push_back(CAT->getSize().getZExtValue());
      }
      NextMT = AT->getElementType();
      Tptr = AT->getElementType().getTypePtr();
    };
  }

  if (IsBFS) {
    if (isTypeInAnalysisScope(NextMT.getTypePtrOrNull())) {
      MembersType.push_back(NextMT);
      saveCodePinTypeDeps(VarInfo.HashKey, NextMT,
                          DpctGlobalInfo::getCodePinTypeDepsVec(), false,
                          PrintPolicy);
      saveCodePinTypeDeps(VarInfo.VarName, NextMT,
                          DpctGlobalInfo::getCodePinDumpFuncDepsVec(), true,
                          PrintPolicy);
    }
  }

  PrintPolicy.SuppressScope = 1;
  std::string MemberOrBaseTypeName = NextMT.getAsString(PrintPolicy);
  PrintPolicy.SuppressScope = 0;
  MemberOrBaseInfo.UserDefinedTypeFlag =
      isTypeInAnalysisScope(NextMT.getTypePtrOrNull());
  MemberOrBaseInfo.TypeNameInCuda = MemberOrBaseTypeName;
  MemberOrBaseInfo.TypeNameInSycl = DpctGlobalInfo::getReplacedTypeName(
      NextMT, DpctGlobalInfo::getContext(), true);
  MemberOrBaseInfo.MemberName = Name;
  MemberOrBaseInfo.CodePinMemberName = Name + "_codepin";
  MemberOrBaseInfo.IsBaseMember = IsBaseMember;
  if (IsBase) {
    VarInfo.Bases.push_back(MemberOrBaseInfo);
  } else {
    VarInfo.Members.push_back(MemberOrBaseInfo);
  }
}

void GenCodePinHeaderRule::collectMemberInfo(
    QualType T, VarInfoForCodePin &VI, std::vector<QualType> &MembersType,
    bool IsBaseMember, clang::PrintingPolicy &PrintPolicy) {
  const CXXRecordDecl *RD = nullptr;
  if (const RecordType *RT = T->getAs<RecordType>()) {
    if (RD = dyn_cast<CXXRecordDecl>(RT->getDecl())) {
      RD = RD->getDefinition();
    }
  }
  if (!RD) {
    return;
  }
  if (const TypedefType *TT = T->getAs<TypedefType>()) {
    VI.IsTypeDef = true;
    llvm::StringRef TypeName =
        TT->getCanonicalTypeInternal()->getAsRecordDecl()->getName();
    if (TypeName.empty())
      VI.OrgTypeName =
          "dpct_type_" + getHashStrFromLoc(RD->getBeginLoc()).substr(0, 6);
    else
      VI.OrgTypeName = TypeName.str();
  }
  VI.IsValid = true;
  if (VI.VarRecordType.empty()) {
    VI.VarRecordType = getRecordTypeStr(RD);
  }
  if (VI.Namespaces.empty()) {
    if (const NamespaceDecl *ND =
            dyn_cast<NamespaceDecl>(RD->getDeclContext())) {
      getNameSpace(ND, VI.Namespaces);
    }
  }

  if (const ClassTemplateSpecializationDecl *SD =
          dyn_cast<ClassTemplateSpecializationDecl>(RD)) {
    processTemplateTypeForCodePin(SD, VI, MembersType, IsBaseMember,
                                  PrintPolicy);
  }

  if (VI.HashKey.empty()) {
    VI.HashKey = getStrFromLoc(RD->getBeginLoc());
  }

  auto InsertCurrentTypeIntoDepVec =
      [](std::string Key,
         std::vector<std::pair<std::string, std::vector<std::string>>> &Vec) {
        auto TIter =
            std::find_if(Vec.begin(), Vec.end(),
                         [&Key](const auto &E) { return E.first == Key; });
        if (TIter == Vec.end()) {
          Vec.push_back({Key, {}});
        }
      };
  InsertCurrentTypeIntoDepVec(VI.HashKey,
                              DpctGlobalInfo::getCodePinTypeDepsVec());
  InsertCurrentTypeIntoDepVec(VI.VarName,
                              DpctGlobalInfo::getCodePinDumpFuncDepsVec());

  for (const auto &Base : RD->bases()) {
    QualType BaseType = Base.getType();
    processCodePinTypeMemberOrBase(BaseType, "", VI, MembersType, true, true,
                                   IsBaseMember, PrintPolicy);
    if (isTypeInAnalysisScope(BaseType.getTypePtrOrNull())) {
      collectMemberInfo(BaseType, VI, MembersType, true, PrintPolicy);
    }
  }

  for (auto F : RD->fields()) {
    processCodePinTypeMemberOrBase(F->getType(), F->getName().str(), VI,
                                   MembersType, false, true, IsBaseMember,
                                   PrintPolicy);
  }
}

void GenCodePinHeaderRule::collectInfoForCodePinDumpFunction(QualType T) {
  auto &Ctx = DpctGlobalInfo::getContext();
  T = T.getLocalUnqualifiedType();
  clang::PrintingPolicy PrintPolicy(Ctx.getLangOpts());
  PrintPolicy.SuppressTagKeyword = 1;
  PrintPolicy.SuppressDefaultTemplateArgs = 1;
  if (T->isPointerType()) {
    collectInfoForCodePinDumpFunction(T->getPointeeType());
    return;
  } else if (T->isArrayType()) {
    if (const clang::ArrayType *AT = T->getAsArrayTypeUnsafe()) {
      collectInfoForCodePinDumpFunction(AT->getElementType());
      return;
    }
  }

  std::vector<QualType> CurrentTypeQueue, NextTypeQueue;
  CurrentTypeQueue.push_back(T);
  bool TopTypeFlag = true;
  while (!CurrentTypeQueue.empty()) {
    for (auto &QT : CurrentTypeQueue) {
      QT = QT.getLocalUnqualifiedType();
      if (!QT->getAs<RecordType>() ||
          !isTypeInAnalysisScope(QT.getTypePtrOrNull())) {
        continue;
      }
      std::string TypeName = QT.getAsString(PrintPolicy);
      auto &Vec = DpctGlobalInfo::getCodePinTypeInfoVec();
      auto Iter =
          std::find_if(Vec.begin(), Vec.end(), [&TypeName](const auto &pair) {
            return pair.first == TypeName;
          });
      if (Iter != Vec.end()) {
        continue;
      }
      VarInfoForCodePin VarInfo;
      if (TopTypeFlag) {
        TopTypeFlag = false;
        VarInfo.TopTypeFlag = true;
      }
      VarInfo.VarName = TypeName;
      collectMemberInfo(QT, VarInfo, NextTypeQueue, false, PrintPolicy);
      PrintPolicy.SuppressScope = 1;
      std::string TypenameWithoutScope = QT.getAsString(PrintPolicy);
      if (QT->isTypedefNameType())
        TypenameWithoutScope =
            QT->getAs<TypedefType>()->getDecl()->getName().str();
      auto Pos = TypenameWithoutScope.find('<');
      VarInfo.VarNameWithoutScopeAndTemplateArgs =
          TypenameWithoutScope.substr(0, Pos);
      if (Pos != std::string::npos) {
        VarInfo.TemplateInstArgs = TypenameWithoutScope.substr(Pos);
      }
      PrintPolicy.SuppressScope = 0;
      Vec.push_back({TypeName, VarInfo});
    }
    CurrentTypeQueue = NextTypeQueue;
    NextTypeQueue.clear();
  }
}

void GenCodePinHeaderRule::runRule(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  if (auto KCall =
          getAssistNodeAsType<CUDAKernelCallExpr>(Result, "kernelCall")) {
    auto DC = KCall->getDirectCallee();
    if (!DC) {
      return;
    }
    int ParamNum = DC->getNumParams();
    for (int i = 0; i < ParamNum; i++) {
      auto Decl = DC->getParamDecl(i);
      if (isUserDefinedDecl(Decl)) {
        collectInfoForCodePinDumpFunction(Decl->getType());
      }
    }
  }
}

} // namespace dpct
} // namespace clang