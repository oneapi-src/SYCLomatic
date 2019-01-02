//===--- TextModification.cpp ---------------------------*- C++ -*---===//
//
// Copyright (C) 2018 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

#include "AnalysisInfo.h"
#include "Utility.h"

#include "clang/AST/DeclTemplate.h"
#include "clang/AST/ExprCXX.h"

///
// KernelInfoMap KernelTransAssist::KernelNameInfoMap;

namespace clang {
namespace syclct {
std::string SyclctGlobalInfo::InRoot = std::string();
const ASTContext *SyclctGlobalInfo::Context = nullptr;
const SourceManager *SyclctGlobalInfo::SM = nullptr;
const std::string MemVarInfo::ExternVariableName = "syclct_extern_memory";
const std::string MemVarInfo::AccessorSuffix = "_acc";

void SyclctGlobalInfo::emplaceKernelAndDeviceReplacement(TransformSetTy &TS,
                                                         StmtStringMap &SSM) {
  for (auto &Kernel : KernelMap) {
    Kernel.second->buildInfo(TS);
    TS.emplace_back(new ReplaceKernelCallExpr(Kernel.second, &SSM));
  }
  for (auto &DF : FuncMap)
    TS.emplace_back(DF.second->getTextModification());
}

std::string
KernelCallExpr::getExternMemSize(const CUDAKernelCallExpr *KernelCall) {
  if (auto Arg = KernelCall->getConfig()->getArg(2))
    if (!Arg->isDefaultArgument())
      return getStmtSpelling(Arg, SyclctGlobalInfo::getContext());
  return "";
}

std::string KernelCallExpr::getAccessorDecl(const std::string &Indent,
                                            const std::string &NL) {
  std::string Result;
  auto VM = getVarMap();
  if (VM->hasExternShared()) {
    auto ExternVariable = VM->getMap(MemVarInfo::Extern).begin()->second;
    Result = Indent + ExternVariable->getMemoryDecl(ExternMemSize) + NL;
    Result += Indent + ExternVariable->getAccessorDecl() + NL;
  }

  Result += getAccessorDecl(MemVarInfo::Local, Indent, NL);
  Result += getAccessorDecl(MemVarInfo::Global, Indent, NL);
  return Result;
}

void CallFunctionExpr::addFunctionDecl(
    std::shared_ptr<DeviceFunctionInfo> FuncDecl) {
  FuncDeclMap.insert(typename GlobalMap<DeviceFunctionInfo>::value_type(
      FuncDecl->getLoc(), FuncDecl));
}

void CallFunctionExpr::addFunctionDecl(const FunctionDecl *FD) {
  if (SyclctGlobalInfo::isInRoot(FD->getBeginLoc()))
    addFunctionDecl(
        SyclctGlobalInfo::getInstance().registerDeviceFunctionInfo(FD));
}

void CallFunctionExpr::addNamedDecl(const NamedDecl *ND) {
  if (auto FD = dyn_cast<FunctionDecl>(ND))
    addFunctionDecl(FD);
  else if (auto CD = dyn_cast<CXXRecordDecl>(ND))
    addRecordDecl(CD);
  else if (auto TD = dyn_cast<TemplateDecl>(ND))
    addTemplateDecl(TD);
}

void CallFunctionExpr::addTemplateFunctionDecl(
    const FunctionTemplateDecl *FTD) {
  addFunctionDecl(FTD->getAsFunction());
  for (auto FD : FTD->specializations())
    addFunctionDecl(FD);
}

void CallFunctionExpr::addRecordDecl(const CXXRecordDecl *D) {
  for (auto Method : D->methods())
    if (getName(Method) == Name)
      addFunctionDecl(Method);
}

void CallFunctionExpr::addTemplateClassDecl(const ClassTemplateDecl *CTD) {
  addRecordDecl(CTD->getTemplatedDecl());
  for (auto D : CTD->specializations())
    addRecordDecl(D);
}

void CallFunctionExpr::addTemplateDecl(const TemplateDecl *TD) {
  if (auto FTD = dyn_cast<FunctionTemplateDecl>(TD))
    addTemplateFunctionDecl(FTD);
  else if (auto CTD = dyn_cast<ClassTemplateDecl>(TD))
    addTemplateClassDecl(CTD);
}

void CallFunctionExpr::addTemplateType(const TemplateArgument &TA) {
  switch (TA.getKind()) {
  case TemplateArgument::Type:
    return TemplateArgs.push_back(TA.getAsType());
  case TemplateArgument::Expression:
    return TemplateArgs.push_back(TA.getAsExpr());
  case TemplateArgument::Integral:
    return TemplateArgs.push_back(TA.getAsIntegral());
  default:
    llvm_unreachable("unexpected template type");
  }
}

void CallFunctionExpr::getTemplateSpecializationInfo(const FunctionDecl *FD) {
  if (auto Args = FD->getTemplateSpecializationArgs()) {
    for (auto TemplateArg : Args->asArray())
      addTemplateType(TemplateArg);
  }
}

void CallFunctionExpr::getTemplateArguments(
    const ArrayRef<TemplateArgumentLoc> &TemplateArray) {
  for (auto Arg : TemplateArray)
    addTemplateType(Arg.getArgument());
}

void CallFunctionExpr::buildCallExprInfo(const CallExpr *CE) {
  if (auto CallDecl = CE->getDirectCallee()) {
    Name = getName(CallDecl);
    addFunctionDecl(CallDecl);
    getTemplateSpecializationInfo(CallDecl);
  } else if (auto Unresolved = dyn_cast<UnresolvedLookupExpr>(
                 CE->getCallee()->IgnoreImpCasts())) {
    Name = Unresolved->getName().getAsString();
    getTemplateArguments(Unresolved->template_arguments());
    for (auto D : Unresolved->decls())
      addNamedDecl(D);
  } else if (auto DependentScope = dyn_cast<CXXDependentScopeMemberExpr>(
                 CE->getCallee()->IgnoreImpCasts())) {
    Name = DependentScope->getMember().getAsString();
    if (auto TST =
            DependentScope->getBaseType()->getAs<TemplateSpecializationType>())
      addTemplateDecl(TST->getTemplateName().getAsTemplateDecl());
    getTemplateArguments(DependentScope->template_arguments());
  }
}

std::string CallFunctionExpr::getName(const NamedDecl *D) {
  if (auto ID = D->getIdentifier())
    return ID->getName().str();
  return "";
}

void CallFunctionExpr::buildInfo(TransformSetTy &TS) {
  if (FuncDeclMap.empty())
    return;
  for (auto &DeviceFunc : FuncDeclMap) {
    DeviceFunc.second->buildInfo(TS);
    VarMap->merge(DeviceFunc.second->getVarMap(), TemplateArgs);
    DeviceFunc.second->setVarMap(VarMap);
  }
  TS.emplace_back(new InsertText(RParenLoc, getArguments()));
}

void DeviceFunctionInfo::buildInfo(TransformSetTy &TS) {
  static std::vector<TemplateArgumentInfo> NullTemplate;
  if (hasBuilt())
    return;
  for (auto &Call : CallExprMap) {
    Call.second->buildInfo(TS);
    VarMap->merge(Call.second->getVarMap(), NullTemplate);
  }
  setBuilt();
}

SourceLocation DeviceFunctionInfo::getRParenLoc(const FunctionDecl *FD) {
  auto Params = FD->parameters();
  if (Params.empty()) {
    auto FuncName = FD->getNameInfo();
    return FuncName.getLoc().getLocWithOffset(FuncName.getAsString().size() +
                                              1);
  } else {
    auto EndParam = *(Params.end() - 1);
    return EndParam->getEndLoc().getLocWithOffset(EndParam->getName().size());
  }
}

std::shared_ptr<MemVarInfo> MemVarInfo::buildMemVarInfo(const VarDecl *Var) {
  if (auto Func = Var->getParentFunctionOrMethod()) {
    auto VI = std::make_shared<MemVarInfo>(Var);
    SyclctGlobalInfo::getInstance()
        .registerDeviceFunctionInfo(dyn_cast<FunctionDecl>(Func))
        ->addVar(VI);
    return VI;
  }

  return SyclctGlobalInfo::getInstance().registerMemVarInfo(Var);
}

MemVarInfo::VarAttrKind MemVarInfo::getAttr(const AttrVec &Attrs) {
  for (auto VarAttr : Attrs) {
    auto Kind = VarAttr->getKind();
    if (Kind == attr::CUDAConstant)
      return Constant;
    else if (Kind == attr::CUDADevice)
      return Device;
    else if (Kind == attr::CUDAShared)
      return Shared;
  }
  llvm_unreachable("unknow variable attribute");
}

std::string MemVarInfo::getMemoryType() {
  switch (Attr) {
  case clang::syclct::MemVarInfo::Device:
    static std::string DeviceMemory = "syclct::device_memory";
    return DeviceMemory + getType()->getAsTemplateArguments();
  case clang::syclct::MemVarInfo::Constant:
    static std::string ConstantMemory = "syclct::constant_memory";
    return ConstantMemory + getType()->getAsTemplateArguments();
  case clang::syclct::MemVarInfo::Shared:
    static std::string SharedMemory = "syclct::shared_memory";
    static std::string ExternSharedMemory = "syclct::extern_shared_memory";
    if (isExtern())
      return ExternSharedMemory;
    return SharedMemory + getType()->getAsTemplateArguments();
  default:
    llvm_unreachable("unknow variable attribute");
  }
}

const std::string &MemVarInfo::getMemoryAttr() {
  switch (Attr) {
  case clang::syclct::MemVarInfo::Device:
    static std::string DeviceMemory = "syclct::device";
    return DeviceMemory;
  case clang::syclct::MemVarInfo::Constant:
    static std::string ConstantMemory = "syclct::constant";
    return ConstantMemory;
  case clang::syclct::MemVarInfo::Shared:
    static std::string SharedMemory = "syclct::shared";
    return SharedMemory;
  default:
    llvm_unreachable("unknow variable attribute");
  }
}

std::string MemVarInfo::getDeclarationReplacement() {
  switch (Scope) {
  case clang::syclct::MemVarInfo::Local:
    return "";
  case clang::syclct::MemVarInfo::Extern:
    return "auto " + getName() + " = " + ExternVariableName + ".reinterpret<" +
           getType()->getName() + ">();";
  case clang::syclct::MemVarInfo::Global:
    const static std::string NullString;
    return getMemoryDecl(NullString);
  default:
    llvm_unreachable("unknow variable scope");
  }
}

void TypeInfo::setTemplateType(const std::vector<TemplateArgumentInfo> &TA) {
  assert(TemplateIndex < TA.size());
  if (isTemplate())
    TemplateType = TA[TemplateIndex].getAsType();
}
} // namespace syclct
} // namespace clang
