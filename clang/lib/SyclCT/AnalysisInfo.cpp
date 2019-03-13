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
#include "ExprAnalysis.h"
#include "Utility.h"

#include "clang/AST/DeclTemplate.h"
#include "clang/AST/ExprCXX.h"

///
// KernelInfoMap KernelTransAssist::KernelNameInfoMap;

namespace clang {
namespace syclct {
std::string SyclctGlobalInfo::InRoot = std::string();
ASTContext *SyclctGlobalInfo::Context = nullptr;
SourceManager *SyclctGlobalInfo::SM = nullptr;
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

void SyclctGlobalInfo::registerCudaMalloc(const CallExpr *CE) {
  if (auto MallocVar = CudaMallocInfo::getMallocVar(CE->getArg(0)))
    registerCudaMallocInfo(MallocVar)->setSizeExpr(CE->getArg(1));
}

std::shared_ptr<CudaMallocInfo>
SyclctGlobalInfo::findCudaMalloc(const Expr *E) {
  if (auto Src = CudaMallocInfo::getMallocVar(E))
    return findCudaMallocInfo(Src);
  return std::shared_ptr<CudaMallocInfo>();
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
    Result += Indent + ExternVariable->getAccessorDecl(ExternMemSize) + NL;
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

void CallFunctionExpr::addTemplateType(const TemplateArgumentLoc &TAL) {
  switch (TAL.getArgument().getKind()) {
  case TemplateArgument::Type:
    return TemplateArgs.push_back(TAL.getTypeSourceInfo()->getType());
  case TemplateArgument::Expression:
    return TemplateArgs.push_back(TAL.getSourceExpression());
  case TemplateArgument::Integral:
    return TemplateArgs.push_back(TAL.getSourceIntegralExpression());
  default:
    llvm_unreachable("unexpected template type");
  }
}

void CallFunctionExpr::buildCallExprInfo(const CallExpr *CE) {
  if (auto CallDecl = CE->getDirectCallee()) {
    Name = getName(CallDecl);
    addFunctionDecl(CallDecl);
    if (auto DRE = dyn_cast<DeclRefExpr>(CE->getCallee()->IgnoreImpCasts()))
      buildTemplateArguments(DRE->template_arguments());
  } else if (auto Unresolved = dyn_cast<UnresolvedLookupExpr>(
                 CE->getCallee()->IgnoreImpCasts())) {
    Name = Unresolved->getName().getAsString();
    buildTemplateArguments(Unresolved->template_arguments());
    for (auto D : Unresolved->decls())
      addNamedDecl(D);
  } else if (auto DependentScope = dyn_cast<CXXDependentScopeMemberExpr>(
                 CE->getCallee()->IgnoreImpCasts())) {
    Name = DependentScope->getMember().getAsString();
    if (auto TST =
            DependentScope->getBaseType()->getAs<TemplateSpecializationType>())
      addTemplateDecl(TST->getTemplateName().getAsTemplateDecl());
    buildTemplateArguments(DependentScope->template_arguments());
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
  TS.emplace_back(new InsertText(RParenLoc, getExtraArguments()));
  for (auto &Arg : Args) {
    if (auto TM = Arg.getTextModification())
      TS.emplace_back(TM);
  }
}

void DeviceFunctionInfo::buildInfo(TransformSetTy &TS) {
  static std::vector<TemplateArgumentInfo> NullTemplate;
  if (hasBuilt())
    return;
  setBuilt();
  for (auto &Call : CallExprMap) {
    Call.second->buildInfo(TS);
    VarMap->merge(Call.second->getVarMap(), NullTemplate);
  }
}

void DeviceFunctionInfo::computeParenLoc() {
  auto &SM = SyclctGlobalInfo::getSourceManager();

  // Compute location of the left parenthesis
  auto FuncNameLoc = FuncDecl->getNameInfo().getLoc();
  LParenLoc =
      Lexer::findNextToken(FuncNameLoc, SM, LangOptions())->getLocation();

  // Compute location of the right parenthesis
  if (!hasParams()) {
    auto &SM = SyclctGlobalInfo::getSourceManager();
    auto Token = Lexer::findNextToken(LParenLoc, SM, LangOptions());
    while (Token->isNot(tok::r_paren)) {
      Token = Lexer::findNextToken(Token->getLocation(), SM, LangOptions());
    }
    RParenLoc = Token->getLocation();
  } else {
    auto EndParam = *(FuncDecl->parameters().end() - 1);
    RParenLoc =
        EndParam->getEndLoc().getLocWithOffset(EndParam->getName().size());
  }
}

TextModification *DeviceFunctionInfo::getTextModification() {
  if (!hasParams()) {
    auto &SM = SyclctGlobalInfo::getSourceManager();
    auto Token = Lexer::findNextToken(LParenLoc, SM, LangOptions());
    // Remove the parameter "void" in the function declaration
    if (Token->is(tok::raw_identifier) &&
        Token->getRawIdentifier().equals("void"))
      return new ReplaceToken(Token->getLocation(), getParameters());
  }

  return new InsertText(RParenLoc, getParameters());
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
  case clang::syclct::MemVarInfo::Device: {
    static std::string DeviceMemory = "syclct::device_memory";
    return DeviceMemory + getType()->getAsTemplateArguments();
  }
  case clang::syclct::MemVarInfo::Constant: {
    static std::string ConstantMemory = "syclct::constant_memory";
    return ConstantMemory + getType()->getAsTemplateArguments();
  }
  case clang::syclct::MemVarInfo::Shared: {
    static std::string SharedMemory = "syclct::shared_memory";
    static std::string ExternSharedMemory = "syclct::extern_shared_memory";
    if (isExtern())
      return ExternSharedMemory;
    return SharedMemory + getType()->getAsTemplateArguments();
  }
  default:
    llvm_unreachable("unknow variable attribute");
  }
}

const std::string &MemVarInfo::getMemoryAttr() {
  switch (Attr) {
  case clang::syclct::MemVarInfo::Device: {
    static std::string DeviceMemory = "syclct::device";
    return DeviceMemory;
  }
  case clang::syclct::MemVarInfo::Constant: {
    static std::string ConstantMemory = "syclct::constant";
    return ConstantMemory;
  }
  case clang::syclct::MemVarInfo::Shared: {
    static std::string SharedMemory = "syclct::shared";
    return SharedMemory;
  }
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
  case clang::syclct::MemVarInfo::Global: {
    const static std::string NullString;
    return getMemoryDecl(NullString);
  }
  default:
    llvm_unreachable("unknow variable scope");
  }
}

TypeInfo::TypeInfo(const QualType &Type)
    : Type(Type), Pointer(false), Template(false), TemplateIndex(0) {
  setArrayInfo();
  setPointerInfo();
  setTemplateInfo();
  setName();
}

std::string TypeInfo::getRangeArgument(const std::string&MemSize,bool MustArguments) {
  std::string Arg = "(";
  for (auto R : Range) {
    if (auto CAT = dyn_cast<ConstantArrayType>(R))
      Arg += CAT->getSize().toString(10, false) + ", ";
    else if (auto VAT = dyn_cast<VariableArrayType>(R))
      Arg +=
          getStmtSpelling(VAT->getSizeExpr(), SyclctGlobalInfo::getContext()) +
          ", ";
    else if (auto DAT = dyn_cast<DependentSizedArrayType>(R)) {
      SizeAnalysis.analysis(DAT->getSizeExpr());
      Arg += SizeAnalysis.getReplacedString() + ", ";
    } else if (MemSize.empty())
      llvm_unreachable("array size should not be zero");
    else
      Arg += MemSize + ", ";
  }
  return (Arg.size() == 1) ? (MustArguments ? (Arg + ")") : "")
                           : Arg.replace(Arg.size() - 2, 2, ")");
}

void TypeInfo::setTemplateType(const std::vector<TemplateArgumentInfo> &TA) {
  assert(TemplateIndex < TA.size());
  if (isTemplate())
    TemplateType = TA[TemplateIndex].getAsType();
  SizeAnalysis.setTemplateArgsList(TA);
}

void ArgumentInfo::getReplacement() {
  if (auto ME = dyn_cast<MemberExpr>(Arg->IgnoreParenImpCasts())) {
    if (auto B = dyn_cast<DeclRefExpr>(ME->getBase())) {
      if (B->getDecl()->getType().getAsString(SyclctGlobalInfo::getInstance()
                                                  .getContext()
                                                  .getPrintingPolicy()) ==
          "dim3") {
        Replacement = B->getDecl()->getName().str() +
                      MapNames::Dim3MemberNamesMap
                          .find(ME->getMemberDecl()->getName().str())
                          ->second;
      }
    }
  }
}

void CudaMallocInfo::replaceType(const Expr *SizeExpr) {
  SizeExpr = SizeExpr->IgnoreImpCasts();
  if (auto BinaryOpt = dyn_cast<BinaryOperator>(SizeExpr)) {
    replaceType(BinaryOpt->getRHS());
    replaceType(BinaryOpt->getLHS());
  } else if (auto UnaryOrTraits =
                 dyn_cast<UnaryExprOrTypeTraitExpr>(SizeExpr)) {
    if (UnaryOrTraits->getKind() == UnaryExprOrTypeTrait::UETT_SizeOf) {
      auto TypeArgument = UnaryOrTraits->getArgumentTypeInfo();
      auto Itr = MapNames::TypeNamesMap.find(
          TypeArgument->getType().getAsString(SyclctGlobalInfo::getInstance()
                                                  .getContext()
                                                  .getPrintingPolicy()));
      if (Itr != MapNames::TypeNamesMap.end()) {
        replaceSizeString(TypeArgument->getTypeLoc().getBeginLoc(),
                          UnaryOrTraits->getRParenLoc().getLocWithOffset(-1),
                          Itr->second);
      }
    }
  }
}

void CudaMallocInfo::replaceSizeString(const SourceLocation &Begin,
                                       const SourceLocation &End,
                                       const std::string &NewTypeName) {
  int BeginOffset =
      Begin.getRawEncoding() - SizeExpr->getBeginLoc().getRawEncoding();
  if ((BeginOffset >= 0) && (BeginOffset < (int)Size.size()))
    Size.replace(BeginOffset, End.getRawEncoding() - Begin.getRawEncoding() + 1,
                 NewTypeName);
}
} // namespace syclct
} // namespace clang
