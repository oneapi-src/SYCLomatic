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
#include "Debug.h"
#include "ExprAnalysis.h"
#include "Utility.h"

#include "clang/AST/DeclTemplate.h"
#include "clang/AST/ExprCXX.h"

namespace clang {
namespace syclct {
std::string SyclctGlobalInfo::InRoot = std::string();
ASTContext *SyclctGlobalInfo::Context = nullptr;
SourceManager *SyclctGlobalInfo::SM = nullptr;
const std::string MemVarInfo::ExternVariableName = "syclct_extern_memory";
const std::string MemVarInfo::AccessorSuffix = "_acc";

void SyclctFileInfo::buildReplacements() {
  makeCanonical(FilePath);
  if (!isChildPath(SyclctGlobalInfo::getInRoot(), FilePath))
    return;
  for (auto &Kernel : KernelMap)
    Kernel.second->buildInfo();
}

void SyclctFileInfo::emplaceReplacements(tooling::Replacements &ReplSet) {
  for (auto &D : FuncMap)
    D.second->emplaceReplacement();
  Repls.emplaceIntoReplSet(ReplSet);
}

void SyclctGlobalInfo::insertCudaMalloc(const CallExpr *CE) {
  if (auto MallocVar = CudaMallocInfo::getMallocVar(CE->getArg(0)))
    insertCudaMallocInfo(MallocVar)->setSizeExpr(CE->getArg(1));
}

std::shared_ptr<CudaMallocInfo>
SyclctGlobalInfo::findCudaMalloc(const Expr *E) {
  if (auto Src = CudaMallocInfo::getMallocVar(E))
    return findCudaMallocInfo(Src);
  return std::shared_ptr<CudaMallocInfo>();
}

std::string KernelCallExpr::analysisExcutionConfig(const Expr *Config) {
  ArgumentAnalysis Analysis(Config);
  Analysis.analysis();
  return Analysis.getReplacedString();
}

void KernelCallExpr::buildExecutionConfig(
    const CUDAKernelCallExpr *KernelCall) {
  auto Config = KernelCall->getConfig();
  ExecutionConfig.NDSize = analysisExcutionConfig(Config->getArg(0));
  ExecutionConfig.WGSize = analysisExcutionConfig(Config->getArg(1));
  ExecutionConfig.ExternMemSize = analysisExcutionConfig(Config->getArg(2));
}

void KernelCallExpr::buildKernelInfo(const CUDAKernelCallExpr *KernelCall) {
  auto &SM = SyclctGlobalInfo::getSourceManager();
  SourceLocation Begin = KernelCall->getBeginLoc();
  LocInfo.NL = getNL(KernelCall->getEndLoc(), SM);
  LocInfo.Indent = getIndent(Begin, SM).str();
  LocInfo.LocHash = getHashAsString(Begin.printToString(SM)).substr(0, 6);
  buildExecutionConfig(KernelCall);
}

void KernelCallExpr::getAccessorDecl(FormatStmtBlock &Block) {
  auto &VM = getVarMap();
  if (VM.hasExternShared()) {
    auto ExternVariable = VM.getMap(MemVarInfo::Extern).begin()->second;
    Block.pushStmt(
        ExternVariable->getAccessorDecl(ExecutionConfig.ExternMemSize));
  }
  getAccessorDecl(Block, MemVarInfo::Local);
  getAccessorDecl(Block, MemVarInfo::Global);
}

void KernelCallExpr::getAccessorDecl(FormatStmtBlock &Block,
                                     MemVarInfo::VarScope Scope) {
  assert(Scope != MemVarInfo::Extern);
  static const std::string NullString;
  for (auto VI : getVarMap().getMap(Scope)) {
    if (Scope == MemVarInfo::Local)
      Block.pushStmt(VI.second->getMemoryDecl());
    Block.pushStmt(VI.second->getAccessorDecl(NullString));
  }
}

inline void KernelCallExpr::buildKernelPointerArgBufferAndOffsetStmt(
    const std::string &ArgName, StmtList &Buffers) {
  Buffers.emplace_back("std::pair<syclct::buffer_t, size_t> " + ArgName +
                       "_buf = syclct::get_buffer_and_offset(" + ArgName +
                       ");");
  Buffers.emplace_back("size_t " + ArgName + "_offset = " + ArgName +
                       "_buf.second;");
}

inline void
KernelCallExpr::buildKernelPointerArgAccessorStmt(const std::string &ArgName,
                                                  StmtList &Accessors) {
  Accessors.emplace_back(
      "auto " + ArgName + "_acc = " + ArgName +
      "_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);");
}

inline void
KernelCallExpr::buildKernelPointerArgRedeclStmt(const std::string &ArgName,
                                                const std::string &TypeName,
                                                StmtList &Redecls) {
  Redecls.emplace_back(TypeName + " *" + ArgName + " = (" + TypeName + "*)(&" +
                       ArgName + "_acc[0] + " + ArgName + "_offset);");
}

void KernelCallExpr::buildKernelPointerArgsStmt(StmtList &Buffers,
                                                StmtList &Accessors,
                                                StmtList &Redecls) {
  for (auto &Arg : getKernelPointerVarList()) {
    auto &ArgName = Arg->getName();
    buildKernelPointerArgBufferAndOffsetStmt(ArgName, Buffers);
    buildKernelPointerArgAccessorStmt(ArgName, Accessors);
    buildKernelPointerArgRedeclStmt(
        ArgName, Arg->getType()->getTemplateSpecializationName(), Redecls);
  }
}

std::string KernelCallExpr::getReplacement() {
  std::string Result;

  StmtList Buffers, Accessors, Redecls;
  buildKernelPointerArgsStmt(Buffers, Accessors, Redecls);

#define FMT_STMT_BLOCK                                                         \
  FormatStmtBlock &BlockPrev = Block;                                          \
  FormatStmtBlock Block(BlockPrev);

  Result += "{" + LocInfo.NL;
  {
    FormatStmtBlock Block(LocInfo.NL, LocInfo.Indent, Result);
    for (auto &BufferStmt : Buffers)
      Block.pushStmt(BufferStmt);
    Block.pushStmt("syclct::get_default_queue().submit(");
    {
      FMT_STMT_BLOCK
      Block.pushStmt("[&](cl::sycl::handler &cgh) {");
      {
        FMT_STMT_BLOCK
        getAccessorDecl(Block);
        for (auto &AccStmt : Accessors)
          Block.pushStmt(AccStmt);
        Block.pushStmt(
            "cgh.parallel_for<syclct_kernel_name<class " + getName() + "_" +
            LocInfo.LocHash +
            (hasTemplateArgs() ? (", " + getTemplateArguments(true)) : "") +
            ">>(");
        {
          FMT_STMT_BLOCK
          Block.pushStmt("cl::sycl::nd_range<3>((" + ExecutionConfig.NDSize +
                         " * " + ExecutionConfig.WGSize + "), " +
                         ExecutionConfig.WGSize + "),");
          Block.pushStmt("[=](cl::sycl::nd_item<3> " +
                         SyclctGlobalInfo::getItemName() + ") {");
          {
            FMT_STMT_BLOCK
            for (auto &Redecl : Redecls)
              Block.pushStmt(Redecl);
            Block.pushStmt(getName() +
                           (hasTemplateArgs()
                                ? ("<" + getTemplateArguments() + ">")
                                : "") +
                           "(" + getArguments() + ");");
          }
          Block.pushStmt("});");
        }
      }
      Block.pushStmt(isSync() ? "}).wait();" : "});");
    }
  }
  Result += LocInfo.Indent + "}" + LocInfo.NL;

  return Result;
}

void KernelCallExpr::buildInfo() {
  CallFunctionExpr::buildInfo();

  // TODO: Output debug info.
  SyclctGlobalInfo::getInstance().addReplacement(
      std::make_shared<ExtReplacement>(getFilePath(), getBegin(), 0,
                                       getReplacement(), nullptr));
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
    FuncInfo = DeviceFunctionDecl::LinkRedecls(CallDecl);
    if (auto DRE = dyn_cast<DeclRefExpr>(CE->getCallee()->IgnoreImpCasts()))
      buildTemplateArguments(DRE->template_arguments());
  } else if (auto Unresolved = dyn_cast<UnresolvedLookupExpr>(
                 CE->getCallee()->IgnoreImpCasts())) {
    Name = Unresolved->getName().getAsString();
    FuncInfo = DeviceFunctionDecl::LinkUnresolved(Unresolved);
    buildTemplateArguments(Unresolved->template_arguments());
  } else if (auto DependentScope = dyn_cast<CXXDependentScopeMemberExpr>(
                 CE->getCallee()->IgnoreImpCasts())) {
    Name = DependentScope->getMember().getAsString();
    buildTemplateArguments(DependentScope->template_arguments());
  }
}

std::string CallFunctionExpr::getName(const NamedDecl *D) {
  if (auto ID = D->getIdentifier())
    return ID->getName().str();
  return "";
}

void CallFunctionExpr::buildInfo() {
  if (!FuncInfo)
    return;
  FuncInfo->buildInfo();
  VarMap.merge(FuncInfo->getVarMap(), TemplateArgs);
}

void CallFunctionExpr::emplaceReplacement() {
  buildInfo();
  Args.emplaceReplacements();
  SyclctGlobalInfo::getInstance().addReplacement(
      std::make_shared<ExtReplacement>(FilePath, RParenLoc, 0,
                                       getExtraArguments(), nullptr));
}

std::string CallFunctionExpr::getTemplateArguments(bool WithScalarWrapped) {
  const static std::string ScalarWrapperPrefix = "syclct_kernel_scalar<",
                           ScalarWrapperSuffix = ">, ";
  std::string Result;
  for (auto &TA : TemplateArgs) {
    if (WithScalarWrapped && !TA.isType())
      Result += ScalarWrapperPrefix + TA.getAsString() + ScalarWrapperSuffix;
    else
      Result += TA.getAsString() + ", ";
  }
  return (Result.empty()) ? Result : Result.erase(Result.size() - 2);
}

ArgumentsInfo::ArgumentsInfo(const CallExpr *C) {
  if (C->getStmtClass() == Stmt::CUDAKernelCallExprClass) {
    KernelArgumentAnalysis::MapTy Map;
    KernelArgumentAnalysis A(Map);
    buildArgsInfo(C->arguments(), A);
    for (auto &V : Map) {
      if (V.second->getType()->isPointer())
        KernelArgs.emplace_back(V.second);
    }
  } else {
    ArgumentAnalysis A;
    buildArgsInfo(C->arguments(), A);
  }
}

void DeviceFunctionInfo::merge(std::shared_ptr<DeviceFunctionInfo> Other) {
  if (this == Other.get())
    return;
  VarMap.merge(Other->getVarMap());
  mergeCallMap(Other->CallExprMap);
}

void DeviceFunctionInfo::mergeCallMap(
    const GlobalMap<CallFunctionExpr> &Other) {
  for (const auto &Call : Other)
    CallExprMap.insert(Call);
}

void DeviceFunctionInfo::buildInfo() {
  if (isBuilt())
    return;
  setBuilt();
  for (auto &Call : CallExprMap) {
    Call.second->emplaceReplacement();
    VarMap.merge(Call.second->getVarMap());
  }
  Params = VarMap.getDeclParam(hasParams());
}

inline void DeviceFunctionDecl::emplaceReplacement() {
  // TODO: Output debug info.
  SyclctGlobalInfo::getInstance().addReplacement(
      std::make_shared<ExtReplacement>(FilePath, ReplaceOffset, ReplaceLength,
                                       FuncInfo->getParameters(), nullptr));
}

void DeviceFunctionDecl::buildReplaceLocInfo(const FunctionDecl *FD) {
  auto &SM = SyclctGlobalInfo::getSourceManager();
  auto &LO = SyclctGlobalInfo::getContext().getLangOpts();

  SourceLocation NextToken;
  if (FD->param_empty())
    NextToken = FD->getNameInfo().getEndLoc();
  else {
    auto EndParam = *(FD->param_end() - 1);
    NextToken = EndParam->getLocation();
  }
  auto Tok = Lexer::findNextToken(NextToken, SM, LO);
  while (Tok.hasValue()) {
    static const llvm::StringRef VoidId = "void";
    switch (Tok->getKind()) {
    case tok::r_paren:
      ReplaceOffset = SM.getFileOffset(Tok->getLocation());
      return;
    case tok::raw_identifier:
      if (Tok->getRawIdentifier() == VoidId) {
        ReplaceOffset = SM.getFileOffset(Tok->getLocation());
        ReplaceLength = Tok->getLength();
        return;
      }
    default:
      Tok = Lexer::findNextToken(Tok->getLocation(), SM, LO);
    }
  }
}

void DeviceFunctionDecl::LinkDecl(const FunctionDecl *FD, DeclList &List,
                                  std::shared_ptr<DeviceFunctionInfo> &Info) {
  if (!SyclctGlobalInfo::isInRoot(FD->getBeginLoc()))
    return;
  auto D = SyclctGlobalInfo::getInstance().insertDeviceFunctionDecl(FD);
  if (Info) {
    if (auto FuncInfo = D->getFuncInfo())
      Info->merge(FuncInfo);
    D->setFuncInfo(Info);
  } else if (auto FuncInfo = D->getFuncInfo())
    Info = FuncInfo;
  else
    List.push_back(D);
}

void DeviceFunctionDecl::LinkRedecls(
    const FunctionDecl *FD, DeclList &List,
    std::shared_ptr<DeviceFunctionInfo> &Info) {
  LinkDeclRange(FD->redecls(), List, Info);
}

void DeviceFunctionDecl::LinkDecl(const FunctionTemplateDecl *FTD,
                                  DeclList &List,
                                  std::shared_ptr<DeviceFunctionInfo> &Info) {
  LinkDeclRange(FTD->specializations(), List, Info);
}

void DeviceFunctionDecl::LinkDecl(const NamedDecl *ND, DeclList &List,
                                  std::shared_ptr<DeviceFunctionInfo> &Info) {
  switch (ND->getKind()) {
  case Decl::Function:
    return LinkRedecls(static_cast<const FunctionDecl *>(ND), List, Info);
  case Decl::FunctionTemplate:
    return LinkDecl(static_cast<const FunctionTemplateDecl *>(ND), List, Info);
  default:
    syclct_unreachable("unexpected name decl type");
  }
}

std::shared_ptr<MemVarInfo> MemVarInfo::buildMemVarInfo(const VarDecl *Var) {
  if (auto Func = Var->getParentFunctionOrMethod()) {
    auto LocInfo = SyclctGlobalInfo::getLocInfo(Var);
    auto VI = std::make_shared<MemVarInfo>(LocInfo.second, LocInfo.first, Var);
    DeviceFunctionDecl::LinkRedecls(dyn_cast<FunctionDecl>(Func))->addVar(VI);
    return VI;
  }

  return SyclctGlobalInfo::getInstance().insertMemVarInfo(Var);
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
    return getMemoryType(DeviceMemory, getType());
  }
  case clang::syclct::MemVarInfo::Constant: {
    static std::string ConstantMemory = "syclct::constant_memory";
    return getMemoryType(ConstantMemory, getType());
  }
  case clang::syclct::MemVarInfo::Shared: {
    static std::string SharedMemory = "syclct::shared_memory";
    static std::string ExternSharedMemory = "syclct::extern_shared_memory";
    if (isExtern())
      return ExternSharedMemory;
    return getMemoryType(SharedMemory, getType());
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
           getType()->getBaseName() + ">();";
  case clang::syclct::MemVarInfo::Global: {
    const static std::string NullString;
    return getMemoryDecl(NullString);
  }
  default:
    syclct_unreachable("unknow variable scope");
  }
}

TypeInfo::TypeInfo(QualType Type)
    : IsPointer(false), IsTemplate(false), TemplateIndex(0),
      TemplateList(nullptr) {
  setArrayInfo(Type);
  setPointerInfo(Type);
  setTemplateInfo(Type);
  setName(Type);
}

std::string TypeInfo::getRangeArgument(const std::string &MemSize,
                                       bool MustArguments) {
  std::string Arg = "(";
  for (auto R : Range) {
    if (auto CAT = dyn_cast<ConstantArrayType>(R))
      Arg += CAT->getSize().toString(10, false);
    else if (auto VAT = dyn_cast<VariableArrayType>(R))
      Arg +=
          getStmtSpelling(VAT->getSizeExpr(), SyclctGlobalInfo::getContext());
    else if (auto DAT = dyn_cast<DependentSizedArrayType>(R)) {
      ArraySizeExprAnalysis SizeAnalysis(DAT->getSizeExpr(), TemplateList);
      SizeAnalysis.analysis();
      Arg += SizeAnalysis.getReplacedString();
    } else if (MemSize.empty())
      syclct_unreachable(
          "array size should not be incomplete while non mem size");
    else
      Arg += MemSize;
    Arg += ", ";
  }
  return (Arg.size() == 1) ? (MustArguments ? (Arg + ")") : "")
                           : Arg.replace(Arg.size() - 2, 2, ")");
}

void TypeInfo::setTemplateType(const std::vector<TemplateArgumentInfo> &TA) {
  assert(TemplateIndex < TA.size());
  if (isTemplate())
    TemplateType = TA[TemplateIndex].getAsType();
  TemplateList = &TA;
}

void TypeInfo::setArrayInfo(QualType &Type) {
  while (Type->isArrayType()) {
    Range.push_back(dyn_cast<ArrayType>(Type));
    Type = Type->getAsArrayTypeUnsafe()->getElementType();
  }
}

void TypeInfo::setPointerInfo(QualType &Type) {
  while (Type->isPointerType()) {
    IsPointer = true;
    Type = Type->getPointeeType();
  }
}
void TypeInfo::setTemplateInfo(QualType &Type) {
  if (auto TemplateType =
          dyn_cast<TemplateTypeParmType>(Type->getCanonicalTypeInternal())) {
    IsTemplate = true;
    TemplateIndex = TemplateType->getIndex();
  }
}
void TypeInfo::setName(QualType &Type) {
  auto &PP = SyclctGlobalInfo::getContext().getPrintingPolicy();
  BaseNameWithoutQualifiers = Type.getUnqualifiedType().getAsString(PP);

  OrginalBaseType = BaseNameWithoutQualifiers;
  if (!isTemplate())
    MapNames::replaceName(MapNames::TypeNamesMap, BaseNameWithoutQualifiers);
  auto Q = Type.getLocalQualifiers();
  if (Q.isEmptyWhenPrinted(PP))
    BaseName = BaseNameWithoutQualifiers;
  else
    BaseName = Type.getLocalQualifiers().getAsString(PP) + " " +
               BaseNameWithoutQualifiers;
}
} // namespace syclct
} // namespace clang
