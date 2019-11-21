//===--- AnalysisInfo.cpp -------------------------------*- C++ -*---===//
//
// Copyright (C) 2018 - 2019 Intel Corporation. All rights reserved.
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
namespace dpct {
std::string DpctGlobalInfo::InRoot = std::string();
// TODO: implement one of this for each source language.
std::string DpctGlobalInfo::CudaPath = std::string();
UsmLevel DpctGlobalInfo::UsmLvl = UsmLevel::none;
CompilerInstance *DpctGlobalInfo::CI = nullptr;
ASTContext *DpctGlobalInfo::Context = nullptr;
SourceManager *DpctGlobalInfo::SM = nullptr;
bool DpctGlobalInfo::KeepOriginCode = false;
const std::string MemVarInfo::ExternVariableName = "dpct_local";
const int TextureObjectInfo::ReplaceTypeLength = strlen("cudaTextureObject_t");

bool DpctFileInfo::isInRoot() { return DpctGlobalInfo::isInRoot(FilePath); }
// TODO: implement one of this for each source language.
bool DpctFileInfo::isInCudaPath() {
  return DpctGlobalInfo::isInCudaPath(FilePath);
}

void DpctFileInfo::buildLinesInfo() {
  if (FilePath.empty())
    return;
  auto &SM = DpctGlobalInfo::getSourceManager();
  auto FID = SM.getOrCreateFileID(SM.getFileManager().getFile(FilePath).get(),
                                  SrcMgr::C_User);
  auto Content = SM.getSLocEntry(FID).getFile().getContentCache();
  if (!Content->SourceLineCache)
    SM.getLineNumber(FID, 0);
  auto LineCache = Content->SourceLineCache;
  auto NumLines = Content->NumLines;
  const char *Buffer = nullptr;
  if (!LineCache) {
    return;
  }
  if (DpctGlobalInfo::isKeepOriginCode())
    Buffer = Content->getBuffer(SM.getDiagnostics(), SM.getFileManager())
                 ->getBufferStart();
  for (unsigned L = 1; L < Content->NumLines; ++L)
    Lines.emplace_back(L, LineCache, Buffer);
  Lines.emplace_back(NumLines, LineCache[NumLines - 1], Content->getSize(),
                     Buffer);
}

void DpctFileInfo::buildReplacements() {
  if (!isInRoot())
    return;
  for (auto &Kernel : KernelMap)
    Kernel.second->buildInfo();
}

void DpctFileInfo::emplaceReplacements(tooling::Replacements &ReplSet) {
  for (auto &D : FuncMap)
    D.second->emplaceReplacement();
  Repls.emplaceIntoReplSet(ReplSet);
}

void DpctGlobalInfo::insertCudaMalloc(const CallExpr *CE) {
  if (auto MallocVar = CudaMallocInfo::getMallocVar(CE->getArg(0)))
    insertCudaMallocInfo(MallocVar)->setSizeExpr(CE->getArg(1));
}
void DpctGlobalInfo::insertCublasAlloc(const CallExpr *CE) {
  if (auto MallocVar = CudaMallocInfo::getMallocVar(CE->getArg(2)))
    insertCudaMallocInfo(MallocVar)->setSizeExpr(CE->getArg(0), CE->getArg(1));
}
std::shared_ptr<CudaMallocInfo> DpctGlobalInfo::findCudaMalloc(const Expr *E) {
  if (auto Src = CudaMallocInfo::getMallocVar(E))
    return findCudaMallocInfo(Src);
  return std::shared_ptr<CudaMallocInfo>();
}

void KernelCallExpr::buildExecutionConfig(
    const CUDAKernelCallExpr *KernelCall) {
  auto Config = KernelCall->getConfig();
  ArgumentAnalysis A;
  for (unsigned Idx = 0; Idx < 4; ++Idx) {
    A.analyze(Config->getArg(Idx));
    ExecutionConfig.Config[Idx] = A.getReplacedString();
  }
}

void KernelCallExpr::buildKernelInfo(const CUDAKernelCallExpr *KernelCall) {
  auto &SM = DpctGlobalInfo::getSourceManager();
  SourceLocation Begin = KernelCall->getBeginLoc();
  LocInfo.NL = getNL();
  LocInfo.Indent = getIndent(Begin, SM).str();
  LocInfo.LocHash = getHashAsString(Begin.printToString(SM)).substr(0, 6);
  buildExecutionConfig(KernelCall);
}

void KernelCallExpr::addAccessorDecl() {
  auto &VM = getVarMap();
  if (VM.hasExternShared()) {
    addAccessorDecl(VM.getMap(MemVarInfo::Extern).begin()->second);
  }
  addAccessorDecl(MemVarInfo::Local);
  addAccessorDecl(MemVarInfo::Global);
  for (auto &Tex : VM.getTextureMap())
    SubmitStmts.emplace_back(Tex.second->getAccessorDecl());
  for (auto &Tex : getTextureObjectList()) {
    if (Tex) {
      SubmitStmts.emplace_back(Tex->getAccessorDecl());
    }
  }
}

void KernelCallExpr::addAccessorDecl(MemVarInfo::VarScope Scope) {
  for (auto &VI : getVarMap().getMap(Scope)) {
    addAccessorDecl(VI.second);
  }
}

void KernelCallExpr::addAccessorDecl(std::shared_ptr<MemVarInfo> VI) {
  if (VI->isShared()) {
    SubmitStmts.emplace_back(VI->getRangeDecl(ExecutionConfig.ExternMemSize));
  } else if (!VI->isGlobal()) {
    SubmitStmts.emplace_back(VI->getMemoryDecl(ExecutionConfig.ExternMemSize));
  } else if (getFilePath() != VI->getFilePath() && !VI->isShared()) {
    // Global variable definition and global variable reference are not in the
    // same file, and are not a share varible, insert extern variable
    // declaration.
    SubmitStmts.emplace_back(VI->getExternGlobalVarDecl());
  }
  SubmitStmts.emplace_back(VI->getAccessorDecl());
}

void KernelCallExpr::buildKernelArgsStmt() {
  for (auto &Arg : getArgsInfo()) {
    if (Arg.isPointer) {
      auto BufferName = Arg.getIdStringWithSuffix("buf");
      OuterStmts.emplace_back(buildString(
          "std::pair<dpct::buffer_t, size_t> ", BufferName,
          " = dpct::get_buffer_and_offset(", Arg.getArgString(), ");"));
      OuterStmts.emplace_back(buildString("size_t ",
                                          Arg.getIdStringWithSuffix("offset"),
                                          " = ", BufferName, ".second;"));
      SubmitStmts.emplace_back(buildString(
          "auto ", Arg.getIdStringWithSuffix("acc"), " = ", BufferName,
          ".first.get_access<cl::sycl::access::mode::read_write>(cgh);"));
      KernelStmts.emplace_back(buildString(
          Arg.getTypeString(), Arg.getIdStringWithIndex(), " = (",
          Arg.getTypeString(), ")(&", Arg.getIdStringWithSuffix("acc"),
          "[0] + ", Arg.getIdStringWithSuffix("offset"), ");"));
      KernelArgs += Arg.getIdStringWithIndex() + ", ";
    } else if (Arg.isRedeclareRequired) {
      OuterStmts.emplace_back(buildString("auto ", Arg.getIdStringWithIndex(),
                                          " = ", Arg.getArgString(), ";"));
      KernelArgs += Arg.getIdStringWithIndex() + ", ";
    } else {
      KernelArgs += Arg.getArgString() + ", ";
    }
  }
  if (!KernelArgs.empty())
    KernelArgs.erase(KernelArgs.length() - 2, 2);
}

void KernelCallExpr::print(KernelPrinter &Printer) {
  std::unique_ptr<KernelPrinter::Block> Block;
  if (!OuterStmts.empty()) {
    Block = std::move(Printer.block(true));
    for (auto &S : OuterStmts)
      Printer.line(S);
  }
  printSubmit(Printer);
  Block.reset();
  if (!getEvent().empty() && isSync())
    Printer.line(getEvent(), ".wait();");
}

void KernelCallExpr::printSubmit(KernelPrinter &Printer) {
  Printer.indent();
  if (!getEvent().empty()) {
    Printer << getEvent() << " = ";
  }
  if (ExecutionConfig.Stream == "0") {
    Printer << "dpct::get_default_queue";
    if (DpctGlobalInfo::getUsmLevel() == UsmLevel::restricted) {
      Printer << "_wait";
    }
    Printer << "().";
  } else {
    if (ExecutionConfig.Stream[0] == '*' || ExecutionConfig.Stream[0] == '&') {
      Printer << "(" << ExecutionConfig.Stream << ")";
    } else {
      Printer << ExecutionConfig.Stream;
    }
    Printer << "->";
  }
  (Printer << "submit(").newLine();
  printSubmitLamda(Printer);
}

void KernelCallExpr::printSubmitLamda(KernelPrinter &Printer) {
  auto Lamda = Printer.block();
  Printer.line("[&](cl::sycl::handler &cgh) {");
  {
    auto Body = Printer.block();
    for (const auto &S : SubmitStmts) {
      Printer.line(S);
    }
    printParallelFor(Printer);
  }
  Printer.line("});");
}

void KernelCallExpr::printParallelFor(KernelPrinter &Printer) {
  Printer.line("cgh.parallel_for<dpct_kernel_name<class ", getName(), "_",
               LocInfo.LocHash,
               (hasTemplateArgs() ? (", " + getTemplateArguments(true)) : ""),
               ">>(");
  auto B = Printer.block();
  Printer.line("cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get("
               "2), dpct_global_range.get(1), dpct_global_range.get(0)), "
               "cl::sycl::range<3>(dpct_local_range.get(2), "
               "dpct_local_range.get(1), dpct_local_range.get(0))),");
  Printer.line("[=](cl::sycl::nd_item<3> ", DpctGlobalInfo::getItemName(),
               ") {");
  printKenel(Printer);
  Printer.line("});");
}

void KernelCallExpr::printKenel(KernelPrinter &Printer) {
  auto B = Printer.block();
  for (auto &S : KernelStmts)
    Printer.line(S);
  Printer.indent() << getName()
                   << (hasTemplateArgs()
                           ? buildString("<", getTemplateArguments(), ">")
                           : "")
                   << "(" << KernelArgs << getExtraArguments() << ");";
  Printer.newLine();
}

std::string KernelCallExpr::getReplacement() {
  addAccessorDecl();
  addStreamDecl();
  buildKernelArgsStmt();
  addNdRangeDecl();

  std::string Result;
  llvm::raw_string_ostream OS(Result);
  KernelPrinter Printer(LocInfo.NL, LocInfo.Indent, OS);
  print(Printer);
  return OS.str();
}

void KernelCallExpr::buildInfo() {
  CallFunctionExpr::buildInfo();

  // TODO: Output debug info.
  DpctGlobalInfo::getInstance().addReplacement(std::make_shared<ExtReplacement>(
      getFilePath(), getBegin(), 0, getReplacement(), nullptr));
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
    llvm::dbgs()
        << "[CallFunctionExpr::addTemplateType] Unexpected template type: "
        << TAL.getArgument().getKind();
    return;
  }
}

void CallFunctionExpr::buildCallExprInfo(const CallExpr *CE) {
  HasArgs = CE->getNumArgs();
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

std::shared_ptr<TextureObjectInfo> CallFunctionExpr::addTextureObjectArg(
    unsigned ArgIdx, const DeclRefExpr *TexRef, bool isKernelCall) {
  if (TextureObjectInfo::isTextureObject(TexRef)) {
    if (isKernelCall) {
      if (auto VD = dyn_cast<VarDecl>(TexRef->getDecl())) {
        return addTextureObjectArgInfo(ArgIdx,
                                       std::make_shared<TextureObjectInfo>(VD));
      }
    } else if (auto PVD = dyn_cast<ParmVarDecl>(TexRef->getDecl())) {
      return addTextureObjectArgInfo(ArgIdx,
                                     std::make_shared<TextureObjectInfo>(PVD));
    }
  }
  return std::shared_ptr<TextureObjectInfo>();
}

void CallFunctionExpr::mergeTextureObjectTypeInfo() {
  for (unsigned Idx = 0; Idx < TextureObjectList.size(); ++Idx) {
    if (auto &Obj = TextureObjectList[Idx]) {
      Obj->setType(FuncInfo->getTextureTypeInfo(Idx));
    }
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
  mergeTextureObjectTypeInfo();
}

void CallFunctionExpr::emplaceReplacement() {
  buildInfo();
  DpctGlobalInfo::getInstance().addReplacement(std::make_shared<ExtReplacement>(
      FilePath, RParenLoc, 0, getExtraArguments(), nullptr));
}

std::string CallFunctionExpr::getTemplateArguments(bool WithScalarWrapped) {
  std::string Result;
  llvm::raw_string_ostream OS(Result);
  for (auto &TA : TemplateArgs) {
    if (WithScalarWrapped && !TA.isType())
      appendString(OS, "dpct_kernel_scalar<", TA.getAsString(), ">, ");
    else
      appendString(OS, TA.getAsString(), ", ");
  }
  OS.flush();
  return (Result.empty()) ? Result : Result.erase(Result.size() - 2);
}

void DeviceFunctionInfo::merge(std::shared_ptr<DeviceFunctionInfo> Other) {
  if (this == Other.get())
    return;
  VarMap.merge(Other->getVarMap());
  dpct::merge(CallExprMap, Other->CallExprMap);
  mergeTextureTypeList(Other->TextureObjectTypeList);
}

void DeviceFunctionInfo::mergeTextureTypeList(
    const std::vector<std::shared_ptr<TextureTypeInfo>> &Other) {
  auto SelfItr = TextureObjectTypeList.begin();
  auto BranchItr = Other.begin();
  while ((SelfItr != TextureObjectTypeList.end()) &&
         (BranchItr != Other.end())) {
    if (!(*SelfItr))
      *SelfItr = *BranchItr;
    ++SelfItr;
    ++BranchItr;
  }
  TextureObjectTypeList.insert(SelfItr, BranchItr, Other.end());
}

void DeviceFunctionInfo::mergeCalledTexObj(
    const std::vector<std::shared_ptr<TextureObjectInfo>> &TexObjList) {
  for (auto &Ty : TexObjList) {
    if (Ty) {
      TextureObjectTypeList[Ty->getParamIdx()] = Ty->getType();
    }
  }
}

void DeviceFunctionInfo::buildInfo() {
  if (isBuilt())
    return;
  setBuilt();
  for (auto &Call : CallExprMap) {
    Call.second->emplaceReplacement();
    VarMap.merge(Call.second->getVarMap());
    mergeCalledTexObj(Call.second->getTextureObjectList());
  }
  ExtraParams = VarMap.getExtraDeclParam(hasParams());
}

inline void DeviceFunctionDecl::emplaceReplacement() {
  // TODO: Output debug info.
  DpctGlobalInfo::getInstance().addReplacement(std::make_shared<ExtReplacement>(
      FilePath, ReplaceOffset, ReplaceLength, FuncInfo->getExtraParameters(),
      nullptr));
  for (auto &Obj : TextureObjectList) {
    if (Obj) {
      Obj->setType(FuncInfo->getTextureTypeInfo(Obj->getParamIdx()));
      Obj->addParamDeclReplacement();
    }
  }
}

void DeviceFunctionDecl::buildReplaceLocInfo(const FunctionDecl *FD) {
  if (FD->isImplicit())
    return;

  auto &SM = DpctGlobalInfo::getSourceManager();
  auto &LO = DpctGlobalInfo::getContext().getLangOpts();

  SourceLocation NextToken;
  if (FD->param_empty())
    NextToken = FD->getNameInfo().getEndLoc();
  else {
    auto EndParam = *(FD->param_end() - 1);
    NextToken = EndParam->getEndLoc();
  }
  Token Tok;
  auto Result = Lexer::getRawToken(NextToken, Tok, SM, LO, true);
  while (!Result) {
    static const llvm::StringRef VoidId = "void";
    switch (Tok.getKind()) {
    case tok::r_paren:
      ReplaceOffset = SM.getFileOffset(Tok.getLocation());
      return;
    case tok::raw_identifier:
      if (Tok.getRawIdentifier() == VoidId) {
        ReplaceOffset = SM.getFileOffset(Tok.getLocation());
        ReplaceLength = Tok.getLength();
        return;
      }
    default:
      Result = Lexer::getRawToken(Tok.getEndLoc(), Tok, SM, LO, true);
    }
  }
}

void DeviceFunctionDecl::LinkDecl(const FunctionDecl *FD, DeclList &List,
                                  std::shared_ptr<DeviceFunctionInfo> &Info) {
  if (!DpctGlobalInfo::isInRoot(FD->getBeginLoc()))
    return;
  auto D = DpctGlobalInfo::getInstance().insertDeviceFunctionDecl(FD);
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
  case Decl::UsingShadow:
    return LinkDecl(
        static_cast<const UsingShadowDecl *>(ND)->getUnderlyingDecl(), List,
        Info);
    break;
  default:
    llvm::dbgs() << "[DeviceFunctionDecl::LinkDecl] Unexpected decl type: "
                 << ND->getDeclKindName();
    return;
  }
}

std::shared_ptr<MemVarInfo> MemVarInfo::buildMemVarInfo(const VarDecl *Var) {
  if (auto Func = Var->getParentFunctionOrMethod()) {
    auto LocInfo = DpctGlobalInfo::getLocInfo(Var);
    auto VI = std::make_shared<MemVarInfo>(LocInfo.second, LocInfo.first, Var);
    DeviceFunctionDecl::LinkRedecls(dyn_cast<FunctionDecl>(Func))->addVar(VI);
    return VI;
  }

  return DpctGlobalInfo::getInstance().insertMemVarInfo(Var);
}

MemVarInfo::VarAttrKind MemVarInfo::getAddressAttr(const AttrVec &Attrs) {
  for (auto VarAttr : Attrs) {
    auto Kind = VarAttr->getKind();
    if (Kind == attr::CUDAConstant)
      return Constant;
    else if (Kind == attr::CUDADevice)
      return Device;
    else if (Kind == attr::CUDAShared)
      return Shared;
  }
  return Host;
}

std::string MemVarInfo::getMemoryType() {
  switch (Attr) {
  case clang::dpct::MemVarInfo::Device: {
    static std::string DeviceMemory = "dpct::device_memory";
    return getMemoryType(DeviceMemory, getType());
  }
  case clang::dpct::MemVarInfo::Constant: {
    static std::string ConstantMemory = "dpct::constant_memory";
    return getMemoryType(ConstantMemory, getType());
  }
  case clang::dpct::MemVarInfo::Shared: {
    static std::string SharedMemory = "dpct::local_memory";
    static std::string ExternSharedMemory = "dpct::extern_local_memory";
    if (isExtern())
      return ExternSharedMemory;
    return getMemoryType(SharedMemory, getType());
  }
  default:
    llvm::dbgs() << "[MemVarInfo::getMemoryType] Unexpected attribute.";
    return "";
  }
}

const std::string &MemVarInfo::getMemoryAttr() {
  switch (Attr) {
  case clang::dpct::MemVarInfo::Device: {
    static std::string DeviceMemory = "dpct::device";
    return DeviceMemory;
  }
  case clang::dpct::MemVarInfo::Constant: {
    static std::string ConstantMemory = "dpct::constant";
    return ConstantMemory;
  }
  case clang::dpct::MemVarInfo::Shared: {
    static std::string SharedMemory = "dpct::local";
    return SharedMemory;
  }
  default:
    llvm::dbgs() << "[MemVarInfo::getMemoryAttr] Unexpected attribute.";
    static std::string NullString;
    return NullString;
  }
}

std::string MemVarInfo::getDeclarationReplacement() {
  switch (Scope) {
  case clang::dpct::MemVarInfo::Local:
    return "";
  case clang::dpct::MemVarInfo::Extern:
    return buildString("auto ", getName(), " = ", ExternVariableName,
                       ".reinterpret<", getType()->getBaseName(), ">();");
  case clang::dpct::MemVarInfo::Global: {
    if (isShared())
      return "";
    return getMemoryDecl();
  }
  default:
    llvm::dbgs() << "[MemVarInfo::getMemoryType] Unexpected scope.";
    return "";
  }
}

CtTypeInfo::CtTypeInfo(const QualType &Ty) : CtTypeInfo() { setTypeInfo(Ty); }

CtTypeInfo::CtTypeInfo(const TypeLoc &TL, bool NeedSizeFold) : CtTypeInfo() {
  setTypeInfo(TL, NeedSizeFold);
}

std::string CtTypeInfo::getRangeArgument(const std::string &MemSize,
                                         bool MustArguments) {
  std::string Arg = "(";
  for (auto &R : Range) {
    auto Size = R.getSize();
    if (Size.empty()) {
      if (MemSize.empty()) {
        Arg += "1";
      } else {
        Arg += MemSize;
      }
    } else
      Arg += Size;
    Arg += ", ";
  }
  return (Arg.size() == 1) ? (MustArguments ? (Arg + ")") : "")
                           : Arg.replace(Arg.size() - 2, 2, ")");
}

void CtTypeInfo::setTemplateType(const std::vector<TemplateArgumentInfo> &TA) {
  assert(TemplateIndex < TA.size());
  if (isTemplate())
    TemplateType = TA[TemplateIndex].getAsType();
  TemplateList = &TA;
  for (auto &R : Range)
    R.setTemplateList(TA);
}

void CtTypeInfo::setTypeInfo(const TypeLoc &TL, bool NeedSizeFold) {
  if (auto CATL = TL.getAs<ConstantArrayTypeLoc>()) {
    if (NeedSizeFold) {
      Range.emplace_back(getFoldedArraySize(CATL));
    } else {
      Range.emplace_back(getUnfoldedArraySize(CATL));
    }
    setTypeInfo(CATL.getElementLoc(), NeedSizeFold);
  } else
    setTypeInfo(TL.getType());
}

void CtTypeInfo::setTypeInfo(QualType Ty) {
  setArrayInfo(Ty);
  setPointerInfo(Ty);
  setReferenceInfo(Ty);
  setTemplateInfo(Ty);
  setName(Ty);
}

std::string CtTypeInfo::getUnfoldedArraySize(const ConstantArrayTypeLoc &TL) {
  ExprAnalysis A;
  A.analyze(TL.getSizeExpr());
  return A.getReplacedString();
}

void CtTypeInfo::setArrayInfo(QualType &Ty) {
  ExprAnalysis A;
  while (Ty->isArrayType()) {
    if (auto CAT = dyn_cast<ConstantArrayType>(Ty))
      Range.emplace_back(getFoldedArraySize(CAT));
    else if (auto DSAT = dyn_cast<DependentSizedArrayType>(Ty)) {
      A.analyze(DSAT->getSizeExpr());
      Range.emplace_back(A.getTemplateDependentStringInfo());
    } else if (dyn_cast<IncompleteArrayType>(Ty))
      Range.emplace_back();
    Ty = Ty->getAsArrayTypeUnsafe()->getElementType();
  }
}
void CtTypeInfo::setPointerInfo(QualType &Ty) {
  while (Ty->isPointerType()) {
    IsPointer = true;
    Ty = Ty->getPointeeType();
  }
}
void CtTypeInfo::setReferenceInfo(QualType &Ty) {
  while (Ty->isReferenceType()) {
    IsReference = true;
    Ty = Ty->getPointeeType();
  }
}

void CtTypeInfo::setTemplateInfo(QualType &Ty) {
  if (auto TemplateType =
          dyn_cast<TemplateTypeParmType>(Ty->getCanonicalTypeInternal())) {
    IsTemplate = true;
    TemplateIndex = TemplateType->getIndex();
  }
}

void CtTypeInfo::setName(QualType &Ty) {
  auto &PP = DpctGlobalInfo::getContext().getPrintingPolicy();
  BaseNameWithoutQualifiers = Ty.getUnqualifiedType().getAsString(PP);

  OrginalBaseType = BaseNameWithoutQualifiers;
  if (!isTemplate())
    MapNames::replaceName(MapNames::TypeNamesMap, BaseNameWithoutQualifiers);
  auto Q = Ty.getLocalQualifiers();
  if (Q.isEmptyWhenPrinted(PP))
    BaseName = BaseNameWithoutQualifiers;
  else
    BaseName = buildString(Ty.getLocalQualifiers().getAsString(PP), " ",
                           BaseNameWithoutQualifiers);
}

void SizeInfo::setTemplateList(
    const std::vector<TemplateArgumentInfo> &TemplateList) {
  if (TDSI)
    Size = TDSI->getReplacedString(TemplateList);
}
} // namespace dpct
} // namespace clang
