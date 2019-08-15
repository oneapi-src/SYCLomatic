//===--- TextModification.cpp ---------------------------*- C++ -*---===//
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
std::string DpctGlobalInfo::CudaPath = std::string();
CompilerInstance *DpctGlobalInfo::CI = nullptr;
ASTContext *DpctGlobalInfo::Context = nullptr;
SourceManager *DpctGlobalInfo::SM = nullptr;
bool DpctGlobalInfo::KeepOriginCode = false;
const std::string MemVarInfo::ExternVariableName = "dpct_extern_memory";

bool DpctFileInfo::isInRoot() { return DpctGlobalInfo::isInRoot(FilePath); }
bool DpctFileInfo::isInCudaPath() { return DpctGlobalInfo::isInCudaPath(FilePath); }

void DpctFileInfo::buildLinesInfo() {
  if (FilePath.empty())
    return;
  auto &SM = DpctGlobalInfo::getSourceManager();
  auto FID = SM.getOrCreateFileID(SM.getFileManager().getFile(FilePath),
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
    Buffer = Content->getBuffer(SM.getDiagnostics(), SM)->getBufferStart();
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
std::shared_ptr<CudaMallocInfo>
DpctGlobalInfo::findCudaMalloc(const Expr *E) {
  if (auto Src = CudaMallocInfo::getMallocVar(E))
    return findCudaMallocInfo(Src);
  return std::shared_ptr<CudaMallocInfo>();
}

std::string KernelCallExpr::analysisExcutionConfig(const Expr *Config) {
  ArgumentAnalysis Analysis(Config);
  Analysis.analyze();
  return Analysis.getReplacedString();
}

void KernelCallExpr::buildExecutionConfig(
    const CUDAKernelCallExpr *KernelCall) {
  auto Config = KernelCall->getConfig();
  ExecutionConfig.NDSize = analysisExcutionConfig(Config->getArg(0));
  ExecutionConfig.WGSize = analysisExcutionConfig(Config->getArg(1));
  ExecutionConfig.ExternMemSize = analysisExcutionConfig(Config->getArg(2));
  ExecutionConfig.Stream = analysisExcutionConfig(Config->getArg(3));
}

void KernelCallExpr::buildKernelInfo(const CUDAKernelCallExpr *KernelCall) {
  auto &SM = DpctGlobalInfo::getSourceManager();
  SourceLocation Begin = KernelCall->getBeginLoc();
  LocInfo.NL = getNL();
  LocInfo.Indent = getIndent(Begin, SM).str();
  LocInfo.LocHash = getHashAsString(Begin.printToString(SM)).substr(0, 6);
  buildExecutionConfig(KernelCall);
}

void KernelCallExpr::getAccessorDecl(FormatStmtBlock &Block) {
  auto &VM = getVarMap();
  if (VM.hasExternShared()) {
    getAccessorDecl(Block, VM.getMap(MemVarInfo::Extern).begin()->second);
  }
  getAccessorDecl(Block, MemVarInfo::Local);
  getAccessorDecl(Block, MemVarInfo::Global);
  for (auto &Tex : VM.getTextureMap())
    Block.pushStmt(Tex.second->getAccessorDecl());
}

void KernelCallExpr::getAccessorDecl(FormatStmtBlock &Block,
                                     MemVarInfo::VarScope Scope) {
  for (auto &VI : getVarMap().getMap(Scope)) {
    getAccessorDecl(Block, VI.second);
  }
}

void KernelCallExpr::getAccessorDecl(FormatStmtBlock &Block,
                                     std::shared_ptr<MemVarInfo> VI) {
  if (!VI->isGlobal())
    Block.pushStmt(VI->getMemoryDecl(ExecutionConfig.ExternMemSize));
  if (VI->isShared())
    Block.pushStmt(VI->getRangeDecl());
  if (getFilePath() != VI->getFilePath() && !VI->isShared()) {
    // Global variable definition and global variable reference are not in the same
    // file, and are not a share varible, insert extern variable declaration.
    Block.pushStmt(VI->getExternGlobalVarDecl());
  }
  Block.pushStmt(VI->getAccessorDecl());
}

void KernelCallExpr::getStreamDecl(FormatStmtBlock &Block) {
  if (getVarMap().hasStream())
    Block.pushStmt("cl::sycl::stream ", DpctGlobalInfo::getStreamName(),
                   "(64 * 1024, 80, cgh);");
}

inline void KernelCallExpr::buildKernelPointerArgBufferAndOffsetStmt(
    const std::string &RefName, const std::string &ArgName, StmtList &Buffers) {
  Buffers.emplace_back(
      buildString("std::pair<dpct::buffer_t, size_t> ", ArgName,
                  "_buf = dpct::get_buffer_and_offset(", RefName, ");"));
  Buffers.emplace_back(
      buildString("size_t ", ArgName, "_offset = ", ArgName, "_buf.second;"));
}

inline void
KernelCallExpr::buildKernelPointerArgAccessorStmt(const std::string &ArgName,
                                                  StmtList &Accessors) {
  Accessors.emplace_back(buildString(
      "auto ", ArgName, "_acc = ", ArgName,
      "_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);"));
}

inline void
KernelCallExpr::buildKernelPointerArgRedeclStmt(const std::string &ArgName,
                                                const std::string &TypeName,
                                                StmtList &Redecls) {
  Redecls.emplace_back(buildString(TypeName, " *", ArgName, " = (", TypeName,
                                   "*)(&", ArgName, "_acc[0] + ", ArgName,
                                   "_offset);"));
}

void KernelCallExpr::buildKernelPointerArgsStmt(StmtList &Buffers,
                                                StmtList &Accessors,
                                                StmtList &Redecls) {
  for (auto &Arg : PointerArgsList) {
    auto &ArgName = Arg->getName();
    buildKernelPointerArgBufferAndOffsetStmt(Arg->getRefString(), ArgName,
                                             Buffers);
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

  llvm::raw_string_ostream OS(Result);
  appendString(OS, "{", LocInfo.NL);
  {
    FormatStmtBlock Block(LocInfo.NL, LocInfo.Indent, OS);
    for (auto &BufferStmt : Buffers)
      Block.pushStmt(BufferStmt);

    // For default stream
    if (ExecutionConfig.Stream == "0") {
      if (!getEvent().empty())
        Block.pushStmt(getEvent() + " = dpct::get_default_queue().submit(");
      else
        Block.pushStmt("dpct::get_default_queue().submit(");
    } else { // For non-default stream
      if (ExecutionConfig.Stream[0] == '*' || ExecutionConfig.Stream[0] == '&')
        Block.pushStmt("(", ExecutionConfig.Stream, ").submit(");
      else
        Block.pushStmt(ExecutionConfig.Stream, ".submit(");
    }

    {
      FMT_STMT_BLOCK
      Block.pushStmt("[&](cl::sycl::handler &cgh) {");
      {
        FMT_STMT_BLOCK
        getStreamDecl(Block);
        getAccessorDecl(Block);
        for (auto &AccStmt : Accessors)
          Block.pushStmt(AccStmt);
        for (auto &Ref : RefArgsList)
          Block.pushStmt("auto ", Ref->getDerefName(), " = ", Ref->getName(),
                         ";");
        Block.pushStmt(
            "cgh.parallel_for<dpct_kernel_name<class ", getName(), "_",
            LocInfo.LocHash,
            (hasTemplateArgs() ? (", " + getTemplateArguments(true)) : ""),
            ">>(");
        {
          FMT_STMT_BLOCK
          Block.pushStmt("cl::sycl::nd_range<3>((", ExecutionConfig.NDSize,
                         " * ", ExecutionConfig.WGSize, "), ",
                         ExecutionConfig.WGSize, "),");
          Block.pushStmt("[=](cl::sycl::nd_item<3> ",
                         DpctGlobalInfo::getItemName(), ") {");
          {
            FMT_STMT_BLOCK
            for (auto &Redecl : Redecls)
              Block.pushStmt(Redecl);
            Block.pushStmt(getName(),
                           (hasTemplateArgs()
                                ? buildString("<", getTemplateArguments(), ">")
                                : ""),
                           "(", getArguments(), ");");
          }
          Block.pushStmt("});");
        }
      }
      Block.pushStmt("});");
    }
  }
  appendString(OS, LocInfo.Indent, "}", LocInfo.NL);
  if (!getEvent().empty() && isSync())
    appendString(OS, LocInfo.Indent, getEvent() + ".wait();", LocInfo.NL);
  return OS.str();
}

void KernelCallExpr::buildInfo() {
  CallFunctionExpr::buildInfo();

  // TODO: Output debug info.
  DpctGlobalInfo::getInstance().addReplacement(
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

void CallFunctionExpr::buildCallExprInfo(const CallExpr *CE,
                                         ArgumentAnalysis &A) {
  Args.buildArgsInfo(CE, A);
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
  DpctGlobalInfo::getInstance().addReplacement(
      std::make_shared<ExtReplacement>(FilePath, RParenLoc, 0,
                                       getExtraArguments(), nullptr));
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
  DpctGlobalInfo::getInstance().addReplacement(
      std::make_shared<ExtReplacement>(FilePath, ReplaceOffset, ReplaceLength,
                                       FuncInfo->getParameters(), nullptr));
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
  default:
    dpct_unreachable("unexpected name decl type");
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
  case clang::dpct::MemVarInfo::Device: {
    static std::string DeviceMemory = "dpct::device_memory";
    return getMemoryType(DeviceMemory, getType());
  }
  case clang::dpct::MemVarInfo::Constant: {
    static std::string ConstantMemory = "dpct::constant_memory";
    return getMemoryType(ConstantMemory, getType());
  }
  case clang::dpct::MemVarInfo::Shared: {
    static std::string SharedMemory = "dpct::shared_memory";
    static std::string ExternSharedMemory = "dpct::extern_shared_memory";
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
  case clang::dpct::MemVarInfo::Device: {
    static std::string DeviceMemory = "dpct::device";
    return DeviceMemory;
  }
  case clang::dpct::MemVarInfo::Constant: {
    static std::string ConstantMemory = "dpct::constant";
    return ConstantMemory;
  }
  case clang::dpct::MemVarInfo::Shared: {
    static std::string SharedMemory = "dpct::shared";
    return SharedMemory;
  }
  default:
    llvm_unreachable("unknow variable attribute");
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
    return getMemoryDecl();
  }
  default:
    dpct_unreachable("unknow variable scope");
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
      if (MemSize.empty())
        dpct_unreachable("array size should not be empty "
                           "when external mem size is not set");
      Arg += MemSize;
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
