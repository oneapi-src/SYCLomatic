//===--- AnalysisInfo.cpp -------------------------------*- C++ -*---===//
//
// Copyright (C) 2018 - 2020 Intel Corporation. All rights reserved.
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
#include "Diagnostics.h"

#include "clang/AST/DeclTemplate.h"
#include "clang/AST/ExprCXX.h"

#include <deque>

#define TYPELOC_CAST(Target) static_cast<const Target &>(TL)

namespace clang {
extern std::function<bool(SourceLocation)> IsInRootFunc;

namespace dpct {
std::string DpctGlobalInfo::InRoot = std::string();
// TODO: implement one of this for each source language.
std::string DpctGlobalInfo::CudaPath = std::string();
UsmLevel DpctGlobalInfo::UsmLvl = UsmLevel::none;
std::unordered_set<int> DpctGlobalInfo::DeviceRNGReturnNumSet;
format::FormatRange DpctGlobalInfo::FmtRng = format::FormatRange::none;
DPCTFormatStyle DpctGlobalInfo::FmtST = DPCTFormatStyle::llvm;
bool DpctGlobalInfo::EnableCtad = false;
bool DpctGlobalInfo::EnableComments = false;
CompilerInstance *DpctGlobalInfo::CI = nullptr;
ASTContext *DpctGlobalInfo::Context = nullptr;
SourceManager *DpctGlobalInfo::SM = nullptr;
FileManager   *DpctGlobalInfo::FM = nullptr;
bool DpctGlobalInfo::KeepOriginCode = false;
bool DpctGlobalInfo::SyclNamedLambda = false;
std::map<const char *, std::shared_ptr<DpctGlobalInfo::MacroExpansionRecord>>
    DpctGlobalInfo::ExpansionRangeToMacroRecord;
std::map<const char *, std::shared_ptr<DpctGlobalInfo::MacroDefRecord>>
    DpctGlobalInfo::MacroTokenToMacroDefineLoc;
std::map<std::string, SourceLocation> DpctGlobalInfo::EndOfEmptyMacros;
std::map<std::string, SourceLocation> DpctGlobalInfo::BeginOfEmptyMacros;
std::map<MacroInfo *, bool> DpctGlobalInfo::MacroDefines;
std::set<std::string> DpctGlobalInfo::IncludingFileSet;
std::set<std::string> DpctGlobalInfo::FileSetInCompiationDB;
const std::string MemVarInfo::ExternVariableName = "dpct_local";
const int TextureObjectInfo::ReplaceTypeLength = strlen("cudaTextureObject_t");
bool DpctGlobalInfo::GuessIndentWidthMatcherFlag = false;
unsigned int DpctGlobalInfo::IndentWidth = 0;
std::unordered_map<std::string, int> DpctGlobalInfo::LocationInitIndexMap;
int DpctGlobalInfo::CurrentMaxIndex = 0;
int DpctGlobalInfo::CurrentIndexInRule = 0;
clang::format::FormatStyle DpctGlobalInfo::CodeFormatStyle;
bool DpctGlobalInfo::HasFoundDeviceChanged = false;
std::unordered_map<int, DpctGlobalInfo::HelperFuncReplInfo>
    DpctGlobalInfo::HelperFuncReplInfoMap;
int DpctGlobalInfo::HelperFuncReplInfoIndex = 1;
std::unordered_map<std::string, DpctGlobalInfo::TempVariableDeclCounter>
    DpctGlobalInfo::TempVariableDeclCounterMap;
std::unordered_set<std::string> DpctGlobalInfo::TempVariableHandledSet;
bool DpctGlobalInfo::UsingDRYPattern = true;
bool DpctGlobalInfo::SpBLASUnsupportedMatrixTypeFlag = false;

DpctGlobalInfo::DpctGlobalInfo() {
  IsInRootFunc = DpctGlobalInfo::checkInRoot;
}

std::shared_ptr<KernelCallExpr>
DpctGlobalInfo::buildLaunchKernelInfo(const CallExpr *LaunchKernelCall) {
  auto LocInfo = getLocInfo(LaunchKernelCall->getBeginLoc());
  auto FileInfo = insertFile(LocInfo.first);
  if (FileInfo->findNode<KernelCallExpr>(LocInfo.second))
    return std::shared_ptr<KernelCallExpr>();

  auto KernelInfo =
      KernelCallExpr::buildFromCudaLaunchKernel(LocInfo, LaunchKernelCall);
  if (KernelInfo) {
    FileInfo->insertNode(LocInfo.second, KernelInfo);
  }

  return KernelInfo;
}

bool DpctFileInfo::isInRoot() { return DpctGlobalInfo::isInRoot(FilePath); }
// TODO: implement one of this for each source language.
bool DpctFileInfo::isInCudaPath() {
  return DpctGlobalInfo::isInCudaPath(FilePath);
}

void DpctFileInfo::buildLinesInfo() {
  if (FilePath.empty())
    return;
  auto &SM = DpctGlobalInfo::getSourceManager();

  auto FE = SM.getFileManager().getFile(FilePath);
  if (std::error_code ec = FE.getError())
    return;
  auto FID = SM.getOrCreateFileID(FE.get(), SrcMgr::C_User);
  auto Content = SM.getSLocEntry(FID).getFile().getContentCache();
  if (!Content->SourceLineCache)
    SM.getLineNumber(FID, 0);
  auto LineCache = Content->SourceLineCache;
  auto NumLines = Content->NumLines;
  FileSize = Content->getSize();
  const char *Buffer = nullptr;
  if (!LineCache) {
    return;
  }
  if (DpctGlobalInfo::isKeepOriginCode())
    Buffer = Content->getBuffer(SM.getDiagnostics(), SM.getFileManager())
                 ->getBufferStart();
  for (unsigned L = 1; L < Content->NumLines; ++L)
    Lines.emplace_back(L, LineCache, Buffer);
  Lines.emplace_back(NumLines, LineCache[NumLines - 1], FileSize, Buffer);
}

void DpctFileInfo::buildReplacements() {
  if (!isInRoot())
    return;
  for (auto &Kernel : KernelMap)
    Kernel.second->buildInfo();

  // Below four maps are used for device RNG API migration
  for (auto &StateTypeInfo : DeviceRandomStateTypeMap)
    StateTypeInfo.second.buildInfo(FilePath, StateTypeInfo.first);

  for (auto &InitAPIInfo : DeviceRandomInitAPIMap)
    InitAPIInfo.second.buildInfo(FilePath, InitAPIInfo.first);

  buildDeviceDistrDeclInfo();
  for (auto &Info : DeviceRandomGenerateAPIMap)
    Info.second.buildInfo(FilePath, Info.first);
  for (auto &Info : DeviceRandomDistrDeclMap)
    Info.second.buildInfo(FilePath, Info.first);

  // DPCT need collect the information in curandGenerator_t decl,
  // curandCreateGenerator API call and curandSetPseudoRandomGeneratorSeed API
  // call, then can migrate them to MKL API.
  for (auto &RandomEngine : RandomEngineMap)
    RandomEngine.second->buildInfo();

  if (DpctGlobalInfo::getSpBLASUnsupportedMatrixTypeFlag()) {
    for (auto &SpBLASWarningLocOffset : SpBLASSet) {
      DiagnosticsUtils::report(getFilePath(), SpBLASWarningLocOffset,
                               Diagnostics::UNSUPPORT_MATRIX_TYPE);
    }
  }
}

void DpctFileInfo::emplaceReplacements(ReplTy &ReplSet) {
  if(!isInRoot())
    return;
  for (auto &D : FuncMap)
    D.second->emplaceReplacement();
  if(!Repls.empty())
    Repls.emplaceIntoReplSet(ReplSet[FilePath]);
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

void DpctGlobalInfo::insertRandomEngine(const Expr *E) {
  if (auto Src = RandomEngineInfo::getHandleVar(E)) {
    insertRandomEngineInfo(Src);
  }
}
std::shared_ptr<RandomEngineInfo>
DpctGlobalInfo::findRandomEngine(const Expr *E) {
  if (auto Src = RandomEngineInfo::getHandleVar(E)) {
    return findRandomEngineInfo(Src);
  }
  return std::shared_ptr<RandomEngineInfo>();
}

int KernelCallExpr::calculateOriginArgsSize() const {
  int Size = 0;
  for (auto &ArgInfo : ArgsInfo) {
    Size += ArgInfo.ArgSize;
  }
  return Size;
}

void KernelCallExpr::buildKernelInfo(const CUDAKernelCallExpr *KernelCall) {
  buildLocationInfo(KernelCall);
  buildExecutionConfig(KernelCall->getConfig()->arguments());
  buildNeedBracesInfo(KernelCall);
}

void KernelCallExpr::buildLocationInfo(const CallExpr *KernelCall) {
  auto &SM = DpctGlobalInfo::getSourceManager();
  SourceLocation Begin = KernelCall->getBeginLoc();
  LocInfo.NL = getNL();
  LocInfo.Indent = getIndent(Begin, SM).str();
  LocInfo.LocHash = getHashAsString(Begin.printToString(SM)).substr(0, 6);
}

void KernelCallExpr::buildNeedBracesInfo(const CallExpr *KernelCall) {
  NeedBraces = true;
  auto &Context = dpct::DpctGlobalInfo::getContext();
  // if parenet is CompoundStmt, then find if it has more than 1 children.
  // else if parent is ExprWithCleanups, then do futher check.
  // else it must be case like:  if/for/while(1) kernel-call, pair of
  // braces are needed.
  auto Parents = Context.getParents(*KernelCall);
  while (Parents.size() == 1) {
    if (auto *Parent = Parents[0].get<CompoundStmt>()) {
      NeedBraces = (Parent->size() > 1);
      return;
    } else if (Parents[0].get<ExprWithCleanups>()) {
      // treat ExprWithCleanups same as CUDAKernelCallExpr when they shows
      // up together
      Parents = Context.getParents(Parents[0]);
    } else {
      return;
    }
  }
}

void KernelCallExpr::addAccessorDecl() {
  auto &VM = getVarMap();
  if (VM.hasExternShared()) {
    addAccessorDecl(VM.getMap(MemVarInfo::Extern).begin()->second);
  }
  addAccessorDecl(MemVarInfo::Local);
  addAccessorDecl(MemVarInfo::Global);
  for (auto &Tex : VM.getTextureMap()) {
    SubmitStmtsList.TextureList.emplace_back(Tex.second->getAccessorDecl());
    SubmitStmtsList.SamplerList.emplace_back(Tex.second->getSamplerDecl());
  }
  for (auto &Tex : getTextureObjectList()) {
    if (Tex) {
      if (!Tex->getType()) {
        // Type PlaceHolder
        Tex->setType("PlaceHolder/*Fix the type mannually*/", 1);
      }
      SubmitStmtsList.TextureList.emplace_back(Tex->getAccessorDecl());
      SubmitStmtsList.SamplerList.emplace_back(Tex->getSamplerDecl());
    }
  }
}

void KernelCallExpr::addAccessorDecl(MemVarInfo::VarScope Scope) {
  for (auto &VI : getVarMap().getMap(Scope)) {
    addAccessorDecl(VI.second);
  }
}

bool KernelCallExpr::isIncludedFile(const std::string &CurrentFile,
                                    const std::string &CheckingFile) {
  auto CurrentFileInfo = DpctGlobalInfo::getInstance().insertFile(CurrentFile);
  auto CheckingFileInfo =
      DpctGlobalInfo::getInstance().insertFile(CheckingFile);

  std::deque<std::shared_ptr<DpctFileInfo>> Q(
      CurrentFileInfo->getIncludedFilesInfoSet().begin(),
      CurrentFileInfo->getIncludedFilesInfoSet().end());

  while (!Q.empty()) {
    if (Q.front() == nullptr) {
      continue;
    } else if (Q.front() == CheckingFileInfo) {
      return true;
    } else {
      Q.insert(Q.end(), Q.front()->getIncludedFilesInfoSet().begin(),
               Q.front()->getIncludedFilesInfoSet().end());
      Q.pop_front();
    }
  }
  return false;
}

void KernelCallExpr::addAccessorDecl(std::shared_ptr<MemVarInfo> VI) {
  if (VI->isShared()) {
    if (VI->getType()->getDimension() > 1) {
      SubmitStmtsList.RangeList.emplace_back(
          VI->getRangeDecl(ExecutionConfig.ExternMemSize));
    }
  } else if (!VI->isGlobal()) {
    SubmitStmtsList.MemoryList.emplace_back(
        VI->getMemoryDecl(ExecutionConfig.ExternMemSize));
  } else if (getFilePath() != VI->getFilePath() &&
             !isIncludedFile(getFilePath(), VI->getFilePath())) {
    // Global variable definition and global variable reference are not in the
    // same file, and are not a share varible, insert extern variable
    // declaration.
    SubmitStmtsList.ExternList.emplace_back(VI->getExternGlobalVarDecl());
  }
  VI->appendAccessorOrPointerDecl(ExecutionConfig.ExternMemSize,
                                  SubmitStmtsList.AccessorList,
                                  SubmitStmtsList.PtrList);
}

void KernelCallExpr::buildKernelArgsStmt() {
  size_t ArgCounter = 0;
  for (auto &Arg : getArgsInfo()) {
    // if current arg is the first arg with default value, insert extra args
    // before current arg
    if (getFuncInfo()) {
      if (ArgCounter == getFuncInfo()->NonDefaultParamNum) {
        KernelArgs += getExtraArguments();
      }
    }
    if(ArgCounter != 0)
      KernelArgs += ", ";

    if (Arg.IsPointer) {
      auto BufferName = Arg.getIdStringWithSuffix("buf");
      // If Arg is used as lvalue after its most recent memory allocation,
      // offsets are necessary; otherwise, offsets are not necessary.
      if (Arg.IsUsedAsLvalueAfterMalloc) {
        OuterStmts.emplace_back(
            buildString("std::pair<dpct::buffer_t, size_t> ", BufferName,
                        " = dpct::get_buffer_and_offset(", Arg.getArgString(),
                        Arg.IsDefinedOnDevice ? ".get_ptr());" : ");"));
        SubmitStmtsList.AccessorList.emplace_back(buildString(
            "auto ", Arg.getIdStringWithSuffix("acc"), " = ", BufferName,
            ".first.get_access<" + MapNames::getClNamespace() +
                "::access::mode::read_write>(cgh);"));
        OuterStmts.emplace_back(buildString("size_t ",
                                            Arg.getIdStringWithSuffix("offset"),
                                            " = ", BufferName, ".second;"));

        // If we found this is a RNG state type, we add the vec_size here.
        std::string TypeStr = Arg.getTypeString();
        if (Arg.IsDeviceRandomGeneratorType) {
          if (DpctGlobalInfo::getDeviceRNGReturnNumSet().size() == 1)
            TypeStr = TypeStr + "<" +
                      std::to_string(
                          *DpctGlobalInfo::getDeviceRNGReturnNumSet().begin()) +
                      "> *";
          else
            TypeStr = TypeStr + "<PlaceHolder/*Fix the vec_size mannually*/> *";
        }

        KernelStmts.emplace_back(buildString(
            TypeStr, Arg.getIdStringWithIndex(), " = (", TypeStr,
                        ")(&", Arg.getIdStringWithSuffix("acc"),
            "[0] + ", Arg.getIdStringWithSuffix("offset"), ");"));
        KernelArgs += Arg.getIdStringWithIndex();
      } else {
        OuterStmts.emplace_back(buildString(
            "dpct::buffer_t ", BufferName, " = dpct::get_buffer(",
            Arg.getArgString(), Arg.IsDefinedOnDevice ? ".get_ptr());" : ");"));
        SubmitStmtsList.AccessorList.emplace_back(buildString(
            "auto ", Arg.getIdStringWithSuffix("acc"), " = ", BufferName,
            ".get_access<" + MapNames::getClNamespace() +
                "::access::mode::read_write>(cgh);"));
        KernelArgs += buildString("(", Arg.getTypeString(), ")(&",
                                  Arg.getIdStringWithSuffix("acc"), "[0])");
      }
    } else if (Arg.IsRedeclareRequired || IsInMacroDefine) {
      std::string ReDeclStr = buildString("auto ", Arg.getIdStringWithIndex(),
                                          " = ", Arg.getArgString());
      if (!Arg.IsDefinedOnDevice) {
        ReDeclStr = ReDeclStr + ";";
      } else {
        if (Arg.IsKernelParamPtr) {
          ReDeclStr = ReDeclStr + ".get_ptr();";
        } else {
          ReDeclStr = ReDeclStr + "[0];";
        }
      }
      SubmitStmtsList.CommandGroupList.emplace_back(ReDeclStr);
      KernelArgs += Arg.getIdStringWithIndex();
    } else if (Arg.Texture) {
      ParameterStream OS;
      Arg.Texture->getKernelArg(OS);
      KernelArgs += OS.Str;
    } else {
      KernelArgs += Arg.getArgString();
    }
    ArgCounter += 1;
  }

  // if all params have no default value, insert extra args in the end of params
  if (getFuncInfo()) {
    if (ArgCounter == getFuncInfo()->NonDefaultParamNum) {
      KernelArgs = KernelArgs + getExtraArguments();
    }
  }

  if (KernelArgs.empty()) {
    KernelArgs += getExtraArguments();
  }
}

void KernelCallExpr::print(KernelPrinter &Printer) {
  std::unique_ptr<KernelPrinter::Block> Block;
  if (!OuterStmts.empty()) {
    if (NeedBraces)
      Block = std::move(Printer.block(true));
    else
      Block = std::move(Printer.block(false));
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
    Printer << QueueStr << ".";
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
  Printer.line("[&](" + MapNames::getClNamespace() + "::handler &cgh) {");
  {
    auto Body = Printer.block();
    SubmitStmtsList.print(Printer);
    printParallelFor(Printer);
  }
  Printer.line("});");
}

void KernelCallExpr::printParallelFor(KernelPrinter &Printer) {
  if (!SubmitStmtsList.NdRangeList.empty() &&
      DpctGlobalInfo::isCommentsEnabled())
    Printer.line("// run the kernel within defined ND range");
  if (DpctGlobalInfo::isSyclNamedLambda()) {
    Printer.line(
        "cgh.parallel_for<dpct_kernel_name<class ", getName(), "_",
        LocInfo.LocHash,
        (hasWrittenTemplateArgs() ? (", " + getTemplateArguments(true)) : ""),
        ">>(");
  } else {
    Printer.line("cgh.parallel_for(");
  }
  auto B = Printer.block();
  DpctGlobalInfo::printCtadClass(Printer.indent(),
                                 MapNames::getClNamespace() + "::nd_range", 3)
      << "(";
  static std::string CanIgnoreRangeStr =
      DpctGlobalInfo::getCtadClass(MapNames::getClNamespace() + "::range", 3) +
      "(1, 1, 1)";
  if (ExecutionConfig.DeclGlobalRange) {
    printReverseRange(Printer, "dpct_global_range");
  } else if (ExecutionConfig.GroupSize == CanIgnoreRangeStr) {
    printKernelRange(Printer, ExecutionConfig.LocalSize, "dpct_local_range",
                     ExecutionConfig.DeclLocalRange,
                     ExecutionConfig.LocalDirectRef);
  } else if (ExecutionConfig.LocalSize == CanIgnoreRangeStr) {
    printKernelRange(Printer, ExecutionConfig.GroupSize, "dpct_group_range",
                     ExecutionConfig.DeclGroupRange,
                     ExecutionConfig.GroupDirectRef);
  } else {
    printKernelRange(Printer, ExecutionConfig.GroupSize, "dpct_group_range",
                     ExecutionConfig.DeclGroupRange,
                     ExecutionConfig.GroupDirectRef);
    Printer << " * ";
    printKernelRange(Printer, ExecutionConfig.LocalSize, "dpct_local_range",
                     ExecutionConfig.DeclLocalRange,
                     ExecutionConfig.LocalDirectRef);
  }
  Printer << ", ";
  printKernelRange(Printer, ExecutionConfig.LocalSize, "dpct_local_range",
                   ExecutionConfig.DeclLocalRange,
                   ExecutionConfig.LocalDirectRef);
  (Printer << "), ").newLine();
  Printer.line("[=](" + MapNames::getClNamespace() + "::nd_item<3> ",
               DpctGlobalInfo::getItemName(), ") {");
  printKernel(Printer);
  Printer.line("});");
}

void KernelCallExpr::printKernel(KernelPrinter &Printer) {
  auto B = Printer.block();
  for (auto &S : KernelStmts)
    Printer.line(S);
  Printer.indent() << getName()
                   << (hasWrittenTemplateArgs()
                           ? buildString("<", getTemplateArguments(), ">")
                           : "")
                   << "(" << KernelArgs << ");";
  Printer.newLine();
}

std::string KernelCallExpr::getReplacement() {
  addAccessorDecl();
  addStreamDecl();
  buildKernelArgsStmt();
  addNdRangeDecl();

  if (IsInMacroDefine) {
    LocInfo.NL = "\\" + LocInfo.NL;
  }
  std::string Result;
  llvm::raw_string_ostream OS(Result);
  KernelPrinter Printer(LocInfo.NL, LocInfo.Indent, OS);
  print(Printer);
  return Printer.str();
}


CallFunctionExpr::CallFunctionExpr(unsigned Offset,
                                   const std::string &FilePathIn,
                                   const CallExpr *CE)
    : FilePath(FilePathIn), BeginLoc(Offset),
      TextureObjectList(CE ? CE->getNumArgs() : 0,
                        std::shared_ptr<TextureObjectInfo>()) {}

inline std::string CallFunctionExpr::getExtraArguments() {
  if (!FuncInfo)
    return "";
  return getVarMap().getExtraCallArguments(FuncInfo->NonDefaultParamNum,
                                           FuncInfo->ParamsNum -
                                               FuncInfo->NonDefaultParamNum);
}

const DeclRefExpr *getAddressedRef(const Expr *E) {
  E = E->IgnoreImplicitAsWritten();
  if (auto DRE = dyn_cast<DeclRefExpr>(E)) {
    if (DRE->getDecl()->getKind() == Decl::Function) {
      return DRE;
    }
  } else if (auto Paren = dyn_cast<ParenExpr>(E)) {
    return getAddressedRef(Paren->getSubExpr());
  } else if (auto Cast = dyn_cast<CastExpr>(E)) {
    return getAddressedRef(Cast->getSubExprAsWritten());
  } else if (auto UO = dyn_cast<UnaryOperator>(E)) {
    if (UO->getOpcode() == UO_AddrOf) {
      return getAddressedRef(UO->getSubExpr());
    }
  }
  return nullptr;
}

std::shared_ptr<KernelCallExpr> KernelCallExpr::buildFromCudaLaunchKernel(
    const std::pair<std::string, unsigned> &LocInfo, const CallExpr *CE) {
  auto LaunchFD = CE->getDirectCallee();
  if (!LaunchFD || LaunchFD->getName() != "cudaLaunchKernel") {
    return std::shared_ptr<KernelCallExpr>();
  }
  if (auto Callee = getAddressedRef(CE->getArg(0))) {
    auto Kernel = std::shared_ptr<KernelCallExpr>(
        new KernelCallExpr(LocInfo.second, LocInfo.first));
    Kernel->buildCalleeInfo(Callee);
    Kernel->buildLocationInfo(CE);
    Kernel->buildExecutionConfig(ArrayRef<const Expr *>{
        CE->getArg(1), CE->getArg(2), CE->getArg(4), CE->getArg(5)});
    Kernel->buildNeedBracesInfo(CE);
    auto FD =
        dyn_cast_or_null<FunctionDecl>(Callee->getReferencedDeclOfCallee());
    auto FuncInfo = Kernel->getFuncInfo();
    if (FD && FuncInfo) {
      auto ArgsArray = ExprAnalysis::ref(CE->getArg(3));
      if (!isa<DeclRefExpr>(CE->getArg(3)->IgnoreImplicitAsWritten())) {
        ArgsArray = "(" + ArgsArray + ")";
      }
      Kernel->resizeTextureObjectList(FD->getNumParams());
      for (auto &Parm : FD->parameters()) {
        Kernel->ArgsInfo.emplace_back(Parm, ArgsArray, Kernel.get());
      }
    }
    return Kernel;
  }
  return std::shared_ptr<KernelCallExpr>();
}

void KernelCallExpr::buildInfo() {
  CallFunctionExpr::buildInfo();
  TotalArgsSize =
      getVarMap().calculateExtraArgsSize() + calculateOriginArgsSize();

  if (TotalArgsSize >
      MapNames::KernelArgTypeSizeMap.at(KernelArgType::MaxParameterSize))
    DiagnosticsUtils::report(getFilePath(), getBegin(),
                             Diagnostics::EXCEED_MAX_PARAMETER_SIZE);
  // TODO: Output debug info.
  DpctGlobalInfo::getInstance().addReplacement(std::make_shared<ExtReplacement>(
      getFilePath(), getBegin(), 0, getReplacement(), nullptr));
}

void CallFunctionExpr::buildTemplateArgumentsFromTypeLoc(const TypeLoc &TL) {
  if (!TL)
    return;
  switch (TL.getTypeLocClass()) {
  /// e.g. X<T>;
  case TypeLoc::TemplateSpecialization:
    return buildTemplateArgumentsFromSpecializationType(
        TYPELOC_CAST(TemplateSpecializationTypeLoc));
  /// e.g.: X<T1>::template Y<T2>
  case TypeLoc::DependentTemplateSpecialization:
    return buildTemplateArgumentsFromSpecializationType(
        TYPELOC_CAST(DependentTemplateSpecializationTypeLoc));
  default:
    break;
  }
}

void KernelCallExpr::setIsInMacroDefine(const CUDAKernelCallExpr *KernelCall) {
  auto &SM = DpctGlobalInfo::getSourceManager();
  auto calleeSpelling = KernelCall->getCallee()->getBeginLoc();
  if (SM.isMacroArgExpansion(calleeSpelling)) {
    calleeSpelling = SM.getImmediateExpansionRange(calleeSpelling).getBegin();
  }
  calleeSpelling = SM.getSpellingLoc(calleeSpelling);
  auto ItMatch = dpct::DpctGlobalInfo::getExpansionRangeToMacroRecord().find(
      SM.getCharacterData(calleeSpelling));
  if (ItMatch != dpct::DpctGlobalInfo::getExpansionRangeToMacroRecord().end()) {
    IsInMacroDefine = true;
  }
}

#define TYPE_CAST(qual_type, type) dyn_cast<type>(qual_type)
#define ARG_TYPE_CAST(type) TYPE_CAST(ArgType, type)
#define PARM_TYPE_CAST(type) TYPE_CAST(ParmType, type)

bool TemplateArgumentInfo::isPlaceholderType(QualType QT) {
  if (auto BT = QT->getAs<BuiltinType>()) {
    if (BT->isPlaceholderType() || BT->isDependentType())
      return true;
  }
  return false;
}

template <class T>
void setTypeTemplateArgument(std::vector<TemplateArgumentInfo> &TAILis,
                             unsigned Idx, T Ty) {
  auto &TA = TAILis[Idx];
  if (TA.isNull())
    TA.setAsType(Ty);
}
template <class T>
void setNonTypeTemplateArgument(std::vector<TemplateArgumentInfo> &TAILis,
                                unsigned Idx, T Ty) {
  auto &TA = TAILis[Idx];
  if (TA.isNull())
    TA.setAsNonType(Ty);
}

/// Return true if Ty is TypedefType.
bool getInnerType(QualType &Ty, TypeLoc &TL) {
  if (auto TypedefTy = dyn_cast<TypedefType>(Ty)) {
    if (!TemplateArgumentInfo::isPlaceholderType(TypedefTy->desugar())) {
      Ty = TypedefTy->desugar();
      TL = TypedefTy->getDecl()->getTypeSourceInfo()->getTypeLoc();
      return true;
    }
  } else if (auto ElaboratedTy = dyn_cast<ElaboratedType>(Ty)) {
    Ty = ElaboratedTy->getNamedType();
    if (TL)
      TL = TYPELOC_CAST(ElaboratedTypeLoc).getNamedTypeLoc();
    return true;
  }
  return false;
}

void deduceTemplateArgumentFromType(std::vector<TemplateArgumentInfo> &TAIList,
                                    QualType ParmType, QualType ArgType,
                                    TypeLoc TL = TypeLoc());

template <class NonTypeValueT>
void deduceNonTypeTemplateArgument(std::vector<TemplateArgumentInfo> &TAIList,
                                   const Expr *Parm,
                                   const NonTypeValueT &Value) {
  Parm = Parm->IgnoreImplicitAsWritten();
  if (auto DRE = dyn_cast<DeclRefExpr>(Parm)) {
    if (auto NTTPD = dyn_cast<NonTypeTemplateParmDecl>(DRE->getDecl())) {
      setNonTypeTemplateArgument(TAIList, NTTPD->getIndex(), Value);
    }
  } else if (auto C = dyn_cast<ConstantExpr>(Parm)) {
    deduceNonTypeTemplateArgument(TAIList, C->getSubExpr(), Value);
  } else if (auto S = dyn_cast<SubstNonTypeTemplateParmExpr>(Parm)) {
    deduceNonTypeTemplateArgument(TAIList, S->getReplacement(), Value);
  }
}

void deduceTemplateArgumentFromTemplateArgs(
    std::vector<TemplateArgumentInfo> &TAIList, const TemplateArgument &Parm,
    const TemplateArgument &Arg,
    const TemplateArgumentLoc &ArgLoc = TemplateArgumentLoc()) {
  switch (Parm.getKind()) {
  case TemplateArgument::Expression:
    switch (Arg.getKind()) {
    case TemplateArgument::Expression:
      deduceNonTypeTemplateArgument(TAIList, Parm.getAsExpr(), Arg.getAsExpr());
      return;
    case TemplateArgument::Integral:
      if (ArgLoc.getArgument().isNull())
        deduceNonTypeTemplateArgument(TAIList, Parm.getAsExpr(),
                                      Arg.getAsIntegral());
      else
        deduceNonTypeTemplateArgument(TAIList, Parm.getAsExpr(),
                                      ArgLoc.getSourceExpression());
      break;
    default:
      break;
    }
    break;
  case TemplateArgument::Type:
    if (Arg.getKind() != TemplateArgument::Type)
      return;
    if (ArgLoc.getArgument().isNull()) {
      deduceTemplateArgumentFromType(TAIList, Parm.getAsType(),
                                     Arg.getAsType());
    } else {
      deduceTemplateArgumentFromType(TAIList, Parm.getAsType(),
                                     ArgLoc.getTypeSourceInfo()->getType(),
                                     ArgLoc.getTypeSourceInfo()->getTypeLoc());
    }
  default:
    break;
  }
}

void deduceTemplateArgumentFromTemplateSpecialization(
    std::vector<TemplateArgumentInfo> &TAIList,
    const TemplateSpecializationType *ParmType, QualType ArgType,
    TypeLoc TL = TypeLoc()) {
  switch (ArgType->getTypeClass()) {
  case Type::Record:
    if (auto CTSD = dyn_cast<ClassTemplateSpecializationDecl>(
            ARG_TYPE_CAST(RecordType)->getDecl())) {
      if (CTSD->getTypeAsWritten() &&
          CTSD->getTypeAsWritten()->getType()->getTypeClass() ==
              Type::TemplateSpecialization) {
        auto TL = CTSD->getTypeAsWritten()->getTypeLoc();
        auto &TSTL = TYPELOC_CAST(TemplateSpecializationTypeLoc);
        for (unsigned i = 0; i < ParmType->getNumArgs(); ++i) {
          deduceTemplateArgumentFromTemplateArgs(
              TAIList, ParmType->getArg(i), TSTL.getArgLoc(i).getArgument(),
              TSTL.getArgLoc(i));
        }
      }
    }
    break;
  case Type::TemplateSpecialization:
    if (TL) {
      auto &TSTL = TYPELOC_CAST(TemplateSpecializationTypeLoc);
      for (unsigned i = 0; i < ParmType->getNumArgs(); ++i) {
        deduceTemplateArgumentFromTemplateArgs(TAIList, ParmType->getArg(i),
                                               TSTL.getArgLoc(i).getArgument(),
                                               TSTL.getArgLoc(i));
      }
    } else {
      auto TST = ARG_TYPE_CAST(TemplateSpecializationType);
      for (unsigned i = 0; i < ParmType->getNumArgs(); ++i) {
        deduceTemplateArgumentFromTemplateArgs(TAIList, ParmType->getArg(i),
                                               TST->getArg(i));
      }
    }
    break;
  default:
    break;
  }
}

TypeLoc getPointeeTypeLoc(TypeLoc TL) {
  if (!TL)
    return TL;
  switch (TL.getTypeLocClass()) {
  case TypeLoc::ConstantArray:
  case TypeLoc::DependentSizedArray:
  case TypeLoc::IncompleteArray:
    return TYPELOC_CAST(ArrayTypeLoc).getElementLoc();
  case TypeLoc::Pointer:
    return TYPELOC_CAST(PointerTypeLoc).getPointeeLoc();
  default:
    return TypeLoc();
  }
}

void deduceTemplateArgumentFromArrayElement(
    std::vector<TemplateArgumentInfo> &TAIList, QualType ParmType,
    QualType ArgType, TypeLoc TL = TypeLoc()) {
  const ArrayType *ParmArray = PARM_TYPE_CAST(ArrayType);
  const ArrayType *ArgArray = ARG_TYPE_CAST(ArrayType);
  if (!ParmArray || !ArgArray) {
    return;
  }
  deduceTemplateArgumentFromType(TAIList, ParmArray->getElementType(),
                                 ArgArray->getElementType(),
                                 getPointeeTypeLoc(TL));
}

void deduceTemplateArgumentFromType(std::vector<TemplateArgumentInfo> &TAIList,
                                    QualType ParmType, QualType ArgType,
                                    TypeLoc TL) {
  ParmType = ParmType.getCanonicalType();
  if (!ParmType->isDependentType())
    return;

  if (TL) {
    TL = TL.getUnqualifiedLoc();
    if (TL.getTypePtr()->getTypeClass() != ArgType->getTypeClass() ||
        TL.getTypeLocClass() == TypeLoc::SubstTemplateTypeParm)
      TL = TypeLoc();
  }

  switch (ParmType->getTypeClass()) {
  case Type::TemplateTypeParm:
    if (TL) {
      setTypeTemplateArgument(
          TAIList, PARM_TYPE_CAST(TemplateTypeParmType)->getIndex(), TL);
    } else {
      ArgType.removeLocalCVRQualifiers(ParmType.getCVRQualifiers());
      setTypeTemplateArgument(
          TAIList, PARM_TYPE_CAST(TemplateTypeParmType)->getIndex(), ArgType);
    }
    break;
  case Type::TemplateSpecialization:
    deduceTemplateArgumentFromTemplateSpecialization(
        TAIList, PARM_TYPE_CAST(TemplateSpecializationType), ArgType, TL);
    break;
  case Type::Pointer:
    if (auto ArgPointer = ARG_TYPE_CAST(PointerType)) {
      deduceTemplateArgumentFromType(TAIList, ParmType->getPointeeType(),
                                     ArgPointer->getPointeeType(),
                                     getPointeeTypeLoc(TL));
    } else if (auto ArgArray = ARG_TYPE_CAST(ArrayType)) {
      deduceTemplateArgumentFromType(TAIList, ParmType->getPointeeType(),
                                     ArgArray->getElementType(),
                                     getPointeeTypeLoc(TL));
    }
    break;
  case Type::LValueReference: {
    auto ParmPointeeType =
        PARM_TYPE_CAST(LValueReferenceType)->getPointeeTypeAsWritten();
    if (auto LVRT = ARG_TYPE_CAST(LValueReferenceType)) {
      deduceTemplateArgumentFromType(
          TAIList, ParmPointeeType, LVRT->getPointeeTypeAsWritten(),
          TL ? TYPELOC_CAST(LValueReferenceTypeLoc).getPointeeLoc() : TL);
    } else if (ParmPointeeType.getQualifiers().hasConst()) {
      deduceTemplateArgumentFromType(TAIList, ParmPointeeType, ArgType, TL);
    }
    break;
  }
  case Type::RValueReference:
    if (auto RVRT = ARG_TYPE_CAST(RValueReferenceType)) {
      deduceTemplateArgumentFromType(
          TAIList,
          PARM_TYPE_CAST(RValueReferenceType)->getPointeeTypeAsWritten(),
          RVRT->getPointeeTypeAsWritten(),
          TL ? TYPELOC_CAST(RValueReferenceTypeLoc).getPointeeLoc() : TL);
    }
    break;
  case Type::ConstantArray: {
    auto ArgConstArray = ARG_TYPE_CAST(ConstantArrayType);
    auto ParmConstArray = PARM_TYPE_CAST(ConstantArrayType);
    if (ArgConstArray &&
        ArgConstArray->getSize() == ParmConstArray->getSize()) {
      deduceTemplateArgumentFromArrayElement(TAIList, ParmType, ArgType, TL);
    }
    break;
  }
  case Type::IncompleteArray:
    deduceTemplateArgumentFromArrayElement(TAIList, ParmType, ArgType, TL);
    break;
  case Type::DependentSizedArray: {
    deduceTemplateArgumentFromArrayElement(TAIList, ParmType, ArgType, TL);
    auto ParmSizeExpr = PARM_TYPE_CAST(DependentSizedArrayType)->getSizeExpr();
    if (TL && TL.getTypePtr()->isArrayType()) {
      deduceNonTypeTemplateArgument(TAIList, ParmSizeExpr,
                                    TYPELOC_CAST(ArrayTypeLoc).getSizeExpr());
    } else if (auto DSAT = ARG_TYPE_CAST(DependentSizedArrayType)) {
      deduceNonTypeTemplateArgument(TAIList, ParmSizeExpr, DSAT->getSizeExpr());
    } else if (auto CAT = ARG_TYPE_CAST(ConstantArrayType)) {
      deduceNonTypeTemplateArgument(TAIList, ParmSizeExpr, CAT->getSize());
    }
    break;
  }
  default:
    break;
  }

  if (getInnerType(ArgType, TL)) {
    deduceTemplateArgumentFromType(TAIList, ParmType, ArgType, TL);
  }
}

void deduceTemplateArgument(std::vector<TemplateArgumentInfo> &TAIList,
                            const Expr *Arg, const ParmVarDecl *PVD) {
  auto ArgType = Arg->getType();
  auto ParmType = PVD->getType();

  TypeLoc TL;
  if (auto DRE = dyn_cast<DeclRefExpr>(Arg->IgnoreImplicitAsWritten())) {
    if (auto DD = dyn_cast<DeclaratorDecl>(DRE->getDecl()))
      TL = DD->getTypeSourceInfo()->getTypeLoc();
  }

  deduceTemplateArgumentFromType(TAIList, ParmType, ArgType, TL);
}

void deduceTemplateArguments(const CallExpr *CE,
                             const FunctionTemplateDecl *FTD,
                             std::vector<TemplateArgumentInfo> &TAIList) {
  if (!FTD)
    return;

  if (!DpctGlobalInfo::isInRoot(FTD->getBeginLoc()))
    return;
  auto &TemplateParmsList = *FTD->getTemplateParameters();
  if (TAIList.size() == TemplateParmsList.size())
    return;

  TAIList.resize(TemplateParmsList.size());

  auto ArgItr = CE->arg_begin();
  auto ParmItr = FTD->getTemplatedDecl()->param_begin();
  while (ArgItr != CE->arg_end() &&
         ParmItr != FTD->getTemplatedDecl()->param_end()) {
    deduceTemplateArgument(TAIList, *ArgItr, *ParmItr);
    ++ArgItr;
    ++ParmItr;
  }
  for (size_t i = 0; i < TAIList.size(); ++i) {
    auto &Arg = TAIList[i];
    if (!Arg.isNull())
      continue;
    auto TemplateParm = TemplateParmsList.getParam(i);
    if (auto TTPD = dyn_cast<TemplateTypeParmDecl>(TemplateParm)) {
      if (TTPD->hasDefaultArgument())
        Arg.setAsType(TTPD->getDefaultArgumentInfo()->getTypeLoc());
    } else if (auto NTTPD = dyn_cast<NonTypeTemplateParmDecl>(TemplateParm)) {
      if (NTTPD->hasDefaultArgument())
        Arg.setAsNonType(NTTPD->getDefaultArgument());
    }
  }
}

void deduceTemplateArguments(const CallExpr *CE, const FunctionDecl *FD,
                             std::vector<TemplateArgumentInfo> &TAIList) {
  if (FD)
    return deduceTemplateArguments(CE, FD->getPrimaryTemplate(), TAIList);
}

void deduceTemplateArguments(const CallExpr *CE, const NamedDecl *ND,
                             std::vector<TemplateArgumentInfo> &TAIList) {
  if (!ND)
    return;
  if (auto FTD = dyn_cast<FunctionTemplateDecl>(ND)) {
    deduceTemplateArguments(CE, FTD, TAIList);
  } else if (auto FD = dyn_cast<FunctionDecl>(ND)) {
    deduceTemplateArguments(CE, FD, TAIList);
  } else if (auto UD = dyn_cast<UsingShadowDecl>(ND)) {
    deduceTemplateArguments(CE, UD->getUnderlyingDecl(), TAIList);
  }
}

void CallFunctionExpr::buildCalleeInfo(const Expr *Callee) {
  if (auto CallDecl =
          dyn_cast_or_null<FunctionDecl>(Callee->getReferencedDeclOfCallee())) {
    Name = getName(CallDecl);
    FuncInfo = DeviceFunctionDecl::LinkRedecls(CallDecl);
    if (auto DRE = dyn_cast<DeclRefExpr>(Callee)) {
      buildTemplateArguments(DRE->template_arguments());
    }
  } else if (auto Unresolved = dyn_cast<UnresolvedLookupExpr>(Callee)) {
    Name = Unresolved->getName().getAsString();
    FuncInfo = DeviceFunctionDecl::LinkUnresolved(Unresolved);
    buildTemplateArguments(Unresolved->template_arguments());
  } else if (auto DependentScope =
                 dyn_cast<CXXDependentScopeMemberExpr>(Callee)) {
    Name = DependentScope->getMember().getAsString();
    buildTemplateArguments(DependentScope->template_arguments());
  } else if (auto DSDRE = dyn_cast<DependentScopeDeclRefExpr>(Callee)) {
    Name = DSDRE->getDeclName().getAsString();
    buildTemplateArgumentsFromTypeLoc(DSDRE->getQualifierLoc().getTypeLoc());
  }
}

void CallFunctionExpr::buildCallExprInfo(const CallExpr *CE) {
  buildCalleeInfo(CE->getCallee()->IgnoreParenImpCasts());
  buildTextureObjectArgsInfo(CE);

  bool HasImplicitArg = false;
  if (auto FD = CE->getDirectCallee()) {
    deduceTemplateArguments(CE, FD, TemplateArgs);
    HasImplicitArg = isa<CXXOperatorCallExpr>(CE) && isa<CXXMethodDecl>(FD);
  } else if (auto Unresolved = dyn_cast<UnresolvedLookupExpr>(
                 CE->getCallee()->IgnoreImplicitAsWritten())) {
    if (Unresolved->getNumDecls())
      deduceTemplateArguments(CE, Unresolved->decls_begin().getDecl(),
		  TemplateArgs);
  }

  if (HasImplicitArg) {
    HasArgs = CE->getNumArgs() == 1;
  } else {
    HasArgs = CE->getNumArgs();
  }

  if (FuncInfo) {
    if (FuncInfo->ParamsNum == 0) {
      ExtraArgLoc =
          DpctGlobalInfo::getSourceManager().getFileOffset(CE->getRParenLoc());
    } else if (FuncInfo->NonDefaultParamNum == 0) {
      // if all params have default value
      ExtraArgLoc = DpctGlobalInfo::getSourceManager().getFileOffset(
          CE->getArg(HasImplicitArg ? 1 : 0)->getBeginLoc());
    } else {
      // if some params have default value, set ExtraArgLoc to the location
      // before the comma
      if (CE->getNumArgs() > FuncInfo->NonDefaultParamNum - 1) {
        auto &SM = DpctGlobalInfo::getSourceManager();
        auto TokenLoc = Lexer::getLocForEndOfToken(
            SM.getSpellingLoc(
                CE->getArg(FuncInfo->NonDefaultParamNum - 1 + HasImplicitArg)
                    ->getEndLoc()),
            0, SM, DpctGlobalInfo::getContext().getLangOpts());
        ExtraArgLoc =
            DpctGlobalInfo::getSourceManager().getFileOffset(TokenLoc);
      } else {
        ExtraArgLoc = 0;
      }
    }
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

  const std::string &DefFilePath = FuncInfo->getDefinitionFilePath();
  if (!DefFilePath.empty() && DefFilePath != getFilePath()) {
    FuncInfo->setNeedSyclExternMacro();
  }
  FuncInfo->buildInfo();
  VarMap.merge(FuncInfo->getVarMap(), TemplateArgs);
  mergeTextureObjectTypeInfo();
}

void CallFunctionExpr::emplaceReplacement() {
  buildInfo();
  DpctGlobalInfo::getInstance().addReplacement(std::make_shared<ExtReplacement>(
      FilePath, ExtraArgLoc, 0, getExtraArguments(), nullptr));
}

std::string CallFunctionExpr::getTemplateArguments(bool WithScalarWrapped) {
  std::string Result;
  llvm::raw_string_ostream OS(Result);
  for (auto &TA : TemplateArgs) {
    if (TA.isNull() || !TA.isWritten())
      continue;
    if (WithScalarWrapped && !TA.isType())
      appendString(OS, "dpct_kernel_scalar<", TA.getString(), ">, ");
    else
      appendString(OS, TA.getString(), ", ");
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
  IsStatic = Other->IsStatic || IsStatic;
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
}

inline void DeviceFunctionDecl::emplaceReplacement() {
  // TODO: Output debug info.
  auto Repl = std::make_shared<ExtReplacement>(
      FilePath, ReplaceOffset, ReplaceLength,
      FuncInfo->getExtraParameters(FormatInformation), nullptr);
  Repl->setNotFormatFlag();
  DpctGlobalInfo::getInstance().addReplacement(Repl);

  if (FuncInfo->IsSyclExternMacroNeeded()) {
    std::string StrRepl = "SYCL_EXTERNAL ";
    DpctGlobalInfo::getInstance().addReplacement(
        std::make_shared<ExtReplacement>(FilePath, Offset, 0, StrRepl,
                                         nullptr));
  }
  for (auto &Obj : TextureObjectList) {
    if (Obj) {
      Obj->setType(FuncInfo->getTextureTypeInfo(Obj->getParamIdx()));
      if (!Obj->getType()) {
        // Type PlaceHolder
        Obj->setType("PlaceHolder/*Fix the type mannually*/", 1);
      }
      Obj->addParamDeclReplacement();
    }
  }
}

void DeviceFunctionDecl::buildReplaceLocInfo(const FunctionDecl *FD) {
  if (FD->isImplicit()) {
    NonDefaultParamNum = FD->getNumParams();
    return;
  }

  auto &SM = DpctGlobalInfo::getSourceManager();
  auto &LO = DpctGlobalInfo::getContext().getLangOpts();

  // Need to get the last decl if there are many decl of the same function
  NonDefaultParamNum = FD->getMostRecentDecl()->getMinRequiredArguments();
  auto FisrtFD = FD->getFirstDecl();
  SourceLocation NextToken;
  if (NonDefaultParamNum == 0) {
    NextToken = SM.getSpellingLoc(FD->getNameInfo().getEndLoc());
    NextToken = Lexer::getLocForEndOfToken(NextToken, 0, SM, LO);
  }
  else {
    NextToken = FD->getParamDecl(NonDefaultParamNum - 1)->getEndLoc();
    if (SM.isMacroArgExpansion(NextToken)) {
      NextToken =
          SM.getSpellingLoc(SM.getImmediateExpansionRange(NextToken).getEnd());
    }
    NextToken = Lexer::getLocForEndOfToken(NextToken, 0, SM, LO);
  }

  // PARAMETER INSERTING LOCATION RULES:
  // 1. Origin parameters number < 2
  //    Do not add new line until longer than 80. The new line begin is aligned
  //    with the end location of "("
  // 2. Origin parameters number >= 2
  //    2.1 If each parameter is in a single line:
  //           Each added parameter is in a single line.
  //           The new line begin is aligned with the last parameter's line
  //           begin
  //    2.2 There are 2 parameters in one line:
  //           Do not add new line until longer than 80.
  //           The new line begin is aligned with the last parameter's line
  //           begin
  FormatInformation.EnableFormat = true;

  auto ItMatch = dpct::DpctGlobalInfo::getExpansionRangeToMacroRecord().find(
      SM.getCharacterData(NextToken));
  if (ItMatch != dpct::DpctGlobalInfo::getExpansionRangeToMacroRecord().end()) {
    FormatInformation.IsAllParamsOneLine = true;
    if (NextToken.isMacroID()) {
      NextToken =
          SM.getSpellingLoc(SM.getImmediateExpansionRange(NextToken).getEnd());
    }
  } else if (NonDefaultParamNum >= 2) {
    FormatInformation.IsAllParamsOneLine = false;
    auto BeginParam = *(FD->param_begin());
    SourceLocation BeginParamLoc = BeginParam->getBeginLoc();

    if (BeginParamLoc.isMacroID())
      BeginParamLoc = SM.getExpansionLoc(BeginParamLoc);
    if (NextToken.isMacroID())
      NextToken = SM.getExpansionLoc(NextToken);

     unsigned int NeedRemoveLength = 0;
    calculateRemoveLength<CUDAGlobalAttr>(FD, "__global__", NeedRemoveLength,
                                          BeginParamLoc, SM, LO);
    calculateRemoveLength<CUDADeviceAttr>(FD, "__device__", NeedRemoveLength,
                                          BeginParamLoc, SM, LO);
    calculateRemoveLength<CUDAHostAttr>(FD, "__host__", NeedRemoveLength,
                                        BeginParamLoc, SM, LO);

    auto ParamA = FisrtFD->param_begin();
    auto ParamB = ParamA++;
    FormatInformation.IsEachParamNL = true;
    while (ParamA != FisrtFD->param_end() && ParamB != FisrtFD->param_end()) {
      bool InValidFlag = false;
      if (isInSameLine(SM.getExpansionLoc((*ParamA)->getBeginLoc()),
                       SM.getExpansionLoc((*ParamB)->getBeginLoc()), SM,
                       InValidFlag) &&
          !InValidFlag) {
        FormatInformation.IsEachParamNL = false;
        break;
      }
      ParamA++;
      ParamB++;
    }

    if (FormatInformation.IsEachParamNL) {
      FormatInformation.NewLineIndentStr = getIndent(NextToken, SM).str();
      FormatInformation.NewLineIndentLength = getIndent(NextToken, SM).size();
      FormatInformation.CurrentLength = getCurrnetColumn(NextToken, SM);
    } else {
      bool InValidFlag = false;
      if (isInSameLine(BeginParamLoc, NextToken, SM, InValidFlag) &&
          !InValidFlag) {
        FormatInformation.NewLineIndentLength =
            getCurrnetColumn(BeginParamLoc, SM) - 1 - NeedRemoveLength;
        FormatInformation.NewLineIndentStr =
            std::string(FormatInformation.NewLineIndentLength, ' ');
        FormatInformation.CurrentLength =
            getCurrnetColumn(NextToken, SM) - NeedRemoveLength;
      } else {
        FormatInformation.NewLineIndentStr = getIndent(NextToken, SM).str();
        FormatInformation.NewLineIndentLength = getIndent(NextToken, SM).size();
        FormatInformation.CurrentLength = getCurrnetColumn(NextToken, SM);
      }
    }
  } else {
    // NonDefaultParamNum < 2
    FormatInformation.IsAllParamsOneLine = false;
    FormatInformation.IsEachParamNL = false;

    SourceLocation BeginParamLoc;
    if (NonDefaultParamNum == 0) {
      BeginParamLoc = NextToken;
      // In this case, the NextToken is the location of the end of function name,
      // so need get the next token end. Example: void foobar ( );
      //                                                     | |
      //                                  "NextToken" is here^ ^Want to get here
      Token Tok;
      auto Result = Lexer::getRawToken(BeginParamLoc, Tok, SM, LO, true);
      if (!Result)
        BeginParamLoc = Tok.getEndLoc();
    } else {
      auto BeginParam = *(FD->param_begin());
      BeginParamLoc = BeginParam->getBeginLoc();
    }

    if (BeginParamLoc.isMacroID())
      BeginParamLoc = SM.getExpansionLoc(BeginParamLoc);

    unsigned int NeedRemoveLength = 0;
    calculateRemoveLength<CUDAGlobalAttr>(FD, "__global__", NeedRemoveLength,
                                          BeginParamLoc, SM, LO);
    calculateRemoveLength<CUDADeviceAttr>(FD, "__device__", NeedRemoveLength,
                                          BeginParamLoc, SM, LO);
    calculateRemoveLength<CUDAHostAttr>(FD, "__host__", NeedRemoveLength,
                                        BeginParamLoc, SM, LO);

    FormatInformation.NewLineIndentLength =
        getCurrnetColumn(BeginParamLoc, SM) - 1 - NeedRemoveLength;
    FormatInformation.NewLineIndentStr =
        std::string(FormatInformation.NewLineIndentLength, ' ');

    if (NonDefaultParamNum == 0) {
      FormatInformation.CurrentLength =
          getCurrnetColumn(BeginParamLoc, SM) - NeedRemoveLength;
    } else {
      FormatInformation.CurrentLength =
          getCurrnetColumn(NextToken, SM) - NeedRemoveLength;
    }
  }

  // Find the correct ReplaceOffset to insert new parameter
  Token Tok;
  auto Result = Lexer::getRawToken(NextToken, Tok, SM, LO, true);
  while (!Result) {
    static const llvm::StringRef VoidId = "void";
    switch (Tok.getKind()) {
    case tok::r_paren:
      ReplaceOffset = SM.getFileOffset(Tok.getLocation());
      return;
    case tok::comma:
      ReplaceOffset = SM.getFileOffset(Tok.getLocation());
      return;
    case tok::raw_identifier:
      if (Tok.getRawIdentifier() == VoidId) {
        ReplaceOffset = SM.getFileOffset(Tok.getLocation());
        ReplaceLength = Tok.getLength();
        return;
      }
      else {
        ReplaceOffset = SM.getFileOffset(Tok.getLocation());
        return;
      }
    // May fall through
    default:
      Result = Lexer::getRawToken(Tok.getEndLoc(), Tok, SM, LO, true);
    }
  }
}

void DeviceFunctionDecl::setFuncInfo(std::shared_ptr<DeviceFunctionInfo> Info) {
  FuncInfo = Info;
  if (IsDefFilePathNeeded)
    FuncInfo->setDefinitionFilePath(FilePath);
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
  LinkDecl(FTD->getAsFunction(), List, Info);
  LinkDeclRange(FTD->specializations(), List, Info);
}

void DeviceFunctionDecl::LinkDecl(const NamedDecl *ND, DeclList &List,
                                  std::shared_ptr<DeviceFunctionInfo> &Info) {
  switch (ND->getKind()) {
  case Decl::CXXMethod:
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
    DpctDiags() << "[DeviceFunctionDecl::LinkDecl] Unexpected decl type: "
      << ND->getDeclKindName() << "\n";
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
    if (Kind == attr::CUDAManaged)
      return Managed;
  }
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
  case clang::dpct::MemVarInfo::Managed: {
    static std::string ManagedMemory = "dpct::shared_memory";
    return getMemoryType(ManagedMemory, getType());
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
  case clang::dpct::MemVarInfo::Managed: {
    static std::string ManagedMemory = "dpct::shared";
    return ManagedMemory;
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
    return buildString("auto ", getName(), " = (", getType()->getBaseName(),
                       " *)", ExternVariableName, ";");
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

std::string MemVarMap::getExtraCallArguments(bool HasPreParam, bool HasPostParam) const {
  return getArgumentsOrParameters<CallArgument>(HasPreParam, HasPostParam);
}
std::string MemVarMap::getExtraDeclParam(bool HasPreParam, bool HasPostParam,
                                         FormatInfo FormatInformation) const {
  return getArgumentsOrParameters<DeclParameter>(HasPreParam, HasPostParam,
                                                 FormatInformation);
}
std::string MemVarMap::getKernelArguments(bool HasPreParam, bool HasPostParam) const {
  return getArgumentsOrParameters<KernelArgument>(HasPreParam, HasPostParam);
}

CtTypeInfo::CtTypeInfo(const TypeLoc &TL, bool NeedSizeFold, bool IsShared)
    : IsPointer(false), IsTemplate(false), IsShared(IsShared) {
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

void CtTypeInfo::setTypeInfo(const TypeLoc &TL, bool NeedSizeFold) {
  switch (TL.getTypeLocClass()) {
  case TypeLoc::Qualified:
    BaseName = TL.getType().getLocalQualifiers().getAsString(
        DpctGlobalInfo::getContext().getPrintingPolicy());
    return setTypeInfo(TYPELOC_CAST(QualifiedTypeLoc).getUnqualifiedLoc(),
                       NeedSizeFold);
  case TypeLoc::ConstantArray:
    return setArrayInfo(TYPELOC_CAST(ConstantArrayTypeLoc), NeedSizeFold);
  case TypeLoc::DependentSizedArray:
    return setArrayInfo(TYPELOC_CAST(DependentSizedArrayTypeLoc), NeedSizeFold);
  case TypeLoc::IncompleteArray:
    return setArrayInfo(TYPELOC_CAST(IncompleteArrayTypeLoc), NeedSizeFold);
  case TypeLoc::Pointer:
    if (IsShared)
      break;
    IsPointer = true;
    return setTypeInfo(TYPELOC_CAST(PointerTypeLoc).getPointeeLoc());
  case TypeLoc::LValueReference:
  case TypeLoc::RValueReference:
    IsReference = true;
    return setTypeInfo(TYPELOC_CAST(ReferenceTypeLoc).getPointeeLoc());
  default:
    break;
  }
  setName(TL);
}

void CtTypeInfo::setArrayInfo(const IncompleteArrayTypeLoc &TL,
                              bool NeedSizeFold) {
  Range.emplace_back();
  setTypeInfo(TL.getElementLoc(), NeedSizeFold);
}

void CtTypeInfo::setArrayInfo(const DependentSizedArrayTypeLoc &TL,
                              bool NeedSizeFold) {
  ExprAnalysis EA;
  EA.analyze(TL.getSizeExpr());
  Range.emplace_back(EA.getTemplateDependentStringInfo());
  setTypeInfo(TL.getElementLoc(), NeedSizeFold);
}

void CtTypeInfo::setArrayInfo(const ConstantArrayTypeLoc &TL,
                              bool NeedSizeFold) {
  if (NeedSizeFold) {
    Range.emplace_back(getFoldedArraySize(TL));
  } else {
    Range.emplace_back(getUnfoldedArraySize(TL));
  }
  setTypeInfo(TL.getElementLoc(), NeedSizeFold);
}

std::string CtTypeInfo::getUnfoldedArraySize(const ConstantArrayTypeLoc &TL) {
  ExprAnalysis A;
  A.analyze(TL.getSizeExpr());
  return A.getReplacedString();
}

void CtTypeInfo::setName(const TypeLoc &TL) {
  ExprAnalysis EA;
  EA.analyze(TL);
  TDSI = EA.getTemplateDependentStringInfo();

  BaseNameWithoutQualifiers = TDSI->getSourceString();

  if (BaseName.empty())
    BaseName = BaseNameWithoutQualifiers;
  else
    BaseName = buildString(BaseName, " ", BaseNameWithoutQualifiers);

  IsTemplate = TL.getTypePtr()->isDependentType();
}

std::shared_ptr<CtTypeInfo> CtTypeInfo::applyTemplateArguments(
    const std::vector<TemplateArgumentInfo> &TA) {
  auto NewType = std::make_shared<CtTypeInfo>(*this);
  if (TDSI)
    NewType->TDSI = TDSI->applyTemplateArguments(TA);
  for (auto &R : NewType->Range)
    R.setTemplateList(TA);
  return NewType;
}

void SizeInfo::setTemplateList(
    const std::vector<TemplateArgumentInfo> &TemplateList) {
  if (TDSI)
    TDSI = TDSI->applyTemplateArguments(TemplateList);
}

void RandomEngineInfo::buildInfo() {
  if (!NeedPrint)
    return;

  // insert engine arguments
  if (IsClassMember || IsArray) {
    // replace type
    DpctGlobalInfo::getInstance().addReplacement(
        std::make_shared<ExtReplacement>(DeclFilePath, TypeBeginOffset,
                                         TypeLength, TypeReplacement + "*",
                                         nullptr));
    if (IsQuasiEngine) {
      DpctGlobalInfo::getInstance().addReplacement(
          std::make_shared<ExtReplacement>(
              CreateCallFilePath, CreateAPIBegin, CreateAPILength,
              (IsArray ? GeneratorName : DeclaratorDeclName) + " = new " +
                  TypeReplacement + "(" + QueueStr + ", " + DimExpr + ")",
              nullptr));
    } else {
      DpctGlobalInfo::getInstance().addReplacement(
          std::make_shared<ExtReplacement>(
              CreateCallFilePath, CreateAPIBegin, CreateAPILength,
              (IsArray ? GeneratorName : DeclaratorDeclName) + " = new " +
                  TypeReplacement + "(" + QueueStr + ", " + SeedExpr + ")",
              nullptr));
    }
  } else {
    // replace type
    DpctGlobalInfo::getInstance().addReplacement(
        std::make_shared<ExtReplacement>(DeclFilePath, TypeBeginOffset,
                                         TypeLength, TypeReplacement, nullptr));
    if (IsQuasiEngine) {
      DpctGlobalInfo::getInstance().addReplacement(
          std::make_shared<ExtReplacement>(
              DeclFilePath, IdentifierEndOffset, 0,
              "(" + QueueStr + ", " + DimExpr + ")", nullptr));
    } else {
      DpctGlobalInfo::getInstance().addReplacement(
          std::make_shared<ExtReplacement>(
              DeclFilePath, IdentifierEndOffset, 0,
              "(" + QueueStr + ", " + SeedExpr + ")", nullptr));
    }
  }
}

void DeviceRandomStateTypeInfo::buildInfo(std::string FilePath,
                                          unsigned int Offset) {
  if (DpctGlobalInfo::getDeviceRNGReturnNumSet().size() == 1) {
    DpctGlobalInfo::getInstance().addReplacement(
        std::make_shared<ExtReplacement>(
            FilePath, Offset, Length,
            GeneratorType + "<" +
                std::to_string(
                    *DpctGlobalInfo::getDeviceRNGReturnNumSet().begin()) +
                ">",
            nullptr));
  } else {
    DpctGlobalInfo::getInstance().addReplacement(
        std::make_shared<ExtReplacement>(FilePath, Offset, Length,
                                         GeneratorType +
                                             "<PlaceHolder/*Fix the vec_size "
                                             "mannually*/>",
                                         nullptr));
  }
}

void DeviceRandomInitAPIInfo::buildInfo(std::string FilePath,
                                        unsigned int Offset) {
  std::string VecSizeStr;
  bool IsOneNumber = false;
  if (DpctGlobalInfo::getDeviceRNGReturnNumSet().size() == 1) {
    int VecSize = *DpctGlobalInfo::getDeviceRNGReturnNumSet().begin();
    if (VecSize == 1)
      IsOneNumber = true;
    VecSizeStr = std::to_string(VecSize);
  } else {
    VecSizeStr = "PlaceHolder/*Fix the vec_size mannually*/";
  }

  std::string ReplStr =
      RNGStateName + " = " + GeneratorType + "<" + VecSizeStr + ">(" + RNGSeed +
      ", {static_cast<std::uint64_t>(" + RNGOffset +
      (IsOneNumber ? "" : " * " + VecSizeStr) +
      "), static_cast<std::uint64_t>(" + RNGSubseq + " * 8)})";

  DpctGlobalInfo::getInstance().addReplacement(std::make_shared<ExtReplacement>(
      FilePath, Offset, Length, ReplStr, nullptr));
}

void DeviceRandomGenerateAPIInfo::buildInfo(std::string FilePath,
                                            unsigned int Offset) {
  std::string ReplStr =
      "mkl::rng::device::generate(" + DistrName + ", " + RNGStateName + ")";
  DpctGlobalInfo::getInstance().addReplacement(std::make_shared<ExtReplacement>(
      FilePath, Offset, Length, ReplStr, nullptr));
}

void DeviceRandomDistrInfo::buildInfo(std::string FilePath,
                                      unsigned int Offset) {
  std::string InsertStr = DistrType + "<" + ValueType + "> " + DistrName + ";" +
                          getNL() + IndentStr;
  DpctGlobalInfo::getInstance().addReplacement(std::make_shared<ExtReplacement>(
      FilePath, Offset, 0, InsertStr, nullptr));
}

bool isInRoot(SourceLocation SL) { return DpctGlobalInfo::isInRoot(SL); }
} // namespace dpct
} // namespace clang
