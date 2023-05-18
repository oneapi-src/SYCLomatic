//===--------------- MigrationAction.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MigrationAction.h"

#include "clang/Frontend/ASTUnit.h"
#include "clang/Frontend/CompilerInstance.h"

#include "MigrationRuleManager.h"
#include "MisleadingBidirectional.h"

#ifdef _WIN32
#include <Windows.h>
#else // __WIN32
#include <fstream>
#endif // __WIN32

extern bool StopOnParseErr;

namespace clang {
namespace dpct {

namespace {
constexpr size_t MinAvailableMemorySize = 512 * 1024 * 1024; // 512Mb
constexpr size_t MinAvailableMemoryPercent = 25;             // 25 percent

/// Check whether available memory size is enough to cache more translate unit
/// info.
/// Return true if available phy memory is larger than \p
/// MinAvailableMemorySize and available phy memory percents is larger than \p
/// MinAvailableMemoryPercent. Otherwise return false.
bool canCacheMoreTranslateUnit();

#ifdef _WIN32
bool canCacheMoreTranslateUnit() {
  MEMORYSTATUSEX MStatus;
  MStatus.dwLength = sizeof(MStatus);
  if (!GlobalMemoryStatusEx(&MStatus))
    return false;

  return MinAvailableMemoryPercent < 100 - MStatus.dwMemoryLoad &&
         MStatus.ullAvailPhys > MinAvailableMemorySize;
}
#else  // _WIN32
bool canCacheMoreTranslateUnit() {
  std::ifstream File("/proc/meminfo", std::ios::in);

  ///  Always return false if can not open meminfo file.
  if (!File.is_open())
    return false;

  std::string Dummy;
  size_t Total = 0, Available = 0;

  try {
    /// 1st line:  "MemTotal:       **** kB"
    File >> Dummy >> Total >> Dummy;
    if (!Total)
      return false;

    /// 2nd line: "MemFree:        **** kB"
    File >> Dummy >> Dummy >> Dummy;
    /// 3rd line: "MemAvailable:   **** kB"
    File >> Dummy >> Available >> Dummy;

    return Available * 100 / Total > MinAvailableMemoryPercent &&
           Available * 1024 > MinAvailableMemorySize;
  } catch (std::exception &) {
    /// Return false if any exception
    return false;
  }
}
#endif // _WIN32
} // namespace

DpctConsumer::DpctConsumer(TranslationUnitInfo *TUI, Preprocessor &PP)
    : Info(TUI) {
  PP.addPPCallbacks(std::make_unique<IncludesCallbacks>(
      Info->Transforms, Info->IncludeMapSet, PP.getSourceManager()));
  if (DpctGlobalInfo::getCheckUnicodeSecurityFlag()) {
    Handler =
        std::make_unique<MisleadingBidirectionalHandler>(Info->Transforms);
    PP.addCommentHandler(Handler.get());
  }
}

void DpctConsumer::Initialize(ASTContext &Context) {
  // Set Context for build information
  DpctGlobalInfo::setContext(Context);
  auto &SM = Context.getSourceManager();
  auto Path = DpctGlobalInfo::getAbsolutePath(SM.getMainFileID());
  assert(Path && "Can not find absolute path");
  DpctGlobalInfo::getInstance().setMainFile(
      Info->MainFile = DpctGlobalInfo::getInstance().insertFile(Path.value()));
}

void DpctConsumer::HandleCXXExplicitFunctionInstantiation(
    const FunctionDecl *Specialization, const FunctionTypeLoc &FTL,
    const ParsedAttributes &Attrs, const TemplateArgumentListInfo &TAList) {
  if (!FTL || !Specialization)
    return;
  ExplicitInstantiationDecl::processFunctionTypeLoc(FTL);
  ExplicitInstantiationDecl::processTemplateArgumentList(TAList);
  if (Specialization->getTemplateSpecializationKind() !=
      TSK_ExplicitInstantiationDefinition)
    return;
  if (Specialization->hasAttr<CUDADeviceAttr>() ||
      Specialization->hasAttr<CUDAGlobalAttr>()) {
    DeviceFunctionDecl::LinkExplicitInstantiation(Specialization, FTL, Attrs,
                                                  TAList);
  }
}

std::unique_ptr<ASTConsumer>
DpctFrontEndAction::CreateASTConsumer(CompilerInstance &CI, StringRef InFile) {
  return std::make_unique<DpctConsumer>(Info, CI.getPreprocessor());
}

void DpctFrontEndAction::EndSourceFileAction() {
  getCompilerInstance().getASTContext().getParentMapContext().clear();
}

DpctToolAction::DpctToolAction(llvm::raw_ostream &DS, ReplTy &Replacements,
                               const std::string &RuleNames,
                               std::vector<PassKind> Passes)
    : Global(DpctGlobalInfo::getInstance()), Repls(Replacements),
      Passes(std::move(Passes)), DiagnosticStream(DS) {
  if (RuleNames.empty())
    return;
  auto Names = split(RuleNames, ',');
  for (const auto &Name : Names) {
    MigrationRuleNames.push_back(Name);
  }
}

StringRef DpctToolAction::getStagingName(PassKind Pass) {
  static std::string StageName[static_cast<unsigned>(PassKind::PK_End)] = {
      "Analyzing", "Migrating"};
  return StageName[static_cast<unsigned>(Pass)];
}

void DpctToolAction::printFileStaging(StringRef Staging, StringRef File) {
  std::string Msg;
  llvm::raw_string_ostream Out(Msg);
  Out << Staging << ": " << File << "\n";
  PrintMsg(Out.str(), false);
}

std::shared_ptr<TranslationUnitInfo> DpctToolAction::createTranslationUnitInfo(
    std::shared_ptr<CompilerInvocation> Invocation, bool &Success) {
  std::shared_ptr<TranslationUnitInfo> Ret;
  printFileStaging("Parsing",
           Invocation->getFrontendOpts().Inputs[0].getFile().str());
  if (runWithCrashGuard(
          [&]() { Ret = createTranslationUnitInfoImpl(Invocation, Success); },
          "Error: dpct internal error. Parsing file \"" +
              Invocation->getFrontendOpts().Inputs[0].getFile().str() +
              "\" causing the error skipped. Migration continues.\n"))
    return Ret;
  return std::shared_ptr<TranslationUnitInfo>();
}

std::shared_ptr<TranslationUnitInfo> DpctToolAction::createTranslationUnitInfoImpl(
    std::shared_ptr<CompilerInvocation> Invocation, bool &Success) {
  auto DiagConsumer = new TextDiagnosticPrinter(
      DiagnosticStream, &Invocation->getDiagnosticOpts());
  auto Info = std::make_shared<TranslationUnitInfo>();
  auto Diags = CompilerInstance::createDiagnostics(
      &Invocation->getDiagnosticOpts(), DiagConsumer,
      /*ShouldOwnClient=*/false, &Invocation->getCodeGenOpts());
  DpctGlobalInfo::setColorOption(Invocation->getDiagnosticOpts().ShowColors);
  Info->AST = ASTUnit::create(Invocation, Diags, CaptureDiagsKind::None, false);
  DpctFrontEndAction FEAction(Info.get());
  auto Ret = ASTUnit::LoadFromCompilerInvocationAction(
      Invocation, std::make_shared<PCHContainerOperations>(), Diags, &FEAction,
      Info->AST.get());
  Success = !DiagConsumer->getNumErrors();
  if (Ret && (bool)&Info->AST->getASTContext())
    return Info;
  return std::shared_ptr<TranslationUnitInfo>();
}

bool DpctToolAction::runInvocation(
    std::shared_ptr<CompilerInvocation> Invocation, FileManager *Files,
    std::shared_ptr<PCHContainerOperations> PCHContainerOps,
    DiagnosticConsumer *DiagConsumer) {
  if (!Invocation)
    return false;

  if (canCacheMoreTranslateUnit()) {
    bool Success;
    if (auto Info = createTranslationUnitInfo(Invocation, Success)) {
      IOTUs.emplace_back(Info);
    }
    return Success;
  }
  IOTUs.emplace_back(Invocation);
  return true;
}

void DpctToolAction::traversTranslationUnit(PassKind Pass,
                                            TranslationUnitInfo &Info) {
  auto &Context = Info.AST->getASTContext();
  auto &Transforms = Info.Transforms;
  auto &IncludeMap = Info.IncludeMapSet;
  Info.AST->getDiagnostics().getClient()->BeginSourceFile(
      Context.getLangOpts());
  DpctGlobalInfo::setContext(Context);
  DpctGlobalInfo::getInstance().setMainFile(Info.MainFile);
  MigrationRuleManager MRM(Pass, Transforms);
  Global.getProcessedFile().insert(Info.MainFile->getFilePath());
  printFileStaging(getStagingName(Pass), Info.MainFile->getFilePath());
  MRM.matchAST(Context, MigrationRuleNames);
  for (const auto &I : Transforms) {
    auto Repl = I->getReplacement(Context);

    // When processing __constant__ between two executions, tool may set the
    // replacement from TextModification as nullptr to ignore this
    // replacement.
    if (Repl == nullptr)
      continue;

    // If a file has replacement, all include statement need change, so we add
    // them to global replacement here.
    const auto FilePath = Repl->getFilePath().str();
    auto Find = IncludeMap.find(FilePath);
    if (Find != IncludeMap.end()) {
      for (const auto &Entry : Find->second) {
        Global.addReplacement(Entry->getReplacement(Context));
      }
      IncludeMap.erase(FilePath);
    }
    Global.addReplacement(Repl);

    StaticsInfo::printReplacements(Transforms, Context);
  }
  Transforms.clear();
}

void DpctToolAction::runPass(PassKind Pass) {
  for (auto &Info : IOTUs) {
    std::shared_ptr<TranslationUnitInfo> TU = Info.TU;
    if (!TU) {
      bool Dummy;
      TU = createTranslationUnitInfo(Info.CI, Dummy);
      if (!TU)
        continue;
    }
    traversTranslationUnit(Pass, *TU);
  }

  if (Pass == PassKind::PK_Analysis) {
    runWithCrashGuard([&]() { Global.buildKernelInfo(); }, PostProcessFaultMsg);
  }
}

void DpctToolAction::runPasses() {
  for (auto Pass : Passes) {
    runPass(Pass);
  }

  runWithCrashGuard(
      [&]() {
        Global.buildReplacements();
        Global.postProcess();
        Global.emplaceReplacements(Repls);
      },
      PostProcessFaultMsg);
}

const std::string DpctToolAction::PostProcessFaultMsg =
    "Error: dpct internal error. dpct tries to recover and "
    "write the migration result.\n";

} // namespace dpct
} // namespace clang
