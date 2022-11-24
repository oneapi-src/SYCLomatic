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

extern bool StopOnParseErr;

namespace clang {
namespace dpct {

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
      Info->MainFile =
          DpctGlobalInfo::getInstance().insertFile(Path.value()));
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

bool DpctToolAction::runInvocation(
    std::shared_ptr<CompilerInvocation> Invocation, FileManager *Files,
    std::shared_ptr<PCHContainerOperations> PCHContainerOps,
    DiagnosticConsumer *DiagConsumer) {
  auto Info = std::make_unique<TranslationUnitInfo>();
  auto Diags = CompilerInstance::createDiagnostics(
      &Invocation->getDiagnosticOpts(), DiagConsumer,
      /*ShouldOwnClient=*/false, &Invocation->getCodeGenOpts());
  DpctGlobalInfo::setColorOption(Invocation->getDiagnosticOpts().ShowColors);
  Info->AST = ASTUnit::create(Invocation, Diags, CaptureDiagsKind::None, false);
  DpctFrontEndAction FEAction(Info.get());
  auto Ret = ASTUnit::LoadFromCompilerInvocationAction(
          Invocation, PCHContainerOps, Diags, &FEAction, Info->AST.get());
  if (Ret && (bool)&Info->AST->getASTContext())
    ASTs.push_back(std::move(Info));
  return !DiagConsumer->getNumErrors();
}

void DpctToolAction::runPass(PassKind Pass) {
  for (auto &Info : ASTs) {
    std::unordered_set<std::string> DuplicateFilter;
    auto &Context = Info->AST->getASTContext();
    auto &Transforms = Info->Transforms;
    auto &IncludeMap = Info->IncludeMapSet;
    auto DiagClient = new TextDiagnosticPrinter(
        DiagnosticStream, &Info->AST->getDiagnostics().getDiagnosticOptions());
    Info->AST->getDiagnostics().setClient(DiagClient);
    DiagClient->BeginSourceFile(Context.getLangOpts(),
                                &Info->AST->getPreprocessor());
    Context.getParentMapContext().clear();
    DpctGlobalInfo::setContext(Context);
    DpctGlobalInfo::getInstance().setMainFile(Info->MainFile);
    MigrationRuleManager MRM(Pass, Transforms);
    Global.getProcessedFile().insert(Info->MainFile->getFilePath());
    MRM.matchAST(Context, MigrationRuleNames);
    for (const auto &I : Transforms) {
      auto Repl = I->getReplacement(Context);

      // When processing __constant__ between two executions, tool may set the
      // replacement from TextModification as nullptr to ignore this
      // replacement.
      if (Repl == nullptr)
        continue;

      // For file path got in AST may be different with the one in
      // preprocessing stage, here only the file name is used to retrieve
      // IncludeMapSet.
      const std::string FileName =
          llvm::sys::path::filename(Repl->getFilePath()).str();
      if (DuplicateFilter.find(FileName) == end(DuplicateFilter)) {
        DuplicateFilter.insert(FileName);
        auto Find = IncludeMap.find(FileName);
        if (Find != IncludeMap.end()) {
          for (const auto &Entry : Find->second) {
            Global.addReplacement(Entry->getReplacement(Context));
          }
        }
      }
      Global.addReplacement(Repl);

      StaticsInfo::printReplacements(Transforms, Context);
    }
    Transforms.clear();
  }
  if (Pass == PassKind::PK_Analysis) {
    int RetJmp = 0;
    CHECKPOINT_ReplacementPostProcess_ENTRY(RetJmp);
    if (RetJmp == 0) {
      try {
        Global.buildKernelInfo();
      } catch (std::exception &) {
        std::string FaultMsg = "Error: dpct internal error. dpct tries to "
                               "recover and write the migration result.\n";
        llvm::errs() << FaultMsg;
      }
    }

    CHECKPOINT_ReplacementPostProcess_EXIT();
  }
}

void DpctToolAction::runPasses() {
  for (auto Pass : Passes) {
    runPass(Pass);
  }

  int RetJmp = 0;
  CHECKPOINT_ReplacementPostProcess_ENTRY(RetJmp);
  if (RetJmp == 0) {
    try {
      Global.buildReplacements();
      Global.postProcess();
      Global.emplaceReplacements(Repls);
    } catch (std::exception &) {
      std::string FaultMsg = "Error: dpct internal error. dpct tries to "
                             "recover and write the migration result.\n";
      llvm::errs() << FaultMsg;
    }
  }

  CHECKPOINT_ReplacementPostProcess_EXIT();
}

} // namespace dpct
} // namespace clang
