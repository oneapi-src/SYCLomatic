//===--------------- MigrationAction.h ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MIGRATION_ACTION_H
#define MIGRATION_ACTION_H

#include "clang/Frontend/FrontendAction.h"
#include "clang/Tooling/Tooling.h"

#include "ASTTraversal.h"
#include "MisleadingBidirectional.h"

namespace clang {
namespace dpct {

class DpctFileInfo;

struct TranslationUnitInfo {
  std::unique_ptr<ASTUnit> AST;
  TransformSetTy Transforms;
  IncludeMapSetTy IncludeMapSet;
  std::shared_ptr<DpctFileInfo> MainFile;
};

class DpctConsumer : public ASTConsumer {
  TranslationUnitInfo *Info;
  std::unique_ptr<MisleadingBidirectionalHandler> Handler;

public:
  DpctConsumer(TranslationUnitInfo *TUI, Preprocessor &PP);
  void Initialize(ASTContext &Context) override;

  void HandleCXXExplicitFunctionInstantiation(
      const FunctionDecl *Specialization, const FunctionTypeLoc &FTL,
      const ParsedAttributes &Attrs,
      const TemplateArgumentListInfo &TAList) override;
};

class DpctFrontEndAction : public ASTFrontendAction {
public:
  DpctFrontEndAction(TranslationUnitInfo *TUI) : Info(TUI) {}
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef InFile) override;

  void EndSourceFileAction() override;

private:
  TranslationUnitInfo *Info;
};

class DpctToolAction : public tooling::ToolAction {
public:
  DpctToolAction(llvm::raw_ostream &DS, ReplTy &Replacements,
                 const std::string &RuleNames, std::vector<PassKind> Passes);
  /// Perform an action for an invocation.
  bool runInvocation(std::shared_ptr<CompilerInvocation> Invocation,
                     FileManager *Files,
                     std::shared_ptr<PCHContainerOperations> PCHContainerOps,
                     DiagnosticConsumer *DiagConsumer) override;

  void runPasses();

private:
  void runPostprocess(const std::string& StaginMsg, llvm::function_ref<void()>);
  void runPass(PassKind Pass);
  void traversTranslationUnit(PassKind Pass, TranslationUnitInfo &Info);

  std::shared_ptr<TranslationUnitInfo>
  createTranslationUnitInfoImpl(std::shared_ptr<CompilerInvocation> Invocation,
                                bool &Success);
  std::shared_ptr<TranslationUnitInfo>
  createTranslationUnitInfo(std::shared_ptr<CompilerInvocation> Invocation,
                            bool &Success);

  void printFileStaging(StringRef Staging, StringRef FileName);

  static StringRef getStagingName(PassKind Pass);

  struct InvocationOrTranslationUnit {
    std::shared_ptr<CompilerInvocation>CI;
    std::shared_ptr<TranslationUnitInfo>TU;

    InvocationOrTranslationUnit(std::shared_ptr<CompilerInvocation> C)
        : CI(C) {}
    InvocationOrTranslationUnit(std::shared_ptr<TranslationUnitInfo> T)
        : TU(T) {}
  };
  DpctGlobalInfo &Global;
  ReplTy &Repls;
  std::vector<std::string> MigrationRuleNames;
  std::vector<PassKind> Passes;
  std::vector<InvocationOrTranslationUnit> IOTUs;
  llvm::raw_ostream &DiagnosticStream;

  static const std::string PostProcessFaultMsg;
};

} // namespace dpct
} // namespace clang

#endif // MIGRATION_ACTION_H