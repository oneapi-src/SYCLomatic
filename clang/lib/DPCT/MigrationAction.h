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

class DpctConsumer : public ASTConsumer {
public:
  void Initialize(ASTContext &Context) override {
    // Set Context for build information
    DpctGlobalInfo::setContext(Context);
  }

  void HandleCXXExplicitFunctionInstantiation(
      const FunctionDecl *Specialization, const FunctionTypeLoc &FTL,
      const ParsedAttributes &Attrs,
      const TemplateArgumentListInfo &TAList) override;
};

class DpctFrontEndAction : public ASTFrontendAction {
public:
  DpctFrontEndAction(TransformSetTy &Transforms, IncludeMapSetTy &IncludeMapSet)
      : TransformSet(Transforms), IncludeMap(IncludeMapSet) {}
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef InFile) override;
  bool BeginSourceFileAction(CompilerInstance &) override;

private:
  TransformSetTy &TransformSet;
  IncludeMapSetTy &IncludeMap;
  std::unique_ptr<MisleadingBidirectionalHandler> Handler;
};

class DpctToolAction : public tooling::ToolAction {
public:
  DpctToolAction(ReplTy &Replacements, const std::string &RuleNames, std::vector<PassKind> Passes);
  /// Perform an action for an invocation.
  bool runInvocation(std::shared_ptr<CompilerInvocation> Invocation,
                     FileManager *Files,
                     std::shared_ptr<PCHContainerOperations> PCHContainerOps,
                     DiagnosticConsumer *DiagConsumer) override;

  void runPasses();

private:
  void runPass(PassKind Pass);

  struct TranslationUnitInfo {
    std::unique_ptr<ASTUnit> AST;
    TransformSetTy Transforms;
    IncludeMapSetTy IncludeMapSet;
  };
  DpctGlobalInfo &Global;
  ReplTy &Repls;
  std::vector<std::string> MigrationRuleNames;
  std::vector<PassKind> Passes;
  std::vector<std::unique_ptr<TranslationUnitInfo>> ASTs;
};

} // namespace dpct
} // namespace clang

#endif // MIGRATION_ACTION_H