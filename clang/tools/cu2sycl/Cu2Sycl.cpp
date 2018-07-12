//===--- Cu2Sycl.cpp ---------------------------------*- C++ -*---===//
//
// Copyright (C) 2018 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

#include "clang/AST/ASTConsumer.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Refactoring.h"

#include "Translation.h"

using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::cu2sycl;
using namespace clang::tooling;

using ReplTy = std::map<std::string, Replacements>;

class Cu2SyclConsumer : public ASTConsumer {
public:
  Cu2SyclConsumer(ReplTy &R) : Repl(R) {
    ATM.emplaceTranslationRule(new ThreadIdxMatcher);
    ATM.emplaceTranslationRule(new BlockDimMatcher);
  }

  void HandleTranslationUnit(ASTContext &Context) override {
    // The translation process is separated into two stages:
    // 1) Analysis of AST and identification of applicable translation rules
    // 2) Generation of actual textual Replacements
    // Such separation makes it possible to post-process the list of identified
    // translation rules before applying them.
    ATM.matchAST(Context, TransformSet);

    const SourceManager &SM = Context.getSourceManager();
    for (const auto &I : TransformSet) {
      Replacement R = I->getReplacement(SM);
      // TODO: this check is invalid and in the wrong place.
      if (R.getFilePath() !=
          SM.getFileEntryForID(SM.getMainFileID())->getName())
        continue;
      if (auto Err = Repl[R.getFilePath()].add(R))
        llvm_unreachable("Error occured");
    }
  }

private:
  ASTTraversalManager ATM;
  TransformSetTy TransformSet;
  ReplTy &Repl;
};

class Cu2SyclAction {
  ReplTy &Repl;

public:
  Cu2SyclAction(ReplTy &R) : Repl(R) {}

  std::unique_ptr<ASTConsumer> newASTConsumer() {
    return llvm::make_unique<Cu2SyclConsumer>(Repl);
  }
};

int main(int argc, const char **argv) {
  llvm::cl::OptionCategory OptCat("CUDA to SYCL translator");
  clang::tooling::CommonOptionsParser OptParser(argc, argv, OptCat);
  RefactoringTool Tool(OptParser.getCompilations(),
                       OptParser.getSourcePathList());
  Cu2SyclAction Action(Tool.getReplacements());
  return Tool.runAndSave(newFrontendActionFactory(&Action).get());
}
