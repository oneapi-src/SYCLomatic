//===--- CudaToSycl.cpp ---------------------------------*- C++ -*---===//
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

#include "Analysis.h"
#include "Refinement.h"
#include "Translation.h"

using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::cu2sycl;
using namespace clang::tooling;

using ReplTy = std::map<std::string, Replacements>;

class CudaToSycl : public ASTConsumer {
public:
  CudaToSycl(ReplTy &R) : Repl(R) {
    AMan.emplaceAnalysis(new KernelInvocationAnalysis);
    TMan.emplaceCudaMatcher(new ThreadIdxMatcher);
    TMan.emplaceCudaMatcher(new BlockDimMatcher);
    RMan.emplaceOptimization(new ItemLinearIDMatcher);
  }

  void HandleTranslationUnit(ASTContext &Context) override {
    // For each translation unit the following steps are performed:
    // 1) Analysis : gathering information about the AST.
    // 2) Translation : detection of CUDA constructs in AST and planning
    //            straightforward replacements for them.
    // 3) Refinement : recognition of complex CUDA/SYCL patterns that can
    //            be translated more efficiently when grouped together or when
    //            context (analysis) is taken into account.
    // 4) Text replacement : generation of final tooling::Replacement objects
    //            that will be applied to source code by clang::tooling
    //            infrastructure.

    AMan.matchAST(Context);
    TMan.matchAST(Context, TransformSet);
    RMan.run(TransformSet, AMan);

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
  AnalysisManager AMan;
  TranslationManager TMan;
  RefinementManager RMan;

  TransformSetTy TransformSet;
  ReplTy &Repl;
};

class CudaToSyclAction {
  ReplTy &Repl;

public:
  CudaToSyclAction(ReplTy &R) : Repl(R) {}

  std::unique_ptr<ASTConsumer> newASTConsumer() {
    return llvm::make_unique<CudaToSycl>(Repl);
  }
};

int main(int argc, const char **argv) {
  llvm::cl::OptionCategory OptCat("CUDA to SYCL translator");
  clang::tooling::CommonOptionsParser OptParser(argc, argv, OptCat);
  RefactoringTool Tool(OptParser.getCompilations(),
                       OptParser.getSourcePathList());
  CudaToSyclAction Action(Tool.getReplacements());
  return Tool.runAndSave(newFrontendActionFactory(&Action).get());
}
