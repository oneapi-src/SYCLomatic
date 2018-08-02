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

#include "ASTTraversal.h"

using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::cu2sycl;
using namespace clang::tooling;

llvm::cl::OptionCategory OptCat("SYCL Compatibility Tool");
llvm::cl::extrahelp
    CommonHelp(clang::tooling::CommonOptionsParser::HelpMessage);
llvm::cl::opt<std::string> Passes(
    "passes", llvm::cl::desc("Comma separated list of transformation passes"),
    llvm::cl::value_desc("\"FunctionAttrsRule,...\""), llvm::cl::cat(OptCat));

using ReplTy = std::map<std::string, Replacements>;

class Cu2SyclConsumer : public ASTConsumer {
public:
  Cu2SyclConsumer(ReplTy &R) : Repl(R) {
    if (Passes != "") {
      std::vector<std::string> Names;
      // Separate string into list by comma
      {
        std::size_t Current, Previous = 0;
        Current = Passes.find(',');
        while (Current != std::string::npos) {
          Names.push_back(Passes.substr(Previous, Current - Previous));
          Previous = Current + 1;
          Current = Passes.find(',', Previous);
        }
        Names.push_back(Passes.substr(Previous, Current - Previous));
      }
      for (auto const &Name : Names) {
        auto *ID = ASTTraversalMetaInfo::getID(Name);
        if (!ID) {
          llvm::errs() << "[ERROR] Rule not found: \"" << Name
                       << "\" - check \"-passes\" option\n";
          std::exit(1);
        }
        ATM.emplaceTranslationRule(ID);
      }
    } else {
      ATM.emplaceAllRules();
    }
  }

  void HandleTranslationUnit(ASTContext &Context) override {
    // The translation process is separated into two stages:
    // 1) Analysis of AST and identification of applicable translation rules
    // 2) Generation of actual textual Replacements
    // Such separation makes it possible to post-process the list of identified
    // translation rules before applying them.
    ATM.matchAST(Context, TransformSet);

    const SourceManager &SM = Context.getSourceManager();
    std::vector<Replacement> ReplSet;
    for (const auto &I : TransformSet) {
      Replacement R = I->getReplacement(SM);
      // TODO: This check filters out headers, which is wrong.
      // TODO: It'd be better not to generate replacements for system headers
      // instead of filtering them.
      if (R.getFilePath() ==
          SM.getFileEntryForID(SM.getMainFileID())->getName())
        ReplSet.emplace_back(std::move(R));
    }

    for (const Replacement &R : ReplacementFilter(ReplSet))
      if (auto Err = Repl[R.getFilePath()].add(R))
        llvm_unreachable("Adding the replacement: Error occured ");
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
  clang::tooling::CommonOptionsParser OptParser(argc, argv, OptCat);
  RefactoringTool Tool(OptParser.getCompilations(),
                       OptParser.getSourcePathList());
  Cu2SyclAction Action(Tool.getReplacements());
  return Tool.runAndSave(newFrontendActionFactory(&Action).get());
}
