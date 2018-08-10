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
#include "SaveNewFiles.h"
#include "ValidateArguments.h"
#include "Utility.h"
#include <string>

using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::cu2sycl;
using namespace clang::tooling;

using ReplTy = std::map<std::string, Replacements>;

using namespace llvm::cl;

static OptionCategory Cu2SyclCat("SYCL Compatibility Tool");
static extrahelp CommonHelp(clang::tooling::CommonOptionsParser::HelpMessage);
static opt<std::string>
    Passes("passes", desc("Comma separated list of transformation passes"),
           value_desc("\"FunctionAttrsRule,...\""), cat(Cu2SyclCat));
static opt<std::string>
    InRoot("in-root",
           desc("Path to root of project to be translated"
                " (header files not under this root will not be translated)"),
           value_desc("/path/to/input/root/"), cat(Cu2SyclCat),
           llvm::cl::Optional);
static opt<std::string>
    OutRoot("out-root",
            desc("Path directory where generated files will be placed"
                 " (directory will be created if it does not exist)"),
            value_desc("/path/to/output/root/"), cat(Cu2SyclCat),
            llvm::cl::Optional);

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
      std::string RPath = R.getFilePath();
      makeCanonical(RPath);
      if (isChildPath(InRoot, RPath))
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
  CommonOptionsParser OptParser(argc, argv, Cu2SyclCat);

  if (!makeCanonicalOrSetDefaults(InRoot, OutRoot,
                                  OptParser.getSourcePathList()))
    exit(-1);

  if (!validatePaths(InRoot, OptParser.getSourcePathList()))
    exit(-1);

  RefactoringTool Tool(OptParser.getCompilations(),
                       OptParser.getSourcePathList());
  Cu2SyclAction Action(Tool.getReplacements());
  if (int RunResult = Tool.run(newFrontendActionFactory(&Action).get())) {
    return RunResult;
  }
  // if run was successful
  return saveNewFiles(Tool, InRoot, OutRoot);
}
