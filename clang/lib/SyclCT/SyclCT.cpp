//===--- SyclCT.cpp -------------------------------------*- C++ -*---===//
//
// Copyright (C) 2018 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

#include "clang/SyclCT/SyclCT.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Refactoring.h"

#include "ASTTraversal.h"
#include "SaveNewFiles.h"
#include "Utility.h"
#include "ValidateArguments.h"
#include <string>

#include "ToolChains/Cuda.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Options.h"
#include <algorithm>
#include <cstring>
#include <vector>

using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::syclct;
using namespace clang::tooling;

using ReplTy = std::map<std::string, Replacements>;

using namespace llvm::cl;

static OptionCategory SyclCTCat("SYCL Compatibility Tool");
static extrahelp CommonHelp(clang::tooling::CommonOptionsParser::HelpMessage);
static opt<std::string>
    Passes("passes", desc("Comma separated list of transformation passes"),
           value_desc("\"FunctionAttrsRule,...\""), cat(SyclCTCat));
static opt<std::string>
    InRoot("in-root",
           desc("Path to root of project to be translated"
                " (header files not under this root will not be translated)"),
           value_desc("/path/to/input/root/"), cat(SyclCTCat),
           llvm::cl::Optional);
static opt<std::string>
    OutRoot("out-root",
            desc("Path directory where generated files will be placed"
                 " (directory will be created if it does not exist)"),
            value_desc("/path/to/output/root/"), cat(SyclCTCat),
            llvm::cl::Optional);
std::string CudaPath; // Global value for the CUDA install path.

class SyclCTConsumer : public ASTConsumer {
public:
  SyclCTConsumer(ReplTy &R, const CompilerInstance &CI)
      : ATM(CI), Repl(R), PP(CI.getPreprocessor()) {
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

    std::vector<Replacement> ReplSet;
    for (const auto &I : TransformSet) {
      Replacement R = I->getReplacement(Context);
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

  void Initialize(ASTContext &Context) override {
    PP.addPPCallbacks(llvm::make_unique<IncludesCallbacks>(
        TransformSet, Context.getSourceManager()));
  }

private:
  ASTTraversalManager ATM;
  TransformSetTy TransformSet;
  ReplTy &Repl;
  Preprocessor &PP;
};

class SyclCTAction : public ASTFrontendAction {
  ReplTy &Repl;

public:
  SyclCTAction(ReplTy &R) : Repl(R) {}

  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef InFile) override {
    return llvm::make_unique<SyclCTConsumer>(Repl, CI);
  }

  bool usesPreprocessorOnly() const override { return false; }
};

// Object of this class will be handed to RefactoringTool::run and will create
// the Action.
class SyclCTActionFactory : public FrontendActionFactory {
  ReplTy &Repl;

public:
  SyclCTActionFactory(ReplTy &R) : Repl(R) {}
  FrontendAction *create() override { return new SyclCTAction{Repl}; }
};

std::string getCudaInstallPath(int argc, const char **argv) {
  std::vector<const char *> Argv;
  Argv.reserve(argc);
  // do not copy "--" so the driver sees a possible --cuda-path option
  std::copy_if(argv, argv + argc, back_inserter(Argv),
               [](const char *s) { return std::strcmp(s, "--"); });

  // Output parameters to indicate errors in parsing. Not checked here,
  // OptParser will handle errors.
  unsigned MissingArgIndex, MissingArgCount;
  std::unique_ptr<llvm::opt::OptTable> Opts = driver::createDriverOptTable();
  llvm::opt::InputArgList ParsedArgs =
      Opts->ParseArgs(Argv, MissingArgIndex, MissingArgCount);

  // Create minimalist CudaInstallationDetector and return the InstallPath.
  DiagnosticsEngine E(nullptr, nullptr, nullptr, false);
  driver::Driver Driver("", llvm::sys::getDefaultTargetTriple(), E, nullptr);
  driver::CudaInstallationDetector CudaDetector(
      Driver, llvm::Triple(Driver.getTargetTriple()), ParsedArgs);

  std::string Path = CudaDetector.getInstallPath();
  makeCanonical(Path);
  return Path;
}

int run(int argc, const char **argv) {
  // CommonOptionsParser will adjust argc to the index of "--"
  CudaPath = getCudaInstallPath(argc, argv);
  CommonOptionsParser OptParser(argc, argv, SyclCTCat);

  if (!makeCanonicalOrSetDefaults(InRoot, OutRoot,
                                  OptParser.getSourcePathList()))
    exit(-1);

  if (!validatePaths(InRoot, OptParser.getSourcePathList()))
    exit(-1);

  RefactoringTool Tool(OptParser.getCompilations(),
                       OptParser.getSourcePathList());

  // Made "-- -x cuda --cuda-host-only" option set by default, .i.e commandline
  // "syclct -in-root ./ -out-root ./ ./topologyQuery.cu  --  -x  cuda
  // --cuda-host-only  -I../common/inc" became "syclct -in-root ./ -out-root ./
  // ./topologyQuery.cu  -- -I../common/inc"
  Tool.appendArgumentsAdjuster(getInsertArgumentAdjuster(
      "--cuda-host-only", ArgumentInsertPosition::BEGIN));

  Tool.appendArgumentsAdjuster(getInsertArgumentAdjuster(
      "cuda", ArgumentInsertPosition::BEGIN));

  Tool.appendArgumentsAdjuster(getInsertArgumentAdjuster(
      "-x", ArgumentInsertPosition::BEGIN));

  SyclCTActionFactory Factory(Tool.getReplacements());
  if (int RunResult = Tool.run(&Factory)) {
    return RunResult;
  }
  // if run was successful
  return saveNewFiles(Tool, InRoot, OutRoot);
}
