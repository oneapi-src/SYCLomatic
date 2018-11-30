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
#include "Debug.h"
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

// static opt<bool, true> Verbose("v", desc("Show verbose compiling message"),
//                                cat(SyclCTCat), location(IsVerbose));

std::string CudaPath; // Global value for the CUDA install path.

class SyclCTConsumer : public ASTConsumer {
public:
  SyclCTConsumer(ReplTy &R, const CompilerInstance &CI, StringRef InFile)
      : ATM(CI, InRoot), Repl(R), PP(CI.getPreprocessor()) {
    int RequiredRType;
    SourceProcessType FileType = GetSourceFileType(InFile);

    if (FileType & (TypeCudaSource | TypeCudaHeader)) {
      RequiredRType = ApplyToCudaFile;
    } else if (FileType & (TypeCppSource | TypeCppHeader)) {
      RequiredRType = ApplyToCppFile;
    }

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
      std::vector<std::vector<std::string>> Rules;

      for (auto const &Name : Names) {
        auto *ID = ASTTraversalMetaInfo::getID(Name);
        auto MapEntry = ASTTraversalMetaInfo::getConstructorTable()[ID];
        auto RuleObj = (TranslationRule *)MapEntry();
        CommonRuleProperty RuleProperty = RuleObj->GetRuleProperty();
        auto RType = RuleProperty.RType;
        auto RulesDependon = RuleProperty.RulesDependon;

        // Add rules should be run on the source file
        if (RType & RequiredRType) {
          std::vector<std::string> Vec;
          Vec.push_back(Name);
          for (auto const &RuleName : RulesDependon) {
            Vec.push_back(RuleName);
          }
          Rules.push_back(Vec);
        }
      }

      std::vector<std::string> SortedRules = ruleTopoSort(Rules);
      for (std::vector<std::string>::reverse_iterator it = SortedRules.rbegin();
           it != SortedRules.rend(); it++) {
        auto *RuleID = ASTTraversalMetaInfo::getID(*it);
        if (!RuleID) {
          llvm::errs() << "[ERROR] Rule\"" << *it << "\" not found\n";
          std::exit(1);
        }
        ATM.emplaceTranslationRule(RuleID);
      }

    } else {
      ATM.emplaceAllRules(RequiredRType);
    }
  }

  /// try to merge replacemnt when meet: (modify in samefile, modify in same
  /// place, is insert(length==0), know the insert order).
  std::vector<ExtReplacement>
  MergeReplacementPass(std::vector<ExtReplacement> ReplSet) {
    std::vector<ExtReplacement> ReplSetMerged;
    for (ExtReplacement &R1 : ReplSet) {
      bool Merged = false;
      if (R1.getMerged()) {
        continue;
      }
      for (ExtReplacement &R2 : ReplSet) {
        if (!R2.getMerged() && R1.getOffset() == R2.getOffset() &&
            R1.getLength() == R2.getLength() &&
            R1.getInsertPosition() != R2.getInsertPosition() &&
            R1.getFilePath() == R2.getFilePath()) {
          if (R1.getInsertPosition() == InsertPositionLeft) {
            ExtReplacement RMerge(R1.getFilePath(), R1.getOffset(),
                                  R1.getLength(),
                                  StringRef(R1.getReplacementText().str() +
                                            R2.getReplacementText().str()),
                                  // TODO: class for merged transformations
                                  nullptr);
            ReplSetMerged.emplace_back(std::move(RMerge));
            R2.setMerged(true);
            R1.setMerged(true);
            Merged = true;
            break;
          } else {
            ExtReplacement RMerge(R1.getFilePath(), R1.getOffset(),
                                  R1.getLength(),
                                  StringRef(R2.getReplacementText().str() +
                                            R1.getReplacementText().str()),
                                  // TODO: class for merged transformations
                                  nullptr);
            ReplSetMerged.emplace_back(std::move(RMerge));
            R2.setMerged(true);
            R1.setMerged(true);
            Merged = true;
            break;
          }
        }
      }
      if (!Merged) {
        ReplSetMerged.emplace_back(std::move(R1));
      }
    }
    return ReplSetMerged;
  }

  void HandleTranslationUnit(ASTContext &Context) override {
    // The translation process is separated into two stages:
    // 1) Analysis of AST and identification of applicable translation rules
    // 2) Generation of actual textual Replacements
    // Such separation makes it possible to post-process the list of identified
    // translation rules before applying them.
    ATM.matchAST(Context, TransformSet, SSM);

    // Sort the transformations according to the sort key of the individual
    // transformations.  Sorted from low->high Key values
    std::sort(TransformSet.begin(), TransformSet.end(),
              TextModification::Compare);

    std::vector<ExtReplacement> ReplSet;
    for (const auto &I : TransformSet) {
      ExtReplacement R = I->getReplacement(Context);
      // TODO: This check filters out headers, which is wrong.
      // TODO: It'd be better not to generate replacements for system headers
      // instead of filtering them.
      std::string RPath = R.getFilePath();
      makeCanonical(RPath);
      if (isChildPath(InRoot, RPath)) {
        // TODO: Staticstics
        ReplSet.emplace_back(std::move(R));
      } else {
        // TODO: Staticstics
      }
    }

    std::vector<ExtReplacement> ReplSetMerged = MergeReplacementPass(ReplSet);
    ReplacementFilter FilteredReplacements(ReplSetMerged);

    for (const ExtReplacement &R : FilteredReplacements) {
      if (auto Err = Repl[R.getFilePath()].add(R)) {
        llvm::dbgs() << Err << "\n";
        syclct_unreachable("Adding the replacement: Error occured ");
      }
    }
    DebugInfo::printReplacements(FilteredReplacements, Context);
  }

  void Initialize(ASTContext &Context) override {
    PP.addPPCallbacks(llvm::make_unique<IncludesCallbacks>(
        TransformSet, Context.getSourceManager(), ATM));
  }

private:
  ASTTraversalManager ATM;
  TransformSetTy TransformSet;
  StmtStringMap SSM;
  ReplTy &Repl;
  Preprocessor &PP;
};

class SyclCTAction : public ASTFrontendAction {
  ReplTy &Repl;

public:
  SyclCTAction(ReplTy &R) : Repl(R) {}

  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef InFile) override {
    return llvm::make_unique<SyclCTConsumer>(Repl, CI, InFile);
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
  CommonOptionsParser OptParser(argc, argv, SyclCTCat);

  if (!makeCanonicalOrSetDefaults(InRoot, OutRoot,
                                  OptParser.getSourcePathList()))
    exit(-1);

  if (!validatePaths(InRoot, OptParser.getSourcePathList()))
    exit(-1);

  CudaPath = getCudaInstallPath(argc, argv);
  SYCLCT_DEBUG_WITH_TYPE(
      "CudaPath", SyclctDbgs() << "Cuda Path found: " << CudaPath << "\n");

  RefactoringTool Tool(OptParser.getCompilations(),
                       OptParser.getSourcePathList());

  // Made "-- -x cuda --cuda-host-only" option set by default, .i.e commandline
  // "syclct -in-root ./ -out-root ./ ./topologyQuery.cu  --  -x  cuda
  // --cuda-host-only  -I../common/inc" became "syclct -in-root ./ -out-root ./
  // ./topologyQuery.cu  -- -I../common/inc"
  Tool.appendArgumentsAdjuster(getInsertArgumentAdjuster(
      "--cuda-host-only", ArgumentInsertPosition::BEGIN));

  Tool.appendArgumentsAdjuster(
      getInsertArgumentAdjuster("cuda", ArgumentInsertPosition::BEGIN));

  Tool.appendArgumentsAdjuster(
      getInsertArgumentAdjuster("-x", ArgumentInsertPosition::BEGIN));

  SyclCTActionFactory Factory(Tool.getReplacements());
  if (int RunResult = Tool.run(&Factory)) {
    DebugInfo::ShowStatus(RunResult);
    return RunResult;
  }

  // if run was successful
  int status = saveNewFiles(Tool, InRoot, OutRoot);
  DebugInfo::ShowStatus(status);
  return status;
}
