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
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include "ASTTraversal.h"
#include "AnalysisInfo.h"
#include "Config.h"
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
#include <fstream>
#include <vector>

#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/Version.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Rewrite/Core/Rewriter.h"
using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::syclct;
using namespace clang::tooling;

using ReplTy = std::map<std::string, Replacements>;
using namespace llvm::cl;

const char *const CtHelpMessage =
    "\n"
    "<source0> ... specify the paths of source files. These paths are\n"
    "\tlooked up in the compile command database. If the path of a file is\n"
    "\tabsolute, it needs to point into CMake's source tree. If the path is\n"
    "\trelative, the current working directory needs to be in the CMake\n"
    "\tsource tree and the file must be in a subdirectory of the current\n"
    "\tworking directory. \"./\" prefixes in the relative files will be\n"
    "\tautomatically removed, but the rest of a relative path must be a\n"
    "\tsuffix of a path in the compile command database.\n"
    "\n";

static OptionCategory SyclCTCat("SYCL Compatibility Tool");
static extrahelp CommonHelp(CtHelpMessage);
static opt<std::string>
    Passes("passes", desc("Comma separated list of migration passes that "
                          "customed to migrate the input file, only specified "
                          "mirgration pass will be applied during migration."),
           value_desc("FunctionAttrsRule,..."), cat(SyclCTCat));
static opt<std::string> InRoot(
    "in-root", desc("Path to root of project to be migrated"
                    " (header files not under this root will not be migrated)"),
    value_desc("/path/to/input/root/"), cat(SyclCTCat), llvm::cl::Optional);
static opt<std::string> OutRoot(
    "out-root", desc("Path directory where generated files will be placed"
                     " (directory will be created if it does not exist)"),
    value_desc("/path/to/output/root/"), cat(SyclCTCat), llvm::cl::Optional);

static opt<std::string> ReportType(
    "report-type",
    desc("Specifies migration report type. You can specify one or more reports "
         "to be generated: \"apis\" migration report provides information "
         "about API names which were or were not migrated and how many times. "
         "Migration report file will have \"apis\" suffix added, if "
         "report-file-prefix option is passed. "
         "\"stats\" provides high level migration statistics: how much Lines "
         "Of Code (LOC) migrated to DPC++, LOC migrated to Compatibility API, "
         "LOC not needed to migrate, LOC the tool was not able to migrate. "
         "Migration report file will have \"stats\" suffix added, if "
         "report-file-prefix option is passed. "
         "\"all\" generates all types of report, each type will go to a "
         "separate file with corresponding suffix added, if report-file-prefix "
         "option is passed. "
         "Default is \"stats\"."),
    value_desc("[all|apis|stats|apis,stats,...]"), cat(SyclCTCat),
    llvm::cl::Optional);

static opt<std::string> ReportFormat(
    "report-format",
    desc("Specifies CSV or human-readable format of the report. If "
         "report-file-prefix option is passed: for CSV format \"csv\" "
         "extension will be used in file name, for \"formatted\" report "
         "\"log\" extension will be used. Default is CSV. "),
    value_desc("[csv|formatted]"), cat(SyclCTCat), llvm::cl::Optional);

static opt<std::string> ReportFilePrefix(
    "report-file-prefix",
    desc("Specifies the prefix for the file name, where the migration report "
         "will be written.  If this option is not passed, the report will go "
         "to stdout. Depending on the report type and format, additional file "
         "extensions will be added, like: prefix.apis.<log|csv>, "
         "prefix.stats.<log|csv>. The report file will be created in the "
         "folder, specified by -out-root. Default is stdout."),
    value_desc("prefix"), cat(SyclCTCat), llvm::cl::Optional);
bool ReportOnlyFlag = false;
static opt<bool, true>
    ReportOnly("report-only",
               llvm::cl::desc("Instructs the tool to produce only report and "
                              "not produce the DPC++ code. By default both the "
                              "DPC++ code and the report will be generated."),
               cat(SyclCTCat), llvm::cl::location(ReportOnlyFlag));

bool KeepOriginalCodeFlag = false;

static opt<bool, true> ShowOrigCode(
    "keep-original-code",
    llvm::cl::desc("Keep original code in comments of SYCL file, default: off"),
    cat(SyclCTCat), llvm::cl::location(KeepOriginalCodeFlag));

static opt<std::string> DiagsContent(
    "report-diags-content",
    desc("Specify diags report verbose level: simple migration pass level info "
         "or detail transformation info happen in migration pass."),
    value_desc("[pass|transformation]"), cat(SyclCTCat), llvm::cl::Optional,
    llvm::cl::Hidden);

static std::string WarningDesc("Comma separated list of warnings to be"
                               " suppressed, valid warning ids range from " +
                               std::to_string((size_t)Warnings::BEGIN) +
                               " to " +
                               std::to_string((size_t)Warnings::END - 1));
opt<std::string> SuppressWarnings("suppress-warnings", desc(WarningDesc),
                                  value_desc("WarningID,..."), cat(SyclCTCat));

bool SuppressWarningsAllFlag = false;
static std::string WarningAllDesc("Suppress all warnings of the migration");
opt<bool, true> SuppressWarningsAll("suppress-warnings-all",
                                    desc(WarningAllDesc), cat(SyclCTCat),
                                    location(SuppressWarningsAllFlag));

std::string CudaPath;          // Global value for the CUDA install path.
std::string SyclctInstallPath; // Installation directory for this tool

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
      // Separate string into list by comma
      auto Names = split(Passes, ',');

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

  std::tuple<std::string /*FilePath*/, unsigned int /*Offset*/,
             unsigned int /*Length*/>
  getPathAndRange(std::string Str) {

    std::size_t PosNext = Str.rfind(':');
    std::size_t Pos = Str.rfind(':', PosNext - 1);
    std::string FilePath = Str.substr(0, Pos);

    unsigned int Offset = std::stoul(Str.substr(Pos + 1, PosNext - Pos - 1));
    unsigned int Length =
        std::stoul(Str.substr(PosNext + 1, Str.length() - PosNext - 1));

    return std::make_tuple(FilePath, Offset, Length);
  }

  /// try to merge replacemnt when meet: (modify in samefile, modify in same
  /// place, is insert(length==0), know the insert order).
  std::vector<ExtReplacement>
  MergeReplacementPass(std::vector<ExtReplacement> ReplSet) {
    std::vector<ExtReplacement> ReplSetMerged;

    std::unordered_map<std::string, std::string> FileMap;
    for (ExtReplacement &R : ReplSet) {
      std::string Key = R.getFilePath().str() + ":" +
                        std::to_string(R.getOffset()) + ":" +
                        std::to_string(R.getLength());

      std::unordered_map<std::string, std::string>::iterator Iter =
          FileMap.find(Key);

      if (Iter != FileMap.end()) {
        auto Data = getPathAndRange(Iter->first);
        std::string FilePath = std::get<0>(Data);
        unsigned int Length = std::get<2>(Data);

        std::string ReplTextR = R.getReplacementText().str();

        if (R.getInsertPosition() == InsertPositionLeft) {
          Iter->second = R.isEqualExtRepl(Length, ReplTextR)
                             ? ReplTextR
                             : Iter->second + ReplTextR;
        } else {
          Iter->second = R.isEqualExtRepl(Length, ReplTextR)
                             ? ReplTextR
                             : ReplTextR + Iter->second;
        }
      } else {
        FileMap[Key] = R.getReplacementText().str();
      }
    }

    for (auto const &Elem : FileMap) {

      auto Data = getPathAndRange(Elem.first);
      std::string FilePath = std::get<0>(Data);
      unsigned int Offset = std::get<1>(Data);
      unsigned int Length = std::get<2>(Data);

      ReplSetMerged.emplace_back(FilePath, Offset, Length, Elem.second,
                                 // TODO: class for merged transformations
                                 nullptr);
    }

    return ReplSetMerged;
  }

  /// Filter out some style comment to avoid compiler complain on nested
  /// comments. \returns string proccessed.
  std::string removeComments(const std::string &Line) {

    std::string CodeLine = Line;
    size_t Beg = CodeLine.find("/*");
    size_t End = CodeLine.find("*/");

    // Eg. "comments*/ program\n" => " program\n"
    if (Beg == std::string::npos && End != std::string::npos) {
      CodeLine.erase(0, End + 2);
    }

    while (CodeLine.find("/*") != std::string::npos) {
      Beg = CodeLine.find("/*");
      End = CodeLine.find("*/", Beg);
      if (End != std::string::npos) {
        // Eg. "program /*comments*/ program\n" => "program  program\n"
        CodeLine.erase(Beg, (End - Beg) + 2);
      } else {
        // Eg. " program /*comments\n" => " program "
        CodeLine.erase(Beg, (CodeLine.size() - Beg) + 2);
      }
    }
    return CodeLine;
  }

  /// Merge comments replacement in CommentsReplSet
  /// with code replacements in CodeRelSet if possible.
  /// \return code replacements merged.
  std::vector<ExtReplacement>
  MergeCommmetsPass(std::vector<ExtReplacement> &CommentsReplSet,
                    std::vector<ExtReplacement> &CodeRelSet) {

    std::vector<ExtReplacement> ReplSetMerged;

    for (ExtReplacement &CommentsRepl : CommentsReplSet) {
      bool Merged = false;
      for (ExtReplacement &CodeRel : CodeRelSet) {
        if (CommentsRepl.getOffset() == CodeRel.getOffset() &&
            0 == CodeRel.getLength() &&
            CommentsRepl.getFilePath() == CodeRel.getFilePath()) {
          // Coderep is  insert: merge style: (Comment + Code)
          ExtReplacement ReplMerged(
              CommentsRepl.getFilePath(), CommentsRepl.getOffset(), 0,
              StringRef(CommentsRepl.getReplacementText().str() +
                        CodeRel.getReplacementText().str()),
              CodeRel.getParentTM());
          ReplSetMerged.emplace_back(std::move(ReplMerged));
          Merged = true;
          CodeRel.setMerged(true);
          // Note: Can not break here, for there maybe multiple same location
          // replacements in CodeRelSet
        } else if (CommentsRepl.getOffset() >= CodeRel.getOffset() &&
                   CommentsRepl.getOffset() <
                       CodeRel.getOffset() + CodeRel.getLength() &&
                   CommentsRepl.getFilePath() == CodeRel.getFilePath()) {
          // Coderep is replacement: merge style: (Code + Comments)
          // Coderep is remove: merge style: (Comments)
          ExtReplacement ReplMerged(
              CommentsRepl.getFilePath(), CodeRel.getOffset(),
              CodeRel.getLength(),
              StringRef(CodeRel.getReplacementText().empty()
                            ? CodeRel.getReplacementText().str() +
                                  CommentsRepl.getReplacementText().str()
                            : CodeRel.getReplacementText().str()),
              CodeRel.getParentTM());
          ReplSetMerged.emplace_back(std::move(ReplMerged));
          Merged = true;
          CodeRel.setMerged(true);
          // Note: Can not break here, for there maybe multiple same location
          // replacements in CodeRelSet
        }
      }
      if (!Merged) {
        ReplSetMerged.emplace_back(std::move(CommentsRepl));
      }
    }

    std::vector<ExtReplacement> ReplSetTotal = ReplSetMerged;
    for (ExtReplacement &R1 : CodeRelSet) {
      if (!R1.getMerged()) {
        ReplSetTotal.emplace_back(std::move(R1));
      }
    }
    return ReplSetTotal;
  }

  std::vector<ExtReplacement>
  keepOriginalCode(SourceManager &SM, std::vector<ExtReplacement> &ReplSet) {

    std::vector<ExtReplacement> CommentsReplSet;
    std::unordered_set<unsigned int> DuplicateFilter;
    std::map<StringRef /*FilePath*/, StringRef /*Code*/> CodeCache;
    std::map<unsigned int /*line position*/, bool /*line needs comments*/>
        CommentsMap;
    // Generate comment-replacement for code line which SYCLCT has modified.
    for (auto I = ReplSet.begin(), E = ReplSet.end(); I != E; ++I) {

      StringRef FilePath = I->getFilePath();
      StringRef Code;
      I->setMerged(false); // To initialize the merged flag to avoid dirty value
                           // in MergeCommmetsPass()

      std::map<StringRef /*FilePath*/, StringRef /*Code*/>::iterator Iter =
          CodeCache.find(FilePath);
      if (Iter != CodeCache.end()) {
        Code = Iter->second;
      } else {
        const FileEntry *FileEntry = SM.getFileManager().getFile(FilePath);
        FileID FID = SM.getOrCreateFileID(FileEntry, SrcMgr::C_User);
        Code = SM.getBufferData(FID);
        CodeCache[FilePath] = Code;
      }

      auto BeginPos = Code.find_last_of('\n', I->getOffset());
      auto EndPos = Code.find('\n', I->getOffset() + I->getLength());
      BeginPos = (BeginPos != StringRef::npos ? BeginPos + 1 : I->getOffset());
      EndPos = (EndPos != StringRef::npos ? EndPos : Code.size());
      StringRef Line = Code.substr(BeginPos, EndPos - BeginPos);

      if (I->isComments() || I->getLength() == 0) {
        // Comments Repalcement and Insert Replacement do not need generate
        // original code replacement.
        continue;
      }

      // Insert comments in each line
      if (DuplicateFilter.find(BeginPos) == end(DuplicateFilter)) {
        DuplicateFilter.insert(BeginPos);
        std::string NewReplacementText;
        if (Line.endswith("\\")) {
          // To handle the situation that '\\' appeared in end of row in a macro
          // statement, lines like:
          // #define ERROR_CHECK(call) \
          //    if((call) != 0) {      \
          //        int err = 0;       \
          //        \ my_abort(err); }
          NewReplacementText =
              "/* SYCLCT_ORIG " + removeComments(Line.str()) + "*/ \\\n";
        } else {
          NewReplacementText =
              "/* SYCLCT_ORIG " + removeComments(Line.str()) + "*/\n";
        }

        ExtReplacement NewR(FilePath, BeginPos, 0, NewReplacementText,
                            I->getParentTM() ? I->getParentTM() : nullptr);
        CommentsReplSet.emplace_back(std::move(NewR));
      }
    }
    return CommentsReplSet;
  }

  void HandleTranslationUnit(ASTContext &Context) override {
    // Set Context for build information
    SyclctGlobalInfo::setContext(Context);
    SyclctGlobalInfo::setInRoot(InRoot);
    // The migration process is separated into two stages:
    // 1) Analysis of AST and identification of applicable migration rules
    // 2) Generation of actual textual Replacements
    // Such separation makes it possible to post-process the list of identified
    // migration rules before applying them.
    ATM.matchAST(Context, TransformSet, SSM);

    SyclctGlobalInfo::getInstance().emplaceKernelAndDeviceReplacement(
        TransformSet, SSM);

    // Sort the transformations according to the sort key of the individual
    // transformations.  Sorted from low->high Key values
    std::stable_sort(TransformSet.begin(), TransformSet.end(),
                     TextModification::Compare);

    std::vector<ExtReplacement> ReplSet;
    for (const auto &I : TransformSet) {
      ExtReplacement R = I->getReplacement(Context);
      // TODO: This check filters out headers, which is wrong.
      // TODO: It'd be better not to generate replacements for system headers
      // instead of filtering them.
      std::string RPath = R.getFilePath();
      if (RPath.empty()) {
        llvm::errs() << "[NOTE] rule \"" << R.getParentTM()->getName()
                     << "\" created null code replacement.\n";
        continue;
      }

      makeCanonical(RPath);
      if (isChildPath(InRoot, RPath) || isSamePath(InRoot, RPath)) {
        // TODO: Staticstics
        ReplSet.emplace_back(std::move(R));
      } else {
        // TODO: Staticstics
      }
    }

    // 1. Merge Pass
    std::vector<ExtReplacement> ReplSetMerged = MergeReplacementPass(ReplSet);

    // 2. Filter pass
    ReplacementFilter FilteredReplacements(ReplSetMerged);
    std::vector<ExtReplacement> ReplSetFiltered;
    for (const ExtReplacement &R : FilteredReplacements) {
      ReplSetFiltered.emplace_back(R);
    }

    // 3. May trigger: MergeCommmetsPass
    if (KeepOriginalCodeFlag) { // To keep original code in comments of SYCL
                                // files
      std::vector<ExtReplacement> CommentsReplSet =
          keepOriginalCode(Context.getSourceManager(), ReplSetFiltered);
      ReplSetFiltered = MergeCommmetsPass(CommentsReplSet, ReplSetFiltered);
    }

    // Finally Replacement set
    for (ExtReplacement &R : ReplSetFiltered) {
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

  ~SyclCTConsumer() {
    // Clean EmittedTransformations for input file migrated.
    ASTTraversalMetaInfo::getEmittedTransformations().clear();
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

std::string getInstallPath(clang::tooling::ClangTool &Tool,
                           const char *invokeCommand) {
  SmallString<512> InstalledPath(invokeCommand);

  // Do a PATH lookup, if there are no directory components.
  if (llvm::sys::path::filename(InstalledPath) == InstalledPath) {
    if (llvm::ErrorOr<std::string> Tmp = llvm::sys::findProgramByName(
            llvm::sys::path::filename(InstalledPath.str()))) {
      InstalledPath = *Tmp;
    }
  }

  makeCanonical(InstalledPath);
  StringRef InstalledPathParent(llvm::sys::path::parent_path(InstalledPath));
  // Move up to parent directory of bin directory
  StringRef InstallPath = llvm::sys::path::parent_path(InstalledPathParent);
  return InstallPath.str();
}

// E.g. Path is "/usr/local/cuda/samples" and "cuda" is a symlink of cuda-8.0
// GetRealPath will return /usr/local/cuda-8.0/samples
std::string GetRealPath(clang::tooling::RefactoringTool &Tool, StringRef Path) {
  // Set up Rewriter and to get source manager.
  LangOptions DefaultLangOptions;
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
  TextDiagnosticPrinter DiagnosticPrinter(llvm::errs(), &*DiagOpts);
  DiagnosticsEngine Diagnostics(
      IntrusiveRefCntPtr<DiagnosticIDs>(new DiagnosticIDs()), &*DiagOpts,
      &DiagnosticPrinter, false);
  SourceManager Sources(Diagnostics, Tool.getFiles());
  Rewriter Rewrite(Sources, DefaultLangOptions);
  const SourceManager &SM = Rewrite.getSourceMgr();

  llvm::SmallString<512> AbsolutePath(Path);
  // Try to get the real file path of the symlink.
  const DirectoryEntry *Dir = SM.getFileManager().getDirectory(
      llvm::sys::path::parent_path(AbsolutePath.str()));
  StringRef DirName = SM.getFileManager().getCanonicalName(Dir);
  SmallVector<char, 512> AbsoluteFilename;
  llvm::sys::path::append(AbsoluteFilename, DirName,
                          llvm::sys::path::filename(AbsolutePath.str()));
  return llvm::StringRef(AbsoluteFilename.data(), AbsoluteFilename.size())
      .str();
}

// To validate the root path of the project to be migrated.
void ValidateInputDirectory(clang::tooling::RefactoringTool &Tool,
                            std::string &InRoot) {
  std::string Path = GetRealPath(Tool, InRoot);

  if (isChildPath(CudaPath, Path)) {
    llvm::errs() << "[ERROR] Input root specified by \"-in-root\" option \""
                 << Path << "\" is in CUDA_PATH folder \"" << CudaPath
                 << "\"\n";
    exit(-1);
  }

  if (isChildPath(Path, SyclctInstallPath) ||
      isSamePath(Path, SyclctInstallPath)) {
    llvm::errs() << "[ERROR] Input folder \"" << Path
                 << "\" is the parent or the same as the folder where DPC++ "
                    "Compatibility Tool is installed \""
                 << SyclctInstallPath << "\"\n";
    exit(-1);
  }
}

unsigned int GetLinesNumber(clang::tooling::RefactoringTool &Tool,
                            StringRef Path) {
  // Set up Rewriter and to get source manager.
  LangOptions DefaultLangOptions;
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
  TextDiagnosticPrinter DiagnosticPrinter(llvm::errs(), &*DiagOpts);
  DiagnosticsEngine Diagnostics(
      IntrusiveRefCntPtr<DiagnosticIDs>(new DiagnosticIDs()), &*DiagOpts,
      &DiagnosticPrinter, false);
  SourceManager Sources(Diagnostics, Tool.getFiles());
  Rewriter Rewrite(Sources, DefaultLangOptions);
  SourceManager &SM = Rewrite.getSourceMgr();

  const FileEntry *Entry = SM.getFileManager().getFile(Path);
  if (!Entry) {
    llvm::errs() << "FilePath Invalide..."
                 << "\n";
    assert(false);
  }

  FileID FID = SM.getOrCreateFileID(Entry, SrcMgr::C_User);

  SourceLocation EndOfFile = SM.getLocForEndOfFile(FID);
  unsigned int LineNumber = SM.getSpellingLineNumber(EndOfFile, nullptr);
  return LineNumber;
}

static void printMetrics(
    clang::tooling::RefactoringTool &Tool,
    std::map<std::string, std::array<unsigned int, 3>> &LOCStaticsMap) {

  for (const auto &Elem : LOCStaticsMap) {
    unsigned TotalLines = GetLinesNumber(Tool, Elem.first);
    unsigned TransToAPI = Elem.second[0];
    unsigned TransToSYCL = Elem.second[1];
    unsigned NotTrans = TotalLines - TransToSYCL - TransToAPI;
    unsigned NotSupport = Elem.second[2];

    SyclctStats() << "\n";
    SyclctStats()
        << "File name, LOC migrated to SYCL, LOC migrated to Compatibility "
           "API, LOC not needed to migrate, LOC not able to migrate";
    SyclctStats() << "\n";
    SyclctStats() << Elem.first + ", " + std::to_string(TransToSYCL) + ", " +
                         std::to_string(TransToAPI) + ", " +
                         std::to_string(NotTrans) + ", " +
                         std::to_string(NotSupport);
    SyclctStats() << "\n";
  }
}

static void saveApisReport(void) {

  if (ReportFilePrefix == "stdout") {
    llvm::outs() << "----------APIS report----------------\n";
    llvm::outs() << "API name, Migrated, Frequency";
    llvm::outs() << "\n";
    for (const auto &Elem : APIStaticsMap) {
      std::string Key = Elem.first;
      unsigned int Count = Elem.second;
      llvm::outs() << Key << "," << std::to_string(Count) << "\n";
    }
    llvm::outs() << "-------------------------------------\n";
  } else {
    std::string RFile = OutRoot + "/" + ReportFilePrefix +
                        (ReportFormat == "csv" ? ".apis.csv" : ".apis.log");
    llvm::sys::fs::create_directories(llvm::sys::path::parent_path(RFile));
    std::ofstream File(RFile);

    File << "API name, Migrated, Frequency" << std::endl;
    for (const auto &Elem : APIStaticsMap) {
      std::string Key = Elem.first;
      unsigned int Count = Elem.second;
      File << Key << "," << std::to_string(Count) << std::endl;
    }
  }
}
static void saveStatsReport(clang::tooling::RefactoringTool &Tool,
                            double Duration) {

  printMetrics(Tool, LOCStaticsMap);
  SyclctStats() << "\nTotal migration time: " + std::to_string(Duration) +
                       " ms\n";
  if (ReportFilePrefix == "stdout") {

    llvm::outs() << "----------Stats report---------------\n";
    llvm::outs() << getSyclctStatsStr() << "\n";
    llvm::outs() << "-------------------------------------\n";
  } else {
    std::string RFile = OutRoot + "/" + ReportFilePrefix +
                        (ReportFormat == "csv" ? ".stats.csv" : ".stats.log");
    llvm::sys::fs::create_directories(llvm::sys::path::parent_path(RFile));
    std::ofstream File(RFile);
    File << getSyclctStatsStr() << "\n";
  }
}

static void saveDiagsReport() {

  // SyclctDiags() << "\n";
  if (ReportFilePrefix == "stdout") {
    llvm::outs() << "--------Diags message----------------\n";
    llvm::outs() << getSyclctDiagsStr() << "\n";
    llvm::outs() << "-------------------------------------\n";
  } else {
    std::string RFile = OutRoot + "/" + ReportFilePrefix + ".diags.log";
    llvm::sys::fs::create_directories(llvm::sys::path::parent_path(RFile));
    std::ofstream File(RFile);
    File << getSyclctDiagsStr() << "\n";
  }
}

std::string printCTVersion() {

  std::string buf;
  llvm::raw_string_ostream OS(buf);

  OS << "\nCompatibility Tool Version: " << SYCLCT_VERSION_MAJOR << "."
     << SYCLCT_VERSION_MINOR << "." << SYCLCT_VERSION_PATCH << " codebase:";

  std::string Path = getClangRepositoryPath();
  std::string Revision = getClangRevision();
  if (!Path.empty() || !Revision.empty()) {
    OS << '(';
    if (!Path.empty())
      OS << Path;
    if (!Revision.empty()) {
      if (!Path.empty())
        OS << ' ';
      OS << Revision;
    }
    OS << ')';
  }

  OS << "\n";
  return OS.str();
}

int run(int argc, const char **argv) {
  // CommonOptionsParser will adjust argc to the index of "--"
  int OriginalArgc = argc;
  llvm::cl::SetVersionPrinter(
      [](llvm::raw_ostream &OS) { OS << printCTVersion() << "\n"; });
  CommonOptionsParser OptParser(argc, argv, SyclCTCat);
  clock_t StartTime = clock();
  if (!makeCanonicalOrSetDefaults(InRoot, OutRoot,
                                  OptParser.getSourcePathList()))
    exit(-1);

  if (!validatePaths(InRoot, OptParser.getSourcePathList()))
    exit(-1);

  bool GenReport = false;
  if (checkReportArgs(ReportType, ReportFormat, ReportFilePrefix,
                      ReportOnlyFlag, GenReport, DiagsContent) == false)
    exit(-1);

  if (GenReport)
    llvm::outs() << "Generate report: "
                 << "report-type:" << ReportType
                 << ", report-format:" << ReportFormat
                 << ", report-file-prefix:" << ReportFilePrefix << "\n";

  CudaPath = getCudaInstallPath(OriginalArgc, argv);
  SYCLCT_DEBUG_WITH_TYPE(
      "CudaPath", SyclctLog() << "Cuda Path found: " << CudaPath << "\n");

  RefactoringTool Tool(OptParser.getCompilations(),
                       OptParser.getSourcePathList());
  SyclctInstallPath = getInstallPath(Tool, argv[0]);

  ValidateInputDirectory(Tool, InRoot);
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

  if (GenReport) {
    // report: apis, stats, diags
    if (ReportType.find("all") != std::string::npos ||
        ReportType.find("apis") != std::string::npos)
      saveApisReport();

    if (ReportType.find("all") != std::string::npos ||
        ReportType.find("stats") != std::string::npos) {
      clock_t EndTime = clock();
      double Duration = (double)(EndTime - StartTime) / (CLOCKS_PER_SEC / 1000);
      saveStatsReport(Tool, Duration);
    }
    // all doesn't include diags.
    if (ReportType.find("diags") != std::string::npos) {
      saveDiagsReport();
    }
    if (ReportOnlyFlag)
      return MigrationSucceeded;
  }

  // if run was successful
  int Status = saveNewFiles(Tool, InRoot, OutRoot);

  DebugInfo::ShowStatus(Status);
  return Status;
}
