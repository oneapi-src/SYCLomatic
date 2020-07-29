//===--- ASTTraversal.cpp --------------------------------*- C++ -*---===//
//
// Copyright (C) 2018 - 2020 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

#include "ASTTraversal.h"
#include "AnalysisInfo.h"
#include "Debug.h"
#include "GAnalytics.h"
#include "SaveNewFiles.h"
#include "Utility.h"
#include "Checkpoint.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Analysis/AnalysisDeclContext.h"
#include "clang/Basic/CharInfo.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Path.h"

#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>

using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::dpct;
using namespace clang::tooling;

extern std::string CudaPath;
extern std::string DpctInstallPath; // Installation directory for this tool
extern llvm::cl::opt<UsmLevel> USMLevel;
extern bool ProcessAllFlag;
extern bool ExplicitClNamespace;

auto parentStmt = []() {
  return anyOf(hasParent(compoundStmt()), hasParent(forStmt()),
               hasParent(whileStmt()), hasParent(doStmt()),
               hasParent(ifStmt()));
};

static const CXXConstructorDecl *getIfConstructorDecl(const Decl *ND) {
  if (const auto *Tmpl = dyn_cast<FunctionTemplateDecl>(ND))
    ND = Tmpl->getTemplatedDecl();
  return dyn_cast<CXXConstructorDecl>(ND);
}

static internal::Matcher<NamedDecl> vectorTypeName() {
  std::vector<std::string> TypeNames(MapNames::SupportedVectorTypes.begin(),
                                     MapNames::SupportedVectorTypes.end());
  return internal::Matcher<NamedDecl>(new internal::HasNameMatcher(TypeNames));
}

unsigned MigrationRule::PairID = 0;

void IncludesCallbacks::ReplaceCuMacro(const Token &MacroNameTok) {
  std::string InRoot = ATM.InRoot;
  std::string InFile = SM.getFilename(MacroNameTok.getLocation()).str();
  bool IsInRoot = !llvm::sys::fs::is_directory(InFile) &&
                  (isChildOrSamePath(InRoot, InFile));

  if (!IsInRoot) {
    return;
  }
  if (!MacroNameTok.getIdentifierInfo()) {
    return;
  }
  std::string MacroName = MacroNameTok.getIdentifierInfo()->getName().str();
  auto Iter = MapNames::MacrosMap.find(MacroName);
  if (Iter != MapNames::MacrosMap.end()) {
    std::string ReplacedMacroName = Iter->second;
    TransformSet.emplace_back(new ReplaceToken(MacroNameTok.getLocation(),
                                               std::move(ReplacedMacroName)));
  }
}

void IncludesCallbacks::MacroDefined(const Token &MacroNameTok,
                                     const MacroDirective *MD) {
  std::string InRoot = ATM.InRoot;
  std::string InFile = SM.getFilename(MacroNameTok.getLocation()).str();
  bool IsInRoot = !llvm::sys::fs::is_directory(InFile) &&
                  (isChildOrSamePath(InRoot, InFile));

  size_t i;
  // Record all macro define locations
  for (i = 0; i < MD->getMacroInfo()->getNumTokens(); i++) {
    std::shared_ptr<dpct::DpctGlobalInfo::MacroDefRecord> R =
      std::make_shared<dpct::DpctGlobalInfo::MacroDefRecord>(
        MacroNameTok.getLocation(), IsInRoot);
    dpct::DpctGlobalInfo::getMacroTokenToMacroDefineLoc()[SM.getCharacterData(
        MD->getMacroInfo()->getReplacementToken(i).getLocation())] = R;
  }

  if (!IsInRoot) {
    return;
  }

  auto MI = MD->getMacroInfo();
  for (auto Iter = MI->tokens_begin(); Iter != MI->tokens_end(); ++Iter) {
    auto II = Iter->getIdentifierInfo();
    if (!II)
      continue;

    if (MapNames::MacrosMap.find(II->getName().str()) !=
        MapNames::MacrosMap.end()) {
      std::string ReplacedMacroName =
          MapNames::MacrosMap.at(II->getName().str());
      TransformSet.emplace_back(
          new ReplaceToken(Iter->getLocation(), std::move(ReplacedMacroName)));
    }

    if (II->hasMacroDefinition() && (II->getName().str() == "__host__" ||
                                     II->getName().str() == "__device__" ||
                                     II->getName().str() == "__global__" ||
                                     II->getName().str() == "__constant__")) {
      TransformSet.emplace_back(removeMacroInvocationAndTrailingSpaces(
          SourceRange(Iter->getLocation(), Iter->getEndLoc())));
    }
  }
}

void IncludesCallbacks::MacroExpands(const Token &MacroNameTok,
                                     const MacroDefinition &MD,
                                     SourceRange Range, const MacroArgs *Args) {
  std::string InRoot = ATM.InRoot;
  std::string InFile = SM.getFilename(MacroNameTok.getLocation()).str();
  bool IsInRoot = !llvm::sys::fs::is_directory(InFile) &&
                  (isChildOrSamePath(InRoot, InFile));
  if (MD.getMacroInfo()->getNumTokens() > 0) {
    if (dpct::DpctGlobalInfo::getMacroDefines().find(MD.getMacroInfo()) ==
        dpct::DpctGlobalInfo::getMacroDefines().end()) {
      // Record all processed macro definition
      dpct::DpctGlobalInfo::getMacroDefines()[MD.getMacroInfo()] = true;
      size_t i;
      // Record all tokens in the macro definition
      for (i = 0; i < MD.getMacroInfo()->getNumTokens(); i++) {
        std::shared_ptr<dpct::DpctGlobalInfo::MacroExpansionRecord> R =
            std::make_shared<dpct::DpctGlobalInfo::MacroExpansionRecord>(
                MacroNameTok.getIdentifierInfo(), MD.getMacroInfo(), Range,
                IsInRoot, i);
        dpct::DpctGlobalInfo::getExpansionRangeToMacroRecord()
            [SM.getCharacterData(
                MD.getMacroInfo()->getReplacementToken(i).getLocation())] = R;
      }
    }
  } else {
    // Extend the Range to include comments/whitespaces before next token
    auto EndLoc = Range.getEnd();
    Token Tok;
    do {
      EndLoc = SM.getExpansionLoc(EndLoc);
      Lexer::getRawToken(
          EndLoc.getLocWithOffset(Lexer::MeasureTokenLength(
              EndLoc, SM, dpct::DpctGlobalInfo::getContext().getLangOpts())),
          Tok, SM, dpct::DpctGlobalInfo::getContext().getLangOpts(), true);
      EndLoc = Tok.getEndLoc();
    } while (Tok.isNot(tok::eof) && Tok.is(tok::comment));

    if (Tok.isNot(tok::eof)) {
      dpct::DpctGlobalInfo::getEndOfEmptyMacros()[getHashStrFromLoc(Tok.getLocation())] = Range.getBegin();
      dpct::DpctGlobalInfo::getBeginOfEmptyMacros()[getHashStrFromLoc(Range.getBegin())] = Range.getEnd();
    }
  }

  if (!IsInRoot) {
    return;
  }

  if (MacroNameTok.getIdentifierInfo() &&
      MacroNameTok.getIdentifierInfo()->getName() == "__CUDA_ARCH__") {
    TransformSet.emplace_back(
      new ReplaceText(Range.getBegin(), 13, "DPCPP_COMPATIBILITY_TEMP"));
  }

  // For the un-specialized struct, there is no AST for the extern function
  // declaration in its member function body in Windows. e.g: template <typename
  // T> struct foo
  //{
  //    __device__ T *getPointer()
  //    {
  //        extern __device__ void error(void); // No AST for this line
  //        error();
  //        return NULL;
  //    }
  // };
  auto TKind = MacroNameTok.getKind();
  if (!MacroNameTok.getIdentifierInfo()) {
    return;
  }
  auto Name = MacroNameTok.getIdentifierInfo()->getName();
  if (TKind == tok::identifier &&
      (Name == "__host__" || Name == "__device__" || Name == "__global__" ||
       Name == "__constant__" || Name == "__launch_bounds__")) {
    TransformSet.emplace_back(removeMacroInvocationAndTrailingSpaces(Range));
  }

  if (TKind == tok::identifier && Name == "__forceinline__") {
    TransformSet.emplace_back(
        new ReplaceToken(Range.getBegin(), "__dpct_inline__"));
  }

  auto Iter = MapNames::HostAllocSet.find(Name.str());
  if (TKind == tok::identifier && Iter != MapNames::HostAllocSet.end()) {
    if (MD.getMacroInfo()->getNumTokens() == 1) {
      auto ReplToken = MD.getMacroInfo()->getReplacementToken(0);
      if (ReplToken.getKind() == tok::numeric_constant) {
        TransformSet.emplace_back(
            new ReplaceToken(Range.getBegin(), "0"));
        DiagnosticsUtils::report(
            Range.getBegin(), Diagnostics::HOSTALLOCMACRO_NO_MEANING,
            dpct::DpctGlobalInfo::getCompilerInstance(), &TransformSet, false,
            Name.str());
      }
    }
  }
}
TextModification *
IncludesCallbacks::removeMacroInvocationAndTrailingSpaces(SourceRange Range) {
  return new ReplaceText(Range.getBegin(),
                         getLenIncludingTrailingSpaces(Range, SM), "", true);
}

void IncludesCallbacks::Ifdef(SourceLocation Loc, const Token &MacroNameTok,
                              const MacroDefinition &MD) {
  ReplaceCuMacro(MacroNameTok);
}
void IncludesCallbacks::Ifndef(SourceLocation Loc, const Token &MacroNameTok,
                               const MacroDefinition &MD) {
  ReplaceCuMacro(MacroNameTok);
}

void IncludesCallbacks::Defined(const Token &MacroNameTok,
                                const MacroDefinition &MD, SourceRange Range) {
  ReplaceCuMacro(MacroNameTok);
}

void IncludesCallbacks::ReplaceCuMacro(SourceRange ConditionRange) {
  auto Begin = SM.getExpansionLoc(ConditionRange.getBegin());
  auto End = SM.getExpansionLoc(ConditionRange.getEnd());
  const char *BP = SM.getCharacterData(Begin);
  const char *EP = SM.getCharacterData(End);
  unsigned int Size = EP - BP + 1;
  std::string E(BP, Size);
  size_t Pos = 0;
  const std::string MacroName = "__CUDA_ARCH__";
  std::string ReplacedMacroName;
  if (MapNames::MacrosMap.find(MacroName) != MapNames::MacrosMap.end()) {
    ReplacedMacroName = MapNames::MacrosMap.at(MacroName);
  } else {
    return;
  }

  std::size_t Found = E.find(MacroName, Pos);
  while (Found != std::string::npos) {
    // found one, insert replace for it
    if (MapNames::MacrosMap.find(MacroName) != MapNames::MacrosMap.end()) {
      SourceLocation IB = Begin.getLocWithOffset(Found);
      SourceLocation IE = IB.getLocWithOffset(MacroName.length());
      CharSourceRange InsertRange(SourceRange(IB, IE), false);
      TransformSet.emplace_back(
          new ReplaceInclude(InsertRange, std::move(ReplacedMacroName)));
    }
    // check next
    Pos = Found + MacroName.length();
    if ((Pos + MacroName.length()) > Size) {
      break;
    }
    Found = E.find(MacroName, Pos);
  }
}
void IncludesCallbacks::If(SourceLocation Loc, SourceRange ConditionRange,
                           ConditionValueKind ConditionValue) {
  std::string InRoot = ATM.InRoot;
  std::string InFile = SM.getFilename(Loc).str();
  bool IsInRoot = !llvm::sys::fs::is_directory(InFile) &&
                  (isChildOrSamePath(InRoot, InFile));

  if (!IsInRoot) {
    return;
  }
  ReplaceCuMacro(ConditionRange);
}
void IncludesCallbacks::Elif(SourceLocation Loc, SourceRange ConditionRange,
                             ConditionValueKind ConditionValue,
                             SourceLocation IfLoc) {
  std::string InRoot = ATM.InRoot;
  std::string InFile = SM.getFilename(Loc).str();
  bool IsInRoot = !llvm::sys::fs::is_directory(InFile) &&
                  (isChildOrSamePath(InRoot, InFile));

  if (!IsInRoot) {
    return;
  }

  ReplaceCuMacro(ConditionRange);
}

bool IncludesCallbacks::ShouldEnter(StringRef FileName, bool IsAngled) {
#ifdef _WIN32
  std::string Name = FileName.str();
  return !IsAngled || !MapNames::isInSet(MapNames::ThrustFileExcludeSet, Name);
#else
  return true;
#endif
}

// A class that uses the RAII idiom to selectively update the locations of the
// last inclusion directives.
class LastInclusionLocationUpdater {
public:
  LastInclusionLocationUpdater(SourceLocation Loc, bool UpdateNeeded = true)
      : Loc(Loc), UpdateNeeded(UpdateNeeded) {}
  ~LastInclusionLocationUpdater() {
    if (UpdateNeeded)
      DpctGlobalInfo::getInstance().setLastIncludeLocation(Loc);
  }
  void update(bool UpdateNeeded) { this->UpdateNeeded = UpdateNeeded; }

private:
  SourceLocation Loc;
  bool UpdateNeeded;
};

void IncludesCallbacks::InclusionDirective(
    SourceLocation HashLoc, const Token &IncludeTok, StringRef FileName,
    bool IsAngled, CharSourceRange FilenameRange, const FileEntry *File,
    StringRef SearchPath, StringRef RelativePath, const Module *Imported,
    SrcMgr::CharacteristicKind FileType) {
  // Record the locations of the first and last inclusion directives in a file
  DpctGlobalInfo::getInstance().setFirstIncludeLocation(HashLoc);
  LastInclusionLocationUpdater Updater{FilenameRange.getEnd()};

  std::string IncludePath = SearchPath.str();
  makeCanonical(IncludePath);

  std::string IncludingFile = DpctGlobalInfo::getLocInfo(HashLoc).first;

  // eg. '/home/path/util.h' -> '/home/path'
  StringRef Directory = llvm::sys::path::parent_path(IncludingFile);
  std::string InRoot = ATM.InRoot;

  bool IsIncludingFileInInRoot = !llvm::sys::fs::is_directory(IncludingFile) &&
                                 (isChildOrSamePath(InRoot, Directory.str()));

  // If the header file included can not be found, just return.
  if (!File) {
    return;
  }

  std::string FilePath;
  if (!File->tryGetRealPathName().empty()) {
    FilePath = File->tryGetRealPathName().str();
  } else {
    llvm::SmallString<512> FilePathAbs(File->getName());
    DpctGlobalInfo::getSourceManager().getFileManager().makeAbsolutePath(
        FilePathAbs);
    llvm::sys::path::native(FilePathAbs);
    llvm::sys::path::remove_dots(FilePathAbs, true);
    FilePath = std::string(FilePathAbs.str());
  }

  std::string DirPath = llvm::sys::path::parent_path(FilePath).str();
  bool IsFileInInRoot = !isChildPath(DpctInstallPath, DirPath) &&
                        (isChildOrSamePath(InRoot, DirPath));

  if (IsFileInInRoot) {
    auto FilePathWithoutSymlinks =
        DpctGlobalInfo::removeSymlinks(SM.getFileManager(), FilePath);
    IncludeFileMap[FilePathWithoutSymlinks] = false;
    dpct::DpctGlobalInfo::getIncludingFileSet().insert(FilePathWithoutSymlinks);
  }

  if (!SM.isWrittenInMainFile(HashLoc) && !IsIncludingFileInInRoot) {
    return;
  }

  // The "FilePath" is included by the "IncludingFile".
  // If "FilePath" is not under the Inroot folder, do not record the including
  // relationship information.
  if (DpctGlobalInfo::isInRoot(FilePath, false))
    DpctGlobalInfo::getInstance().recordIncludingRelationship(IncludingFile,
                                                              FilePath);

  // Record that math header is included in this file
  if (IsAngled && (FileName.compare(StringRef("math.h")) == 0 ||
                   FileName.compare(StringRef("cmath")) == 0)) {
    DpctGlobalInfo::getInstance().setMathHeaderInserted(HashLoc, true);
  }

  // Record that time header is included in this file
  if (IsAngled && (FileName.compare(StringRef("time.h")) == 0 )) {
    DpctGlobalInfo::getInstance().setTimeHeaderInserted(HashLoc, true);
  }

  // Record that algorithm header is included in this file
  if (IsAngled && FileName.compare(StringRef("algorithm")) == 0) {
    DpctGlobalInfo::getInstance().setAlgorithmHeaderInserted(HashLoc, true);
  }

  // Replace with
  // <mkl_blas_sycl.hpp>, <mkl_lapack_sycl.hpp> and <mkl_sycl_types.hpp>
  if ((FileName.compare(StringRef("cublas_v2.h")) == 0) ||
      (FileName.compare(StringRef("cublas.h")) == 0) ||
      (FileName.compare(StringRef("cusolverDn.h")) == 0)) {
    DpctGlobalInfo::getInstance().insertHeader(HashLoc, MKL_BLAS_Solver);
    TransformSet.emplace_back(new ReplaceInclude(
        CharSourceRange(SourceRange(HashLoc, FilenameRange.getEnd()),
                        /*IsTokenRange=*/false),
        ""));
    Updater.update(false);
  }

  // Replace with <mkl_rng_sycl.hpp>
  if ((FileName.compare(StringRef("curand.h")) == 0)) {
    DpctGlobalInfo::getInstance().insertHeader(HashLoc, MKL_RNG);
    TransformSet.emplace_back(new ReplaceInclude(
        CharSourceRange(SourceRange(HashLoc, FilenameRange.getEnd()),
                        /*IsTokenRange=*/false),
        ""));
    Updater.update(false);
  }

  // Replace with <mkl_rng_sycl_device.hpp>
  if ((FileName.compare(StringRef("curand_kernel.h")) == 0)) {
    DpctGlobalInfo::getInstance().insertHeader(HashLoc, MKL_RNG_DEVICE);
    TransformSet.emplace_back(new ReplaceInclude(
        CharSourceRange(SourceRange(HashLoc, FilenameRange.getEnd()),
                        /*IsTokenRange=*/false),
        ""));
    Updater.update(false);
  }

  // Replace with <mkl_spblas_sycl.hpp>
  if ((FileName.compare(StringRef("cusparse.h")) == 0) ||
      (FileName.compare(StringRef("cusparse_v2.h")) == 0)) {
    DpctGlobalInfo::getInstance().insertHeader(HashLoc, MKL_SPBLAS);
    TransformSet.emplace_back(new ReplaceInclude(
        CharSourceRange(SourceRange(HashLoc, FilenameRange.getEnd()),
                        /*IsTokenRange=*/false),
        ""));
    Updater.update(false);
  }

  if (FileName.startswith(StringRef("nccl"))) {
    if (isChildOrSamePath(InRoot, DirPath)) {
      return;
    }
    DiagnosticsUtils::report(
        HashLoc, Diagnostics::MANUAL_MIGRATION_LIBRARY,
        dpct::DpctGlobalInfo::getCompilerInstance(), &TransformSet, false,
        "Intel(R) oneAPI Collective Communications Library");
    Updater.update(false);
  }
  if (FileName.startswith(StringRef("cudnn"))) {
    if (isChildOrSamePath(InRoot, DirPath)) {
      return;
    }
    DiagnosticsUtils::report(
        HashLoc, Diagnostics::MANUAL_MIGRATION_LIBRARY,
        dpct::DpctGlobalInfo::getCompilerInstance(), &TransformSet, false,
        "Intel(R) oneAPI Deep Neural Network Library (oneDNN)");
    Updater.update(false);
  }

  if (!isChildPath(CudaPath, IncludePath) &&
      IncludePath.compare(0, 15, "/usr/local/cuda", 15)) {

    // Replace "#include "*.cuh"" with "include "*.dp.hpp""
    if (!IsAngled && FileName.endswith(".cuh")) {
      CharSourceRange InsertRange(SourceRange(HashLoc, FilenameRange.getEnd()),
                                  /* IsTokenRange */ false);
      std::string NewFileName = "#include \"" +
                                FileName.drop_back(strlen(".cuh")).str() +
                                ".dp.hpp\"";
      TransformSet.emplace_back(
          new ReplaceInclude(InsertRange, std::move(NewFileName)));
      return;
    }

    // Replace "#include "*.cu"" with "include "*.dp.cpp""
    if (!IsAngled && FileName.endswith(".cu")) {
      CharSourceRange InsertRange(SourceRange(HashLoc, FilenameRange.getEnd()),
                                  /* IsTokenRange */ false);
      std::string NewFileName =
          "#include \"" + FileName.drop_back(strlen(".cu")).str() + ".dp.cpp\"";
      TransformSet.emplace_back(
          new ReplaceInclude(InsertRange, std::move(NewFileName)));
      return;
    }

    // To generate replacement of replacing "#include "*.c"" with "include
    // "*.c.dp.cpp"".
    if (!IsAngled && FileName.endswith(".c")) {
      CharSourceRange InsertRange(SourceRange(HashLoc, FilenameRange.getEnd()),
                                  /* IsTokenRange */ false);
      std::string NewFileName = "#include \"" + FileName.str() + ".dp.cpp\"";

      // For file path in preprocessing stage may be different with the one in
      // syntax analysis stage, here only file name is used as the key.
      const std::string Name = llvm::sys::path::filename(FileName).str();
      IncludeMapSet[Name].push_back(std::make_unique<ReplaceInclude>(
          InsertRange, std::move(NewFileName)));
    }
  }

  // Extra process thrust headers, map to PSTL mapping headers in runtime.
  // For multi thrust header files, only insert once for PSTL mapping header.
  if (IsAngled && (FileName.find("thrust/") != std::string::npos)) {
    if (!DpstdHeaderInserted) {
      std::string Replacement = std::string("<dpct/dpstd_utils.hpp>") +
                                getNL() + "#include <dpstd/execution>" +
                                getNL() + "#include <dpstd/algorithm>";
      DpstdHeaderInserted = true;
      TransformSet.emplace_back(
          new ReplaceInclude(FilenameRange, std::move(Replacement)));
    } else {
      // Replace the complete include directive with an empty string.
      TransformSet.emplace_back(new ReplaceInclude(
          CharSourceRange(SourceRange(HashLoc, FilenameRange.getEnd()),
                          /*IsTokenRange=*/false),
          ""));
      Updater.update(false);
    }
    return;
  }

  //  TODO: implement one of this for each source language.
  // If it's not an include from the SDK, leave it,
  // unless it's runtime header, in which case it will be replaced.
  // In other words, runtime header will be replaced regardless of where it's
  // coming from.
  if (!isChildOrSamePath(CudaPath, IncludePath) &&
      IncludePath.compare(0, 15, "/usr/local/cuda", 15)) {
    if (!(IsAngled && FileName.compare(StringRef("cuda_runtime.h")) == 0)) {
      return;
    }
  }

  // Replace the complete include directive with an empty string.
  // Also remove the trailing spaces to end of the line.
  TransformSet.emplace_back(new ReplaceInclude(
      CharSourceRange(SourceRange(HashLoc, FilenameRange.getEnd()),
                      /*IsTokenRange=*/false),
      "", true));
  Updater.update(false);
}

void IncludesCallbacks::FileChanged(SourceLocation Loc, FileChangeReason Reason,
                                    SrcMgr::CharacteristicKind FileType,
                                    FileID PrevFID) {
  // Record the location when a file is entered
  if (Reason == clang::PPCallbacks::EnterFile) {
    DpctGlobalInfo::getInstance().setFileEnterLocation(Loc);

    std::string InRoot = ATM.InRoot;
    std::string InFile = SM.getFilename(Loc).str();
    bool IsInRoot = !llvm::sys::fs::is_directory(InFile) &&
                    (isChildOrSamePath(InRoot, InFile));

    if (!IsInRoot) {
      return;
    }

    InFile = getAbsolutePath(InFile);
    makeCanonical(InFile);
    if (ProcessAllFlag || GetSourceFileType(InFile) & TypeCudaSource) {
      IncludeFileMap[DpctGlobalInfo::removeSymlinks(SM.getFileManager(), InFile)] = false;
    }
  }
}

void MigrationRule::print(llvm::raw_ostream &OS) {
  const auto &EmittedTransformations = getEmittedTransformations();
  if (EmittedTransformations.empty()) {
    return;
  }

  OS << "[" << getName() << "]" << getNL();
  constexpr char Indent[] = "  ";
  for (const TextModification *TM : EmittedTransformations) {
    OS << Indent;
    TM->print(OS, getCompilerInstance().getASTContext(),
              /* Print parent */ false);
  }
}

void MigrationRule::printStatistics(llvm::raw_ostream &OS) {
  const auto &EmittedTransformations = getEmittedTransformations();
  if (EmittedTransformations.empty()) {
    return;
  }

  OS << "<Statistics of " << getName() << ">" << getNL();
  std::unordered_map<std::string, size_t> TMNameCountMap;
  for (const TextModification *TM : EmittedTransformations) {
    const std::string Name = TM->getName();
    if (TMNameCountMap.count(Name) == 0) {
      TMNameCountMap.emplace(std::make_pair(Name, 1));
    } else {
      ++TMNameCountMap[Name];
    }
  }

  constexpr char Indent[] = "  ";
  for (const auto &Pair : TMNameCountMap) {
    const std::string &Name = Pair.first;
    const size_t &Numbers = Pair.second;
    OS << Indent << "Emitted # of replacement <" << Name << ">: " << Numbers
       << getNL();
  }
}

void MigrationRule::emplaceTransformation(const char *RuleID,
                                          TextModification *TM) {
  ASTTraversalMetaInfo::getEmittedTransformations()[RuleID].emplace_back(TM);
  TransformSet->emplace_back(TM);
}

void IterationSpaceBuiltinRule::registerMatcher(MatchFinder &MF) {
  // TODO: check that threadIdx is not a local variable.
  MF.addMatcher(
      memberExpr(hasObjectExpression(opaqueValueExpr(hasSourceExpression(
                     declRefExpr(to(varDecl(hasAnyName("threadIdx", "blockDim",
                                                       "blockIdx", "gridDim"))
                                        .bind("varDecl")))))),
                 hasAncestor(functionDecl().bind("func")))
          .bind("memberExpr"),
      this);
  MF.addMatcher(declRefExpr(to(varDecl(hasAnyName("warpSize")).bind("varDecl")))
                    .bind("declRefExpr"),
                this);
}

void IterationSpaceBuiltinRule::run(const MatchFinder::MatchResult &Result) {
  CHECKPOINT_ASTMATCHER_RUN_ENTRY();
  const MemberExpr *ME = getNodeAsType<MemberExpr>(Result, "memberExpr");
  const VarDecl *VD = nullptr;
  const DeclRefExpr *DRE = nullptr;
  bool IsME = false;
  if (ME) {
    if (auto FD = getAssistNodeAsType<FunctionDecl>(Result, "func"))
      DeviceFunctionDecl::LinkRedecls(FD)->setItem();
    VD = getAssistNodeAsType<VarDecl>(Result, "varDecl", false);
    if (!VD) {
      return;
    }
    IsME = true;
  } else if ((DRE = getNodeAsType<DeclRefExpr>(Result, "declRefExpr"))) {
    VD = getAssistNodeAsType<VarDecl>(Result, "varDecl", false);
    if (!VD) {
      return;
    }
    std::string InFile = dpct::DpctGlobalInfo::getSourceManager()
                             .getFilename(VD->getBeginLoc())
                             .str();
    if (!isChildOrSamePath(DpctInstallPath, InFile)) {
      return;
    }
  } else {
    return;
  }

  std::string Replacement = getItemName();
  StringRef BuiltinName = VD->getName();

  if (BuiltinName == "threadIdx")
    Replacement += ".get_local_id(";
  else if (BuiltinName == "blockDim")
    Replacement += ".get_local_range().get(";
  else if (BuiltinName == "blockIdx")
    Replacement += ".get_group(";
  else if (BuiltinName == "gridDim")
    Replacement += ".get_group_range(";
  else if (BuiltinName == "warpSize")
    Replacement += ".get_sub_group().get_local_range().get(0)";
  else {
    llvm::dbgs() << "[" << getName()
                 << "] Unexpected builtin variable: " << BuiltinName;
    return;
  }

  if (IsME) {
    ValueDecl *Field = ME->getMemberDecl();
    StringRef FieldName = Field->getName();
    unsigned Dimension;
    if (FieldName == "__fetch_builtin_x")
      Dimension = 2;
    else if (FieldName == "__fetch_builtin_y")
      Dimension = 1;
    else if (FieldName == "__fetch_builtin_z")
      Dimension = 0;
    else {
      llvm::dbgs() << "[" << getName()
                   << "] Unexpected field name: " << FieldName;
      return;
    }
    Replacement += std::to_string(Dimension);
    Replacement += ")";
  }
  if (IsME) {
    emplaceTransformation(new ReplaceStmt(ME, std::move(Replacement)));
  } else {
    emplaceTransformation(new ReplaceStmt(DRE, std::move(Replacement)));
  }
}

REGISTER_RULE(IterationSpaceBuiltinRule)

void ErrorHandlingIfStmtRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(
      // Match if-statement that has no else and has a condition of either an
      // operator!= or a variable of type enum.
      ifStmt(unless(hasElse(anything())),
             hasCondition(
                 anyOf(binaryOperator(hasOperatorName("!=")).bind("op!="),
                       ignoringImpCasts(
                           declRefExpr(hasType(hasCanonicalType(enumType())))
                               .bind("var")))))
          .bind("errIf"),
      this);
  MF.addMatcher(
      // Match if-statement that has no else and has a condition of
      // operator==.
      ifStmt(unless(hasElse(anything())),
             hasCondition(binaryOperator(hasOperatorName("==")).bind("op==")))
          .bind("errIfSpecial"),
      this);
}

static bool isVarRef(const Expr *E) {
  if (auto D = dyn_cast<DeclRefExpr>(E))
    return isa<VarDecl>(D->getDecl());
  else
    return false;
}

static std::string getVarType(const Expr *E) {
  return E->getType().getCanonicalType().getUnqualifiedType().getAsString();
}

static bool isCudaFailureCheck(const BinaryOperator *Op, bool IsEq = false) {
  auto Lhs = Op->getLHS()->IgnoreImplicit();
  auto Rhs = Op->getRHS()->IgnoreImplicit();

  const Expr *Literal = nullptr;

  if (isVarRef(Lhs) && (getVarType(Lhs) == "enum cudaError" ||
                        getVarType(Lhs) == "enum cudaError_enum")) {
    Literal = Rhs;
  } else if (isVarRef(Rhs) && (getVarType(Rhs) == "enum cudaError" ||
                               getVarType(Rhs) == "enum cudaError_enum")) {
    Literal = Lhs;
  } else
    return false;

  if (auto IntLit = dyn_cast<IntegerLiteral>(Literal)) {
    if (IsEq ^ (IntLit->getValue() != 0))
      return false;
  } else if (auto D = dyn_cast<DeclRefExpr>(Literal)) {
    auto EnumDecl = dyn_cast<EnumConstantDecl>(D->getDecl());
    if (!EnumDecl)
      return false;
    if (IsEq ^ (EnumDecl->getInitVal() != 0))
      return false;
  } else {
    // The expression is neither an int literal nor an enum value.
    return false;
  }

  return true;
}

static bool isCudaFailureCheck(const DeclRefExpr *E) {
  return isVarRef(E) && (getVarType(E) == "enum cudaError" ||
                         getVarType(E) == "enum cudaError_enum");
}

void ErrorHandlingIfStmtRule::run(const MatchFinder::MatchResult &Result) {
  CHECKPOINT_ASTMATCHER_RUN_ENTRY();
  static std::vector<std::string> NameList = {"errIf", "errIfSpecial"};
  const IfStmt *If = getNodeAsType<IfStmt>(Result, "errIf");
  if (!If)
    if (!(If = getNodeAsType<IfStmt>(Result, "errIfSpecial")))
      return;
  auto EmitNotRemoved = [&](SourceLocation SL, const Stmt *R) {
    report(SL, Diagnostics::STMT_NOT_REMOVED, false);
  };
  auto isErrorHandlingSafeToRemove = [&](const Stmt *S) {
    if (const auto *CE = dyn_cast<CallExpr>(S)) {
      if (!CE->getDirectCallee()) {
        EmitNotRemoved(S->getSourceRange().getBegin(), S);
        return false;
      }
      auto Name = CE->getDirectCallee()->getNameAsString();
      static const llvm::StringSet<> SafeCallList = {
          "printf", "puts", "exit", "cudaDeviceReset", "fprintf"};
      if (SafeCallList.find(Name) == SafeCallList.end()) {
        EmitNotRemoved(S->getSourceRange().getBegin(), S);
        return false;
      }
#if 0
    //TODO: enable argument check
    for (const auto *S : CE->arguments()) {
      if (!isErrorHandlingSafeToRemove(S->IgnoreImplicit()))
        return false;
    }
#endif
      return true;
    }
#if 0
  //TODO: enable argument check
  else if (isa <DeclRefExpr>(S))
    return true;
  else if (isa<IntegerLiteral>(S))
    return true;
  else if (isa<StringLiteral>(S))
    return true;
#endif
    EmitNotRemoved(S->getSourceRange().getBegin(), S);
    return false;
  };

  auto isErrorHandling = [&](const Stmt *Block) {
    if (!isa<CompoundStmt>(Block))
      return isErrorHandlingSafeToRemove(Block);
    const CompoundStmt *CS = cast<CompoundStmt>(Block);
    for (const auto *S : CS->children()) {
      if (auto *E = dyn_cast_or_null<Expr>(S)) {
        if (!isErrorHandlingSafeToRemove(E->IgnoreImplicit())) {
          return false;
        }
      }
    }
    return true;
  };

  if (![&] {
        bool IsIfstmtSpecialCase = false;
        SourceLocation Ip;
        if (auto Op = getNodeAsType<BinaryOperator>(Result, "op!=")) {
          if (!isCudaFailureCheck(Op))
            return false;
        } else if (auto Op = getNodeAsType<BinaryOperator>(Result, "op==")) {
          if (!isCudaFailureCheck(Op, true))
            return false;
          IsIfstmtSpecialCase = true;
          Ip = Op->getBeginLoc();

        } else {
          auto CondVar = getNodeAsType<DeclRefExpr>(Result, "var");
          if (!isCudaFailureCheck(CondVar))
            return false;
        }
        // We know that it's error checking condition, check the body
        if (!isErrorHandling(If->getThen())) {
          if (IsIfstmtSpecialCase) {
            report(Ip, Diagnostics::IFSTMT_SPECIAL_CASE, false);
          } else {
            report(If->getSourceRange().getBegin(),
                   Diagnostics::IFSTMT_NOT_REMOVED, false);
          }
          return false;
        }
        return true;
      }()) {

    return;
  }

  emplaceTransformation(new ReplaceStmt(If, ""));

  // if the last token right after the ifstmt is ";"
  // remove the token
  auto &SM = DpctGlobalInfo::getSourceManager();
  auto EndLoc = Lexer::getLocForEndOfToken(
      SM.getSpellingLoc(If->getEndLoc()), 0, SM, Result.Context->getLangOpts());
  Token Tok;
  Lexer::getRawToken(EndLoc, Tok, SM, Result.Context->getLangOpts(), true);
  if (Tok.getKind() == tok::semi) {
    emplaceTransformation(new ReplaceText(EndLoc, 1, ""));
  }
}

REGISTER_RULE(ErrorHandlingIfStmtRule)

void ErrorHandlingHostAPIRule::registerMatcher(MatchFinder &MF) {
  auto isMigratedHostAPI = [&]() {
    return allOf(
        anyOf(returns(asString("cudaError_t")),
              returns(asString("cublasStatus_t")),
              returns(asString("nvgraphStatus_t")),
              returns(asString("cusparseStatus_t")),
              returns(asString("cusolverStatus_t")),
              returns(asString("curandStatus_t"))),
        // cudaGetLastError returns cudaError_t but won't fail in the call
        unless(hasName("cudaGetLastError")),
        anyOf(unless(hasAttr(attr::CUDADevice)), hasAttr(attr::CUDAHost)));
  };

  // Match host api call in the condition session of flow control
  MF.addMatcher(
      functionDecl(
          allOf(
              unless(hasDescendant(functionDecl())),
              unless(
                  anyOf(hasAttr(attr::CUDADevice), hasAttr(attr::CUDAGlobal))),
              anyOf(
                  hasDescendant(ifStmt(hasCondition(expr(hasDescendant(
                      callExpr(callee(functionDecl(isMigratedHostAPI())))))))),
                  hasDescendant(doStmt(hasCondition(expr(hasDescendant(
                      callExpr(callee(functionDecl(isMigratedHostAPI())))))))),
                  hasDescendant(whileStmt(hasCondition(expr(hasDescendant(
                      callExpr(callee(functionDecl(isMigratedHostAPI())))))))),
                  hasDescendant(switchStmt(hasCondition(expr(hasDescendant(
                      callExpr(callee(functionDecl(isMigratedHostAPI())))))))),
                  hasDescendant(
                      forStmt(hasCondition(expr(hasDescendant(callExpr(
                          callee(functionDecl(isMigratedHostAPI())))))))))))
          .bind("inConditionHostAPI"),
      this);

  // Match host api call whose return value used inside flow control or return
  MF.addMatcher(
      functionDecl(
          allOf(unless(hasDescendant(functionDecl())),
                unless(anyOf(hasAttr(attr::CUDADevice),
                             hasAttr(attr::CUDAGlobal))),
                hasDescendant(callExpr(allOf(
                    callee(functionDecl(isMigratedHostAPI())),
                    anyOf(hasAncestor(binaryOperator(allOf(
                              hasLHS(declRefExpr()), isAssignmentOperator()))),
                          hasAncestor(varDecl())),
                    anyOf(hasAncestor(ifStmt()), hasAncestor(doStmt()),
                          hasAncestor(switchStmt()), hasAncestor(whileStmt()),
                          hasAncestor(callExpr()), hasAncestor(forStmt())))))))
          .bind("inLoopHostAPI"),
      this);

  MF.addMatcher(
      functionDecl(
          allOf(unless(hasDescendant(functionDecl())),
                unless(anyOf(hasAttr(attr::CUDADevice),
                             hasAttr(attr::CUDAGlobal))),
                hasDescendant(
                    callExpr(allOf(callee(functionDecl(isMigratedHostAPI())),
                                   hasAncestor(returnStmt())))
                        .bind("cccc"))))
          .bind("inReturnHostAPI"),
      this);

  // Match host api call whose return value captured and used
  MF.addMatcher(
      callExpr(allOf(callee(functionDecl(isMigratedHostAPI())),
                     anyOf(hasAncestor(binaryOperator(
                               allOf(hasLHS(declRefExpr().bind("targetLHS")),
                                     isAssignmentOperator()))),
                           hasAncestor(varDecl().bind("targetVarDecl"))),
                     unless(hasDescendant(functionDecl())),
                     hasAncestor(
                         functionDecl(unless(anyOf(hasAttr(attr::CUDADevice),
                                                   hasAttr(attr::CUDAGlobal))))
                             .bind("savedHostAPI"))))
          .bind("referencedHostAPI"),
      this);
}

void ErrorHandlingHostAPIRule::run(const MatchFinder::MatchResult &Result) {
  CHECKPOINT_ASTMATCHER_RUN_ENTRY();
  // if host api call in the condition session of flow control
  // or host api call whose return value used inside flow control or return
  // then add try catch.
  auto FD = getNodeAsType<FunctionDecl>(Result, "inConditionHostAPI");
  if (!FD) {
    FD = getNodeAsType<FunctionDecl>(Result, "inLoopHostAPI");
  }
  if (!FD) {
    FD = getNodeAsType<FunctionDecl>(Result, "inReturnHostAPI");
  }
  if (FD) {
    insertTryCatch(FD);
    return;
  }

  // Check if the return value is saved in an variable,
  // if yes, get the varDecl as the target varDecl TD.
  FD = getAssistNodeAsType<FunctionDecl>(Result, "savedHostAPI");
  if (!FD)
    return;
  auto TVD = getAssistNodeAsType<VarDecl>(Result, "targetVarDecl");
  auto TLHS = getAssistNodeAsType<DeclRefExpr>(Result, "targetLHS");
  const ValueDecl *TD;
  if (TVD || TLHS) {
    TD = TVD ? TVD : TLHS->getDecl();
  }

  if (!TD)
    return;

  // Get the location of the API call to make sure the variable is referenced
  // AFTER the API call.
  auto CE = getAssistNodeAsType<CallExpr>(Result, "referencedHostAPI");

  if (!CE)
    return;

  // For each reference of TD, check if the location is after CE,
  // if yes, add try catch.
  std::vector<const DeclRefExpr *> Refs;
  VarReferencedInFD(FD->getBody(), TD, Refs);
  SourceManager &SM = DpctGlobalInfo::getSourceManager();
  auto CallLoc = SM.getExpansionLoc(CE->getBeginLoc());
  for (auto It = Refs.begin(); It != Refs.end(); ++It) {
    auto RefLoc = SM.getExpansionLoc((*It)->getBeginLoc());
    if (SM.getCharacterData(RefLoc) - SM.getCharacterData(CallLoc) > 0) {
      insertTryCatch(FD);
      return;
    }
  }
}

void ErrorHandlingHostAPIRule::insertTryCatch(const FunctionDecl *FD) {
  SourceManager &SM = DpctGlobalInfo::getSourceManager();
  bool IsLambda = false;
  if (auto CMD = dyn_cast<CXXMethodDecl>(FD)) {
    if (CMD->getParent()->isLambda()) {
      IsLambda = true;
    }
  }

  std::string IndentStr = getIndent(FD->getBeginLoc(), SM).str();

  if (IsLambda) {
    if (auto CSM = dyn_cast<CompoundStmt>(FD->getBody())) {
      //IndentStr = getIndent((*(CSM->body_begin()))->getBeginLoc(), SM).str();
      std::string TryStr = "try{ " + std::string(getNL()) + IndentStr;
      emplaceTransformation(
          new InsertBeforeStmt(*(CSM->body_begin()), std::move(TryStr)));
    }
  } else if (const CXXConstructorDecl *CDecl = getIfConstructorDecl(FD)) {
    emplaceTransformation(new InsertBeforeCtrInitList(CDecl, " try "));
  } else {
    emplaceTransformation(new InsertBeforeStmt(FD->getBody(), " try "));
  }

  std::string ReplaceStr =
      getNL() + IndentStr +
      std::string("catch (" + MapNames::getClNamespace() +
                  "::exception const &exc) {") +
      getNL() + IndentStr + IndentStr +
      std::string("std::cerr << exc.what() << \"Exception caught at file:\" << "
                  "__FILE__ << "
                  "\", line:\" << __LINE__ << std::endl;") +
      getNL() + IndentStr + IndentStr + std::string("std::exit(1);") + getNL() +
      IndentStr + "}";
  if (IsLambda) {
    ReplaceStr += getNL() + IndentStr + "}";
  }
  emplaceTransformation(
      new InsertAfterStmt(FD->getBody(), std::move(ReplaceStr)));
}

REGISTER_RULE(ErrorHandlingHostAPIRule)

void AlignAttrsRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(cxxRecordDecl(hasAttr(attr::Aligned)).bind("classDecl"), this);
}

void AlignAttrsRule::run(const MatchFinder::MatchResult &Result) {
  CHECKPOINT_ASTMATCHER_RUN_ENTRY();
  auto C = getNodeAsType<CXXRecordDecl>(Result, "classDecl");
  if (!C)
    return;
  auto &AV = C->getAttrs();

  for (auto A : AV) {
    if (A->getKind() == attr::Aligned) {
      auto SM = Result.SourceManager;
      auto ExpB = SM->getExpansionLoc(A->getLocation());
      if (!strncmp(SM->getCharacterData(ExpB), "__align__(", 10))
        emplaceTransformation(new ReplaceToken(ExpB, "__dpct_align__"));
    }
  }
}

REGISTER_RULE(AlignAttrsRule)

void FuncAttrsRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(functionDecl(hasAttr(attr::AlwaysInline)).bind("funcDecl"),
                this);
}

void FuncAttrsRule::run(const MatchFinder::MatchResult &Result) {
  CHECKPOINT_ASTMATCHER_RUN_ENTRY();
  auto FD = getNodeAsType<FunctionDecl>(Result, "funcDecl");
  auto SM = Result.SourceManager;
  if (!FD)
    return;
  auto &FA = FD->getAttrs();
  for (auto A : FA) {
    if (A->getKind() == attr::AlwaysInline) {
      // directly used
      auto Loc = SM->getExpansionLoc(A->getLocation());
      if (!strncmp(SM->getCharacterData(Loc), "__forceinline__", 15))
        emplaceTransformation(new ReplaceToken(Loc, "__dpct_inline__"));
      // if is used in another macro
      Loc = SM->getSpellingLoc(
          SM->getImmediateExpansionRange(A->getLocation()).getBegin());
      if (!strncmp(SM->getCharacterData(Loc), "__forceinline__", 15))
        emplaceTransformation(new ReplaceToken(Loc, "__dpct_inline__"));
    }
  }
}

REGISTER_RULE(FuncAttrsRule)

void AtomicFunctionRule::registerMatcher(MatchFinder &MF) {
  std::vector<std::string> AtomicFuncNames(AtomicFuncNamesMap.size());
  std::transform(
      AtomicFuncNamesMap.begin(), AtomicFuncNamesMap.end(),
      AtomicFuncNames.begin(),
      [](const std::pair<std::string, std::string> &p) { return p.first; });

  auto hasAnyAtomicFuncName = [&]() {
    return internal::Matcher<NamedDecl>(
        new internal::HasNameMatcher(AtomicFuncNames));
  };

  // Support all integer type, float and double
  // Type half and half2 are not supported
  auto supportedTypes = [&]() {
    // TODO: investigate usage of __half and __half2 types and support it
    return anyOf(hasType(pointsTo(isInteger())),
                 hasType(pointsTo(asString("float"))),
                 hasType(pointsTo(asString("double"))));
  };

  auto supportedAtomicFunctions = [&]() {
    return allOf(hasAnyAtomicFuncName(), hasParameter(0, supportedTypes()));
  };

  auto unsupportedAtomicFunctions = [&]() {
    return allOf(hasAnyAtomicFuncName(),
                 unless(hasParameter(0, supportedTypes())));
  };

  MF.addMatcher(callExpr(callee(functionDecl(supportedAtomicFunctions())))
                    .bind("supportedAtomicFuncCall"),
                this);

  MF.addMatcher(callExpr(callee(functionDecl(unsupportedAtomicFunctions())))
                    .bind("unsupportedAtomicFuncCall"),
                this);
}

void AtomicFunctionRule::ReportUnsupportedAtomicFunc(const CallExpr *CE) {
  if (!CE)
    return;

  std::ostringstream OSS;
  // Atomic functions with __half and half2 are not supported.
  if (!CE->getDirectCallee())
    return;
  OSS << "half version of "
      << MapNames::ITFName.at(CE->getDirectCallee()->getName().str());
  report(CE->getBeginLoc(), Diagnostics::API_NOT_MIGRATED, false, OSS.str());
}

// Determine if S is a statement inside
// a if/while/do while/for statement.
bool AtomicFunctionRule::IsStmtInStatement(const clang::Stmt *S, const clang::Decl *Root) {
  auto ParentStmt = getParentStmt(S);
  if (!ParentStmt)
    return false;

  auto &Context = dpct::DpctGlobalInfo::getContext();
  auto Parents = Context.getParents(*S);

  if(Parents.size()<1)
    return false;
  const clang::Decl *Parent = Parents[0].get<Decl>();
  auto ParentStmtClass = ParentStmt->getStmtClass();
  bool Ret = ParentStmtClass == Stmt::StmtClass::IfStmtClass ||
             ParentStmtClass == Stmt::StmtClass::WhileStmtClass ||
             ParentStmtClass == Stmt::StmtClass::DoStmtClass ||
             ParentStmtClass == Stmt::StmtClass::ForStmtClass;
  if (Ret)
    return true;
  else if (Parent != Root)
    return IsStmtInStatement(ParentStmt, Root);
  else
    return false;
}

// To find if device variable \pExpr has __share__ attribute,
// if it has, HasSharedAttr is set true.
// if \pExpr is in a if/while/do while/for statement
// \pNeedReport is set true.
// To handle six kind of cases:
// case1: extern __shared__ uint32_t share_array[];
//        atomicAdd(&share_array[0], 1);
// case2: extern __shared__ uint32_t share_array[];
//        uint32_t *p = &share_array[0];
//        atomicAdd(p, 1);
// case3: __shared__ uint32_t share_v;
//        atomicAdd(&share_v, 1);
// case4: __shared__ uint32_t share_v;
//        uint32_t *p = &share_v;
//        atomicAdd(p, 1);
// case5: extern __shared__ uint32_t share_array[];
//        atomicAdd(share_array, 1);
// case6: __shared__ uint32_t share_v;
//        uint32_t *p;
//        p = &share_v;
//        atomicAdd(p, 1);
void AtomicFunctionRule::GetShareAttrRecursive(const Expr *Expr,
                                               bool &HasSharedAttr,
                                               bool &NeedReport) {
  if (!Expr)
    return;

  if (dyn_cast<CallExpr>(Expr)) {
    NeedReport = true;
    return;
  }

  if (auto UO = dyn_cast<UnaryOperator>(Expr)) {
    if (UO->getOpcode() == UnaryOperatorKind::UO_AddrOf) {
      Expr = UO->getSubExpr();
    }
  }

  if (auto BO = dyn_cast<BinaryOperator>(Expr)) {
    GetShareAttrRecursive(BO->getLHS(), HasSharedAttr, NeedReport);
    GetShareAttrRecursive(BO->getRHS(), HasSharedAttr, NeedReport);
  }

  if (auto ASE = dyn_cast<ArraySubscriptExpr>(Expr)) {
    Expr = ASE->getBase();
  }
  const clang::Expr *AssignedExpr = NULL;
  const FunctionDecl *FuncDecl =NULL;
  if (auto DRE = dyn_cast<DeclRefExpr>(Expr->IgnoreImplicitAsWritten())) {
    if (isa<ParmVarDecl>(DRE->getDecl()))
      return;

    if (auto VD = dyn_cast<VarDecl>(DRE->getDecl())) {
      if (VD->hasAttr<CUDASharedAttr>()) {
        HasSharedAttr = true;
        return;
      }

      AssignedExpr = VD->getInit();
      if (FuncDecl = dyn_cast<FunctionDecl>(VD->getDeclContext())) {
        std::vector<const DeclRefExpr *> Refs;
        VarReferencedInFD(FuncDecl->getBody(), VD, Refs);
        for (auto const &Ref : Refs) {
          if (Ref == DRE)
            break;

          if (auto BO = dyn_cast<BinaryOperator>(getParentStmt(Ref))) {
            if (BO->getLHS() == Ref && BO->getOpcode() == BO_Assign &&
                !DpctGlobalInfo::checkSpecificBO(DRE, BO))
              AssignedExpr = BO->getRHS();
          }
        }
      }
    }
  }

  if (AssignedExpr) {
    // if AssignedExpr in a if/while/do while/for statement,
    // it is necessary to report a warning message.
    if (IsStmtInStatement(AssignedExpr, FuncDecl)) {
      NeedReport = true;
    }
    GetShareAttrRecursive(AssignedExpr, HasSharedAttr, NeedReport);
  }
}

void AtomicFunctionRule::MigrateAtomicFunc(
    const CallExpr *CE, const ast_matchers::MatchFinder::MatchResult &Result) {
  if (!CE)
    return;

  // TODO: 1. Investigate are there usages of atomic functions on local address
  //          space
  //       2. If item 1. shows atomic functions on local address space is
  //          significant, detect whether this atomic operation operates in
  //          global space or local space (currently, all in global space,
  //          see dpct_atomic.hpp for more details)
  if (!CE->getDirectCallee())
    return;
  const std::string AtomicFuncName = CE->getDirectCallee()->getName().str();
  if(AtomicFuncNamesMap.find(AtomicFuncName) == AtomicFuncNamesMap.end())
    return;
  std::string ReplacedAtomicFuncName = AtomicFuncNamesMap.at(AtomicFuncName);

  // Explicitly cast all arguments except first argument
  const Type *Arg0Type = CE->getArg(0)->getType().getTypePtrOrNull();
  // Atomic operation's first argument is always pointer type
  if (!Arg0Type || !Arg0Type->isPointerType()) {
    return;
  }
  const QualType PointeeType = Arg0Type->getPointeeType();

  std::string TypeName;
  bool IsTemplateType = false;
  if (auto *SubstedType = dyn_cast<SubstTemplateTypeParmType>(PointeeType)) {
    IsTemplateType = true;
    // Type is substituted in template initialization, use the template
    // parameter name
    if (!SubstedType->getReplacedParameter()->getIdentifier()) {
      return;
    }
    TypeName =
        SubstedType->getReplacedParameter()->getIdentifier()->getName().str();
  } else {
    TypeName = PointeeType.getAsString();
  }

  // add exceptions for atomic tranlastion:
  // eg. source code: atomicMin(double), don't migrate it, its user code.
  //     also: atomic_fetch_min<double> is not available in compute++.
  if ((TypeName == "double" && AtomicFuncName != "atomicAdd") ||
      (TypeName == "float" &&
       !(AtomicFuncName == "atomicAdd" || AtomicFuncName == "atomicExch"))) {

    return;
  }

  bool HasSharedAttr = false;
  bool NeedReport = false;
  GetShareAttrRecursive(CE->getArg(0), HasSharedAttr, NeedReport);
  std::string ClNamespace = ExplicitClNamespace ? "cl::sycl" : "sycl";
  std::string SpaceName = ClNamespace + "::access::address_space::local_space";
  std::string ReplAtomicFuncNameWithSpace =
      ReplacedAtomicFuncName + "<" + TypeName + ", " + SpaceName + ">";
  auto SL = CE->getArg(0)->getBeginLoc();
  if (NeedReport)
    report(SL, Diagnostics::SHARE_MEMORY_ATTR_DEDUCE, false,
           getStmtSpelling(CE->getArg(0)), ReplacedAtomicFuncName,
           ReplAtomicFuncNameWithSpace);

  // Inline the code for ingeter types
  static std::unordered_map<std::string, std::string> AtomicMap = {
    {"atomicAdd", "fetch_add"},
    {"atomicSub", "fetch_sub"},
    {"atomicAnd", "fetch_and"},
    {"atomicOr", "fetch_or"},
    {"atomicXor", "fetch_xor"},
    {"atomicMin", "fetch_min"},
    {"atomicMax", "fetch_max"},
  };

  auto IsMacro = CE->getBeginLoc().isMacroID();
  auto Iter = AtomicMap.find(AtomicFuncName);
  if (!IsMacro && !IsTemplateType && PointeeType->isIntegerType() &&
      Iter != AtomicMap.end()) {
    std::string ReplStr{MapNames::getClNamespace() + "::"};
    ReplStr += "atomic<";
    ReplStr += TypeName;
    if (HasSharedAttr) {
      ReplStr += ", ";
      ReplStr += MapNames::getClNamespace();
      ReplStr += "::access::address_space::local_space";
    }
    ReplStr += ">(";
    ReplStr += MapNames::getClNamespace();
    if (HasSharedAttr)
      ReplStr += "::local_ptr<";
    else
      ReplStr += "::global_ptr<";
    ReplStr += TypeName;
    ReplStr += ">(";
    // Take care of __shared__ variables because their types are
    // changed to pointers
    bool Arg0NeedDeref = false;
    auto *UO = dyn_cast<UnaryOperator>(CE->getArg(0));
    if (UO && UO->getOpcode() == clang::UO_AddrOf) {
      if (auto DRE = dyn_cast<DeclRefExpr>(UO->getSubExpr()->IgnoreImpCasts()))
        Arg0NeedDeref = IsTypeChangedToPointer(DRE);
    }
    // Deref the expression if it is the unary operator of a shared simple
    // variable
    if (Arg0NeedDeref) {
      std::ostringstream OS;
      printDerefOp(OS, CE->getArg(0));
      ReplStr += OS.str();
    } else {
      ReplStr += getStmtSpelling(CE->getArg(0));
    }
    ReplStr += ")).";
    ReplStr += Iter->second;
    ReplStr += "(";

    auto Arg1NeedDeref = false;
    if (auto DRE = dyn_cast<DeclRefExpr>(CE->getArg(1)->IgnoreImpCasts()))
      Arg1NeedDeref = IsTypeChangedToPointer(DRE);
    if (Arg1NeedDeref) {
      std::ostringstream OS;
      printDerefOp(OS, CE->getArg(1));
      ReplStr += OS.str();
    } else {
      ReplStr += getStmtSpelling(CE->getArg(1));
    }
    ReplStr += ")";

    emplaceTransformation(new ReplaceStmt(CE, std::move(ReplStr)));
    return;
  }

  if (HasSharedAttr) {
    ReplacedAtomicFuncName = ReplAtomicFuncNameWithSpace;
  }

  emplaceTransformation(new ReplaceCalleeName(
      CE, std::move(ReplacedAtomicFuncName), AtomicFuncName));

  const unsigned NumArgs = CE->getNumArgs();
  for (unsigned i = 0; i < NumArgs; ++i) {
    const Expr *Arg = CE->getArg(i);
    if (auto *ImpCast = dyn_cast<ImplicitCastExpr>(Arg)) {
      if (ImpCast->getCastKind() != clang::CK_LValueToRValue) {
        if (i == 0) {
          if (dyn_cast<DeclRefExpr>(Arg->IgnoreImpCasts())) {
            emplaceTransformation(
                new InsertBeforeStmt(Arg, "(" + TypeName + "*)"));
          } else {
            insertAroundStmt(Arg, "(" + TypeName + "*)(", ")");
          }
        } else {
          if (dyn_cast<IntegerLiteral>(Arg->IgnoreImpCasts()) ||
              dyn_cast<DeclRefExpr>(Arg->IgnoreImpCasts())) {
            emplaceTransformation(
                new InsertBeforeStmt(Arg, "(" + TypeName + ")"));
          } else {
            insertAroundStmt(Arg, "(" + TypeName + ")(", ")");
          }
        }
      }
    }
  }
}

void AtomicFunctionRule::run(const MatchFinder::MatchResult &Result) {
  CHECKPOINT_ASTMATCHER_RUN_ENTRY();
  ReportUnsupportedAtomicFunc(
      getNodeAsType<CallExpr>(Result, "unsupportedAtomicFuncCall"));

  MigrateAtomicFunc(getNodeAsType<CallExpr>(Result, "supportedAtomicFuncCall"),
                    Result);
}

REGISTER_RULE(AtomicFunctionRule)

void ThrustFunctionRule::registerMatcher(MatchFinder &MF) {
  std::vector<std::string> ThrustFuncNames(MapNames::ThrustFuncNamesMap.size());
  std::transform(
      MapNames::ThrustFuncNamesMap.begin(), MapNames::ThrustFuncNamesMap.end(),
      ThrustFuncNames.begin(),
      [](const std::pair<std::string, MapNames::ThrustFuncReplInfo> &p) {
        return p.first;
      });

  auto hasAnyThrustFuncName = [&]() {
    return internal::Matcher<NamedDecl>(
        new internal::HasNameMatcher(ThrustFuncNames));
  };

  MF.addMatcher(callExpr(callee(functionDecl(
                             hasAnyThrustFuncName(),
                             hasDeclContext(namespaceDecl(hasName("thrust"))))))
                    .bind("thrustFuncCall"),
               this);
}

TextModification *removeArg(const CallExpr *C, unsigned n,
                            const SourceManager &SM);

void ThrustFunctionRule::run(const MatchFinder::MatchResult &Result) {
  CHECKPOINT_ASTMATCHER_RUN_ENTRY();
  auto UniqueName = [](const Stmt *S) {
    auto &SM = DpctGlobalInfo::getSourceManager();
    SourceLocation Loc = S->getBeginLoc();
    return getHashAsString(Loc.printToString(SM)).substr(0, 6);
  };

  if (const CallExpr *CE = getNodeAsType<CallExpr>(Result, "thrustFuncCall")) {
    // handle the a regular call expr
    const std::string ThrustFuncName = CE->getDirectCallee()->getName().str();
    const unsigned NumArgs = CE->getNumArgs();
    auto QT= CE->getArg(0)->getType();
    LangOptions LO;
    std::string ArgT = QT.getAsString(PrintingPolicy(LO));

    auto ReplInfo = MapNames::ThrustFuncNamesMap.find(ThrustFuncName);
    if(ReplInfo == MapNames::ThrustFuncNamesMap.end())
        return;
    auto NewName = ReplInfo->second.ReplName;

    if (ThrustFuncName == "copy_if" &&
        (ArgT.find("execution_policy_base") == std::string::npos &&
             NumArgs == 5 ||
         NumArgs > 5)) {
      NewName = "dpct::" + ThrustFuncName;
      emplaceTransformation(
          new ReplaceCalleeName(CE, std::move(NewName), ThrustFuncName));
    } else if (ThrustFuncName == "transform_reduce") {
      // The initial value and the reduce functor are provided before the
      // transform functor in std::transform_reduce, which differs from
      // thrust::transform_reduce.
      if (NumArgs == 5) {
        emplaceTransformation(removeArg(CE, 2, *Result.SourceManager));

        dpct::ExprAnalysis EA;
        EA.analyze(CE->getArg(2));
        std::string Str = ", " + EA.getReplacedString();
        emplaceTransformation(
            new InsertAfterStmt(CE->getArg(4), std::move(Str)));

      } else if (NumArgs == 6) {
        emplaceTransformation(removeArg(CE, 3, *Result.SourceManager));
        dpct::ExprAnalysis EA;
        EA.analyze(CE->getArg(3));
        std::string Str = ", " + EA.getReplacedString();
        emplaceTransformation(
            new InsertAfterStmt(CE->getArg(5), std::move(Str)));
      }

      if (ArgT.find("execution_policy_base") != std::string::npos) {
        emplaceTransformation(
            new ReplaceCalleeName(CE, std::move(NewName), ThrustFuncName));
        return;
      }

    } else if (ThrustFuncName == "make_zip_iterator") {
      // dpstd::make_zip_iterator expects the component iterators to be passed
      // directly instead of being wrapped in a tuple as
      // thrust::make_zip_iterator requires.
      std::string NewArg;
      if (auto CCE = dyn_cast<CXXConstructExpr>(CE->getArg(0)))
        if (const CallExpr *SubCE =
                dyn_cast<CallExpr>(CCE->getArg(0)->IgnoreImplicit())) {
          std::string Arg0 = getStmtSpelling(SubCE->getArg(0));
          std::string Arg1 = getStmtSpelling(SubCE->getArg(1));
          NewArg = Arg0 + ", " + Arg1;
        }

      if (NewArg.empty()) {
        std::string Arg0 =
            "std::get<0>(" + getStmtSpelling(CE->getArg(0)) + ")";
        std::string Arg1 =
            "std::get<1>(" + getStmtSpelling(CE->getArg(0)) + ")";
        NewArg = Arg0 + ", " + Arg1;
      }

      emplaceTransformation(removeArg(CE, 0, *Result.SourceManager));
      emplaceTransformation(
          new InsertAfterStmt(CE->getArg(0), std::move(NewArg)));

    } else if (ArgT.find("execution_policy_base") != std::string::npos) {
      emplaceTransformation(
          new ReplaceCalleeName(CE, std::move(NewName), ThrustFuncName));
      return;
    }

    if (ThrustFuncName == "exclusive_scan") {
      DpctGlobalInfo::getInstance().insertHeader(CE->getBeginLoc(), Numeric);
      emplaceTransformation(new InsertText(CE->getEndLoc(), ", 0"));
    }

    emplaceTransformation(
        new ReplaceCalleeName(CE, std::move(NewName), ThrustFuncName));
    if(CE->getNumArgs()<=0)
        return;
    auto ExtraParam = ReplInfo->second.ExtraParam;
    if (!ExtraParam.empty()) {
      // This is a temporary fix until, the Intel(R) oneAPI DPC++ Compiler and
      // Intel(R) oneAPI DPC++ Library support creating a SYCL execution policy
      // without creating a unique one for every use
      if (ExtraParam == "dpstd::execution::sycl") {
        std::string Name = UniqueName(CE);
        if (checkWhetherIsDuplicate(CE, false))
          return;
        int Index = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();
        buildTempVariableMap(Index, CE, HelperFuncType::DefaultQueue);
        std::string TemplateArg = "";
        if (DpctGlobalInfo::isSyclNamedLambda())
          TemplateArg = std::string("<class Policy_") + UniqueName(CE) + ">";
        ExtraParam = "dpstd::execution::make_device_policy" +
                      TemplateArg + "({{NEEDREPLACEQ" +
                     std::to_string(Index) + "}})";
      }
      emplaceTransformation(
          new InsertBeforeStmt(CE->getArg(0), ExtraParam + ", "));
    }
  }
}

REGISTER_RULE(ThrustFunctionRule)

void ThrustCtorExprRule::registerMatcher(MatchFinder &MF) {

  auto hasAnyThrustRecord = []() {
    return cxxRecordDecl(hasName("complex"),
                         hasDeclContext(namespaceDecl(hasName("thrust"))));
  };

  MF.addMatcher(
      cxxConstructExpr(hasType(hasAnyThrustRecord())).bind("thrustCtorExpr"),
      this);
}

void ThrustCtorExprRule::run(const MatchFinder::MatchResult &Result) {
  CHECKPOINT_ASTMATCHER_RUN_ENTRY();
  if (const CXXConstructExpr *CE =
          getNodeAsType<CXXConstructExpr>(Result, "thrustCtorExpr")) {
    // handle constructor expressions
    std::string ExprStr = getStmtSpelling(CE);
    if (ExprStr.substr(0, 8) != "thrust::") {
      return;
    }
    auto P = ExprStr.find('<');
    if (P != std::string::npos) {
      std::string FuncName = ExprStr.substr(8, P - 8);
      auto ReplInfo = MapNames::ThrustFuncNamesMap.find(FuncName);
      if (ReplInfo == MapNames::ThrustFuncNamesMap.end()) {
        return;
      }
      std::string ReplName = ReplInfo->second.ReplName;
      if (ReplName == "std::complex") {
        DpctGlobalInfo::getInstance().insertHeader(CE->getBeginLoc(), Complex);
      }
      emplaceTransformation(
          new ReplaceText(CE->getBeginLoc(), P, std::move(ReplName)));
    }
  }
}

REGISTER_RULE(ThrustCtorExprRule)

// Rule for types replacements in var declarations and field declarations
void TypeInDeclRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(
      typeLoc(
          loc(qualType(hasDeclaration(namedDecl(anyOf(
              hasAnyName(
                  "cudaError", "cufftResult_t", "curandStatus", "cublasStatus",
                  "CUstream_st", "complex", "device_vector", "device_ptr",
                  "host_vector", "cublasHandle_t", "CUevent_st", "__half",
                  "half", "__half2", "half2", "cudaMemoryAdvise",
                  "cudaError_enum", "cudaDeviceProp", "cudaPitchedPtr",
                  "counting_iterator", "transform_iterator",
                  "permutation_iterator", "iterator_difference",
                  "cusolverDnHandle_t", "device_malloc_allocator", "divides",
                  "tuple", "maximum", "multiplies", "plus", "cudaDataType_t",
                  "cudaError_t", "CUresult", "cudaEvent_t", "cublasStatus_t",
                  "cuComplex", "cuDoubleComplex", "cublasFillMode_t",
                  "cublasDiagType_t", "cublasSideMode_t", "cublasOperation_t",
                  "cusolverStatus_t", "cusolverEigType_t", "cusolverEigMode_t",
                  "curandStatus_t", "cudaStream_t", "cusparseStatus_t",
                  "cusparseDiagType_t", "cusparseFillMode_t",
                  "cusparseIndexBase_t", "cusparseMatrixType_t",
                  "cusparseOperation_t", "cusparseMatDescr_t",
                  "cusparseHandle_t", "CUcontext", "cublasPointerMode_t",
                  "cusparsePointerMode_t", "cublasGemmAlgo_t",
                  "cusparseSolveAnalysisInfo_t", "cudaDataType",
                  "cublasDataType_t", "curandState_t", "curandState",
                  "curandStateXORWOW_t", "curandStatePhilox4_32_10_t",
                  "curandStateMRG32k3a_t", "minus", "negate", "logical_or",
                  "identity"),
              matchesName("cudnn.*|nccl.*")))))))
          .bind("cudaTypeDef"),
      this);
  MF.addMatcher(varDecl(hasTypeLoc(typeLoc(loc(templateSpecializationType(
                            hasAnyTemplateArgument(refersToType(hasDeclaration(
                                namedDecl(hasName("use_default"))))))))))
                    .bind("useDefaultVarDeclInTemplateArg"),
                this);
}

std::string getReplacementForType(std::string TypeStr, bool IsVarDecl = false,
                                  std::string *TypeStrRemovePrefix = nullptr) {
  // divide TypeStr into elements, separated by whitespace
  std::istringstream ISS(TypeStr);
  std::vector<std::string> Strs(std::istream_iterator<std::string>{ISS},
                                std::istream_iterator<std::string>());
  auto it = std::remove_if(Strs.begin(), Strs.end(), [](llvm::StringRef Str) {
    return (Str.contains("&") || Str.contains("*"));
  });
  if (it != Strs.end())
    Strs.erase(it);

  // append possible '>' at the end to the previous element
  while (Strs.size() > 1 && Strs.back() == ">") {
    Strs[Strs.size() - 2] += Strs.back();
    Strs.pop_back();
  }

  std::string TypeName = Strs.back();
  // remove possible template parameters from TypeName
  size_t bracketBeginPos = TypeName.find('<');
  if (bracketBeginPos != std::string::npos) {
    size_t bracketEndPos = TypeName.rfind('>');
    TypeName.erase(bracketBeginPos, bracketEndPos - bracketBeginPos + 1);
  }
  if (TypeStrRemovePrefix != nullptr)
    *TypeStrRemovePrefix = TypeName;
  SrcAPIStaticsMap[TypeName]++;
  auto Search = MapNames::TypeNamesMap.find(TypeName);
  if (Search == MapNames::TypeNamesMap.end())
    return "";

  std::string Replacement = TypeStr;
  if(Replacement.find(TypeName) == std::string::npos)
    return "";

  Replacement = Replacement.substr(Replacement.find(TypeName));
  if (IsVarDecl) {
    return Replacement.replace(0, TypeName.length(), Search->second);
  } else {
    Replacement.replace(0, TypeName.length(), Search->second);
  }

  return Replacement;
}

template <typename T>
bool getLocation(const Type *TypePtr, SourceLocation &SL) {
  auto TType = TypePtr->getAs<T>();
  if (TType) {
    auto TypeDecl = TType->getDecl();
    if (TypeDecl) {
      SL = TypeDecl->getLocation();
      return true;
    } else {
      return false;
    }
  }
  return false;
}

bool getTypeDeclLocation(const Type *TypePtr, SourceLocation &SL) {
  if (getLocation<EnumType>(TypePtr, SL)) {
    return true;
  } else if (getLocation<TypedefType>(TypePtr, SL)) {
    return true;
  } else if (getLocation<RecordType>(TypePtr, SL)) {
    return true;
  }
  return false;
}

bool getTemplateTypeReplacement(std::string TypeStr, std::string &Replacement,
                                unsigned &Len) {
  auto P1 = TypeStr.find('<');
  if (P1 != std::string::npos) {
    auto P2 = Replacement.find('<');
    if (P2 != std::string::npos) {
      Replacement = Replacement.substr(0, P2);
    }
    Len = P1;
    return true;
  }
  return false;
}

bool isAuto(const char *StrChar, unsigned Len) {
  return std::string(StrChar, Len) == "auto";
}

void insertComplexHeader(SourceLocation SL,
                         std::string &Replacement) {
  if (SL.isValid() && Replacement.substr(0, 12) == "std::complex") {
    DpctGlobalInfo::getInstance().insertHeader(SL, Complex);
  }
}

void TypeInDeclRule::processCudaStreamType(const DeclaratorDecl *DD,
                                           const SourceManager *SM,
                                           bool &SpecialCaseHappened) {

  Token Tok;
  Lexer::getRawToken(DD->getBeginLoc(), Tok, *SM, LangOptions());
  auto Tok2Ptr = Lexer::findNextToken(DD->getBeginLoc(), *SM, LangOptions());

  if (Tok2Ptr.hasValue()) {

    auto Tok2 = Tok2Ptr.getValue();

    SourceLocation InsertLoc;
    auto PointerType = deducePointerType(DD, "CUstream_st");
    std::string TypeStr = Tok.getRawIdentifier().str();
    if (Tok.getKind() == tok::raw_identifier && TypeStr == "cudaStream_t") {
      SpecialCaseHappened = true;
      SrcAPIStaticsMap[TypeStr]++;
      // cudaStream_t const
      if (Tok2.getKind() == tok::raw_identifier &&
          Tok2.getRawIdentifier() == "const") {
        emplaceTransformation(new ReplaceToken(Tok.getLocation(), ""));
        emplaceTransformation(
            new ReplaceToken(Tok2.getLocation(), "sycl::queue"));
        InsertLoc = Tok2.getEndLoc().getLocWithOffset(1);
        emplaceTransformation(
            new InsertText(InsertLoc, std::move(PointerType)));
      }
      // cudaStream_t
      else {
        emplaceTransformation(
            new ReplaceToken(Tok.getLocation(), "sycl::queue"));
        InsertLoc = Tok.getEndLoc().getLocWithOffset(1);
        emplaceTransformation(
            new InsertText(InsertLoc, std::move(PointerType)));
      }
    } else if (Tok.getKind() == tok::raw_identifier &&
               Tok.getRawIdentifier() == "const") {

      // const cudaStream_t
      TypeStr = Tok2.getRawIdentifier().str();
      if (Tok.getKind() == tok::raw_identifier && TypeStr == "cudaStream_t") {
        SpecialCaseHappened = true;
        SrcAPIStaticsMap[TypeStr]++;
        emplaceTransformation(new ReplaceToken(Tok.getLocation(), ""));
        emplaceTransformation(
            new ReplaceToken(Tok2.getLocation(), "sycl::queue"));
        InsertLoc = Tok2.getEndLoc().getLocWithOffset(1);
        emplaceTransformation(
            new InsertText(InsertLoc, std::move(PointerType)));
      }
    }
  }
  auto SD = getSiblingDecls(DD);
  for (auto It = SD.begin(); It != SD.end(); ++It) {
    auto DD2 = *It;
    auto L2 = DD2->getLocation();
    auto P = SM->getCharacterData(L2);
    // Find the first non-space char after previous semicolon
    while (*P != ',')
      --P;
    ++P;
    while (isspace(*P))
      ++P;
    // Insert "*" or "*const" right before it
    auto InsertLoc = L2.getLocWithOffset(P - SM->getCharacterData(L2));
    auto PointerType = deducePointerType(DD2, "CUstream_st");
    emplaceTransformation(new InsertText(InsertLoc, std::move(PointerType)));
  }
}

void TypeInDeclRule::reportForNcclAndCudnn(const TypeLoc *TL,
                                           const SourceLocation BeginLoc) {
  auto QT = TL->getType();
  std::string TypeStrRemovePrefix = TL->getType().getAsString();

  auto IsNCCLMatched = TypeStrRemovePrefix.find("nccl") != std::string::npos;
  auto IsCUDNNMatched = TypeStrRemovePrefix.find("cudnn") != std::string::npos;
  if (IsNCCLMatched || IsCUDNNMatched) {
    auto TP = QT.getTypePtr();
    if (TP) {
      SourceLocation SL;
      if (getTypeDeclLocation(TP, SL)) {
        std::string FilePath =
            DpctGlobalInfo::getSourceManager().getFilename(SL).str();
        if (!DpctGlobalInfo::isInRoot(FilePath)) {
          if (IsNCCLMatched)
            report(BeginLoc, Diagnostics::MANUAL_MIGRATION_LIBRARY, false,
                   "Intel(R) oneAPI Collective Communications Library");
          else if (IsCUDNNMatched)
            report(BeginLoc, Diagnostics::MANUAL_MIGRATION_LIBRARY, false,
                   "Intel(R) oneAPI Deep Neural Network Library (oneDNN)");
        }
      }
    }
  }
}

bool TypeInDeclRule::replaceTemplateSpecialization(
    SourceManager *SM, LangOptions &LOpts, SourceLocation BeginLoc,
    const TemplateSpecializationTypeLoc TSL) {

  for (unsigned i = 0; i < TSL.getNumArgs(); ++i) {
    auto UTL =
        TSL.getArgLoc(i).getTypeSourceInfo()->getTypeLoc().getUnqualifiedLoc();

    if (UTL.getTypeLocClass() == clang::TypeLoc::Elaborated) {
      auto ETC = UTL.getAs<ElaboratedTypeLoc>();

      auto ETBeginLoc = ETC.getQualifierLoc().getBeginLoc();
      auto ETEndLoc = ETC.getQualifierLoc().getEndLoc();

      if(ETBeginLoc.isInvalid() || ETEndLoc.isInvalid())
        continue;

      const char *Start = SM->getCharacterData(ETBeginLoc);
      const char *End = SM->getCharacterData(ETEndLoc);
      auto TyLen = End - Start;
      assert(TyLen > 0);

      std::string RealTypeNameStr(Start, TyLen);

      auto Pos = RealTypeNameStr.find('<');
      if (Pos != std::string::npos) {
        RealTypeNameStr = RealTypeNameStr.substr(0, Pos);
        TyLen = Pos;
      }

      std::string Replacement =
          MapNames::findReplacedName(MapNames::TypeNamesMap, RealTypeNameStr);

      if (!Replacement.empty()) {
        SrcAPIStaticsMap[RealTypeNameStr]++;
        emplaceTransformation(
            new ReplaceText(ETBeginLoc, TyLen, std::move(Replacement)));
      }
    }
  }

  Token Tok;
  Lexer::getRawToken(BeginLoc, Tok, *SM, LOpts, true);
  if (!Tok.isAnyIdentifier()) {
    return false;
  }
  auto TypeNameStr = Tok.getRawIdentifier().str();
  // skip to the next identifier after keyword "typename"
  if (TypeNameStr == "typename") {
    Tok = Lexer::findNextToken(BeginLoc, *SM, LOpts).getValue();
    BeginLoc = Tok.getLocation();
  }
  auto LAngleLoc = TSL.getLAngleLoc();
  BeginLoc = SM->getExpansionLoc(BeginLoc);

  const char *Start = SM->getCharacterData(BeginLoc);
  const char *End = SM->getCharacterData(LAngleLoc);
  auto TyLen = End - Start;
  assert(TyLen > 0);
  const std::string RealTypeNameStr(Start, TyLen);
  std::string Replacement =
      MapNames::findReplacedName(MapNames::TypeNamesMap, RealTypeNameStr);
  if (!Replacement.empty()) {
    insertComplexHeader(BeginLoc, Replacement);
    emplaceTransformation(
        new ReplaceText(BeginLoc, TyLen, std::move(Replacement)));
    return true;
  }
  return false;
}

// There's no AST matcher for dealing with DependentNameTypeLocs so
// it is handled 'manually'
bool TypeInDeclRule::replaceDependentNameTypeLoc(SourceManager *SM,
                                                 LangOptions &LOpts,
                                                 const TypeLoc *TL) {
  auto D = DpctGlobalInfo::findAncestor<Decl>(TL);
  TypeSourceInfo *TSI = nullptr;
  if (auto TD = dyn_cast<TypedefDecl>(D))
    TSI = TD->getTypeSourceInfo();
  else if (auto VD = dyn_cast<VarDecl>(D))
    TSI = VD->getTypeSourceInfo();
  else if (auto FD = dyn_cast<FieldDecl>(D))
    TSI = FD->getTypeSourceInfo();
  else
    return false;

  auto TTL = TSI->getTypeLoc();
  auto SR = TTL.getSourceRange();
  auto DTL = TTL.getAs<DependentNameTypeLoc>();
  if (!DTL)
    return false;

  auto NNSL = DTL.getQualifierLoc();
  auto NNTL = NNSL.getTypeLoc();

  auto BeginLoc = SR.getBegin();
  if (NNTL.getTypeLocClass() == clang::TypeLoc::TemplateSpecialization) {
    auto TSL = NNTL.getUnqualifiedLoc().getAs<TemplateSpecializationTypeLoc>();
    if (replaceTemplateSpecialization(SM, LOpts, BeginLoc, TSL)) {
      // Check if "::type" needs replacement (only needed for
      // thrust::iterator_difference)
      Token Tok;
      Lexer::getRawToken(SR.getEnd(), Tok, *SM, LOpts, true);
      auto TypeNameStr =
          Tok.isAnyIdentifier() ? Tok.getRawIdentifier().str() : "";
      Lexer::getRawToken(TSL.getBeginLoc(), Tok, *SM, LOpts, true);
      auto TemplateNameStr =
          Tok.isAnyIdentifier() ? Tok.getRawIdentifier().str() : "";
      if (TypeNameStr == "type" && TemplateNameStr == "iterator_difference") {
        emplaceTransformation(
            new ReplaceText(SR.getEnd(), 4, std::string("difference_type")));
      }
      return true;
    }
  }
  return false;
}

bool TypeInDeclRule::isDeviceRandomStateType(const TypeLoc *TL,
                                             const SourceLocation &SL) {
  std::string TypeStr = TL->getType().getAsString();

  if (MapNames::DeviceRandomGeneratorTypeMap.find(TypeStr) !=
      MapNames::DeviceRandomGeneratorTypeMap.end()) {
    if (TypeStr == "curandState_t" || TypeStr == "curandState" ||
        TypeStr == "curandStateXORWOW_t") {
      report(SL, Diagnostics::DIFFERENT_GENERATOR, false);
    }
    return true;
  } else {
    return false;
  }
}

void TypeInDeclRule::run(const MatchFinder::MatchResult &Result) {
  CHECKPOINT_ASTMATCHER_RUN_ENTRY();
  if (auto TL = getNodeAsType<TypeLoc>(Result, "cudaTypeDef")) {
    auto BeginLoc = TL->getBeginLoc();
    SourceManager *SM = Result.SourceManager;

    if (BeginLoc.isMacroID()) {
      auto SpellingLocation = SM->getSpellingLoc(BeginLoc);
      if (DpctGlobalInfo::replaceMacroName(SpellingLocation)) {
        BeginLoc = SM->getExpansionLoc(BeginLoc);
      } else {
        BeginLoc = SpellingLocation;
      }
    }

    if (isDeviceRandomStateType(TL, BeginLoc)) {
      std::string TypeStr = TL->getType().getAsString();
      auto P = MapNames::DeviceRandomGeneratorTypeMap.find(TypeStr);
      DpctGlobalInfo::getInstance().insertDeviceRandomStateTypeInfo(
          BeginLoc,
          Lexer::MeasureTokenLength(BeginLoc, *SM,
                                    DpctGlobalInfo::getContext().getLangOpts()),
          P->second);
      return;
    }

    reportForNcclAndCudnn(TL, BeginLoc);

    auto LOpts = Result.Context->getLangOpts();

    if (replaceDependentNameTypeLoc(SM, LOpts, TL)) {
      return;
    }

    Token Tok;
    Lexer::getRawToken(BeginLoc, Tok, *SM, LOpts, true);
    if (Tok.isAnyIdentifier()) {

      std::string TypeStr = Tok.getRawIdentifier().str();
      if (TL->getTypeLocClass() == clang::TypeLoc::Elaborated) {
        auto ETC = TL->getAs<ElaboratedTypeLoc>();
        auto NTL = ETC.getNamedTypeLoc();

        if (NTL.getTypeLocClass() == clang::TypeLoc::TemplateSpecialization) {
          auto TSL =
            NTL.getUnqualifiedLoc().getAs<TemplateSpecializationTypeLoc>();
          if (replaceTemplateSpecialization(SM, LOpts, BeginLoc, TSL)) {
            return;
          }
        } else if (NTL.getTypeLocClass() == clang::TypeLoc::Record) {
          auto TSL = NTL.getUnqualifiedLoc().getAs<RecordTypeLoc>();

          const std::string TyName =
              dpct::DpctGlobalInfo::getTypeName(TSL.getType());
          std::string Replacement =
              MapNames::findReplacedName(MapNames::TypeNamesMap, TyName);

          if (!Replacement.empty()) {
            emplaceTransformation(
                new ReplaceToken(BeginLoc, TSL.getEndLoc(), std::move(Replacement)));
            return;
          }
        }
      } else if (TL->getTypeLocClass() == clang::TypeLoc::TemplateSpecialization) {
        // To process the case like "typename thrust::device_vector<int>::iterator itr;".
        auto ND = DpctGlobalInfo::findAncestor<NamedDecl>(TL);
        if(ND){
          auto TSL = TL->getAs<TemplateSpecializationTypeLoc>();
          if (replaceTemplateSpecialization(SM, LOpts, ND->getBeginLoc(), TSL)) {
            return;
          }
        }
      }

      std::string Str =
          MapNames::findReplacedName(MapNames::TypeNamesMap, TypeStr);
      // Add '#include <complex>' directive to the file only once
      if (TypeStr == "cuComplex" || TypeStr == "cuDoubleComplex") {
        DpctGlobalInfo::getInstance().insertHeader(BeginLoc, Complex);
      }

      if (TypeStr == "identity") {
        emplaceTransformation(new ReplaceToken(
            TL->getBeginLoc().getLocWithOffset(Lexer::MeasureTokenLength(
                TL->getBeginLoc(), dpct::DpctGlobalInfo::getSourceManager(),
                dpct::DpctGlobalInfo::getContext().getLangOpts())),
            TL->getEndLoc(), ""));
      }

      const DeclStmt *DS = DpctGlobalInfo::findAncestor<DeclStmt>(TL);
      if (TypeStr == "cusparseMatDescr_t" && DS) {
        for (auto I : DS->decls()) {
          const VarDecl *VDI = dyn_cast<VarDecl>(I);
          if (VDI && VDI->hasInit()) {
            if (VDI->getInitStyle() == VarDecl::InitializationStyle::CInit) {
              const Expr *IE = VDI->getInit();
              // cusparseMatDescr_t descr = InitExpr ;
              //                         |          |
              //                       Begin       End
              SourceLocation End =
                  IE->getEndLoc().getLocWithOffset(Lexer::MeasureTokenLength(
                      SM->getExpansionLoc(IE->getEndLoc()), *SM,
                      DpctGlobalInfo::getContext().getLangOpts()));

              SourceLocation Begin = IE->getEndLoc().getLocWithOffset(-1);
              auto C = SM->getCharacterData(Begin);
              int Offset = 0;
              while (*C != '=') {
                C--;
                Offset--;
              }
              Begin = Begin.getLocWithOffset(Offset);
              unsigned int Len = SM->getDecomposedLoc(End).second -
                                 SM->getDecomposedLoc(Begin).second;
              emplaceTransformation(new ReplaceText(Begin, Len, ""));
            }
          }
        }
      }

      const DeclaratorDecl *DD = nullptr;
      const VarDecl *VarD = DpctGlobalInfo::findAncestor<VarDecl>(TL);
      const FieldDecl *FieldD = DpctGlobalInfo::findAncestor<FieldDecl>(TL);
      const FunctionDecl *FD = DpctGlobalInfo::findAncestor<FunctionDecl>(TL);
      if (FD &&
          (FD->hasAttr<CUDADeviceAttr>() || FD->hasAttr<CUDAGlobalAttr>())) {
        if (TL->getType().getAsString().find("cublasHandle_t") !=
            std::string::npos)
          report(BeginLoc, Diagnostics::HANDLE_IN_DEVICE, false, TypeStr);
      }

      if (VarD) {
        DD = VarD;
      } else if (FieldD) {
        DD = FieldD;
      }

      bool SpecialCaseHappened = false;
      if (DD) {
        if (TL->getType().getAsString().find("cudaStream_t") !=
            std::string::npos) {
          processCudaStreamType(DD, SM, SpecialCaseHappened);
        }
      }
      if (!Str.empty() && !SpecialCaseHappened) {
        SrcAPIStaticsMap[TypeStr]++;
        emplaceTransformation(new ReplaceToken(BeginLoc, std::move(Str)));
        return;
      }
    }
  }
  if (auto VD = getNodeAsType<VarDecl>(
          Result, "useDefaultVarDeclInTemplateArg")) {
    auto TL = VD->getTypeSourceInfo()->getTypeLoc();

    auto TSTL = TL.getAs<TemplateSpecializationTypeLoc>();
    if (!TSTL)
      return;
    auto TST = dyn_cast<TemplateSpecializationType>(VD->getType().getUnqualifiedType());
    if (!TST)
      return;

    bool HasUnremovedPreviousArg = 0;
    for (unsigned i = 0; i < TST->getNumArgs(); i++) {
      if (!dpct::DpctGlobalInfo::getTypeName(TST->getArg(0).getAsType())
               .compare("thrust::use_default")) {
        auto ArgBeginLoc = TSTL.getArgLoc(i).getSourceRange().getBegin();
        auto ArgEndLoc = TSTL.getArgLoc(i).getSourceRange().getEnd();
        if (HasUnremovedPreviousArg && i < TST->getNumArgs() - 1) {
          ArgEndLoc = TSTL.getArgLoc(i - 1).getSourceRange().getBegin();
        }
        emplaceTransformation(new ReplaceToken(ArgBeginLoc, ArgEndLoc, ""));
      } else {
        HasUnremovedPreviousArg = 1;
      }
    }
  }
}

REGISTER_RULE(TypeInDeclRule)

// Rule for types replacements in var. declarations.
void VectorTypeNamespaceRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(typeLoc(loc(qualType(hasDeclaration(
                            anyOf(namedDecl(vectorTypeName()),
                                  typedefDecl(vectorTypeName()))))))
                    .bind("vectorTypeTL"),
                this);
}

void VectorTypeNamespaceRule::run(const MatchFinder::MatchResult &Result) {
  CHECKPOINT_ASTMATCHER_RUN_ENTRY();
  if (auto TL = getNodeAsType<TypeLoc>(Result, "vectorTypeTL")) {
    auto BeginLoc = TL->getBeginLoc();
    SourceManager *SM = Result.SourceManager;

    if (BeginLoc.isMacroID()) {
      auto SpellingLocation = SM->getSpellingLoc(BeginLoc);
      if (DpctGlobalInfo::replaceMacroName(SpellingLocation)) {
        BeginLoc = SM->getExpansionLoc(BeginLoc);
      } else {
        BeginLoc = SpellingLocation;
      }
    }

    const FieldDecl *FD = DpctGlobalInfo::findAncestor<FieldDecl>(TL);
    if (auto D = dyn_cast_or_null<CXXRecordDecl>(getParentDecl(FD))) {
      // To process cases like "union struct_union {float2 data;};".
      auto Type = FD->getType();
      if (D && D->isUnion() && !Type->isPointerType() && !Type->isArrayType()) {
        // To add a default member initializer list "{}" to the
        // vector variant member of the union, because a union contains a
        // non-static data member with a non-trivial default constructor, the
        // default constructor of the union will be deleted by default.
        auto Loc = FD->getEndLoc().getLocWithOffset(Lexer::MeasureTokenLength(
            FD->getEndLoc(), *SM, Result.Context->getLangOpts()));
        emplaceTransformation(new InsertText(Loc, "{}"));
      }
    }

    Token Tok;
    auto LOpts = Result.Context->getLangOpts();
    Lexer::getRawToken(BeginLoc, Tok, *SM, LOpts, true);
    if (Tok.isAnyIdentifier()) {
      const std::string TypeStr = Tok.getRawIdentifier().str();
      std::string Str =
          MapNames::findReplacedName(MapNames::TypeNamesMap, TypeStr);
      if (!Str.empty()) {
        SrcAPIStaticsMap[TypeStr]++;
        emplaceTransformation(new ReplaceToken(BeginLoc, std::move(Str)));
      }
    }
  }
}

REGISTER_RULE(VectorTypeNamespaceRule)

void VectorTypeMemberAccessRule::registerMatcher(MatchFinder &MF) {
  auto memberAccess = [&]() {
    return hasObjectExpression(hasType(qualType(hasCanonicalType(
        recordType(hasDeclaration(cxxRecordDecl(vectorTypeName())))))));
  };

  // int2.x => int2.x()
  MF.addMatcher(
      memberExpr(allOf(memberAccess(), unless(hasParent(binaryOperator(allOf(
                                           hasLHS(memberExpr(memberAccess())),
                                           isAssignmentOperator()))))))
          .bind("VecMemberExpr"),
      this);

  // int2.x += xxx => int2.x() += xxx
  MF.addMatcher(
      binaryOperator(allOf(hasLHS(memberExpr(memberAccess())
                                      .bind("VecMemberExprAssignmentLHS")),
                           isAssignmentOperator()))
          .bind("VecMemberExprAssignment"),
      this);
}

void VectorTypeMemberAccessRule::renameMemberField(const MemberExpr *ME) {
  auto BaseTy = ME->getBase()->getType().getAsString();
  auto &SM = DpctGlobalInfo::getSourceManager();
  if (*(BaseTy.end() - 1) == '1') {
    auto Begin = ME->getOperatorLoc();
    auto End = Lexer::getLocForEndOfToken(
        ME->getMemberLoc(), 0, SM, DpctGlobalInfo::getContext().getLangOpts());
    auto Length = SM.getFileOffset(End) - SM.getFileOffset(Begin);
    return emplaceTransformation(new ReplaceText(Begin, Length, ""));
  }
  std::string MemberName = ME->getMemberNameInfo().getAsString();
  if (MapNames::replaceName(MapNames::MemberNamesMap, MemberName))
    emplaceTransformation(
        new RenameFieldInMemberExpr(ME, std::move(MemberName)));
}

void VectorTypeMemberAccessRule::run(const MatchFinder::MatchResult &Result) {
  CHECKPOINT_ASTMATCHER_RUN_ENTRY();
  if (const MemberExpr *ME =
          getNodeAsType<MemberExpr>(Result, "VecMemberExpr")) {
    auto Parents = Result.Context->getParents(*ME);
    if (Parents.size() == 0) {
      return;
    }
    renameMemberField(ME);
  }

  if (auto ME = getNodeAsType<MemberExpr>(Result, "VecMemberExprAssignmentLHS"))
    renameMemberField(ME);
}

REGISTER_RULE(VectorTypeMemberAccessRule)

namespace clang {
namespace ast_matchers {

AST_MATCHER(FunctionDecl, overloadedVectorOperator) {
  if (!DpctGlobalInfo::isInRoot(Node.getBeginLoc()))
    return false;

  switch (Node.getOverloadedOperator()) {
  default: { return false; }
#define OVERLOADED_OPERATOR_MULTI(...)
#define OVERLOADED_OPERATOR(Name, ...)                                         \
  case OO_##Name: {                                                            \
    break;                                                                     \
  }
#include "clang/Basic/OperatorKinds.def"
#undef OVERLOADED_OPERATOR
#undef OVERLOADED_OPERATOR_MULTI
  }

  // Check parameter is vector type
  auto SupportedParamType = [&](const ParmVarDecl *PD) {
    if(!PD)
        return false;
    const IdentifierInfo *IDInfo =
        PD->getOriginalType().getBaseTypeIdentifier();
    if (!IDInfo)
      return false;

    const std::string TypeName = IDInfo->getName().str();
    return (MapNames::SupportedVectorTypes.find(TypeName) !=
            MapNames::SupportedVectorTypes.end());
  };

  // As long as one parameter is vector type
  for (unsigned i = 0, End = Node.getNumParams(); i != End; ++i) {
    if (SupportedParamType(Node.getParamDecl(i))) {
      return true;
    }
  }

  return false;
}

} // namespace ast_matchers
} // namespace clang

void VectorTypeOperatorRule::registerMatcher(MatchFinder &MF) {
  auto vectorTypeOverLoadedOperator = [&]() {
    return functionDecl(overloadedVectorOperator());
  };

  // Matches user overloaded operator declaration
  MF.addMatcher(vectorTypeOverLoadedOperator().bind("overloadedOperatorDecl"),
                this);

  // Matches call of user overloaded operator
  MF.addMatcher(cxxOperatorCallExpr(callee(vectorTypeOverLoadedOperator()))
                    .bind("callOverloadedOperator"),
                this);
}

const char VectorTypeOperatorRule::NamespaceName[] =
    "dpct_operator_overloading";

void VectorTypeOperatorRule::MigrateOverloadedOperatorDecl(
    const MatchFinder::MatchResult &Result, const FunctionDecl *FD) {
  if (!FD)
    return;

  // Helper function to get the scope of function declartion
  // Eg:
  //
  //    void test();
  //   ^            ^
  //   |            |
  // Begin         End
  //
  //    void test() {}
  //   ^              ^
  //   |              |
  // Begin           End
  auto GetFunctionSourceRange = [&](const SourceManager &SM,
                                    const SourceLocation &StartLoc,
                                    const SourceLocation &EndLoc) {
    const std::pair<FileID, unsigned> StartLocInfo =
        SM.getDecomposedExpansionLoc(StartLoc);
    llvm::StringRef Buffer(SM.getCharacterData(EndLoc));
    const std::string Str = std::string(";") + getNL();
    size_t Offset = Buffer.find_first_of(Str);
    assert(Offset != llvm::StringRef::npos);
    const std::pair<FileID, unsigned> EndLocInfo =
        SM.getDecomposedExpansionLoc(EndLoc.getLocWithOffset(Offset + 1));
    assert(StartLocInfo.first == EndLocInfo.first);

    return SourceRange(
        SM.getComposedLoc(StartLocInfo.first, StartLocInfo.second),
        SM.getComposedLoc(EndLocInfo.first, EndLocInfo.second));
  };

  // Add namespace to user overloaded operator declaration
  // double2& operator+=(double2& lhs, const double2& rhs)
  // =>
  // namespace dpct_operator_overloading {
  //
  // double2& operator+=(double2& lhs, const double2& rhs)
  //
  // }
  const auto &SM = *Result.SourceManager;
  const std::string NL = getNL(FD->getBeginLoc(), SM);

  std::ostringstream Prologue;
  // clang-format off
  Prologue << "namespace " << NamespaceName << " {" << NL
           << NL;
  // clang-format on

  std::ostringstream Epilogue;
  // clang-format off
  Epilogue << NL
           << "}  // namespace " << NamespaceName << NL
           << NL;
  // clang-format on
  SourceRange SR;
  auto P = getParentDecl(FD);
  // Deal with functions as well as function templates
  if (auto FTD = dyn_cast<FunctionTemplateDecl>(P)) {
    SR = GetFunctionSourceRange(SM, FTD->getBeginLoc(), FTD->getEndLoc());
  } else {
    SR = GetFunctionSourceRange(SM, FD->getBeginLoc(), FD->getEndLoc());
  }
  report(SR.getBegin(), Diagnostics::TRNA_WARNING_OVERLOADED_API_FOUND, false);
  emplaceTransformation(new InsertText(SR.getBegin(), Prologue.str()));
  emplaceTransformation(new InsertText(SR.getEnd(), Epilogue.str()));
}

void VectorTypeOperatorRule::MigrateOverloadedOperatorCall(
    const MatchFinder::MatchResult &Result, const CXXOperatorCallExpr *CE) {
  if (!CE)
    return;

  // Explicitly call user overloaded operator
  //
  // For non-assignment operator:
  // a == b
  // =>
  // dpct_operator_overloading::operator==(a, b)
  //
  // For assignment operator:
  // a += b
  // =>
  // a = dpct_operator_overloading::operator+=(a, b)

  const std::string OperatorName =
      BinaryOperator::getOpcodeStr(
          BinaryOperator::getOverloadedOpcode(CE->getOperator()))
          .str();

  std::ostringstream FuncCall;

  if (CE->isAssignmentOp()) {
    const auto &SM = *Result.SourceManager;
    const char *Start = SM.getCharacterData(CE->getBeginLoc());
    const char *End = SM.getCharacterData(CE->getOperatorLoc());
    const std::string LHSText(Start, End - Start);
    FuncCall << LHSText << " = ";
  }

  FuncCall << NamespaceName << "::operator" << OperatorName;

  std::string OperatorReplacement = (CE->getNumArgs() == 1)
                                        ? /* Unary operator */ ""
                                        : /* Binary operator */ ",";
  emplaceTransformation(
      new ReplaceToken(CE->getOperatorLoc(), std::move(OperatorReplacement)));
  insertAroundStmt(CE, FuncCall.str() + "(", ")");
}

void VectorTypeOperatorRule::run(const MatchFinder::MatchResult &Result) {
  CHECKPOINT_ASTMATCHER_RUN_ENTRY();
  // Add namespace to user overloaded operator declaration
  MigrateOverloadedOperatorDecl(
      Result, getNodeAsType<FunctionDecl>(Result, "overloadedOperatorDecl"));

  // Explicitly call user overloaded operator
  MigrateOverloadedOperatorCall(Result, getNodeAsType<CXXOperatorCallExpr>(
                                            Result, "callOverloadedOperator"));
}

REGISTER_RULE(VectorTypeOperatorRule)

void VectorTypeCtorRule::registerMatcher(MatchFinder &MF) {

  // make_int2
  auto makeVectorFunc = [&]() {
    std::vector<std::string> MakeVectorFuncNames;
    for (const std::string &TypeName : MapNames::SupportedVectorTypes) {
      MakeVectorFuncNames.emplace_back("make_" + TypeName);
    }

    return internal::Matcher<NamedDecl>(
        new internal::HasNameMatcher(MakeVectorFuncNames));
  };

  // migrate utility for vector type: eg: make_int2
  MF.addMatcher(
      callExpr(callee(functionDecl(makeVectorFunc()))).bind("VecUtilFunc"),
      this);

}

std::string
VectorTypeCtorRule::getReplaceTypeName(const std::string &TypeName) {
  return std::string(
      MapNames::findReplacedName(MapNames::TypeNamesMap, TypeName));
}

// Determines which case of construction applies and creates replacements for
// the syntax. Returns the constructor node and a boolean indicating if a
// closed brace needs to be appended.
void VectorTypeCtorRule::run(const MatchFinder::MatchResult &Result) {
  CHECKPOINT_ASTMATCHER_RUN_ENTRY();
  if (const CallExpr *CE = getNodeAsType<CallExpr>(Result, "VecUtilFunc")) {
    if (!CE->getDirectCallee())
      return;
    const llvm::StringRef FuncName = CE->getDirectCallee()->getName();
    assert(FuncName.startswith("make_") &&
           "Found non make_<vector type> function");
    emplaceTransformation(new ReplaceStmt(
        CE->getCallee(), getReplaceTypeName(CE->getType().getAsString())));
    return;
  }
}

REGISTER_RULE(VectorTypeCtorRule)

void ReplaceDim3CtorRule::registerMatcher(MatchFinder &MF) {
  // Find dim3 constructors which are part of different casts (representing
  // different syntaxes). This includes copy constructors. All constructors
  // will be visited once.
  MF.addMatcher(cxxConstructExpr(hasType(namedDecl(hasName("dim3"))),
                                 argumentCountIs(1),
                                 unless(hasAncestor(cxxConstructExpr(
                                     hasType(namedDecl(hasName("dim3")))))))
                    .bind("dim3Top"),
                this);

  MF.addMatcher(cxxConstructExpr(
                    hasType(namedDecl(hasName("dim3"))), argumentCountIs(3),
                    anyOf(hasParent(varDecl()), hasParent(exprWithCleanups())),
                    unless(hasAncestor(
                        cxxConstructExpr(hasType(namedDecl(hasName("dim3")))))))
                    .bind("dim3CtorDecl"),
                this);

  MF.addMatcher(
      cxxConstructExpr(hasType(namedDecl(hasName("dim3"))), argumentCountIs(3),
                       // skip fields in a struct.  The source loc is
                       // messed up (points to the start of the struct)
                       unless(hasAncestor(cxxRecordDecl())),
                       unless(hasParent(varDecl())),
                       unless(hasParent(exprWithCleanups())),
                       unless(hasAncestor(cxxConstructExpr(
                           hasType(namedDecl(hasName("dim3")))))))
          .bind("dim3CtorNoDecl"),
      this);

  MF.addMatcher(
      typeLoc(
          loc(qualType(hasDeclaration(anyOf(
              namedDecl(hasAnyName("dim3","cudaExtent", "cudaPos")),
              typedefDecl(hasAnyName("dim3", "cudaExtent", "cudaPos"   )))))))
          .bind("dim3Type"),
      this);
}

ReplaceDim3Ctor *ReplaceDim3CtorRule::getReplaceDim3Modification(
    const MatchFinder::MatchResult &Result) {
  if (auto Ctor = getNodeAsType<CXXConstructExpr>(Result, "dim3CtorDecl")) {
    // dim3 a; or dim3 a(1);
    return new ReplaceDim3Ctor(Ctor, true /*isDecl*/);
  } else if (auto Ctor =
                 getNodeAsType<CXXConstructExpr>(Result, "dim3CtorNoDecl")) {
    // deflt = dim3(3);
    return new ReplaceDim3Ctor(Ctor, false /*isDecl*/);
  } else if (auto Ctor = getNodeAsType<CXXConstructExpr>(Result, "dim3Top")) {
    // dim3 d3_6_3 = dim3(ceil(test.x + NUM), NUM + test.y, NUM + test.z + NUM);
    if (auto A = ReplaceDim3Ctor::getConstructExpr(Ctor->getArg(0))) {
      // strip the top CXXConstructExpr, if there's a CXXConstructExpr further
      // down
      return new ReplaceDim3Ctor(Ctor, A);
    } else {
      // Copy constructor case: dim3 a(copyfrom)
      // No replacements are needed
      return nullptr;
    }
  }

  return nullptr;
}

void ReplaceDim3CtorRule::run(const MatchFinder::MatchResult &Result) {
  CHECKPOINT_ASTMATCHER_RUN_ENTRY();
  ReplaceDim3Ctor *R = getReplaceDim3Modification(Result);
  if (R) {
    // add a transformation that will filter out all nested transformations
    emplaceTransformation(R->getEmpty());
    // all the nested transformations will be applied when R->getReplacement()
    // is called
    emplaceTransformation(R);
  }

  if (auto TL = getNodeAsType<TypeLoc>(Result, "dim3Type")) {
    auto BeginLoc = TL->getBeginLoc();
    SourceManager *SM = Result.SourceManager;

    if (BeginLoc.isMacroID()) {
      auto SpellingLocation = SM->getSpellingLoc(BeginLoc);
      if (DpctGlobalInfo::replaceMacroName(SpellingLocation)) {
        BeginLoc = SM->getExpansionLoc(BeginLoc);
      } else {
        BeginLoc = SpellingLocation;
      }
    }

    Token Tok;
    auto LOpts = Result.Context->getLangOpts();
    Lexer::getRawToken(BeginLoc, Tok, *SM, LOpts, true);
    if (Tok.isAnyIdentifier()) {

      if (TL->getTypeLocClass() == clang::TypeLoc::Elaborated) {
        // To handle case like "struct cudaExtent extent;"
        auto ETC = TL->getUnqualifiedLoc().getAs<ElaboratedTypeLoc>();
        auto NTL = ETC.getNamedTypeLoc();

        if (NTL.getTypeLocClass() == clang::TypeLoc::Record) {
          auto TSL = NTL.getUnqualifiedLoc().getAs<RecordTypeLoc>();

          const std::string TyName =
              dpct::DpctGlobalInfo::getTypeName(TSL.getType());
          std::string Str =
              MapNames::findReplacedName(MapNames::TypeNamesMap, TyName);

          if (!Str.empty()) {
            emplaceTransformation(
                new ReplaceToken(BeginLoc, TSL.getEndLoc(), std::move(Str)));
            return;
          }
        }
      }

      std::string TypeName = Tok.getRawIdentifier().str();
      std::string Str =
          MapNames::findReplacedName(MapNames::TypeNamesMap, TypeName);
      if (auto VD = DpctGlobalInfo::findAncestor<VarDecl>(TL)) {
        auto TypeStr = VD->getType().getAsString();
        if (VD->getKind() == Decl::Var &&
            (TypeStr == "dim3" || TypeStr == "struct cudaExtent" ||
             TypeStr == "struct cudaPos")) {
          std::string Replacement;
          std::string ReplacedType = "range";
          if (TypeStr == "dim3" || TypeStr == "struct cudaExtent") {
            ReplacedType = "range";
          } else if (TypeStr == "struct cudaPos") {
            ReplacedType = "id";
          }

          llvm::raw_string_ostream OS(Replacement);
          DpctGlobalInfo::printCtadClass(
              OS, buildString(MapNames::getClNamespace(), "::", ReplacedType),
              3);
          Str = OS.str();
        }
      }

      if (!Str.empty()) {
        SrcAPIStaticsMap[TypeName]++;
        emplaceTransformation(new ReplaceToken(BeginLoc, std::move(Str)));
        return;
      }
    }
  }
}


REGISTER_RULE(ReplaceDim3CtorRule)

void Dim3MemberFieldsRule::FieldsRename(const MatchFinder::MatchResult &Result,
                                        std::string Str, const MemberExpr *ME) {
  auto SM = Result.SourceManager;

  SourceLocation MemberLoc, OptLoc;
  MemberLoc = SM->getSpellingLoc(ME->getMemberLoc());
  OptLoc = SM->getSpellingLoc(ME->getOperatorLoc());
  bool isArrow = ME->isArrow();
  if(isArrow)
    emplaceTransformation(new ReplaceText(OptLoc, 2, ""));
  else
    emplaceTransformation(new ReplaceText(OptLoc, 1, ""));

  auto Search = MapNames::Dim3MemberNamesMap.find(
      ME->getMemberNameInfo().getAsString());
  if (Search != MapNames::Dim3MemberNamesMap.end()) {
    std::string NewString = Search->second;
    emplaceTransformation(new ReplaceText(MemberLoc, 1, std::move(NewString)));
  }
}

// rule for dim3 types member fields replacements.
void Dim3MemberFieldsRule::registerMatcher(MatchFinder &MF) {
  // dim3->x/y/z => dim3->operator[](0)/(1)/(2)
  MF.addMatcher(
      memberExpr(
          has(implicitCastExpr(hasType(pointsTo(typedefDecl(hasName("dim3")))))
                  .bind("ImplCast")))
          .bind("Dim3MemberPointerExpr"),
      this);

  // dim3.x/y/z => dim3[0]/[1]/[2]
  MF.addMatcher(
      memberExpr(
          hasObjectExpression(hasType(qualType(hasCanonicalType(
              recordType(hasDeclaration(cxxRecordDecl(hasName("dim3")))))))))
          .bind("Dim3MemberDotExpr"),
      this);
}

void Dim3MemberFieldsRule::run(const MatchFinder::MatchResult &Result) {
  CHECKPOINT_ASTMATCHER_RUN_ENTRY();
  if (const MemberExpr *ME =
          getNodeAsType<MemberExpr>(Result, "Dim3MemberPointerExpr")) {
    // E.g.
    // dim3 *pd3;
    // pd3->x;
    // will migrate to:
    // sycl::range<3> *pd3;
    // (*pd3)[0];
    auto Impl = getAssistNodeAsType<ImplicitCastExpr>(Result, "ImplCast");
    insertAroundStmt(Impl, "(*", ")");
    FieldsRename(Result, "->", ME);
  }

  if (const MemberExpr *ME =
          getNodeAsType<MemberExpr>(Result, "Dim3MemberDotExpr")) {
    FieldsRename(Result, ".", ME);
  }
}

REGISTER_RULE(Dim3MemberFieldsRule)

void DevicePropVarRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(
      memberExpr(
          hasObjectExpression(hasType(qualType(hasCanonicalType(recordType(
              hasDeclaration(cxxRecordDecl(hasName("cudaDeviceProp")))))))))
          .bind("DevicePropVar"),
      this);
}

void DevicePropVarRule::run(const MatchFinder::MatchResult &Result) {
  CHECKPOINT_ASTMATCHER_RUN_ENTRY();
  const MemberExpr *ME = getNodeAsType<MemberExpr>(Result, "DevicePropVar");
  if (!ME)
    return;
  auto Parents = Result.Context->getParents(*ME);
  if (Parents.size() < 1)
    return;
  auto MemberName = ME->getMemberNameInfo().getAsString();
  if (MemberName == "sharedMemPerBlock") {
    report(ME->getBeginLoc(), Diagnostics::LOCAL_MEM_SIZE, false);
  } else if (MemberName == "maxGridSize") {
    report(ME->getBeginLoc(), Diagnostics::MAX_GRID_SIZE, false);
  } else if (MemberName == "deviceOverlap") {
    emplaceTransformation(
        new ReplaceToken(ME->getBeginLoc(), ME->getEndLoc(), "true"));
    return;
  }

  auto Search = PropNamesMap.find(MemberName);
  if (Search == PropNamesMap.end()) {
    // TODO report migration error
    return;
  }
  if (Parents[0].get<clang::ImplicitCastExpr>()) {
    // migrate to get_XXX() eg. "b=a.minor" to "b=a.get_minor_version()"
    emplaceTransformation(
        new RenameFieldInMemberExpr(ME, "get_" + Search->second + "()"));
  } else if (auto *BO = Parents[0].get<clang::BinaryOperator>()) {
    // migrate to set_XXX() eg. "a.minor = 1" to "a.set_minor_version(1)"
    if (BO->getOpcode() == clang::BO_Assign) {
      emplaceTransformation(
          new RenameFieldInMemberExpr(ME, "set_" + Search->second));
      emplaceTransformation(new ReplaceText(BO->getOperatorLoc(), 1, "("));
      emplaceTransformation(new InsertAfterStmt(BO, ")"));
    }
  }
  if ((Search->second.compare(0, 13, "major_version") == 0) ||
      (Search->second.compare(0, 13, "minor_version") == 0)) {
    report(ME->getBeginLoc(), Comments::VERSION_COMMENT, false);
  }
  if (Search->second.compare(0, 10, "integrated") == 0) {
    report(ME->getBeginLoc(), Comments::NOT_SUPPORT_API_INTEGRATEDORNOT, false);
  }
}

REGISTER_RULE(DevicePropVarRule)

// Rule for enums constants.
void EnumConstantRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(declRefExpr(to(enumConstantDecl(hasType(enumDecl(hasAnyName(
                                "cudaComputeMode", "cudaMemcpyKind"))))))
                    .bind("EnumConstant"),
                this);

  MF.addMatcher(parmVarDecl(hasType(namedDecl(hasAnyName("cudaMemcpyKind",
                                                         "cudaComputeMode"))))
                    .bind("parmVarDecl"),
                this);
}

void EnumConstantRule::handleComputeMode(std::string EnumName,
                                         const DeclRefExpr *E) {
  report(E->getBeginLoc(), Diagnostics::COMPUTE_MODE, false);
  auto P = getParentStmt(E);
  if (auto ICE = dyn_cast<ImplicitCastExpr>(P)) {
    P = getParentStmt(ICE);
    if (auto BO = dyn_cast<BinaryOperator>(P)) {
      auto LHS = BO->getLHS()->IgnoreImpCasts();
      auto RHS = BO->getRHS()->IgnoreImpCasts();
      const MemberExpr *ME = nullptr;
      if (auto MEL = dyn_cast<MemberExpr>(LHS))
        ME = MEL;
      else if (auto MER = dyn_cast<MemberExpr>(RHS))
        ME = MER;
      if (ME) {
        auto MD = ME->getMemberDecl();
        auto BaseTy = DpctGlobalInfo::getUnqualifiedTypeName(
            ME->getBase()->getType().getUnqualifiedType(),
            DpctGlobalInfo::getContext());
        if (MD->getNameAsString() == "computeMode" &&
            BaseTy == "cudaDeviceProp") {
          if (EnumName == "cudaComputeModeDefault") {
            if (BO->getOpcodeStr() == "==")
              emplaceTransformation(new ReplaceStmt(P, "true"));
            else if (BO->getOpcodeStr() == "!=")
              emplaceTransformation(new ReplaceStmt(P, "false"));
          } else {
            if (BO->getOpcodeStr() == "==")
              emplaceTransformation(new ReplaceStmt(P, "false"));
            else if (BO->getOpcodeStr() == "!=")
              emplaceTransformation(new ReplaceStmt(P, "true"));
          }
          return;
        }
      }
    }
  }
  // default => 1
  // others  => 0
  if (EnumName == "cudaComputeModeDefault") {
    emplaceTransformation(new ReplaceStmt(E, "1"));
    return;
  } else if (EnumName == "cudaComputeModeExclusive" ||
             EnumName == "cudaComputeModeProhibited" ||
             EnumName == "cudaComputeModeExclusiveProcess") {
    emplaceTransformation(new ReplaceStmt(E, "0"));
    return;
  }
}

void EnumConstantRule::run(const MatchFinder::MatchResult &Result) {
  CHECKPOINT_ASTMATCHER_RUN_ENTRY();

  if (const auto PVD = getNodeAsType<ParmVarDecl>(Result, "parmVarDecl")) {

    SourceManager *SM = Result.SourceManager;
    auto LOpts = Result.Context->getLangOpts();
    auto BeginLoc = PVD->getBeginLoc();
    std::string TypeName = PVD->getType().getAsString();

    Token Tok;
    Lexer::getRawToken(BeginLoc, Tok, *SM, LOpts, true);
    if (!Tok.isAnyIdentifier()) {
      return;
    }

    const IdentifierInfo *IdInfo =
        PVD->getOriginalType().getBaseTypeIdentifier();

    if (!IdInfo)
      return;

    std::string BaseTypeName = IdInfo->getName().str();

    auto TypeNameStr = Tok.getRawIdentifier().str();
    int Length = BaseTypeName.length();

    if (TypeNameStr == "enum") {
      const char *startBuf = SM->getCharacterData(BeginLoc);
      auto TypeSpecEnd = PVD->getTypeSpecEndLoc();
      const char *EndBuf = SM->getCharacterData(TypeSpecEnd);
      Length += (EndBuf - startBuf);
    }

    std::string Replacement =
        MapNames::findReplacedName(MapNames::TypeNamesMap, BaseTypeName);

    if (!Replacement.empty()) {
      emplaceTransformation(
          new ReplaceText(BeginLoc, Length, std::move(Replacement)));
    }
  }

  const DeclRefExpr *E = getNodeAsType<DeclRefExpr>(Result, "EnumConstant");
  if (!E)
    return;
  std::string EnumName = E->getNameInfo().getName().getAsString();
  if (EnumName == "cudaComputeModeDefault" ||
      EnumName == "cudaComputeModeExclusive" ||
      EnumName == "cudaComputeModeProhibited" ||
      EnumName == "cudaComputeModeExclusiveProcess") {
    handleComputeMode(EnumName, E);
    return;
  }
  auto Search = EnumNamesMap.find(EnumName);
  if (Search == EnumNamesMap.end()) {
    // TODO report migration error
    return;
  }

  emplaceTransformation(new ReplaceStmt(E, "dpct::" + Search->second));
}

REGISTER_RULE(EnumConstantRule)

void ErrorConstantsRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(declRefExpr(to(enumConstantDecl(hasType(enumDecl(anyOf(
                                hasName("cudaError"), hasName("cufftResult_t"),
                                hasName("cudaError_enum")))))))
                    .bind("ErrorConstants"),
                this);
}

void ErrorConstantsRule::run(const MatchFinder::MatchResult &Result) {
  CHECKPOINT_ASTMATCHER_RUN_ENTRY();
  const DeclRefExpr *DE = getNodeAsType<DeclRefExpr>(Result, "ErrorConstants");
  if (!DE)
    return;
  auto *EC = cast<EnumConstantDecl>(DE->getDecl());
  std::string Repl = EC->getInitVal().toString(10);

  // If the cudaErrorNotReady is one operand of binary operator "==" or "!=",
  // and the other operand is the function call "cudaEventQuery", and the whole
  // binary is in the condition of if/while/for/do/switch,
  // then cudaErrorNotReady will be migrated to "0" while "==" will be migrated
  // to "!=".
  if (EC->getDeclName().getAsString() == "cudaErrorNotReady" &&
      isConditionOfFlowControl(DE, true)) {
    auto &Context = dpct::DpctGlobalInfo::getContext();
    auto ParentNodes = Context.getParents(*DE);
    ast_type_traits::DynTypedNode ParentNode;
    bool MatchFunction = false;
    SourceLocation OperatorLoc;
    const BinaryOperator *BO = nullptr;
    while (!ParentNodes.empty()) {
      ParentNode = ParentNodes[0];
      BO = ParentNode.get<BinaryOperator>();
      if (BO && (BO->getOpcode() == BinaryOperatorKind::BO_EQ ||
                 BO->getOpcode() == BinaryOperatorKind::BO_NE)) {
        auto LHSCall = dyn_cast<CallExpr>(BO->getLHS()->IgnoreImpCasts());
        auto RHSCall = dyn_cast<CallExpr>(BO->getRHS()->IgnoreImpCasts());

        if ((LHSCall && LHSCall->getDirectCallee() &&
             LHSCall->getDirectCallee()
                     ->getNameInfo()
                     .getName()
                     .getAsString() == "cudaEventQuery") ||
            (RHSCall && RHSCall->getDirectCallee() &&
             RHSCall->getDirectCallee()
                     ->getNameInfo()
                     .getName()
                     .getAsString() == "cudaEventQuery")) {
          MatchFunction = true;
          break;
        }
      }
      ParentNodes = Context.getParents(ParentNode);
    }

    if (MatchFunction) {
      Repl = "0";
      std::string OperatorRepl;
      if (BO->getOpcode() == BinaryOperatorKind::BO_EQ)
        OperatorRepl = "!=";
      else
        OperatorRepl = "==";
      emplaceTransformation(
          new ReplaceToken(DpctGlobalInfo::getSourceManager().getSpellingLoc(
                               BO->getOperatorLoc()),
                           std::move(OperatorRepl)));
    }
  }

  emplaceTransformation(new ReplaceStmt(DE, Repl));
}

REGISTER_RULE(ErrorConstantsRule)

void ManualMigrateEnumsRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(declRefExpr(to(enumConstantDecl(matchesName("NCCL_.*"))))
                    .bind("NCCLConstants"),
                this);
  MF.addMatcher(declRefExpr(to(enumConstantDecl(matchesName("CUDNN_.*"))))
                    .bind("CUDNNConstants"),
                this);
}

void ManualMigrateEnumsRule::run(const MatchFinder::MatchResult &Result) {
  CHECKPOINT_ASTMATCHER_RUN_ENTRY();
  if (const DeclRefExpr *DE =
          getNodeAsType<DeclRefExpr>(Result, "NCCLConstants")) {
    auto *ECD = cast<EnumConstantDecl>(DE->getDecl());
    std::string FilePath = DpctGlobalInfo::getSourceManager()
                               .getFilename(ECD->getBeginLoc())
                               .str();
    if (DpctGlobalInfo::isInRoot(FilePath)) {
      return;
    }
    report(dpct::DpctGlobalInfo::getSourceManager().getExpansionLoc(
               DE->getBeginLoc()),
           Diagnostics::MANUAL_MIGRATION_LIBRARY, false,
           "Intel(R) oneAPI Collective Communications Library");
  } else if (const DeclRefExpr *DE =
                 getNodeAsType<DeclRefExpr>(Result, "CUDNNConstants")) {
    auto *ECD = cast<EnumConstantDecl>(DE->getDecl());
    std::string FilePath = DpctGlobalInfo::getSourceManager()
                               .getFilename(ECD->getBeginLoc())
                               .str();
    if (DpctGlobalInfo::isInRoot(FilePath)) {
      return;
    }
    report(dpct::DpctGlobalInfo::getSourceManager().getExpansionLoc(
               DE->getBeginLoc()),
           Diagnostics::MANUAL_MIGRATION_LIBRARY, false,
           "Intel(R) oneAPI Deep Neural Network Library (oneDNN)");
  }
}

REGISTER_RULE(ManualMigrateEnumsRule)

// Rule for BLAS enums.
// Migrate BLAS status values to corresponding int values
// Other BLAS named values are migrated to corresponding named values
void BLASEnumsRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(declRefExpr(to(enumConstantDecl(matchesName(
                                "(CUBLAS_STATUS.*)|(CUDA_R_.*)|(CUDA_C_.*)|("
                                "CUBLAS_GEMM_.*)|(CUBLAS_POINTER_MODE.*)"))))
                    .bind("BLASStatusConstants"),
                this);
  MF.addMatcher(declRefExpr(to(enumConstantDecl(matchesName(
                                "(CUBLAS_OP.*)|(CUBLAS_SIDE.*)|(CUBLAS_FILL_"
                                "MODE.*)|(CUBLAS_DIAG.*)"))))
                    .bind("BLASNamedValueConstants"),
                this);
}

void BLASEnumsRule::run(const MatchFinder::MatchResult &Result) {
  CHECKPOINT_ASTMATCHER_RUN_ENTRY();
  if (const DeclRefExpr *DE =
          getNodeAsType<DeclRefExpr>(Result, "BLASStatusConstants")) {
    auto *EC = cast<EnumConstantDecl>(DE->getDecl());
    emplaceTransformation(new ReplaceStmt(DE, EC->getInitVal().toString(10)));
  }

  if (const DeclRefExpr *DE =
          getNodeAsType<DeclRefExpr>(Result, "BLASNamedValueConstants")) {
    auto *EC = cast<EnumConstantDecl>(DE->getDecl());
    std::string Name = EC->getNameAsString();
    auto Search = MapNames::BLASEnumsMap.find(Name);
    if (Search == MapNames::BLASEnumsMap.end()) {
      llvm::dbgs() << "[" << getName()
                   << "] Unexpected enum variable: " << Name;
      return;
    }
    std::string Replacement = Search->second;
    emplaceTransformation(new ReplaceStmt(DE, std::move(Replacement)));
  }
}

REGISTER_RULE(BLASEnumsRule)

// Rule for RANDOM enums.
// Migrate RANDOM status values to corresponding int values
void RandomEnumsRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(
      declRefExpr(to(enumConstantDecl(matchesName("CURAND_STATUS.*"))))
          .bind("RANDOMStatusConstants"),
      this);
}

void RandomEnumsRule::run(const MatchFinder::MatchResult &Result) {
  CHECKPOINT_ASTMATCHER_RUN_ENTRY();
  if (const DeclRefExpr *DE =
          getNodeAsType<DeclRefExpr>(Result, "RANDOMStatusConstants")) {
    auto *EC = cast<EnumConstantDecl>(DE->getDecl());
    emplaceTransformation(new ReplaceStmt(DE, EC->getInitVal().toString(10)));
  }
}

REGISTER_RULE(RandomEnumsRule)

// Rule for spBLAS enums.
// Migrate spBLAS status values to corresponding int values
// Other spBLAS named values are migrated to corresponding named values
void SPBLASEnumsRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(
      declRefExpr(to(enumConstantDecl(matchesName(
                      "(CUSPARSE_STATUS.*)|("
                      "CUSPARSE_POINTER_MODE.*)|(CUSPARSE_MATRIX_TYPE.*)"))))
          .bind("SPBLASStatusConstants"),
      this);
  MF.addMatcher(
      declRefExpr(to(enumConstantDecl(matchesName(
                      "(CUSPARSE_OPERATION.*)|(CUSPARSE_FILL_MODE.*)|(CUSPARSE_"
                      "DIAG_TYPE.*)|(CUSPARSE_INDEX_BASE.*)"))))
          .bind("SPBLASNamedValueConstants"),
      this);
}

void SPBLASEnumsRule::run(const MatchFinder::MatchResult &Result) {
  CHECKPOINT_ASTMATCHER_RUN_ENTRY();
  if (const DeclRefExpr *DE =
          getNodeAsType<DeclRefExpr>(Result, "SPBLASStatusConstants")) {
    auto *EC = cast<EnumConstantDecl>(DE->getDecl());
    emplaceTransformation(new ReplaceStmt(DE, EC->getInitVal().toString(10)));
  }

  if (const DeclRefExpr *DE =
          getNodeAsType<DeclRefExpr>(Result, "SPBLASNamedValueConstants")) {
    auto *EC = cast<EnumConstantDecl>(DE->getDecl());
    std::string Name = EC->getNameAsString();
    auto Search = MapNames::SPBLASEnumsMap.find(Name);
    if (Search == MapNames::SPBLASEnumsMap.end()) {
      llvm::dbgs() << "[" << getName()
                   << "] Unexpected enum variable: " << Name;
      return;
    }
    std::string Replacement = Search->second;
    emplaceTransformation(new ReplaceStmt(DE, std::move(Replacement)));
  }
}

REGISTER_RULE(SPBLASEnumsRule)

// Rule for spBLAS function calls.
void SPBLASFunctionCallRule::registerMatcher(MatchFinder &MF) {
  auto functionName = [&]() {
    return hasAnyName(
        /*management*/
        "cusparseCreate", "cusparseDestroy", "cusparseSetStream",
        "cusparseGetStream", "cusparseGetPointerMode", "cusparseSetPointerMode",
        /*helper*/
        "cusparseCreateMatDescr", "cusparseDestroyMatDescr",
        "cusparseSetMatType", "cusparseGetMatType", "cusparseSetMatIndexBase",
        "cusparseGetMatIndexBase", "cusparseSetMatDiagType",
        "cusparseGetMatDiagType", "cusparseSetMatFillMode",
        "cusparseGetMatFillMode", "cusparseCreateSolveAnalysisInfo",
        "cusparseDestroySolveAnalysisInfo",
        /*level 2*/
        "cusparseScsrmv", "cusparseDcsrmv", "cusparseCcsrmv", "cusparseZcsrmv",
        "cusparseScsrsv_analysis", "cusparseDcsrsv_analysis",
        "cusparseCcsrsv_analysis", "cusparseZcsrsv_analysis",
        /*level 3*/
        "cusparseScsrmm", "cusparseDcsrmm", "cusparseCcsrmm", "cusparseZcsrmm");
  };
  MF.addMatcher(
      callExpr(allOf(callee(functionDecl(functionName())), parentStmt()))
          .bind("FunctionCall"),
      this);
  MF.addMatcher(callExpr(allOf(callee(functionDecl(functionName())),
                               unless(parentStmt())))
                    .bind("FunctionCallUsed"),
                this);
}

void SPBLASFunctionCallRule::run(const MatchFinder::MatchResult &Result) {
  CHECKPOINT_ASTMATCHER_RUN_ENTRY();
  bool IsAssigned = false;
  const CallExpr *CE = getNodeAsType<CallExpr>(Result, "FunctionCall");
  if (!CE) {
    if (!(CE = getNodeAsType<CallExpr>(Result, "FunctionCallUsed")))
      return;
    IsAssigned = true;
  }

  if (!CE->getDirectCallee())
    return;

  auto &SM = DpctGlobalInfo::getSourceManager();
  auto SL = SM.getExpansionLoc(CE->getBeginLoc());
  std::string Key =
      SM.getFilename(SL).str() + std::to_string(SM.getDecomposedLoc(SL).second);
  DpctGlobalInfo::updateInitSuffixIndexInRule(
      DpctGlobalInfo::getSuffixIndexInitValue(Key));

  std::string FuncName =
      CE->getDirectCallee()->getNameInfo().getName().getAsString();
  StringRef FuncNameRef(FuncName);
  SourceLocation FuncNameBegin(CE->getBeginLoc());
  SourceLocation FuncCallEnd(CE->getEndLoc());
  // TODO: For case like:
  //  #define CHECK_STATUS(x) fun(c)
  //  CHECK_STATUS(anAPICall());
  // Below code can distinguish this kind of function like macro, need refine to
  // cover more cases.
  bool IsMacroArg = SM.isMacroArgExpansion(CE->getBeginLoc()) &&
                    SM.isMacroArgExpansion(CE->getEndLoc());

  // Offset 1 is the length of the last token ")"
  FuncCallEnd = SM.getExpansionLoc(FuncCallEnd).getLocWithOffset(1);
  auto SR = getScopeInsertRange(CE, FuncNameBegin, FuncCallEnd);
  SourceLocation PrefixInsertLoc = SR.getBegin(), SuffixInsertLoc = SR.getEnd();
  bool CanAvoidUsingLambda = false;
  SourceLocation OuterInsertLoc;
  std::string OriginStmtType;
  bool NeedUseLambda = isConditionOfFlowControl(
      CE, OriginStmtType, CanAvoidUsingLambda, OuterInsertLoc);
  bool IsInReturnStmt = isInReturnStmt(CE, OuterInsertLoc);
  bool CanAvoidBrace = false;
  const CompoundStmt *CS = findImmediateBlock(CE);
  if (CS && (CS->size() == 1)) {
    const Stmt *S = *(CS->child_begin());
    if (CE == S || dyn_cast<ReturnStmt>(S))
      CanAvoidBrace = true;
  }

  if (NeedUseLambda || IsMacroArg || IsInReturnStmt) {
    NeedUseLambda = true;
    SourceRange SR = getFunctionRange(CE);
    PrefixInsertLoc = SR.getBegin();
    SuffixInsertLoc = SR.getEnd();
    if (IsInReturnStmt) {
      OriginStmtType = "return";
      CanAvoidUsingLambda = true;
    }
  }

  std::string IndentStr = getIndent(PrefixInsertLoc, SM).str();
  std::string PrefixInsertStr, SuffixInsertStr;

  std::string Msg = "the function call is redundant in DPC++.";
  std::string Repl;
  // This length should be used only when NeedUseLambda is true.
  // If NeedUseLambda is false, Len may longer than the function call length,
  // because in this case, PrefixInsertLoc and SuffixInsertLoc are the begin
  // location of the whole statement and the location after the semi of the
  // statement.
  unsigned int Len = SM.getDecomposedLoc(SuffixInsertLoc).second -
                     SM.getDecomposedLoc(PrefixInsertLoc).second;
  if (FuncName == "cusparseCreate" || FuncName == "cusparseDestroy" ||
      FuncName == "cusparseSetStream" || FuncName == "cusparseGetStream") {
    NeedUseLambda = false;
    if (FuncName == "cusparseCreate") {
      std::string LHS;
      if (isSimpleAddrOf(CE->getArg(0))) {
        LHS = getNameStrRemovedAddrOf(CE->getArg(0));
      } else {
        dpct::ExprAnalysis EA;
        EA.analyze(CE->getArg(0));
        LHS = "*(" + EA.getReplacedString() + ")";
      }
      if (checkWhetherIsDuplicate(CE, false))
        return;
      int Index = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();
      buildTempVariableMap(Index, CE, HelperFuncType::DefaultQueue);
      Repl = LHS + " = &{{NEEDREPLACEQ" + std::to_string(Index) + "}}";
    } else if (FuncName == "cusparseDestroy") {
      dpct::ExprAnalysis EA(CE->getArg(0));
      Repl = EA.getReplacedString() + " = nullptr";
    } else if (FuncName == "cusparseSetStream") {
      dpct::ExprAnalysis EA0(CE->getArg(0));
      dpct::ExprAnalysis EA1(CE->getArg(1));
      Repl = EA0.getReplacedString() + " = " + EA1.getReplacedString();
    } else if (FuncName == "cusparseGetStream") {
      dpct::ExprAnalysis EA0(CE->getArg(0));
      std::string LHS;
      if (isSimpleAddrOf(CE->getArg(1))) {
        LHS = getNameStrRemovedAddrOf(CE->getArg(1));
      } else {
        dpct::ExprAnalysis EA;
        EA.analyze(CE->getArg(1));
        LHS = "*(" + EA.getReplacedString() + ")";
      }
      Repl = LHS + " = " + EA0.getReplacedString();
    }
  } else if (FuncName == "cusparseCreateMatDescr") {
    NeedUseLambda = false;
    std::string LHS;
    if (isSimpleAddrOf(CE->getArg(0))) {
      LHS = getNameStrRemovedAddrOf(CE->getArg(0));
    } else {
      dpct::ExprAnalysis EA;
      EA.analyze(CE->getArg(0));
      LHS = "*(" + EA.getReplacedString() + ")";
    }
    Repl = LHS + " = mkl::index_base::zero";
  } else if (FuncName == "cusparseDestroyMatDescr" ||
             FuncName == "cusparseGetPointerMode" ||
             FuncName == "cusparseSetPointerMode" ||
             FuncName == "cusparseScsrsv_analysis" ||
             FuncName == "cusparseDcsrsv_analysis" ||
             FuncName == "cusparseCcsrsv_analysis" ||
             FuncName == "cusparseZcsrsv_analysis" ||
             FuncName == "cusparseCreateSolveAnalysisInfo" ||
             FuncName == "cusparseDestroySolveAnalysisInfo" ||
             FuncName == "cusparseSetMatType" ||
             FuncName == "cusparseGetMatType" ||
             FuncName == "cusparseSetMatDiagType" ||
             FuncName == "cusparseGetMatDiagType" ||
             FuncName == "cusparseSetMatFillMode" ||
             FuncName == "cusparseGetMatFillMode") {
    if (FuncName == "cusparseSetMatType") {
      Expr::EvalResult ER;
      if (CE->getArg(1)->EvaluateAsInt(ER, *Result.Context)) {
        int64_t Value = ER.Val.getInt().getExtValue();
        if (Value != 0) {
          DpctGlobalInfo::setSpBLASUnsupportedMatrixTypeFlag(true);
        }
      } else {
        DpctGlobalInfo::setSpBLASUnsupportedMatrixTypeFlag(true);
      }
    }

    if (IsAssigned) {
      report(PrefixInsertLoc, Diagnostics::FUNC_CALL_REMOVED_0, false, FuncName,
             Msg);
      if (FuncName == "cusparseGetMatDiagType")
        emplaceTransformation(
            new ReplaceStmt(CE, false, FuncName, false, "(mkl::diag)0"));
      else if (FuncName == "cusparseGetMatFillMode")
        emplaceTransformation(
            new ReplaceStmt(CE, false, FuncName, false, "(mkl::uplo)0"));
      else
        emplaceTransformation(new ReplaceStmt(CE, false, FuncName, false, "0"));
    } else {
      report(PrefixInsertLoc, Diagnostics::FUNC_CALL_REMOVED, false, FuncName,
             Msg);
      emplaceTransformation(new ReplaceStmt(CE, false, FuncName, false, ""));
    }
    return;
  } else if (FuncName == "cusparseSetMatIndexBase" ||
             FuncName == "cusparseGetMatIndexBase") {
    NeedUseLambda = false;
    ExprAnalysis EA0(CE->getArg(0));
    bool IsSet = FuncNameRef.startswith("cusparseSet");
    ExprAnalysis EA1;
    if (IsSet) {
      Repl = EA0.getReplacedString() + " = ";
      EA1.analyze(CE->getArg(1));
      Expr::EvalResult ER;
      if (CE->getArg(1)->EvaluateAsInt(ER, *Result.Context)) {
        int64_t Value = ER.Val.getInt().getExtValue();
        if (Value == 0)
          Repl = Repl + "mkl::index_base::zero";
        else
          Repl = Repl + "mkl::index_base::one";
      } else {
        Repl = Repl + EA1.getReplacedString();
      }
    } else {
      Repl = EA0.getReplacedString();
    }

    // Get API do not return status, so return directly.
    if (IsAssigned && IsSet) {
      insertAroundStmt(CE, "(", ", 0)");
      report(PrefixInsertLoc, Diagnostics::NOERROR_RETURN_COMMA_OP, true);
    }
    emplaceTransformation(new ReplaceStmt(CE, false, FuncName, true, Repl));
    return;
  } else if (FuncName == "cusparseScsrmv" || FuncName == "cusparseDcsrmv" ||
             FuncName == "cusparseCcsrmv" || FuncName == "cusparseZcsrmv") {
    std::string BufferType;
    if (FuncName == "cusparseScsrmv")
      BufferType = "float";
    else if (FuncName == "cusparseDcsrmv")
      BufferType = "double";
    else if (FuncName == "cusparseCcsrmv")
      BufferType = "std::complex<float>";
    else
      BufferType = "std::complex<double>";
    int ArgNum = CE->getNumArgs();
    std::vector<std::string> CallExprArguReplVec;
    for (int i = 0; i < ArgNum; ++i) {
      ExprAnalysis EA;
      EA.analyze(CE->getArg(i));
      CallExprArguReplVec.push_back(EA.getReplacedString());
    }

    std::string CSRValA, CSRRowPtrA, CSRColIndA, X, Y;
    if (DpctGlobalInfo::getUsmLevel() == UsmLevel::none) {
      auto ProcessBuffer = [&](const Expr *E, const std::string TypeStr) {
        std::string Decl;
        std::string BufferName =
            getBufferNameAndDeclStr(E, TypeStr, IndentStr, Decl);
        PrefixInsertStr = PrefixInsertStr + Decl;
        return BufferName;
      };
      CSRValA = ProcessBuffer(CE->getArg(7), BufferType);
      CSRRowPtrA = ProcessBuffer(CE->getArg(8), "int");
      CSRColIndA = ProcessBuffer(CE->getArg(9), "int");
      X = ProcessBuffer(CE->getArg(10), BufferType);
      Y = ProcessBuffer(CE->getArg(12), BufferType);
    } else {
      CSRValA = CallExprArguReplVec[7];
      CSRRowPtrA = CallExprArguReplVec[8];
      CSRColIndA = CallExprArguReplVec[9];
      X = CallExprArguReplVec[10];
      Y = CallExprArguReplVec[12];
    }

    std::string MatrixHandleName =
        "mat_handle_ct" +
        std::to_string(DpctGlobalInfo::getSuffixIndexInRuleThenInc());
    PrefixInsertStr = PrefixInsertStr + "mkl::sparse::matrix_handle_t " +
                      MatrixHandleName + ";" + getNL() + IndentStr;
    PrefixInsertStr = PrefixInsertStr + "mkl::sparse::init_matrix_handle(&" +
                      MatrixHandleName + ");" + getNL() + IndentStr;
    if (DpctGlobalInfo::getUsmLevel() == UsmLevel::none) {
      PrefixInsertStr =
          PrefixInsertStr + "mkl::sparse::set_csr_data(" + MatrixHandleName +
          ", " + CallExprArguReplVec[2] + ", " + CallExprArguReplVec[3] + ", " +
          CallExprArguReplVec[6] + ", " + CSRRowPtrA + ", " + CSRColIndA +
          ", " + CSRValA + ");" + getNL() + IndentStr;
    } else {
      if (FuncName == "cusparseScsrmv" || FuncName == "cusparseDcsrmv")
        PrefixInsertStr = PrefixInsertStr + "mkl::sparse::set_csr_data(" +
                          MatrixHandleName + ", " + CallExprArguReplVec[2] +
                          ", " + CallExprArguReplVec[3] + ", " +
                          CallExprArguReplVec[6] + ", const_cast<int*>(" +
                          CSRRowPtrA + "), const_cast<int*>(" + CSRColIndA +
                          "), const_cast<" + BufferType + "*>(" + CSRValA +
                          "));" + getNL() + IndentStr;
      else
        PrefixInsertStr =
            PrefixInsertStr + "mkl::sparse::set_csr_data(" + MatrixHandleName +
            ", " + CallExprArguReplVec[2] + ", " + CallExprArguReplVec[3] +
            ", " + CallExprArguReplVec[6] + ", const_cast<int*>(" +
            CSRRowPtrA + "), const_cast<int*>(" + CSRColIndA + "), (" +
            BufferType + "*)" + CSRValA + ");" + getNL() + IndentStr;
    }
    SuffixInsertStr = SuffixInsertStr + getNL() + IndentStr +
                      "mkl::sparse::release_matrix_handle(&" +
                      MatrixHandleName + ");";

    std::string TransStr;
    Expr::EvalResult ER;
    if (CE->getArg(1)->EvaluateAsInt(ER, *Result.Context)) {
      int64_t Value = ER.Val.getInt().getExtValue();
      if (Value == 0) {
        TransStr = "mkl::transpose::nontrans";
      } else if (Value == 1) {
        TransStr = "mkl::transpose::trans";
      } else {
        TransStr = "mkl::transpose::conjtrans";
      }
    } else {
      const CStyleCastExpr *CSCE = nullptr;
      if (CSCE = dyn_cast<CStyleCastExpr>(CE->getArg(1))) {
        ExprAnalysis EA(CSCE->getSubExpr());
        TransStr = "dpct::get_transpose(" + EA.getReplacedString() + ")";
      } else {
        TransStr = CallExprArguReplVec[1];
      }
    }

    if (DpctGlobalInfo::getUsmLevel() == UsmLevel::none) {
      Repl = "mkl::sparse::gemv(*" + CallExprArguReplVec[0] + ", " + TransStr +
             ", " + "dpct::get_value(" + CallExprArguReplVec[5] + ", *" +
             CallExprArguReplVec[0] + "), " + MatrixHandleName + ", " + X +
             ", " + "dpct::get_value(" + CallExprArguReplVec[11] + ", *" +
             CallExprArguReplVec[0] + "), " + Y + ")";
    } else {
      if (FuncName == "cusparseScsrmv" || FuncName == "cusparseDcsrmv")
        Repl = "mkl::sparse::gemv(*" + CallExprArguReplVec[0] + ", " +
               TransStr + ", " + "dpct::get_value(" + CallExprArguReplVec[5] +
               ", *" + CallExprArguReplVec[0] + "), " + MatrixHandleName +
               ", const_cast<" + BufferType + "*>(" + X + "), " +
               "dpct::get_value(" + CallExprArguReplVec[11] + ", *" +
               CallExprArguReplVec[0] + "), " + Y + ")";
      else
        Repl = "mkl::sparse::gemv(*" + CallExprArguReplVec[0] + ", " +
               TransStr + ", " + "dpct::get_value(" + CallExprArguReplVec[5] +
               ", *" + CallExprArguReplVec[0] + "), " + MatrixHandleName +
               ", (" + BufferType + "*)" + X + ", " + "dpct::get_value(" +
               CallExprArguReplVec[11] + ", *" + CallExprArguReplVec[0] +
               "), (" + BufferType + "*)" + Y + ")";
    }
  } else if (FuncName == "cusparseScsrmm" || FuncName == "cusparseDcsrmm" ||
             FuncName == "cusparseCcsrmm" || FuncName == "cusparseZcsrmm") {
    std::string BufferType;
    if (FuncName == "cusparseScsrmm")
      BufferType = "float";
    else if (FuncName == "cusparseDcsrmm")
      BufferType = "double";
    else if (FuncName == "cusparseCcsrmm")
      BufferType = "std::complex<float>";
    else
      BufferType = "std::complex<double>";
    int ArgNum = CE->getNumArgs();
    std::vector<std::string> CallExprArguReplVec;
    for (int i = 0; i < ArgNum; ++i) {
      ExprAnalysis EA;
      EA.analyze(CE->getArg(i));
      CallExprArguReplVec.push_back(EA.getReplacedString());
    }

    std::string CSRValA, CSRRowPtrA, CSRColIndA, B, C;
    if (DpctGlobalInfo::getUsmLevel() == UsmLevel::none) {
      auto ProcessBuffer = [&](const Expr *E, const std::string TypeStr) {
        std::string Decl;
        std::string BufferName =
            getBufferNameAndDeclStr(E, TypeStr, IndentStr, Decl);
        PrefixInsertStr = PrefixInsertStr + Decl;
        return BufferName;
      };
      CSRValA = ProcessBuffer(CE->getArg(8), BufferType);
      CSRRowPtrA = ProcessBuffer(CE->getArg(9), "int");
      CSRColIndA = ProcessBuffer(CE->getArg(10), "int");
      B = ProcessBuffer(CE->getArg(11), BufferType);
      C = ProcessBuffer(CE->getArg(14), BufferType);
    } else {
      CSRValA = CallExprArguReplVec[8];
      CSRRowPtrA = CallExprArguReplVec[9];
      CSRColIndA = CallExprArguReplVec[10];
      B = CallExprArguReplVec[11];
      C = CallExprArguReplVec[14];
    }

    std::string MatrixHandleName =
        "mat_handle_ct" +
        std::to_string(DpctGlobalInfo::getSuffixIndexInRuleThenInc());
    PrefixInsertStr = PrefixInsertStr + "mkl::sparse::matrix_handle_t " +
                      MatrixHandleName + ";" + getNL() + IndentStr;
    PrefixInsertStr = PrefixInsertStr + "mkl::sparse::init_matrix_handle(&" +
                      MatrixHandleName + ");" + getNL() + IndentStr;
    if (DpctGlobalInfo::getUsmLevel() == UsmLevel::none) {
      PrefixInsertStr =
          PrefixInsertStr + "mkl::sparse::set_csr_data(" + MatrixHandleName +
          ", " + CallExprArguReplVec[2] + ", " + CallExprArguReplVec[4] + ", " +
          CallExprArguReplVec[7] + ", " + CSRRowPtrA + ", " + CSRColIndA +
          ", " + CSRValA + ");" + getNL() + IndentStr;
    } else {
      if (FuncName == "cusparseScsrmm" || FuncName == "cusparseDcsrmm")
        PrefixInsertStr = PrefixInsertStr + "mkl::sparse::set_csr_data(" +
                          MatrixHandleName + ", " + CallExprArguReplVec[2] +
                          ", " + CallExprArguReplVec[4] + ", " +
                          CallExprArguReplVec[7] + ", const_cast<int*>(" +
                          CSRRowPtrA + "), const_cast<int*>(" + CSRColIndA +
                          "), const_cast<" + BufferType + "*>(" + CSRValA +
                          "));" + getNL() + IndentStr;
      else
        PrefixInsertStr =
            PrefixInsertStr + "mkl::sparse::set_csr_data(" + MatrixHandleName +
            ", " + CallExprArguReplVec[2] + ", " + CallExprArguReplVec[4] +
            ", " + CallExprArguReplVec[7] + ", const_cast<int*>(" +
            CSRRowPtrA + "), const_cast<int*>(" + CSRColIndA + "), (" +
            BufferType + "*)" + CSRValA + ");" + getNL() + IndentStr;
    }
    SuffixInsertStr = SuffixInsertStr + getNL() + IndentStr +
                      "mkl::sparse::release_matrix_handle(&" +
                      MatrixHandleName + ");";

    std::string TransStr;
    Expr::EvalResult ER;
    if (CE->getArg(1)->EvaluateAsInt(ER, *Result.Context)) {
      int64_t Value = ER.Val.getInt().getExtValue();
      if (Value == 0) {
        TransStr = "mkl::transpose::nontrans";
      } else if (Value == 1) {
        TransStr = "mkl::transpose::trans";
      } else {
        TransStr = "mkl::transpose::conjtrans";
      }
    } else {
      const CStyleCastExpr *CSCE = nullptr;
      if (CSCE = dyn_cast<CStyleCastExpr>(CE->getArg(1))) {
        ExprAnalysis EA(CSCE->getSubExpr());
        TransStr = "dpct::get_transpose(" + EA.getReplacedString() + ")";
      } else {
        TransStr = CallExprArguReplVec[1];
      }
    }

    if (DpctGlobalInfo::getUsmLevel() == UsmLevel::none) {
      Repl = "mkl::sparse::gemm(*" + CallExprArguReplVec[0] + ", " + TransStr +
             ", " + "dpct::get_value(" + CallExprArguReplVec[6] + ", *" +
             CallExprArguReplVec[0] + "), " + MatrixHandleName + ", " + B +
             ", " + CallExprArguReplVec[3] + ", " + CallExprArguReplVec[12] +
             ", " + "dpct::get_value(" + CallExprArguReplVec[13] + ", *" +
             CallExprArguReplVec[0] + "), " + C + ", " +
             CallExprArguReplVec[15] + ")";
    } else {
      if (FuncName == "cusparseScsrmm" || FuncName == "cusparseDcsrmm")
        Repl = "mkl::sparse::gemm(*" + CallExprArguReplVec[0] + ", " +
               TransStr + ", " + "dpct::get_value(" + CallExprArguReplVec[6] +
               ", *" + CallExprArguReplVec[0] + "), " + MatrixHandleName +
               ", const_cast<" + BufferType + "*>(" + B + "), " +
               CallExprArguReplVec[3] + ", " + CallExprArguReplVec[12] + ", " +
               "dpct::get_value(" + CallExprArguReplVec[13] + ", *" +
               CallExprArguReplVec[0] + "), " + C + ", " +
               CallExprArguReplVec[15] + ")";
      else
        Repl = "mkl::sparse::gemm(*" + CallExprArguReplVec[0] + ", " +
               TransStr + ", " + "dpct::get_value(" + CallExprArguReplVec[6] +
               ", *" + CallExprArguReplVec[0] + "), " + MatrixHandleName +
               ", (" + BufferType + "*)" + B + ", " + CallExprArguReplVec[3] +
               ", " + CallExprArguReplVec[12] + ", " + "dpct::get_value(" +
               CallExprArguReplVec[13] + ", *" + CallExprArguReplVec[0] +
               "), (" + BufferType + "*)" + C + ", " + CallExprArguReplVec[15] +
               ")";
    }
  }

  if (NeedUseLambda) {
    if (PrefixInsertStr.empty() && SuffixInsertStr.empty()) {
      // If there is one API call in the migrted code, it is unnecessary to
      // use a lambda expression
      NeedUseLambda = false;
    }
  }

  if (FuncNameRef.endswith("csrmv") || FuncNameRef.endswith("csrmm")) {
    if (NeedUseLambda && CanAvoidUsingLambda && !IsMacroArg) {
      DpctGlobalInfo::getInstance().insertSpBLASWarningLocOffset(OuterInsertLoc);
    } else {
      DpctGlobalInfo::getInstance().insertSpBLASWarningLocOffset(PrefixInsertLoc);
    }
  }

  if (NeedUseLambda) {
    if (CanAvoidUsingLambda && !IsMacroArg) {
      std::string InsertStr;
      if (DpctGlobalInfo::getUsmLevel() == UsmLevel::none && !CanAvoidBrace)
        InsertStr = std::string("{") + getNL() + IndentStr + PrefixInsertStr +
                    Repl + ";" + SuffixInsertStr + getNL() + IndentStr + "}" +
                    getNL() + IndentStr;
      else
        InsertStr = PrefixInsertStr + Repl + ";" + SuffixInsertStr + getNL() +
                    IndentStr;
      emplaceTransformation(
          new InsertText(OuterInsertLoc, std::move(InsertStr)));
      report(OuterInsertLoc, Diagnostics::CODE_LOGIC_CHANGED, true,
             OriginStmtType == "if" ? "an " + OriginStmtType
                                    : "a " + OriginStmtType);
      emplaceTransformation(new ReplaceText(FuncNameBegin, Len, "0"));
    } else {
      if (IsAssigned) {
        report(PrefixInsertLoc, Diagnostics::NOERROR_RETURN_LAMBDA, false);
        insertAroundRange(
            PrefixInsertLoc, SuffixInsertLoc,
            std::string("[&](){") + getNL() + IndentStr + PrefixInsertStr,
            std::string(";") + SuffixInsertStr + getNL() + IndentStr +
                "return 0;" + getNL() + IndentStr + std::string("}()"));
      } else {
        insertAroundRange(PrefixInsertLoc, SuffixInsertLoc,
                          std::string("[&](){") + getNL() + IndentStr +
                              PrefixInsertStr,
                          std::string(";") + SuffixInsertStr + getNL() +
                              IndentStr + std::string("}()"));
      }
      emplaceTransformation(new ReplaceText(PrefixInsertLoc, Len,
                                            std::move(Repl), false, FuncName));
    }
  } else {
    if (DpctGlobalInfo::getUsmLevel() == UsmLevel::none && !CanAvoidBrace) {
      if (!PrefixInsertStr.empty()) {
        insertAroundRange(
            PrefixInsertLoc, SuffixInsertLoc,
            std::string("{") + getNL() + IndentStr + PrefixInsertStr,
            SuffixInsertStr + getNL() + IndentStr + std::string("}"));
      }
    } else {
      insertAroundRange(PrefixInsertLoc, SuffixInsertLoc,
                        std::move(PrefixInsertStr), std::move(SuffixInsertStr));
    }
    if (IsAssigned) {
      insertAroundStmt(CE, "(", ", 0)");
      report(PrefixInsertLoc, Diagnostics::NOERROR_RETURN_COMMA_OP, true);
    }

    emplaceTransformation(new ReplaceStmt(CE, false, FuncName, true, Repl));
  }
}

REGISTER_RULE(SPBLASFunctionCallRule)

// Rule for Random function calls. Currently only support host APIs.
void RandomFunctionCallRule::registerMatcher(MatchFinder &MF) {
  auto functionName = [&]() {
    return hasAnyName(
        "curandCreateGenerator", "curandSetPseudoRandomGeneratorSeed",
        "curandSetGeneratorOffset", "curandSetQuasiRandomGeneratorDimensions",
        "curandDestroyGenerator", "curandGenerate", "curandGenerateLongLong",
        "curandGenerateLogNormal", "curandGenerateLogNormalDouble",
        "curandGenerateNormal", "curandGenerateNormalDouble",
        "curandGeneratePoisson", "curandGenerateUniform",
        "curandGenerateUniformDouble");
  };
  MF.addMatcher(
      callExpr(allOf(callee(functionDecl(functionName())), parentStmt()))
          .bind("FunctionCall"),
      this);
  MF.addMatcher(callExpr(allOf(callee(functionDecl(functionName())),
                               unless(parentStmt())))
                    .bind("FunctionCallUsed"),
                this);
}

void RandomFunctionCallRule::run(const MatchFinder::MatchResult &Result) {
  CHECKPOINT_ASTMATCHER_RUN_ENTRY();
  bool IsAssigned = false;
  const CallExpr *CE = getNodeAsType<CallExpr>(Result, "FunctionCall");
  if (!CE) {
    if (!(CE = getNodeAsType<CallExpr>(Result, "FunctionCallUsed")))
      return;
    IsAssigned = true;
  }

  if (!CE->getDirectCallee())
    return;

  auto &SM = DpctGlobalInfo::getSourceManager();
  auto SL = SM.getExpansionLoc(CE->getBeginLoc());
  std::string Key =
      SM.getFilename(SL).str() + std::to_string(SM.getDecomposedLoc(SL).second);
  DpctGlobalInfo::updateInitSuffixIndexInRule(
      DpctGlobalInfo::getSuffixIndexInitValue(Key));


  std::string FuncName =
      CE->getDirectCallee()->getNameInfo().getName().getAsString();
  SourceLocation FuncNameBegin(CE->getBeginLoc());
  SourceLocation FuncCallEnd(CE->getEndLoc());
  // TODO: For case like:
  //  #define CHECK_STATUS(x) fun(c)
  //  CHECK_STATUS(anAPICall());
  // Below code can distinguish this kind of function like macro, need refine to
  // cover more cases.
  bool IsMacroArg = SM.isMacroArgExpansion(CE->getBeginLoc()) &&
                    SM.isMacroArgExpansion(CE->getEndLoc());
  // Offset 1 is the length of the last token ")"
  FuncCallEnd = SM.getExpansionLoc(FuncCallEnd).getLocWithOffset(1);
  auto SR = getScopeInsertRange(CE, FuncNameBegin, FuncCallEnd);
  SourceLocation PrefixInsertLoc = SR.getBegin(), SuffixInsertLoc = SR.getEnd();

  bool CanAvoidUsingLambda = false;
  SourceLocation OuterInsertLoc;
  std::string OriginStmtType;
  bool NeedUseLambda = isConditionOfFlowControl(
      CE, OriginStmtType, CanAvoidUsingLambda, OuterInsertLoc);
  bool IsInReturnStmt = isInReturnStmt(CE, OuterInsertLoc);
  bool CanAvoidBrace = false;
  const CompoundStmt *CS = findImmediateBlock(CE);
  if (CS && (CS->size() == 1)) {
    const Stmt *S = *(CS->child_begin());
    if (CE == S || dyn_cast<ReturnStmt>(S))
      CanAvoidBrace = true;
  }

  if (NeedUseLambda) {
    PrefixInsertLoc = FuncNameBegin;
    SuffixInsertLoc = FuncCallEnd;
  } else if (IsMacroArg) {
    NeedUseLambda = true;
    SourceRange SR = getFunctionRange(CE);
    PrefixInsertLoc = SR.getBegin();
    SuffixInsertLoc = SR.getEnd();
  } else if (IsInReturnStmt) {
    NeedUseLambda = true;
    CanAvoidUsingLambda = true;
    OriginStmtType = "return";
  }

  std::string IndentStr = getIndent(PrefixInsertLoc, SM).str();
  std::string PrefixInsertStr;

  std::string Msg = "the function call is redundant in DPC++.";
  if (FuncName == "curandSetPseudoRandomGeneratorSeed" ||
      FuncName == "curandSetQuasiRandomGeneratorDimensions") {
    if (IsAssigned) {
      report(PrefixInsertLoc, Diagnostics::FUNC_CALL_REMOVED_0, false, FuncName,
             Msg);
      emplaceTransformation(new ReplaceStmt(CE, false, FuncName, false, "0"));
    } else {
      report(PrefixInsertLoc, Diagnostics::FUNC_CALL_REMOVED, false, FuncName,
             Msg);
      emplaceTransformation(new ReplaceStmt(CE, false, FuncName, false, ""));
    }
  }

  if (FuncName == "curandCreateGenerator") {
    auto REInfo = DpctGlobalInfo::getInstance().findRandomEngine(CE->getArg(0));
    if (!REInfo) {
      DpctGlobalInfo::getInstance().insertRandomEngine(CE->getArg(0));
      REInfo = DpctGlobalInfo::getInstance().findRandomEngine(CE->getArg(0));
    }

    std::string EnumStr = getStmtSpelling(CE->getArg(1));
    if (MapNames::RandomEngineTypeMap.find(EnumStr) ==
        MapNames::RandomEngineTypeMap.end()) {
      report(SM.getExpansionLoc(REInfo->getDeclaratorDeclBeginLoc()),
             Diagnostics::UNMIGRATED_TYPE, false, "curandGenerator_t",
             "the migration depends on the second argument of "
             "curandCreateGenerator");
      report(PrefixInsertLoc, Diagnostics::NOT_SUPPORTED_PARAMETER, false,
             FuncName,
             "parameter " + EnumStr + " is unsupported");
      REInfo->setNeedPrint(false);
      return;
    }
    if (EnumStr == "CURAND_RNG_PSEUDO_XORWOW" ||
        EnumStr == "CURAND_RNG_QUASI_SOBOL64" ||
        EnumStr == "CURAND_RNG_QUASI_SCRAMBLED_SOBOL64") {
      report(SM.getExpansionLoc(REInfo->getDeclaratorDeclBeginLoc()),
             Diagnostics::DIFFERENT_GENERATOR, false);
    } else if (EnumStr == "CURAND_RNG_QUASI_SCRAMBLED_SOBOL32") {
      report(SM.getExpansionLoc(REInfo->getDeclaratorDeclBeginLoc()),
             Diagnostics::DIFFERENT_BASIC_GENERATOR, false);
    }

    if (!REInfo->isClassMember() && !REInfo->isArray()) {
      if (IsAssigned) {
        report(PrefixInsertLoc, Diagnostics::FUNC_CALL_REMOVED_0, false,
               FuncName, Msg);
        emplaceTransformation(new ReplaceStmt(CE, false, FuncName, false, "0"));
      } else {
        report(PrefixInsertLoc, Diagnostics::FUNC_CALL_REMOVED, false, FuncName,
               Msg);
        emplaceTransformation(new ReplaceStmt(CE, false, FuncName, false, ""));
      }
    }

    REInfo->setGeneratorName(getDrefName(CE->getArg(0)));
    REInfo->setEngineTypeReplacement(
        MapNames::RandomEngineTypeMap.find(EnumStr)->second);

    if (EnumStr == "CURAND_RNG_QUASI_DEFAULT" ||
        EnumStr == "CURAND_RNG_QUASI_SOBOL32" ||
        EnumStr == "CURAND_RNG_QUASI_SCRAMBLED_SOBOL32" ||
        EnumStr == "CURAND_RNG_QUASI_SOBOL64" ||
        EnumStr == "CURAND_RNG_QUASI_SCRAMBLED_SOBOL64")
      REInfo->setQuasiEngineFlag();

    REInfo->setTypeBeginOffset(
        SM.getDecomposedLoc(
              SM.getExpansionLoc(REInfo->getDeclaratorDeclBeginLoc()))
            .second);
    REInfo->setTypeLength(Lexer::MeasureTokenLength(
        SM.getExpansionLoc(REInfo->getDeclaratorDeclBeginLoc()), SM,
        DpctGlobalInfo::getContext().getLangOpts()));

    unsigned int FuncCallLen =
        SM.getDecomposedLoc(FuncCallEnd).second -
        SM.getDecomposedLoc(FuncNameBegin).second;
    REInfo->setCreateAPILength(FuncCallLen);
    REInfo->setCreateAPIBegin(SM.getDecomposedLoc(FuncNameBegin).second);
    auto EndLoc = REInfo->getDeclaratorDeclEndLoc();
    EndLoc = EndLoc.getLocWithOffset(
        Lexer::MeasureTokenLength(SM.getExpansionLoc(EndLoc), SM,
                                  DpctGlobalInfo::getContext().getLangOpts()));
    REInfo->setIdentifierEndOffset(SM.getDecomposedLoc(EndLoc).second);
    REInfo->setCreateCallFilePath(SM.getFilename(FuncNameBegin).str());

    if (REInfo->isClassMember() || REInfo->isArray()) {
      if (checkWhetherIsDuplicate(CE->getArg(0), false))
        return;
      int Index = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();
      REInfo->setQueueStr("{{NEEDREPLACEQ" + std::to_string(Index) + "}}");
      buildTempVariableMap(Index, CE->getArg(0), HelperFuncType::DefaultQueue);
    } else {
      if (checkWhetherIsDuplicate(RandomEngineInfo::getHandleVar(CE->getArg(0)),
                                  false))
        return;
      int Index = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();
      REInfo->setQueueStr("{{NEEDREPLACEQ" + std::to_string(Index) + "}}");
      buildTempVariableMap(Index, RandomEngineInfo::getHandleVar(CE->getArg(0)),
                           HelperFuncType::DefaultQueue);
    }
  } else if (FuncName == "curandDestroyGenerator") {
    auto REInfo = DpctGlobalInfo::getInstance().findRandomEngine(CE->getArg(0));
    if (!REInfo) {
      DpctGlobalInfo::getInstance().insertRandomEngine(CE->getArg(0));
      REInfo = DpctGlobalInfo::getInstance().findRandomEngine(CE->getArg(0));
    }
    if (REInfo->isClassMember() || REInfo->isArray()) {
      if (IsAssigned) {
        report(PrefixInsertLoc, Diagnostics::NOERROR_RETURN_COMMA_OP, false);
        insertAroundStmt(CE, "(", ", 0)");
      }
      emplaceTransformation(
          new ReplaceStmt(CE, false, FuncName, false,
                          "delete " + getStmtSpelling(CE->getArg(0))));
    } else {
      if (IsAssigned) {
        report(PrefixInsertLoc, Diagnostics::FUNC_CALL_REMOVED_0, false,
               FuncName,
               Msg);
        emplaceTransformation(new ReplaceStmt(CE, false, FuncName, false, "0"));
      } else {
        report(PrefixInsertLoc, Diagnostics::FUNC_CALL_REMOVED, false, FuncName,
               Msg);
        emplaceTransformation(new ReplaceStmt(CE, false, FuncName, false, ""));
      }
    }
  } else if (FuncName == "curandSetPseudoRandomGeneratorSeed") {
    auto REInfo = DpctGlobalInfo::getInstance().findRandomEngine(CE->getArg(0));
    if (!REInfo) {
      DpctGlobalInfo::getInstance().insertRandomEngine(CE->getArg(0));
      REInfo = DpctGlobalInfo::getInstance().findRandomEngine(CE->getArg(0));
    }
    REInfo->setSeedExpr(CE->getArg(1));
  } else if (FuncName == "curandSetQuasiRandomGeneratorDimensions") {
    auto REInfo = DpctGlobalInfo::getInstance().findRandomEngine(CE->getArg(0));
    if (!REInfo) {
      DpctGlobalInfo::getInstance().insertRandomEngine(CE->getArg(0));
      REInfo = DpctGlobalInfo::getInstance().findRandomEngine(CE->getArg(0));
    }
    REInfo->setDimExpr(CE->getArg(1));
  } else if (MapNames::RandomGenerateFuncReplInfoMap.find(FuncName) !=
             MapNames::RandomGenerateFuncReplInfoMap.end()) {
    auto REInfo = DpctGlobalInfo::getInstance().findRandomEngine(CE->getArg(0));
    if (!REInfo) {
      DpctGlobalInfo::getInstance().insertRandomEngine(CE->getArg(0));
      REInfo = DpctGlobalInfo::getInstance().findRandomEngine(CE->getArg(0));
    }
    auto ReplInfoPair = MapNames::RandomGenerateFuncReplInfoMap.find(FuncName);
    MapNames::RandomGenerateFuncReplInfo ReplInfo = ReplInfoPair->second;
    std::string BufferDecl;
    std::string BufferName;
    if (DpctGlobalInfo::getUsmLevel() == UsmLevel::none) {
      BufferName = getBufferNameAndDeclStr(
          CE->getArg(1), ReplInfo.BufferTypeInfo, IndentStr, BufferDecl);
    }
    std::string DistributeDecl;
    std::string DistrName =
        "distr_ct" +
        std::to_string(DpctGlobalInfo::getSuffixIndexInRuleThenInc());
    if (FuncName == "curandGenerateLogNormal" ||
        FuncName == "curandGenerateLogNormalDouble") {
      ExprAnalysis EMean, EDev;
      EMean.analyze(CE->getArg(3));
      EDev.analyze(CE->getArg(4));
      DistributeDecl = ReplInfo.DistributeName + "<" + ReplInfo.DistributeType +
                       "> " + DistrName + "(" + EMean.getReplacedString() +
                       ", " +
                       EDev.getReplacedString() + ", 0.0, 1.0);";
    } else if (FuncName == "curandGenerateNormal" ||
               FuncName == "curandGenerateNormalDouble") {
      ExprAnalysis EMean, EDev;
      EMean.analyze(CE->getArg(3));
      EDev.analyze(CE->getArg(4));
      DistributeDecl = ReplInfo.DistributeName + "<" + ReplInfo.DistributeType +
                       "> " + DistrName + "(" + EMean.getReplacedString() +
                       ", " +
                       EDev.getReplacedString() + ");";
    } else if (FuncName == "curandGeneratePoisson") {
      ExprAnalysis ELambda;
      ELambda.analyze(CE->getArg(3));
      DistributeDecl = ReplInfo.DistributeName + "<" + ReplInfo.DistributeType +
                       "> " + DistrName + "(" + ELambda.getReplacedString() +
                       ");";
    } else {
      DistributeDecl = ReplInfo.DistributeName + "<" + ReplInfo.DistributeType +
                       "> " + DistrName + ";";
    }
    std::string Data;
    if (DpctGlobalInfo::getUsmLevel() == UsmLevel::restricted) {
      PrefixInsertStr = DistributeDecl + getNL() + IndentStr;

      auto TypePtr = CE->getArg(1)->getType().getTypePtr();
      if (!TypePtr || !TypePtr->isPointerType()) {
        Data = "(" + ReplInfo.DistributeType + "*)" +
               getStmtSpelling(CE->getArg(1));
      } else if (TypePtr->getPointeeType().getAsString() ==
                 ReplInfo.DistributeType) {
        Data = getStmtSpelling(CE->getArg(1));
      } else {
        Data = "(" + ReplInfo.DistributeType + "*)" +
               getStmtSpelling(CE->getArg(1));
      }
    } else {
      PrefixInsertStr = BufferDecl + DistributeDecl + getNL() + IndentStr;
      Data = BufferName;
    }
    ExprAnalysis EA;
    EA.analyze(CE->getArg(2));
    std::string ReplStr;

    if (REInfo && (REInfo->isClassMember() || REInfo->isArray())) {
      ReplStr = "mkl::rng::generate(" + DistrName + ", *" +
                getStmtSpelling(CE->getArg(0)) + ", " + EA.getReplacedString() +
                ", " + Data + ")";
    } else {
      ReplStr = "mkl::rng::generate(" + DistrName + ", " +
                getStmtSpelling(CE->getArg(0)) + ", " + EA.getReplacedString() +
                ", " + Data + ")";
    }

    if (NeedUseLambda) {
      if (PrefixInsertStr.empty()) {
        // If there is one API call in the migrted code, it is unnecessary to
        // use a lambda expression
        NeedUseLambda = false;
      }
    }

    if (NeedUseLambda) {
      if (CanAvoidUsingLambda) {
        std::string InsertStr;
        if (DpctGlobalInfo::getUsmLevel() == UsmLevel::none && !CanAvoidBrace)
          InsertStr = std::string("{") + getNL() + IndentStr + PrefixInsertStr +
                      ReplStr + ";" + getNL() + IndentStr + "}" + getNL() +
                      IndentStr;
        else
          InsertStr = PrefixInsertStr + ReplStr + ";" + getNL() + IndentStr;
        emplaceTransformation(
            new InsertText(OuterInsertLoc, std::move(InsertStr)));
        report(OuterInsertLoc, Diagnostics::CODE_LOGIC_CHANGED, true,
               OriginStmtType == "if" ? "an " + OriginStmtType
                                      : "a " + OriginStmtType);
        emplaceTransformation(new ReplaceStmt(CE, "0"));
      } else {
        if (IsAssigned) {
          report(PrefixInsertLoc, Diagnostics::NOERROR_RETURN_LAMBDA, false);
          insertAroundRange(
              PrefixInsertLoc, SuffixInsertLoc,
              std::string("[&](){") + getNL() + IndentStr + PrefixInsertStr,
              std::string(";") + getNL() + IndentStr + "return 0;" + getNL() +
                  IndentStr + std::string("}()"));
        } else {
          insertAroundRange(
              PrefixInsertLoc, SuffixInsertLoc,
              std::string("[&](){") + getNL() + IndentStr + PrefixInsertStr,
              std::string(";") + getNL() + IndentStr + std::string("}()"));
        }
        emplaceTransformation(new ReplaceStmt(CE, std::move(ReplStr)));
      }
    } else {
      if (DpctGlobalInfo::getUsmLevel() == UsmLevel::none && !CanAvoidBrace) {
        if (!PrefixInsertStr.empty()) {
          insertAroundRange(
              PrefixInsertLoc, SuffixInsertLoc,
              std::string("{") + getNL() + IndentStr + PrefixInsertStr,
              getNL() + IndentStr + std::string("}"));
        }
      } else {
        emplaceTransformation(
            new InsertText(PrefixInsertLoc, std::move(PrefixInsertStr)));
      }
      if (IsAssigned) {
        insertAroundStmt(CE, "(", ", 0)");
        report(PrefixInsertLoc, Diagnostics::NOERROR_RETURN_COMMA_OP, true);
      }
      emplaceTransformation(new ReplaceStmt(CE, std::move(ReplStr)));
    }
  } else if (FuncName == "curandSetGeneratorOffset") {
    if (IsAssigned) {
      insertAroundStmt(CE, "(", ", 0)");
      report(FuncNameBegin, Diagnostics::NOERROR_RETURN_COMMA_OP, false);
    }
    std::string Repl =
        "mkl::rng::skip_ahead(" + getStmtSpelling(CE->getArg(0)) + ", ";
    ExprAnalysis EO;
    EO.analyze(CE->getArg(1));
    Repl = Repl + EO.getReplacedString() + ")";
    emplaceTransformation(new ReplaceStmt(CE, std::move(Repl)));
  }
}

REGISTER_RULE(RandomFunctionCallRule)

// Rule for device Random function calls.
void DeviceRandomFunctionCallRule::registerMatcher(MatchFinder &MF) {
  auto functionName = [&]() {
    return hasAnyName("curand_init", "curand_normal2", "curand_normal2_double",
                      "curand_normal_double", "curand_uniform");
  };
  MF.addMatcher(
      callExpr(callee(functionDecl(functionName()))).bind("FunctionCall"),
      this);
}

void DeviceRandomFunctionCallRule::run(const MatchFinder::MatchResult &Result) {
  CHECKPOINT_ASTMATCHER_RUN_ENTRY();
  const CallExpr *CE = getNodeAsType<CallExpr>(Result, "FunctionCall");
  if (!CE)
    return;
  if (!CE->getDirectCallee())
    return;

  auto &SM = DpctGlobalInfo::getSourceManager();
  auto SL = SM.getExpansionLoc(CE->getBeginLoc());
  std::string Key =
      SM.getFilename(SL).str() + std::to_string(SM.getDecomposedLoc(SL).second);
  DpctGlobalInfo::updateInitSuffixIndexInRule(
      DpctGlobalInfo::getSuffixIndexInitValue(Key));

  std::string FuncName =
      CE->getDirectCallee()->getNameInfo().getName().getAsString();
  SourceLocation FuncNameBegin(CE->getBeginLoc());
  SourceLocation FuncCallEnd(CE->getEndLoc());
  // Offset 1 is the length of the last token ")"
  FuncCallEnd = SM.getExpansionLoc(FuncCallEnd).getLocWithOffset(1);
  auto FuncCallLength =
      SM.getCharacterData(FuncCallEnd) - SM.getCharacterData(FuncNameBegin);
  std::string IndentStr = getIndent(FuncNameBegin, SM).str();

  if (FuncName == "curand_init") {
    if (CE->getNumArgs() < 4) {
      report(FuncNameBegin, Diagnostics::API_NOT_MIGRATED, false, FuncName);
      return;
    }

    std::string Arg0Type = CE->getArg(0)->getType().getAsString();
    std::string Arg1Type = CE->getArg(1)->getType().getAsString();
    std::string Arg2Type = CE->getArg(2)->getType().getAsString();
    std::string DRefArg3Type;

    if (Arg0Type == "unsigned long long" && Arg1Type == "unsigned long long" &&
        Arg2Type == "unsigned long long" &&
        CE->getArg(3)->getType()->isPointerType()) {
      DRefArg3Type = CE->getArg(3)->getType()->getPointeeType().getAsString();
      if (MapNames::DeviceRandomGeneratorTypeMap.find(DRefArg3Type) ==
          MapNames::DeviceRandomGeneratorTypeMap.end()) {
        report(FuncNameBegin, Diagnostics::NOT_SUPPORTED_PARAMETER, false,
               FuncName,
               "parameter " + getStmtSpelling(CE->getArg(3)) +
                   " is unsupported");
        return;
      }
    } else {
      report(FuncNameBegin, Diagnostics::API_NOT_MIGRATED, false, FuncName);
      return;
    }

    ExprAnalysis EARNGSeed(CE->getArg(0));
    ExprAnalysis EARNGSubseq(CE->getArg(1));
    ExprAnalysis EARNGOffset(CE->getArg(2));

    auto IsLiteral = [=](const Expr *E) {
      if (dyn_cast<IntegerLiteral>(E->IgnoreCasts()) ||
          dyn_cast<FloatingLiteral>(E->IgnoreCasts()) ||
          dyn_cast<FixedPointLiteral>(E->IgnoreCasts())) {
        return true;
      }
      return false;
    };

    DpctGlobalInfo::getInstance().insertDeviceRandomInitAPIInfo(
        FuncNameBegin, FuncCallLength,
        MapNames::DeviceRandomGeneratorTypeMap.find(DRefArg3Type)->second,
        EARNGSeed.getReplacedString(), EARNGSubseq.getReplacedString(),
        IsLiteral(CE->getArg(1)), EARNGOffset.getReplacedString(),
        IsLiteral(CE->getArg(2)), getDrefName(CE->getArg(3)), IndentStr);
  } else {
    const CompoundStmt *CS = findImmediateBlock(CE);
    if (!CS)
      return;

    SourceLocation DistrInsertLoc =
        SM.getExpansionLoc(CS->body_front()->getBeginLoc());
    std::string DistrIndentStr = getIndent(DistrInsertLoc, SM).str();
    std::string DrefedStateName = getDrefName(CE->getArg(0));

    if (FuncName == "curand_uniform") {
      DpctGlobalInfo::getDeviceRNGReturnNumSet().insert(1);
      DpctGlobalInfo::getInstance().insertDeviceRandomGenerateAPIInfo(
          FuncNameBegin, FuncCallLength, DistrInsertLoc,
          "mkl::rng::device::uniform", "float", DistrIndentStr, DrefedStateName,
          IndentStr);
    } else if (FuncName == "curand_normal2") {
      DpctGlobalInfo::getDeviceRNGReturnNumSet().insert(2);
      DpctGlobalInfo::getInstance().insertDeviceRandomGenerateAPIInfo(
          FuncNameBegin, FuncCallLength, DistrInsertLoc,
          "mkl::rng::device::uniform", "float", DistrIndentStr, DrefedStateName,
          IndentStr);
    } else if (FuncName == "curand_normal2_double") {
      DpctGlobalInfo::getDeviceRNGReturnNumSet().insert(2);
      DpctGlobalInfo::getInstance().insertDeviceRandomGenerateAPIInfo(
          FuncNameBegin, FuncCallLength, DistrInsertLoc,
          "mkl::rng::device::uniform", "double", DistrIndentStr,
          DrefedStateName, IndentStr);
    } else if (FuncName == "curand_normal_double") {
      DpctGlobalInfo::getDeviceRNGReturnNumSet().insert(1);
      DpctGlobalInfo::getInstance().insertDeviceRandomGenerateAPIInfo(
          FuncNameBegin, FuncCallLength, DistrInsertLoc,
          "mkl::rng::device::uniform", "double", DistrIndentStr,
          DrefedStateName, IndentStr);
    }
  }
}

REGISTER_RULE(DeviceRandomFunctionCallRule)


void BLASFunctionCallRule::registerMatcher(MatchFinder &MF) {
  auto functionName = [&]() {
    return hasAnyName(
        "make_cuComplex", "make_cuDoubleComplex",
        /*Regular BLAS API*/
        /*Regular helper*/
        "cublasCreate_v2", "cublasDestroy_v2", "cublasSetVector",
        "cublasGetVector", "cublasSetVectorAsync", "cublasGetVectorAsync",
        "cublasSetMatrix", "cublasGetMatrix", "cublasSetMatrixAsync",
        "cublasGetMatrixAsync", "cublasSetStream_v2", "cublasGetStream_v2",
        "cublasGetPointerMode_v2", "cublasSetPointerMode_v2",
        /*Regular level 1*/
        "cublasIsamax_v2", "cublasIdamax_v2", "cublasIcamax_v2",
        "cublasIzamax_v2", "cublasIsamin_v2", "cublasIdamin_v2",
        "cublasIcamin_v2", "cublasIzamin_v2", "cublasSasum_v2",
        "cublasDasum_v2", "cublasScasum_v2", "cublasDzasum_v2",
        "cublasSaxpy_v2", "cublasDaxpy_v2", "cublasCaxpy_v2", "cublasZaxpy_v2",
        "cublasScopy_v2", "cublasDcopy_v2", "cublasCcopy_v2", "cublasZcopy_v2",
        "cublasSdot_v2", "cublasDdot_v2", "cublasCdotu_v2", "cublasCdotc_v2",
        "cublasZdotu_v2", "cublasZdotc_v2", "cublasSnrm2_v2", "cublasDnrm2_v2",
        "cublasScnrm2_v2", "cublasDznrm2_v2", "cublasSrot_v2", "cublasDrot_v2",
        "cublasCsrot_v2", "cublasZdrot_v2", "cublasSrotg_v2", "cublasDrotg_v2",
        "cublasCrotg_v2", "cublasZrotg_v2", "cublasSrotm_v2", "cublasDrotm_v2",
        "cublasSrotmg_v2", "cublasDrotmg_v2", "cublasSscal_v2",
        "cublasDscal_v2", "cublasCscal_v2", "cublasCsscal_v2", "cublasZscal_v2",
        "cublasZdscal_v2", "cublasSswap_v2", "cublasDswap_v2", "cublasCswap_v2",
        "cublasZswap_v2",
        /*Regular level 2*/
        "cublasSgbmv_v2", "cublasDgbmv_v2", "cublasCgbmv_v2", "cublasZgbmv_v2",
        "cublasSgemv_v2", "cublasDgemv_v2", "cublasCgemv_v2", "cublasZgemv_v2",
        "cublasSger_v2", "cublasDger_v2", "cublasCgeru_v2", "cublasCgerc_v2",
        "cublasZgeru_v2", "cublasZgerc_v2", "cublasSsbmv_v2", "cublasDsbmv_v2",
        "cublasSspmv_v2", "cublasDspmv_v2", "cublasSspr_v2", "cublasDspr_v2",
        "cublasSspr2_v2", "cublasDspr2_v2", "cublasSsymv_v2", "cublasDsymv_v2",
        "cublasSsyr_v2", "cublasDsyr_v2", "cublasSsyr2_v2", "cublasDsyr2_v2",
        "cublasStbmv_v2", "cublasDtbmv_v2", "cublasCtbmv_v2", "cublasZtbmv_v2",
        "cublasStbsv_v2", "cublasDtbsv_v2", "cublasCtbsv_v2", "cublasZtbsv_v2",
        "cublasStpmv_v2", "cublasDtpmv_v2", "cublasCtpmv_v2", "cublasZtpmv_v2",
        "cublasStpsv_v2", "cublasDtpsv_v2", "cublasCtpsv_v2", "cublasZtpsv_v2",
        "cublasStrmv_v2", "cublasDtrmv_v2", "cublasCtrmv_v2", "cublasZtrmv_v2",
        "cublasStrsv_v2", "cublasDtrsv_v2", "cublasCtrsv_v2", "cublasZtrsv_v2",
        "cublasChemv_v2", "cublasZhemv_v2", "cublasChbmv_v2", "cublasZhbmv_v2",
        "cublasChpmv_v2", "cublasZhpmv_v2", "cublasCher_v2", "cublasZher_v2",
        "cublasCher2_v2", "cublasZher2_v2", "cublasChpr_v2", "cublasZhpr_v2",
        "cublasChpr2_v2", "cublasZhpr2_v2",
        /*Regular level 3*/
        "cublasSgemm_v2", "cublasDgemm_v2", "cublasCgemm_v2", "cublasZgemm_v2",
        "cublasHgemm", "cublasCgemm3m", "cublasZgemm3m",
        "cublasHgemmStridedBatched",
        "cublasSgemmStridedBatched", "cublasDgemmStridedBatched",
        "cublasCgemmStridedBatched", "cublasZgemmStridedBatched",
        "cublasSsymm_v2", "cublasDsymm_v2",
        "cublasCsymm_v2", "cublasZsymm_v2", "cublasSsyrk_v2", "cublasDsyrk_v2",
        "cublasCsyrk_v2", "cublasZsyrk_v2", "cublasSsyr2k_v2",
        "cublasDsyr2k_v2", "cublasCsyr2k_v2", "cublasZsyr2k_v2",
        "cublasStrsm_v2", "cublasDtrsm_v2", "cublasCtrsm_v2", "cublasZtrsm_v2",
        "cublasChemm_v2", "cublasZhemm_v2", "cublasCherk_v2", "cublasZherk_v2",
        "cublasCher2k_v2", "cublasZher2k_v2", "cublasSsyrkx", "cublasDsyrkx",
        "cublasStrmm_v2", "cublasDtrmm_v2", "cublasCtrmm_v2", "cublasZtrmm_v2",
        "cublasHgemmBatched", "cublasSgemmBatched", "cublasDgemmBatched",
        "cublasCgemmBatched",
        "cublasZgemmBatched", "cublasStrsmBatched", "cublasDtrsmBatched",
        "cublasCtrsmBatched", "cublasZtrsmBatched",
        /*Extensions*/
        "cublasSgetrfBatched", "cublasDgetrfBatched", "cublasCgetrfBatched",
        "cublasZgetrfBatched", "cublasSgetrsBatched", "cublasDgetrsBatched",
        "cublasCgetrsBatched", "cublasZgetrsBatched", "cublasSgetriBatched",
        "cublasDgetriBatched", "cublasCgetriBatched", "cublasZgetriBatched",
        "cublasSgeqrfBatched", "cublasDgeqrfBatched", "cublasCgeqrfBatched",
        "cublasZgeqrfBatched", "cublasGemmEx", "cublasSgemmEx", "cublasCgemmEx",
        /*Legacy API*/
        "cublasInit", "cublasShutdown", "cublasGetError",
        "cublasSetKernelStream",
        /*level 1*/
        "cublasSnrm2", "cublasDnrm2", "cublasScnrm2", "cublasDznrm2",
        "cublasSdot", "cublasDdot", "cublasCdotu", "cublasCdotc", "cublasZdotu",
        "cublasZdotc", "cublasSscal", "cublasDscal", "cublasCscal",
        "cublasZscal", "cublasCsscal", "cublasZdscal", "cublasSaxpy",
        "cublasDaxpy", "cublasCaxpy", "cublasZaxpy", "cublasScopy",
        "cublasDcopy", "cublasCcopy", "cublasZcopy", "cublasSswap",
        "cublasDswap", "cublasCswap", "cublasZswap", "cublasIsamax",
        "cublasIdamax", "cublasIcamax", "cublasIzamax", "cublasIsamin",
        "cublasIdamin", "cublasIcamin", "cublasIzamin", "cublasSasum",
        "cublasDasum", "cublasScasum", "cublasDzasum", "cublasSrot",
        "cublasDrot", "cublasCsrot", "cublasZdrot", "cublasSrotg",
        "cublasDrotg", "cublasSrotm", "cublasDrotm", "cublasSrotmg",
        "cublasDrotmg",
        /*level 2*/
        "cublasSgemv", "cublasDgemv", "cublasCgemv", "cublasZgemv",
        "cublasSgbmv", "cublasDgbmv", "cublasCgbmv", "cublasZgbmv",
        "cublasStrmv", "cublasDtrmv", "cublasCtrmv", "cublasZtrmv",
        "cublasStbmv", "cublasDtbmv", "cublasCtbmv", "cublasZtbmv",
        "cublasStpmv", "cublasDtpmv", "cublasCtpmv", "cublasZtpmv",
        "cublasStrsv", "cublasDtrsv", "cublasCtrsv", "cublasZtrsv",
        "cublasStpsv", "cublasDtpsv", "cublasCtpsv", "cublasZtpsv",
        "cublasStbsv", "cublasDtbsv", "cublasCtbsv", "cublasZtbsv",
        "cublasSsymv", "cublasDsymv", "cublasChemv", "cublasZhemv",
        "cublasSsbmv", "cublasDsbmv", "cublasChbmv", "cublasZhbmv",
        "cublasSspmv", "cublasDspmv", "cublasChpmv", "cublasZhpmv",
        "cublasSger", "cublasDger", "cublasCgeru", "cublasCgerc", "cublasZgeru",
        "cublasZgerc", "cublasSsyr", "cublasDsyr", "cublasCher", "cublasZher",
        "cublasSspr", "cublasDspr", "cublasChpr", "cublasZhpr", "cublasSsyr2",
        "cublasDsyr2", "cublasCher2", "cublasZher2", "cublasSspr2",
        "cublasDspr2", "cublasChpr2", "cublasZhpr2",
        /*level 3*/
        "cublasSgemm", "cublasDgemm", "cublasCgemm", "cublasZgemm",
        "cublasSsyrk", "cublasDsyrk", "cublasCsyrk", "cublasZsyrk",
        "cublasCherk", "cublasZherk", "cublasSsyr2k", "cublasDsyr2k",
        "cublasCsyr2k", "cublasZsyr2k", "cublasCher2k", "cublasZher2k",
        "cublasSsymm", "cublasDsymm", "cublasCsymm", "cublasZsymm",
        "cublasChemm", "cublasZhemm", "cublasStrsm", "cublasDtrsm",
        "cublasCtrsm", "cublasZtrsm", "cublasStrmm", "cublasDtrmm",
        "cublasCtrmm", "cublasZtrmm");
  };

  MF.addMatcher(callExpr(allOf(callee(functionDecl(functionName())),
                               hasAncestor(functionDecl(
                                   anyOf(hasAttr(attr::CUDADevice),
                                         hasAttr(attr::CUDAGlobal))))))
                    .bind("kernelCall"),
                this);

  MF.addMatcher(
      callExpr(
          allOf(callee(functionDecl(functionName())), parentStmt(),
                hasAncestor(functionDecl(unless(allOf(
                    hasAttr(attr::CUDADevice), hasAttr(attr::CUDAGlobal)))))))
          .bind("FunctionCall"),
      this);
  MF.addMatcher(
      callExpr(
          allOf(callee(functionDecl(functionName())), unless(parentStmt()),
                unless(hasAncestor(varDecl())),
                hasAncestor(functionDecl(unless(allOf(
                    hasAttr(attr::CUDADevice), hasAttr(attr::CUDAGlobal)))))))
          .bind("FunctionCallUsedNotInitializeVarDecl"),
      this);

  MF.addMatcher(
      callExpr(
          allOf(callee(functionDecl(functionName())), hasAncestor(varDecl()),
                hasAncestor(functionDecl(unless(allOf(
                    hasAttr(attr::CUDADevice), hasAttr(attr::CUDAGlobal)))))))
          .bind("FunctionCallUsedInitializeVarDecl"),
      this);
}

void BLASFunctionCallRule::run(const MatchFinder::MatchResult &Result) {
  CHECKPOINT_ASTMATCHER_RUN_ENTRY();
  bool IsAssigned = false;
  bool IsInitializeVarDecl = false;
  bool HasDeviceAttr = false;
  const CallExpr *CE = getNodeAsType<CallExpr>(Result, "kernelCall");
  if (CE) {
    HasDeviceAttr = true;
  } else if (!(CE = getNodeAsType<CallExpr>(Result, "FunctionCall"))) {
    if ((CE = getNodeAsType<CallExpr>(
             Result, "FunctionCallUsedNotInitializeVarDecl"))) {
      IsAssigned = true;
    } else if ((CE = getNodeAsType<CallExpr>(
                    Result, "FunctionCallUsedInitializeVarDecl"))) {
      IsAssigned = true;
      IsInitializeVarDecl = true;
    } else {
      return;
    }
  }

  if (!CE->getDirectCallee())
    return;
  std::string FuncName =
      CE->getDirectCallee()->getNameInfo().getName().getAsString();

  const SourceManager *SM = Result.SourceManager;
  auto SL = SM->getExpansionLoc(CE->getBeginLoc());
  std::string Key = SM->getFilename(SL).str() +
                    std::to_string(SM->getDecomposedLoc(SL).second);
  DpctGlobalInfo::updateInitSuffixIndexInRule(
      DpctGlobalInfo::getSuffixIndexInitValue(Key));

  SourceLocation FuncNameBegin(CE->getBeginLoc());
  SourceLocation FuncCallEnd(CE->getEndLoc());
  // There are some macroes like "#define API API_v2"
  // so the function names we match should have the
  // suffix "_v2".
  bool IsMacroArg = SM->isMacroArgExpansion(CE->getBeginLoc()) &&
                    SM->isMacroArgExpansion(CE->getEndLoc());

  if (FuncNameBegin.isMacroID() && IsMacroArg) {
    FuncNameBegin = SM->getImmediateSpellingLoc(FuncNameBegin);
    FuncNameBegin = SM->getExpansionLoc(FuncNameBegin);
  } else if (FuncNameBegin.isMacroID()) {
    FuncNameBegin = SM->getExpansionLoc(FuncNameBegin);
  }

  if (FuncCallEnd.isMacroID() && IsMacroArg) {
    FuncCallEnd = SM->getImmediateSpellingLoc(FuncCallEnd);
    FuncCallEnd = SM->getExpansionLoc(FuncCallEnd);
  } else if (FuncCallEnd.isMacroID()) {
    FuncCallEnd = SM->getExpansionLoc(FuncCallEnd);
  }

  // Offset 1 is the length of the last token ")"
  FuncCallEnd = FuncCallEnd.getLocWithOffset(1);
  auto SR = getScopeInsertRange(CE, FuncNameBegin, FuncCallEnd);
  SourceLocation PrefixInsertLoc = SR.getBegin(), SuffixInsertLoc = SR.getEnd();

  auto FuncCallLength =
      SM->getCharacterData(FuncCallEnd) - SM->getCharacterData(FuncNameBegin);

  bool CanAvoidUsingLambda = false;
  SourceLocation OuterInsertLoc;
  std::string OriginStmtType;
  bool NeedUseLambda = isConditionOfFlowControl(
      CE, OriginStmtType, CanAvoidUsingLambda, OuterInsertLoc);
  bool IsInReturnStmt = isInReturnStmt(CE, OuterInsertLoc);
  bool CanAvoidBrace = false;
  const CompoundStmt *CS = findImmediateBlock(CE);
  if (CS && (CS->size() == 1)) {
    const Stmt *S = *(CS->child_begin());
    if (CE == S || dyn_cast<ReturnStmt>(S))
      CanAvoidBrace = true;
  }

  if (NeedUseLambda) {
    PrefixInsertLoc = FuncNameBegin;
    SuffixInsertLoc = FuncCallEnd;
  } else if (IsMacroArg) {
    NeedUseLambda = true;
    SourceRange SR = getFunctionRange(CE);
    PrefixInsertLoc = SR.getBegin();
    SuffixInsertLoc = SR.getEnd();
  } else if (IsInReturnStmt) {
    NeedUseLambda = true;
    CanAvoidUsingLambda = true;
    OriginStmtType = "return";
    // For some Legacy BLAS API (return the calculated value), below two
    // variables are needed. Although the function call is in return stmt, it
    // cannot move out and must use lambda.
    PrefixInsertLoc = FuncNameBegin;
    SuffixInsertLoc = FuncCallEnd;
  }

  if (DpctGlobalInfo::getUsmLevel() == UsmLevel::none &&
      (FuncName == "cublasHgemmBatched" || FuncName == "cublasSgemmBatched" ||
       FuncName == "cublasDgemmBatched" ||
       FuncName == "cublasCgemmBatched" || FuncName == "cublasZgemmBatched" ||
       FuncName == "cublasStrsmBatched" || FuncName == "cublasDtrsmBatched" ||
       FuncName == "cublasCtrsmBatched" || FuncName == "cublasZtrsmBatched")) {
    report(FuncNameBegin, Diagnostics::API_NOT_MIGRATED, false, FuncName);
    return;
  }

  std::string IndentStr = getIndent(PrefixInsertLoc, *SM).str();
  // PrefixInsertStr: stmt + NL + indent
  // SuffixInsertStr: NL + indent + stmt
  std::string PrefixInsertStr, SuffixInsertStr;
  CallExprArguReplVec.clear();
  CallExprReplStr = "";
  // TODO: Need to process the situation when scalar pointers (alpha, beta)
  // are device pointers.
  if (MapNames::BatchedBLASFuncReplInfoMap.find(FuncName) !=
      MapNames::BatchedBLASFuncReplInfoMap.end()) {
    auto ReplInfoPair = MapNames::BatchedBLASFuncReplInfoMap.find(FuncName);
    auto ReplInfo = ReplInfoPair->second;
    std::string Replacement = ReplInfo.ReplName;
    BLASEnumInfo EnumInfo =
        BLASEnumInfo(ReplInfo.OperationIndexInfo, ReplInfo.FillModeIndexInfo,
                     ReplInfo.SideModeIndexInfo, ReplInfo.DiagTypeIndexInfo);
    std::string BufferType = ReplInfo.BufferTypeInfo[0];


    if (HasDeviceAttr) {
      report(FuncNameBegin, Diagnostics::FUNCTION_CALL_IN_DEVICE, false,
             MapNames::ITFName.at(FuncName), Replacement);
      return;
    }

    // update the replacement of four enmu arguments
    auto processEnumArgus = [&](const Expr *E, unsigned int Index,
                                std::string &Argu) {
      const CStyleCastExpr *CSCE = nullptr;
      if (CSCE = dyn_cast<CStyleCastExpr>(E)) {
        std::string CurrentArgumentRepl;
        processParamIntCastToBLASEnum(E, CSCE, Index, IndentStr, EnumInfo,
                                      PrefixInsertStr, CurrentArgumentRepl);
        Argu = CurrentArgumentRepl;
      }
    };

    int ArgNum = CE->getNumArgs();
    for (int i = 0; i < ArgNum; ++i) {
      ExprAnalysis EA;
      EA.analyze(CE->getArg(i));
      CallExprArguReplVec.push_back(EA.getReplacedString());
      processEnumArgus(CE->getArg(i), i, CallExprArguReplVec[i]);
    }

    // update the replacement of buffers
    for (size_t i = 0; i < ReplInfo.BufferIndexInfo.size(); ++i) {
      int BufferIndex = ReplInfo.BufferIndexInfo[i];
      if (DpctGlobalInfo::getUsmLevel() == UsmLevel::none) {
        std::string BufferDecl;
        CallExprArguReplVec[BufferIndex] = getBufferNameAndDeclStr(
            CE->getArg(BufferIndex), BufferType, IndentStr, BufferDecl);
        PrefixInsertStr = PrefixInsertStr + BufferDecl;
      } else {
        if (BufferType == "std::complex<float>" ||
            BufferType == "std::complex<double>") {
          if (i == ReplInfo.BufferIndexInfo.size() - 1)
            CallExprArguReplVec[BufferIndex] =
                "(" + BufferType + "**)" + CallExprArguReplVec[BufferIndex];
          else
            CallExprArguReplVec[BufferIndex] = "(const " + BufferType + "**)" +
                                               CallExprArguReplVec[BufferIndex];
        }
      }
    }

    // update the replacement of scalar arguments
    for (size_t i = 0; i < ReplInfo.PointerIndexInfo.size(); ++i) {
      int ScalarIndex = ReplInfo.PointerIndexInfo[i];
      CallExprArguReplVec[ScalarIndex] = "dpct::get_value(" +
                                         CallExprArguReplVec[ScalarIndex] +
                                         ", *" + CallExprArguReplVec[0] + ")";
    }

    //Declare temp variables for m/n/k/lda/ldb/ldc/alpha/beta/transa/transb/groupsize
    //These pointers are accessed on host only and the value will be saved before
    //MKL API returns, so pass the addresses of the variables on stack directly.
    auto declareTempVars = [&](const std::string &TempVarType,
                               std::vector<std::string> Names,
                               std::vector<int> Indexes) {
      auto Num = Names.size();
      PrefixInsertStr = PrefixInsertStr + TempVarType;
      for (size_t i = 0; i < Num; i++) {
        std::string DeclName =
            Names[i] +
            std::to_string(DpctGlobalInfo::getSuffixIndexInRuleThenInc());
        PrefixInsertStr = PrefixInsertStr + " " + DeclName +
                          " = " + CallExprArguReplVec[Indexes[i]] + ",";
        CallExprArguReplVec[Indexes[i]] = "&" + DeclName;
      }
      PrefixInsertStr[PrefixInsertStr.size() - 1] = ';';
      PrefixInsertStr = PrefixInsertStr + getNL() + IndentStr;
    };

    if (FuncName == "cublasHgemmBatched" || FuncName == "cublasSgemmBatched" ||
        FuncName == "cublasDgemmBatched" ||
        FuncName == "cublasCgemmBatched" || FuncName == "cublasZgemmBatched") {
      if (CE->getArg(1)->IgnoreImplicit()->isLValue())
        CallExprArguReplVec[1] = "&" + CallExprArguReplVec[1];
      else
        declareTempVars({"mkl::transpose"}, {"transpose_ct"}, {1});
      if (CE->getArg(2)->IgnoreImplicit()->isLValue())
        CallExprArguReplVec[2] = "&" + CallExprArguReplVec[2];
      else
        declareTempVars({"mkl::transpose"}, {"transpose_ct"}, {2});

      declareTempVars("int64_t",
                      {"m_ct", "n_ct", "k_ct", "lda_ct", "ldb_ct", "ldc_ct",
                       "group_size_ct"},
                      {3, 4, 5, 8, 10, 13, 14});
      declareTempVars(BufferType, {"alpha_ct", "beta_ct"}, {6, 11});

      // insert the group_count to CallExprArguReplVec
      auto InsertIter = CallExprArguReplVec.begin();
      std::advance(InsertIter, 14);
      CallExprArguReplVec.insert(InsertIter, "1");
    } else {
      if (CE->getArg(1)->IgnoreImplicit()->isLValue())
        CallExprArguReplVec[1] = "&" + CallExprArguReplVec[1];
      else
        declareTempVars({"mkl::side"}, {"side_ct"}, {1});
      if (CE->getArg(2)->IgnoreImplicit()->isLValue())
        CallExprArguReplVec[2] = "&" + CallExprArguReplVec[2];
      else
        declareTempVars({"mkl::uplo"}, {"uplo_ct"}, {2});
      if (CE->getArg(3)->IgnoreImplicit()->isLValue())
        CallExprArguReplVec[3] = "&" + CallExprArguReplVec[3];
      else
        declareTempVars({"mkl::transpose"}, {"transpose_ct"}, {3});
      if (CE->getArg(4)->IgnoreImplicit()->isLValue())
        CallExprArguReplVec[2] = "&" + CallExprArguReplVec[4];
      else
        declareTempVars({"mkl::diag"}, {"diag_ct"}, {4});

      declareTempVars("int64_t",
                      {"m_ct", "n_ct", "lda_ct", "ldb_ct", "group_size_ct"},
                      {5, 6, 9, 11, 12});
      declareTempVars(BufferType, {"alpha_ct"}, {7});

      // insert the group_count to CallExprArguReplVec
      auto InsertIter = CallExprArguReplVec.begin();
      std::advance(InsertIter, 12);
      CallExprArguReplVec.insert(InsertIter, "1");
    }
    // add an empty event vector as the last argument
    CallExprArguReplVec.push_back("{}");
    CallExprReplStr = getFinalCallExprStr(Replacement) + CallExprReplStr;

    if (NeedUseLambda) {
      if (PrefixInsertStr.empty() && SuffixInsertStr.empty()) {
        NeedUseLambda = false;
      }
    }
    applyMigrationText(NeedUseLambda, IsMacroArg, CanAvoidBrace,
                       CanAvoidUsingLambda, OriginStmtType, IsAssigned,
                       OuterInsertLoc, PrefixInsertLoc, SuffixInsertLoc,
                       FuncNameBegin, FuncCallEnd, FuncCallLength, IndentStr,
                       PrefixInsertStr, SuffixInsertStr);
  } else if (FuncName == "cublasSsyrkx" || FuncName == "cublasDsyrkx") {
    auto ReplInfoPair = MapNames::BLASFuncReplInfoMap.find(FuncName);
    MapNames::BLASFuncReplInfo ReplInfo = ReplInfoPair->second;
    std::string Replacement = ReplInfo.ReplName;
    BLASEnumInfo EnumInfo(
        ReplInfo.OperationIndexInfo, ReplInfo.FillModeIndexInfo,
        ReplInfo.SideModeIndexInfo, ReplInfo.DiagTypeIndexInfo);
    if (HasDeviceAttr) {
      report(FuncNameBegin, Diagnostics::FUNCTION_CALL_IN_DEVICE, false,
             MapNames::ITFName.at(FuncName), Replacement);
      return;
    }

    // initialize the replacement of each arguments
    int ArgNum = CE->getNumArgs();
    for (int i = 0; i < ArgNum; ++i) {
      ExprAnalysis EA;
      EA.analyze(CE->getArg(i));
      CallExprArguReplVec.push_back(EA.getReplacedString());
    }

    // update the replacement of three buffer arguments
    std::string BufferType = FuncName == "cublasSsyrkx" ? "float" : "double";
    if (DpctGlobalInfo::getUsmLevel() == UsmLevel::none) {
      std::string BufferDecl;
      CallExprArguReplVec[6] = getBufferNameAndDeclStr(
          CE->getArg(6), BufferType, IndentStr, BufferDecl);
      PrefixInsertStr = PrefixInsertStr + BufferDecl;
      CallExprArguReplVec[8] = getBufferNameAndDeclStr(
          CE->getArg(8), BufferType, IndentStr, BufferDecl);
      PrefixInsertStr = PrefixInsertStr + BufferDecl;
      CallExprArguReplVec[11] = getBufferNameAndDeclStr(
          CE->getArg(11), BufferType, IndentStr, BufferDecl);
      PrefixInsertStr = PrefixInsertStr + BufferDecl;
    }


    // update the replacement of two scalar arguments
    CallExprArguReplVec[5] = "dpct::get_value(" + CallExprArguReplVec[5] +
                             ", *" + CallExprArguReplVec[0] + ")";
    CallExprArguReplVec[10] = "dpct::get_value(" + CallExprArguReplVec[10] +
                             ", *" + CallExprArguReplVec[0] + ")";

    // update the replacement of the fillmode enmu argument
    const CStyleCastExpr *CSCE = nullptr;
    if (CSCE = dyn_cast<CStyleCastExpr>(CE->getArg(1))) {
      std::string CurrentArgumentRepl;
      processParamIntCastToBLASEnum(CE->getArg(1), CSCE, 1, IndentStr, EnumInfo,
                                    PrefixInsertStr, CurrentArgumentRepl);
      CallExprArguReplVec[1] = CurrentArgumentRepl;
      CSCE = nullptr;
    }

    // update the replacement of two transpose operation enmu arguments
    // original API: C = alpha*t(A, t_1)*t(t(B, t_1), transpose) + beta*C
    // migrated API: C = alpha*t(A, t_1)*t(B, t_2) + beta*C
    // So the migated API need insert another transpose state variable which is
    // decided by the first transpose state variable.
    // For float and double type, the transpose state variable can only be
    // transpose or non-transpose, so if t_1 == transpose, then t_2 should be
    // non-transpose and vice versa.

    // the first transpose operation enmu argument
    std::string TransTempVarName;
    if (CSCE = dyn_cast<CStyleCastExpr>(CE->getArg(2))) {
      // It is a c-style cast expr, process with processParamIntCastToBLASEnum()
      // If the sub-expr has side-effect, a temp variable will be declared in
      // processParamIntCastToBLASEnum() function, if not, then we can use this
      // expr in the inserted argument directly.
      std::string CurrentArgumentRepl;
      TransTempVarName = processParamIntCastToBLASEnum(
          CE->getArg(2), CSCE, 2, IndentStr, EnumInfo, PrefixInsertStr,
          CurrentArgumentRepl);
      CallExprArguReplVec[2] = CurrentArgumentRepl;
      CSCE = nullptr;
    } else {
      // if it isn't c-style cast, then the expr type should be an enumeration.
      // because the another transpose state argument depends on this argument,
      // if this expr has side-effect, then we need declare a temp variable,
      // if not, then we can use this expr in the inserted argument directly.
      if (CE->getArg(2)->HasSideEffects(DpctGlobalInfo::getContext())) {
        TransTempVarName =
            getTempNameForExpr(CE->getArg(2), true, true) + "transpose_ct" +
            std::to_string(DpctGlobalInfo::getSuffixIndexInRuleThenInc());

        PrefixInsertStr = PrefixInsertStr + "auto " + TransTempVarName + " = " +
                          CallExprArguReplVec[2] + ";" + getNL() + IndentStr;
        CallExprArguReplVec[2] = TransTempVarName;
      }
    }

    // The inserted transpose operation enmu argument (depends on the first one)
    // Save the replacement in the InsertStr first.
    std::string InsertStr;
    if (CSCE = dyn_cast<CStyleCastExpr>(CE->getArg(2))) {
      // Case1: the first operation enmu argument is a c-style cast expr
      if (CE->getArg(2)->HasSideEffects(DpctGlobalInfo::getContext())) {
        // if the expr has side effect, use the temp variable name which returns
        // from processParamIntCastToBLASEnum()
        InsertStr = "(int)" + TransTempVarName +
                    "==0 ? mkl::transpose::trans : mkl::transpose::nontrans";
      } else {
        // if the expr hasn't side effect, use the sub-expr to decide the
        // inserting argument
        std::string SubExprStr;
        bool IsTypeCastInMacro = false;
        if (CSCE->getSubExpr()->getBeginLoc().isMacroID() &&
            isOuterMostMacro(CSCE)) {
          // When type casting syntax is in a macro, analyze the entire CSCE
          // by ExprAnalysis. Then we need add a type cast to int before the
          // inserted argument.
          // Related to BLASFunctionCallRule::processParamIntCastToBLASEnum()
          IsTypeCastInMacro = true;
          ExprAnalysis SEA;
          SEA.analyze(CSCE);
          SubExprStr = SEA.getReplacedString();
        } else {
          SubExprStr = getStmtSpelling(CSCE->getSubExpr());
        }
        Expr::EvalResult ER;
        if (CSCE->getSubExpr()->EvaluateAsInt(ER, *Result.Context) &&
            !CSCE->getSubExpr()->getBeginLoc().isMacroID()) {
          // if the sub-expr can be evaluated, then generate the corresponding
          // replacement directly.
          int64_t Value = ER.Val.getInt().getExtValue();
          if (Value == 0) {
            InsertStr = "mkl::transpose::trans";
          } else if (Value == 1) {
            InsertStr = "mkl::transpose::nontrans";
          } else {
            InsertStr = SubExprStr +
                "==0 ? mkl::transpose::trans : mkl::transpose::nontrans";
          }
        } else {
          // if the sub-expr cannot be evaluated, use the conditional operator
          SubExprStr = IsTypeCastInMacro ? "(int)" + SubExprStr : SubExprStr;
          InsertStr = SubExprStr +
                      "==0 ? mkl::transpose::trans : mkl::transpose::nontrans";
        }
      }
    } else {
      // Case2: the first operation enmu argument isn't c-style cast expr, then
      // the expr type should be an enumeration.
      if (CE->getArg(2)->HasSideEffects(DpctGlobalInfo::getContext())) {
        // if the expr has side effect, use the temp variable name which
        // generated in the previous step.
        InsertStr = TransTempVarName +
                    "==mkl::transpose::nontrans ? mkl::transpose::trans : "
                    "mkl::transpose::nontrans";
      } else {
        // The expr hasn't side effect, if the enumeration is literal, then
        // generate the corresponding replacement directly, if not, use the
        // conditional operator
        std::string TransStr = getStmtSpelling(CE->getArg(2));
        auto TransPair = MapNames::BLASEnumsMap.find(TransStr);
        if (TransPair != MapNames::BLASEnumsMap.end()) {
          TransStr = TransPair->second;
        }
        if (TransStr == "mkl::transpose::nontrans") {
          InsertStr = "mkl::transpose::trans";
        } else if (TransStr == "mkl::transpose::trans") {
          InsertStr = "mkl::transpose::nontrans";
        } else {
          InsertStr =
              TransStr + "==mkl::transpose::nontrans ? mkl::transpose::trans : "
                         "mkl::transpose::nontrans";
        }
      }
    }
    // insert the InsertStr to CallExprArguReplVec
    auto InsertIter = CallExprArguReplVec.begin();
    std::advance(InsertIter, 3);
    CallExprArguReplVec.insert(InsertIter, InsertStr);
    // After this line, the old iterators of CallExprArguReplVec may be
    // invalidation. Need to use new iterators.

    CallExprReplStr = getFinalCallExprStr(Replacement) + CallExprReplStr;

    if (NeedUseLambda) {
      if (PrefixInsertStr.empty() && SuffixInsertStr.empty()) {
        NeedUseLambda = false;
      }
    }

    applyMigrationText(NeedUseLambda, IsMacroArg, CanAvoidBrace,
                       CanAvoidUsingLambda, OriginStmtType, IsAssigned,
                       OuterInsertLoc, PrefixInsertLoc, SuffixInsertLoc,
                       FuncNameBegin, FuncCallEnd, FuncCallLength, IndentStr,
                       PrefixInsertStr, SuffixInsertStr);
  } else if (FuncName == "cublasStrmm_v2" || FuncName == "cublasDtrmm_v2" ||
             FuncName == "cublasCtrmm_v2" || FuncName == "cublasZtrmm_v2") {
    std::string Replacement;
    BLASEnumInfo EnumInfo;
    std::string BufferType;
    if (FuncName == "cublasStrmm_v2" || FuncName == "cublasDtrmm_v2"){
      auto ReplInfoPair = MapNames::BLASFuncReplInfoMap.find(FuncName);
      auto ReplInfo = ReplInfoPair->second;
      Replacement = ReplInfo.ReplName;
      EnumInfo =
          BLASEnumInfo(ReplInfo.OperationIndexInfo, ReplInfo.FillModeIndexInfo,
                       ReplInfo.SideModeIndexInfo, ReplInfo.DiagTypeIndexInfo);
      BufferType = ReplInfo.BufferTypeInfo[0];
    } else {
      auto ReplInfoPair = MapNames::BLASFuncComplexReplInfoMap.find(FuncName);
      auto ReplInfo = ReplInfoPair->second;
      Replacement = ReplInfo.ReplName;
      EnumInfo =
          BLASEnumInfo(ReplInfo.OperationIndexInfo, ReplInfo.FillModeIndexInfo,
                       ReplInfo.SideModeIndexInfo, ReplInfo.DiagTypeIndexInfo);
      BufferType = ReplInfo.BufferTypeInfo[0];
    }

    if (HasDeviceAttr) {
      report(FuncNameBegin, Diagnostics::FUNCTION_CALL_IN_DEVICE, false,
             MapNames::ITFName.at(FuncName), Replacement);
      return;
    }

    // initialize the replacement of each arguments
    int ArgNum = CE->getNumArgs();
    for (int i = 0; i < ArgNum; ++i) {
      ExprAnalysis EA;
      EA.analyze(CE->getArg(i));
      CallExprArguReplVec.push_back(EA.getReplacedString());
    }

    // original API is like: C = A * B
    // migrated API is like: B = A * B
    // So first we need copy all arguments relative to B (including leading
    // dimension and data) to C, then remove all arguments relative to C in the
    // original call.
    // In order to call the memcpy API, if an expression will be used more than
    // twice and have side effect, it need be redecalred. In this API migration,
    // the ptrC, m, n and ldc may need redeclaration.

    // declare a temp variable for ptrC if it has side effect
    std::string PtrCName;
    if (CE->getArg(12)->HasSideEffects(DpctGlobalInfo::getContext())) {
      PtrCName = getTempNameForExpr(CE->getArg(12), true, true) + "ptr_ct" +
                 std::to_string(DpctGlobalInfo::getSuffixIndexInRuleThenInc());
      PrefixInsertStr = PrefixInsertStr + "auto " + PtrCName + " = " +
                        CallExprArguReplVec[12] + ";" + getNL() + IndentStr;
    } else {
      PtrCName = CallExprArguReplVec[12];
    }

    // declare temp variables for m, n and ldc if has side effect
    auto &Context = dpct::DpctGlobalInfo::getContext();
    std::string Arg5Repl, Arg6Repl, Arg13Repl;
    std::string TempArgsDecl;
    auto processTempVars = [&](const Expr *E, std::string Name,
                               std::string &ArgRepl) {
      if (E->HasSideEffects(Context)) {
        ArgRepl = getTempNameForExpr(E, true, true) + Name +
                  std::to_string(DpctGlobalInfo::getSuffixIndexInRuleThenInc());
        TempArgsDecl =
            TempArgsDecl + "auto " + ArgRepl + " = " + getStmtSpelling(E) + ";";
      } else {
        ArgRepl = getStmtSpelling(E);
      }
    };
    processTempVars(CE->getArg(5), "m_ct", Arg5Repl); // Arg5: m
    processTempVars(CE->getArg(6), "n_ct", Arg6Repl); // Arg6: n
    processTempVars(CE->getArg(13), "ld_ct", Arg13Repl); // Arg13: ldc
    CallExprArguReplVec[5] = Arg5Repl;
    CallExprArguReplVec[6] = Arg6Repl;
    CallExprArguReplVec[13] = Arg13Repl;
    if (!TempArgsDecl.empty())
      PrefixInsertStr = PrefixInsertStr + TempArgsDecl + getNL() + IndentStr;

    // generate the data memcpy API call
    PrefixInsertStr = PrefixInsertStr + "dpct::matrix_mem_copy(" + PtrCName +
                      ", " + CallExprArguReplVec[10] + ", " + Arg13Repl + ", " +
                      CallExprArguReplVec[11] + ", " + Arg5Repl + ", " +
                      Arg6Repl + ", dpct::device_to_device, *" +
                      CallExprArguReplVec[0] + ");" + getNL() + IndentStr;

    // update the replacement of four enmu arguments
    auto processEnumArgus = [&](const Expr *E, unsigned int Index,
                                std::string &Argu) {
      const CStyleCastExpr *CSCE = nullptr;
      if (CSCE = dyn_cast<CStyleCastExpr>(E)) {
        std::string CurrentArgumentRepl;
        processParamIntCastToBLASEnum(E, CSCE, Index, IndentStr, EnumInfo,
                                      PrefixInsertStr, CurrentArgumentRepl);
        Argu = CurrentArgumentRepl;
      }
    };
    processEnumArgus(CE->getArg(1), 1, CallExprArguReplVec[1]);
    processEnumArgus(CE->getArg(2), 2, CallExprArguReplVec[2]);
    processEnumArgus(CE->getArg(3), 3, CallExprArguReplVec[3]);
    processEnumArgus(CE->getArg(4), 4, CallExprArguReplVec[4]);

    // update the replacement of two buffers
    // the second buffer is constrcuted with PtrCName
    if (DpctGlobalInfo::getUsmLevel() == UsmLevel::none) {
      std::string BufferDecl;
      CallExprArguReplVec[8] = getBufferNameAndDeclStr(
          CE->getArg(8), BufferType, IndentStr, BufferDecl);
      PrefixInsertStr = PrefixInsertStr + BufferDecl;
      CallExprArguReplVec[12] =
          getBufferNameAndDeclStr(PtrCName, BufferType, IndentStr, BufferDecl);
      PrefixInsertStr = PrefixInsertStr + BufferDecl;
    } else {
      if (FuncName == "cublasCtrmm_v2") {
        CallExprArguReplVec[8] =
            "(std::complex<float>*)" + CallExprArguReplVec[8];
        CallExprArguReplVec[12] = "(std::complex<float>*)" + PtrCName;
      } else if (FuncName == "cublasZtrmm_v2") {
        CallExprArguReplVec[8] =
            "(std::complex<double>*)" + CallExprArguReplVec[8];
        CallExprArguReplVec[12] = "(std::complex<double>*)" + PtrCName;
      } else {
        CallExprArguReplVec[12] = PtrCName;
      }
    }

    // update the replacement of a scalar argument
    CallExprArguReplVec[7] = "dpct::get_value(" + CallExprArguReplVec[7] +
                             ", *" + CallExprArguReplVec[0] + ")";

    // Remove arguments ptrB and ldb from CallExprArguReplVec
    CallExprArguReplVec.erase(CallExprArguReplVec.begin() + 10,
                              CallExprArguReplVec.begin() + 12);
    // After this line, the old iterators of CallExprArguReplVec may be
    // invalidation. Need to use new iterators.

    CallExprReplStr = getFinalCallExprStr(Replacement) + CallExprReplStr;

    if (NeedUseLambda) {
      if (PrefixInsertStr.empty() && SuffixInsertStr.empty()) {
        NeedUseLambda = false;
      }
    }

    applyMigrationText(NeedUseLambda, IsMacroArg, CanAvoidBrace,
                       CanAvoidUsingLambda, OriginStmtType, IsAssigned,
                       OuterInsertLoc, PrefixInsertLoc, SuffixInsertLoc,
                       FuncNameBegin, FuncCallEnd, FuncCallLength, IndentStr,
                       PrefixInsertStr, SuffixInsertStr);
  } else if (FuncName == "cublasSgemmEx" || FuncName == "cublasCgemmEx") {
    std::string Replacement = "mkl::blas::gemm";
    if (HasDeviceAttr) {
      report(FuncNameBegin, Diagnostics::FUNCTION_CALL_IN_DEVICE, false,
             MapNames::ITFName.at(FuncName), Replacement);
      return;
    }

    auto AType = CE->getArg(8);
    auto BType = CE->getArg(11);
    auto CType = CE->getArg(15);

    Expr::EvalResult ATypeER, BTypeER, CTypeER;
    bool CanATypeBeEval =
        AType->EvaluateAsInt(ATypeER, DpctGlobalInfo::getContext());
    bool CanBTypeBeEval =
        BType->EvaluateAsInt(BTypeER, DpctGlobalInfo::getContext());
    bool CanCTypeBeEval =
        CType->EvaluateAsInt(CTypeER, DpctGlobalInfo::getContext());

    int64_t ABTypeValue, CTypeValue;
    if (CanCTypeBeEval && (CanATypeBeEval || CanBTypeBeEval)) {
      CTypeValue = CTypeER.Val.getInt().getExtValue();
      if (CanATypeBeEval)
        ABTypeValue = ATypeER.Val.getInt().getExtValue();
      else
        ABTypeValue = BTypeER.Val.getInt().getExtValue();
    } else {
      report(FuncNameBegin, Diagnostics::UNSUPPORT_PARAM_COMBINATION, false,
             MapNames::ITFName.at(FuncName),
             "not all values of parameters could be evaluated in migration");
      return;
    }

    MapNames::BLASGemmExTypeInfo TypeInfo;
    std::string Key = std::to_string(ABTypeValue) + ":" +
                      std::to_string(CTypeValue);
    if (MapNames::BLASTGemmExTypeInfoMap.find(Key) !=
        MapNames::BLASTGemmExTypeInfoMap.end()) {
      TypeInfo = MapNames::BLASTGemmExTypeInfoMap.find(Key)->second;
    } else {
      report(
          FuncNameBegin, Diagnostics::UNSUPPORT_PARAM_COMBINATION, false,
          MapNames::ITFName.at(FuncName),
          "the combination of matrix data type and scalar type is unsupported");
      return;
    }

    std::vector<int> ArgsIndex{0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 14, 16};
    // initialize the replacement of each arguments
    for (auto I : ArgsIndex) {
      ExprAnalysis EA;
      EA.analyze(CE->getArg(I));
      CallExprArguReplVec.push_back(EA.getReplacedString());
    }

    // update the replacement of four enmu arguments
    BLASEnumInfo EnumInfo = BLASEnumInfo({1, 2}, -1, -1, -1);
    auto processEnumArgus = [&](const Expr *E, unsigned int Index,
                                std::string &Argu) {
      const CStyleCastExpr *CSCE = nullptr;
      if (CSCE = dyn_cast<CStyleCastExpr>(E)) {
        std::string CurrentArgumentRepl;
        processParamIntCastToBLASEnum(E, CSCE, Index, IndentStr, EnumInfo,
                                      PrefixInsertStr, CurrentArgumentRepl);
        Argu = CurrentArgumentRepl;
      }
    };
    processEnumArgus(CE->getArg(1), 1, CallExprArguReplVec[1]);
    processEnumArgus(CE->getArg(2), 2, CallExprArguReplVec[2]);

    // update the replacement of three buffers
    if (DpctGlobalInfo::getUsmLevel() == UsmLevel::none) {
      std::string BufferDecl;
      CallExprArguReplVec[7] = getBufferNameAndDeclStr(
          CE->getArg(7), TypeInfo.ABType, IndentStr, BufferDecl);
      PrefixInsertStr = PrefixInsertStr + BufferDecl;
      CallExprArguReplVec[9] = getBufferNameAndDeclStr(
          CE->getArg(10), TypeInfo.ABType, IndentStr, BufferDecl);
      PrefixInsertStr = PrefixInsertStr + BufferDecl;
      CallExprArguReplVec[12] = getBufferNameAndDeclStr(
          CE->getArg(14), TypeInfo.CType, IndentStr, BufferDecl);
      PrefixInsertStr = PrefixInsertStr + BufferDecl;
    } else {
      CallExprArguReplVec[7] =
          "(" + TypeInfo.ABType + "*)" + CallExprArguReplVec[7];
      CallExprArguReplVec[9] =
          "(" + TypeInfo.ABType + "*)" + CallExprArguReplVec[9];
      CallExprArguReplVec[12] =
          "(" + TypeInfo.CType + "*)" + CallExprArguReplVec[12];
    }

    // update the replacement of two scalar arguments
    CallExprArguReplVec[6] = "dpct::get_value(" + CallExprArguReplVec[6] +
                             ", *" + CallExprArguReplVec[0] + ")";
    CallExprArguReplVec[11] = "dpct::get_value(" + CallExprArguReplVec[11] +
                              ", *" + CallExprArguReplVec[0] + ")";
    if (Key == "2:2") {
      CallExprArguReplVec[6] = MapNames::getClNamespace() + "::vec<float, 1>{" +
                               CallExprArguReplVec[6] + "}.convert<" +
                               MapNames::getClNamespace() + "::half, " +
                               MapNames::getClNamespace() +
                               "::rounding_mode::automatic>()[0]";
      CallExprArguReplVec[11] = MapNames::getClNamespace() +
                                "::vec<float, 1>{" + CallExprArguReplVec[11] +
                                "}.convert<" + MapNames::getClNamespace() +
                                "::half, " + MapNames::getClNamespace() +
                                "::rounding_mode::automatic>()[0]";
    }

    CallExprReplStr = getFinalCallExprStr(Replacement) + CallExprReplStr;

    if (NeedUseLambda) {
      if (PrefixInsertStr.empty() && SuffixInsertStr.empty()) {
        NeedUseLambda = false;
      }
    }
    applyMigrationText(NeedUseLambda, IsMacroArg, CanAvoidBrace,
                       CanAvoidUsingLambda, OriginStmtType, IsAssigned,
                       OuterInsertLoc, PrefixInsertLoc, SuffixInsertLoc,
                       FuncNameBegin, FuncCallEnd, FuncCallLength, IndentStr,
                       PrefixInsertStr, SuffixInsertStr);
  } else if (FuncName == "cublasGemmEx") {
    std::string Replacement = "mkl::blas::gemm";
     if (HasDeviceAttr) {
      report(FuncNameBegin, Diagnostics::FUNCTION_CALL_IN_DEVICE, false,
             MapNames::ITFName.at(FuncName), Replacement);
      return;
    }

// MKL API does not have computeType and algo parameters.
// computeType(alpha/beta)               AType/BType     CType           IsSupportInMKL
// CUDA_R_16F(2)/CUBLAS_COMPUTE_16F(64)  CUDA_R_16F(2)   CUDA_R_16F(2)   yes
// CUDA_R_32I(10)                        CUDA_R_8I(3)    CUDA_R_32I(10)  no
// CUDA_R_32F(0)/CUBLAS_COMPUTE_32F(68)  CUDA_R_16F(2)   CUDA_R_16F(2)   no (can cast alpha/beta to half)
// CUDA_R_32F(0)                         CUDA_R_8I(3)    CUDA_R_32I(10)  no
// CUDA_R_32F(0)/CUBLAS_COMPUTE_32F(68)  CUDA_R_16F(2)   CUDA_R_32F(0)   yes
// CUDA_R_32F(0)/CUBLAS_COMPUTE_32F(68)  CUDA_R_32F(0)   CUDA_R_32F(0)   yes
// CUDA_R_64F(1)/CUBLAS_COMPUTE_64F(70)  CUDA_R_64F(1)   CUDA_R_64F(1)   yes
// CUDA_C_32F(4)                         CUDA_C_8I(7)    CUDA_C_32F(4)   no
// CUDA_C_32F(4)                         CUDA_C_32F(4)   CUDA_C_32F(4)   yes
// CUDA_C_64F(5)                         CUDA_C_64F(5)   CUDA_C_64F(5)   yes

    auto AType = CE->getArg(8);
    auto BType = CE->getArg(11);
    auto CType = CE->getArg(15);
    auto ComputeType = CE->getArg(17);

    Expr::EvalResult ATypeER, BTypeER, CTypeER, ComputeTypeER;
    bool CanATypeBeEval =
        AType->EvaluateAsInt(ATypeER, DpctGlobalInfo::getContext());
    bool CanBTypeBeEval =
        BType->EvaluateAsInt(BTypeER, DpctGlobalInfo::getContext());
    bool CanCTypeBeEval =
        CType->EvaluateAsInt(CTypeER, DpctGlobalInfo::getContext());
    bool CanComputeTypeBeEval =
        ComputeType->EvaluateAsInt(ComputeTypeER, DpctGlobalInfo::getContext());

    int64_t ABTypeValue, CTypeValue, ComputeTypeValue;
    if (CanCTypeBeEval && CanComputeTypeBeEval &&
        (CanATypeBeEval || CanBTypeBeEval)) {
      CTypeValue = CTypeER.Val.getInt().getExtValue();
      ComputeTypeValue = ComputeTypeER.Val.getInt().getExtValue();
      if (CanATypeBeEval)
        ABTypeValue = ATypeER.Val.getInt().getExtValue();
      else
        ABTypeValue = BTypeER.Val.getInt().getExtValue();
    } else {
      report(FuncNameBegin, Diagnostics::UNSUPPORT_PARAM_COMBINATION, false,
             MapNames::ITFName.at(FuncName),
             "not all values of parameters could be evaluated in migration");
      return;
    }

    MapNames::BLASGemmExTypeInfo TypeInfo;
    std::string Key = std::to_string(ComputeTypeValue) + ":" +
                      std::to_string(ABTypeValue) + ":" +
                      std::to_string(CTypeValue);
    if(MapNames::BLASGemmExTypeInfoMap.find(Key) !=
        MapNames::BLASGemmExTypeInfoMap.end()){
      TypeInfo = MapNames::BLASGemmExTypeInfoMap.find(Key)->second;
    } else {
      report(
          FuncNameBegin, Diagnostics::UNSUPPORT_PARAM_COMBINATION, false,
          MapNames::ITFName.at(FuncName),
          "the combination of matrix data type and scalar type is unsupported");
      return;
    }

    std::vector<int> ArgsIndex{0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 14, 16};
    // initialize the replacement of each arguments
    for (auto I : ArgsIndex) {
      ExprAnalysis EA;
      EA.analyze(CE->getArg(I));
      CallExprArguReplVec.push_back(EA.getReplacedString());
    }

    // update the replacement of four enmu arguments
    BLASEnumInfo EnumInfo = BLASEnumInfo({1, 2}, -1, -1, -1);
    auto processEnumArgus = [&](const Expr *E, unsigned int Index,
                                std::string &Argu) {
      const CStyleCastExpr *CSCE = nullptr;
      if (CSCE = dyn_cast<CStyleCastExpr>(E)) {
        std::string CurrentArgumentRepl;
        processParamIntCastToBLASEnum(E, CSCE, Index, IndentStr, EnumInfo,
                                      PrefixInsertStr, CurrentArgumentRepl);
        Argu = CurrentArgumentRepl;
      }
    };
    processEnumArgus(CE->getArg(1), 1, CallExprArguReplVec[1]);
    processEnumArgus(CE->getArg(2), 2, CallExprArguReplVec[2]);

    // update the replacement of three buffers
    if (DpctGlobalInfo::getUsmLevel() == UsmLevel::none) {
      std::string BufferDecl;
      CallExprArguReplVec[7] = getBufferNameAndDeclStr(
          CE->getArg(7), TypeInfo.ABType, IndentStr, BufferDecl);
      PrefixInsertStr = PrefixInsertStr + BufferDecl;
      CallExprArguReplVec[9] = getBufferNameAndDeclStr(
          CE->getArg(10), TypeInfo.ABType, IndentStr, BufferDecl);
      PrefixInsertStr = PrefixInsertStr + BufferDecl;
      CallExprArguReplVec[12] = getBufferNameAndDeclStr(
          CE->getArg(14), TypeInfo.CType, IndentStr, BufferDecl);
      PrefixInsertStr = PrefixInsertStr + BufferDecl;
    } else {
      CallExprArguReplVec[7] =
          "(" + TypeInfo.ABType + "*)" + CallExprArguReplVec[7];
      CallExprArguReplVec[9] =
          "(" + TypeInfo.ABType + "*)" + CallExprArguReplVec[9];
      CallExprArguReplVec[12] =
          "(" + TypeInfo.CType + "*)" + CallExprArguReplVec[12];
    }

    // update the replacement of two scalar arguments
    CallExprArguReplVec[6] = "dpct::get_value((" + TypeInfo.OriginScalarType +
                             "*)" + CallExprArguReplVec[6] + ", *" +
                             CallExprArguReplVec[0] + ")";
    CallExprArguReplVec[11] = "dpct::get_value((" + TypeInfo.OriginScalarType +
                              "*)" + CallExprArguReplVec[11] + ", *" +
                              CallExprArguReplVec[0] + ")";
    if (Key == "0:2:2" || Key == "68:2:2") {
      CallExprArguReplVec[6] = MapNames::getClNamespace() + "::vec<float, 1>{" +
                               CallExprArguReplVec[6] + "}.convert<" +
                               MapNames::getClNamespace() + "::half, " +
                               MapNames::getClNamespace() +
                               "::rounding_mode::automatic>()[0]";
      CallExprArguReplVec[11] = MapNames::getClNamespace() +
                                "::vec<float, 1>{" + CallExprArguReplVec[11] +
                                "}.convert<" + MapNames::getClNamespace() +
                                "::half, " + MapNames::getClNamespace() +
                                "::rounding_mode::automatic>()[0]";
    }

    CallExprReplStr = getFinalCallExprStr(Replacement) + CallExprReplStr;

    if (NeedUseLambda) {
      if (PrefixInsertStr.empty() && SuffixInsertStr.empty()) {
        NeedUseLambda = false;
      }
    }
    applyMigrationText(NeedUseLambda, IsMacroArg, CanAvoidBrace,
                       CanAvoidUsingLambda, OriginStmtType, IsAssigned,
                       OuterInsertLoc, PrefixInsertLoc, SuffixInsertLoc,
                       FuncNameBegin, FuncCallEnd, FuncCallLength, IndentStr,
                       PrefixInsertStr, SuffixInsertStr);
  } else if (MapNames::BLASFuncReplInfoMap.find(FuncName) !=
      MapNames::BLASFuncReplInfoMap.end()) {
    auto ReplInfoPair = MapNames::BLASFuncReplInfoMap.find(FuncName);
    MapNames::BLASFuncReplInfo ReplInfo = ReplInfoPair->second;
    std::string Replacement = ReplInfo.ReplName;
    BLASEnumInfo EnumInfo(
        ReplInfo.OperationIndexInfo, ReplInfo.FillModeIndexInfo,
        ReplInfo.SideModeIndexInfo, ReplInfo.DiagTypeIndexInfo);
    if (HasDeviceAttr) {
      report(FuncNameBegin, Diagnostics::FUNCTION_CALL_IN_DEVICE, false,
             MapNames::ITFName.at(FuncName), Replacement);
      return;
    }
    int ArgNum = CE->getNumArgs();
    for (int i = 0; i < ArgNum; ++i) {
      int IndexTemp = -1;
      std::string CurrentArgumentRepl;
      const CStyleCastExpr *CSCE = nullptr;
      if (isReplIndex(i, ReplInfo.BufferIndexInfo, IndexTemp)) {
        if (DpctGlobalInfo::getUsmLevel() == UsmLevel::restricted) {
          if (ReplInfo.BufferTypeInfo[IndexTemp] == "int") {
            std::string ResultTempPtr =
                "res_temp_ptr_ct" +
                std::to_string(DpctGlobalInfo::getSuffixIndexInRuleThenInc());
            PrefixInsertStr =
                PrefixInsertStr + "int64_t* " + ResultTempPtr + " = " +
                MapNames::getClNamespace() + "::malloc_shared<int64_t>(" +
                "1, dpct::get_default_queue());" + getNL() + IndentStr;
            SuffixInsertStr = SuffixInsertStr + getNL() + IndentStr + "*" +
                              getStmtSpelling(CE->getArg(i)) + " = (int)*" +
                              ResultTempPtr + ";" + getNL() + IndentStr +
                              MapNames::getClNamespace() + "::free(" +
                              ResultTempPtr + ", dpct::get_default_queue());";
            CurrentArgumentRepl = ResultTempPtr;
          } else {
            CurrentArgumentRepl = getStmtSpelling(CE->getArg(i));
          }
        } else {
          std::string BufferDecl = "";
          std::string BufferName = "";
          BufferName = getBufferNameAndDeclStr(
              CE->getArg(i), ReplInfo.BufferTypeInfo[IndexTemp], IndentStr,
              BufferDecl);
          PrefixInsertStr = PrefixInsertStr + BufferDecl;

          if (ReplInfo.BufferTypeInfo[IndexTemp] == "int") {
            std::string ResultTempBuf =
                "res_temp_buf_ct" +
                std::to_string(DpctGlobalInfo::getSuffixIndexInRuleThenInc());
            PrefixInsertStr = PrefixInsertStr + MapNames::getClNamespace() +
                              "::buffer<int64_t> " + ResultTempBuf + "(" +
                              MapNames::getClNamespace() + "::range<1>(1));" +
                              getNL() + IndentStr;
            SuffixInsertStr =
                SuffixInsertStr + getNL() + IndentStr + BufferName +
                ".get_access<" + MapNames::getClNamespace() +
                "::access::mode::" + "write>()[0] = (int)" + ResultTempBuf +
                "." + "get_access<" + MapNames::getClNamespace() +
                "::access::mode::read>()[0];";
            CurrentArgumentRepl = ResultTempBuf;
          } else {
            CurrentArgumentRepl = BufferName;
          }

        }
      } else if (isReplIndex(i, ReplInfo.PointerIndexInfo, IndexTemp)) {
        ExprAnalysis EA(CE->getArg(i));
        CurrentArgumentRepl = "dpct::get_value(" + EA.getReplacedString() +
                              ", *" + CallExprArguReplVec[0] + ")";
      } else if ((CSCE = dyn_cast<CStyleCastExpr>(CE->getArg(i)))) {
        processParamIntCastToBLASEnum(CE->getArg(i), CSCE, i, IndentStr,
                                      EnumInfo, PrefixInsertStr,
                                      CurrentArgumentRepl);
      } else {
        ExprAnalysis EA;
        EA.analyze(CE->getArg(i));
        CurrentArgumentRepl = EA.getReplacedString();
      }
      CallExprArguReplVec.push_back(CurrentArgumentRepl);
    }

    if (DpctGlobalInfo::getUsmLevel() == UsmLevel::restricted) {
      if (FuncName == "cublasSrotm_v2") {
        CallExprArguReplVec[6] =
            "const_cast<float*>(" + CallExprArguReplVec[6] + ")";
      } else if (FuncName == "cublasDrotm_v2") {
        CallExprArguReplVec[6] =
            "const_cast<double*>(" + CallExprArguReplVec[6] + ")";
      }
      addWait(FuncName, CE, SuffixInsertStr, IndentStr);
    }
    CallExprReplStr = getFinalCallExprStr(Replacement) + CallExprReplStr;


    if (NeedUseLambda) {
      if (PrefixInsertStr.empty() && SuffixInsertStr.empty()) {
        // If there is one API call in the migrted code, it is unnecessary to
        // use a lambda expression
        NeedUseLambda = false;
      }
    }

    applyMigrationText(NeedUseLambda, IsMacroArg, CanAvoidBrace,
                       CanAvoidUsingLambda, OriginStmtType, IsAssigned,
                       OuterInsertLoc, PrefixInsertLoc, SuffixInsertLoc,
                       FuncNameBegin, FuncCallEnd, FuncCallLength, IndentStr,
                       PrefixInsertStr, SuffixInsertStr);
  } else if (MapNames::BLASFuncComplexReplInfoMap.find(FuncName) !=
             MapNames::BLASFuncComplexReplInfoMap.end()) {
    auto ReplInfoPair = MapNames::BLASFuncComplexReplInfoMap.find(FuncName);
    MapNames::BLASFuncComplexReplInfo ReplInfo = ReplInfoPair->second;
    std::string Replacement = ReplInfo.ReplName;
    BLASEnumInfo EnumInfo(
        ReplInfo.OperationIndexInfo, ReplInfo.FillModeIndexInfo,
        ReplInfo.SideModeIndexInfo, ReplInfo.DiagTypeIndexInfo);
    if (HasDeviceAttr) {
      report(FuncNameBegin, Diagnostics::FUNCTION_CALL_IN_DEVICE, false,
             MapNames::ITFName.at(FuncName), Replacement);
      return;
    }

    int ArgNum = CE->getNumArgs();

    for (int i = 0; i < ArgNum; ++i) {
      int IndexTemp = -1;
      std::string CurrentArgumentRepl;
      const CStyleCastExpr *CSCE = nullptr;
      if (isReplIndex(i, ReplInfo.BufferIndexInfo, IndexTemp)) {
        if (DpctGlobalInfo::getUsmLevel() == UsmLevel::restricted) {
          if (ReplInfo.BufferTypeInfo[IndexTemp] == "int") {
            std::string ResultTempPtr =
                "res_temp_ptr_ct" +
                std::to_string(DpctGlobalInfo::getSuffixIndexInRuleThenInc());
            PrefixInsertStr =
                PrefixInsertStr + "int64_t* " + ResultTempPtr + " = " +
                MapNames::getClNamespace() + "::malloc_shared<int64_t>(" +
                "1, dpct::get_default_queue());" + getNL() + IndentStr;
            SuffixInsertStr = SuffixInsertStr + getNL() + IndentStr + "*" +
                              getStmtSpelling(CE->getArg(i)) + " = (int)*" +
                              ResultTempPtr + ";" + getNL() + IndentStr +
                              MapNames::getClNamespace() + "::free(" +
                              ResultTempPtr + ", dpct::get_default_queue());";
            CurrentArgumentRepl = ResultTempPtr;
          } else if (ReplInfo.BufferTypeInfo[IndexTemp] ==
                         "std::complex<float>" ||
                     ReplInfo.BufferTypeInfo[IndexTemp] ==
                         "std::complex<double>") {
            CurrentArgumentRepl = "(" + ReplInfo.BufferTypeInfo[IndexTemp] +
                                  "*)" + getStmtSpelling(CE->getArg(i));
          } else {
            CurrentArgumentRepl = getStmtSpelling(CE->getArg(i));
          }
        } else {
          std::string BufferDecl = "";
          std::string BufferName = "";
          BufferName = getBufferNameAndDeclStr(
              CE->getArg(i), ReplInfo.BufferTypeInfo[IndexTemp], IndentStr,
              BufferDecl);
          PrefixInsertStr = PrefixInsertStr + BufferDecl;

          if (ReplInfo.BufferTypeInfo[IndexTemp] == "int") {
            std::string ResultTempBuf =
                "res_temp_buf_ct" +
                std::to_string(DpctGlobalInfo::getSuffixIndexInRuleThenInc());
            PrefixInsertStr = PrefixInsertStr + MapNames::getClNamespace() +
                              "::buffer<int64_t> " + ResultTempBuf + "(" +
                              MapNames::getClNamespace() + "::range<1>(1));" +
                              getNL() + IndentStr;
            SuffixInsertStr =
                SuffixInsertStr + getNL() + IndentStr + BufferName +
                ".get_access<" + MapNames::getClNamespace() +
                "::access::mode::" + "write>()[0] = (int)" + ResultTempBuf +
                "." +
                "get_access<" + MapNames::getClNamespace() +
                "::access::mode::read>()[0];";
            CurrentArgumentRepl = ResultTempBuf;
          } else {
            CurrentArgumentRepl = BufferName;
          }
        }
      } else if (isReplIndex(i, ReplInfo.PointerIndexInfo, IndexTemp)) {
        ExprAnalysis EA(CE->getArg(i));
        CurrentArgumentRepl = "dpct::get_value(" + EA.getReplacedString() +
                              ", *" + CallExprArguReplVec[0] + ")";
      } else if ((CSCE = dyn_cast<CStyleCastExpr>(CE->getArg(i)))) {
        processParamIntCastToBLASEnum(CE->getArg(i), CSCE, i, IndentStr,
                                      EnumInfo, PrefixInsertStr,
                                      CurrentArgumentRepl);
      } else {
        ExprAnalysis EA;
        EA.analyze(CE->getArg(i));
        CurrentArgumentRepl = EA.getReplacedString();
      }

      CallExprArguReplVec.push_back(CurrentArgumentRepl);
    }

    if (DpctGlobalInfo::getUsmLevel() == UsmLevel::restricted) {
      addWait(FuncName, CE, SuffixInsertStr, IndentStr);
    }

    CallExprReplStr = getFinalCallExprStr(Replacement) + CallExprReplStr;

    if (NeedUseLambda) {
      if (PrefixInsertStr.empty() && SuffixInsertStr.empty()) {
        // If there is one API call in the migrted code, it is unnecessary to
        // use a lambda expression
        NeedUseLambda = false;
      }
    }

    applyMigrationText(NeedUseLambda, IsMacroArg, CanAvoidBrace,
                       CanAvoidUsingLambda, OriginStmtType, IsAssigned,
                       OuterInsertLoc, PrefixInsertLoc, SuffixInsertLoc,
                       FuncNameBegin, FuncCallEnd, FuncCallLength, IndentStr,
                       PrefixInsertStr, SuffixInsertStr);
  } else if (MapNames::LegacyBLASFuncReplInfoMap.find(FuncName) !=
             MapNames::LegacyBLASFuncReplInfoMap.end()) {
    auto ReplInfoPair = MapNames::LegacyBLASFuncReplInfoMap.find(FuncName);
    MapNames::BLASFuncComplexReplInfo ReplInfo = ReplInfoPair->second;
    CallExprReplStr = CallExprReplStr + ReplInfo.ReplName +
                      "(*dpct::get_current_device().get_saved_queue()";
    std::string IndentStr =
        getIndent(PrefixInsertLoc, (Result.Context)->getSourceManager()).str();

    std::string VarType;
    std::string VarName;
    std::string DeclOutOfBrace;
    const VarDecl *VD = 0;
    if (IsInitializeVarDecl) {
      VD = getAncestralVarDecl(CE);
      if (VD) {
        VarType = VD->getType().getAsString();
        if (VarType == "cuComplex") {
          VarType = MapNames::getClNamespace() + "::float2";
        }
        if (VarType == "cuDoubleComplex") {
          VarType = MapNames::getClNamespace() + "::double2";
        }
        VarName = VD->getNameAsString();
      } else {
        assert(0 && "Fail to get VarDecl information");
        return;
      }
      DeclOutOfBrace = VarType + " " + VarName + ";" + getNL() + IndentStr;
    }
    std::vector<std::string> ParamsStrsVec =
        getParamsAsStrs(CE, *(Result.Context));
    int ArgNum = CE->getNumArgs();
    for (int i = 0; i < ArgNum; ++i) {
      int IndexTemp = -1;
      if (isReplIndex(i, ReplInfo.BufferIndexInfo, IndexTemp)) {
        if (DpctGlobalInfo::getUsmLevel() == UsmLevel::restricted) {
          if ((FuncName == "cublasSrotm" || FuncName == "cublasDrotm") &&
              i == 5) {
            CallExprReplStr = CallExprReplStr + ", const_cast<" +
                              ReplInfo.BufferTypeInfo[IndexTemp] + "*>(" +
                              getStmtSpelling(CE->getArg(5)) + ")";
          } else if (ReplInfo.BufferTypeInfo[IndexTemp] ==
                         "std::complex<float>" ||
                     ReplInfo.BufferTypeInfo[IndexTemp] ==
                         "std::complex<double>") {
            CallExprReplStr = CallExprReplStr + ", (" +
                              ReplInfo.BufferTypeInfo[IndexTemp] + "*)" +
                              ParamsStrsVec[i];
          } else {
            CallExprReplStr = CallExprReplStr + ", " + ParamsStrsVec[i];
          }
        } else {
          std::string BufferDecl;
          std::string BufferName = getBufferNameAndDeclStr(
              CE->getArg(i), ReplInfo.BufferTypeInfo[IndexTemp], IndentStr,
              BufferDecl);
          CallExprReplStr = CallExprReplStr + ", " + BufferName;
          PrefixInsertStr = PrefixInsertStr + BufferDecl;
        }
      } else if (isReplIndex(i, ReplInfo.PointerIndexInfo, IndexTemp)) {
        if (ReplInfo.PointerTypeInfo[IndexTemp] == "float" ||
            ReplInfo.PointerTypeInfo[IndexTemp] == "double") {
          // This code path is only for legacy cublasSrotmg and cublasDrotmg
          if (isSimpleAddrOf(CE->getArg(i)->IgnoreImplicit())) {
            CallExprReplStr =
                CallExprReplStr + ", " +
                getNameStrRemovedAddrOf(CE->getArg(i)->IgnoreImplicit());
          } else if (isCOCESimpleAddrOf(CE->getArg(i)->IgnoreImplicit())) {
            CallExprReplStr =
                CallExprReplStr + ", " +
                getNameStrRemovedAddrOf(CE->getArg(i)->IgnoreImplicit(), true);
          } else {
            CallExprReplStr = CallExprReplStr + ", *(" + ParamsStrsVec[i] + ")";
          }
        } else {
          CallExprReplStr =
              CallExprReplStr + ", " + ReplInfo.PointerTypeInfo[IndexTemp] +
              "((" + ParamsStrsVec[i] + ").x(),(" + ParamsStrsVec[i] + ").y())";
        }
      } else if (isReplIndex(i, ReplInfo.OperationIndexInfo, IndexTemp)) {
        Expr::EvalResult ER;
        if (CE->getArg(i)->EvaluateAsInt(ER, *Result.Context) &&
            !CE->getArg(i)->getBeginLoc().isMacroID()) {
          int64_t Value = ER.Val.getInt().getExtValue();
          if (Value == 'N' || Value == 'n') {
            CallExprReplStr = CallExprReplStr + ", mkl::transpose::nontrans";
          } else if (Value == 'T' || Value == 't') {
            CallExprReplStr = CallExprReplStr + ", mkl::transpose::trans";
          } else {
            CallExprReplStr = CallExprReplStr + ", mkl::transpose::conjtrans";
          }
        } else {
          std::string TransParamName;
          if (CE->getArg(i)->HasSideEffects(DpctGlobalInfo::getContext())) {
            TransParamName =
                "transpose_ct" +
                std::to_string(DpctGlobalInfo::getSuffixIndexInRuleThenInc());
            PrefixInsertStr = PrefixInsertStr + "auto " + TransParamName +
                              " = " + ParamsStrsVec[i] + ";" + getNL() +
                              IndentStr;
          } else {
            TransParamName = ParamsStrsVec[i];
          }
          CallExprReplStr = CallExprReplStr + ", " + "(" + TransParamName +
                            "=='N'||" + TransParamName +
                            "=='n') ? mkl::transpose::nontrans: ((" +
                            TransParamName + "=='T'||" + TransParamName +
                            "=='t') ? mkl::transpose::"
                            "trans : mkl::transpose::conjtrans)";
        }
      } else if (ReplInfo.FillModeIndexInfo == i) {
        Expr::EvalResult ER;
        if (CE->getArg(i)->EvaluateAsInt(ER, *Result.Context) &&
            !CE->getArg(i)->getBeginLoc().isMacroID()) {
          int64_t Value = ER.Val.getInt().getExtValue();
          if (Value == 'U' || Value == 'u') {
            CallExprReplStr = CallExprReplStr + ", mkl::uplo::upper";
          } else {
            CallExprReplStr = CallExprReplStr + ", mkl::uplo::lower";
          }
        } else {
          std::string FillParamName;
          if (CE->getArg(i)->HasSideEffects(DpctGlobalInfo::getContext())) {
            FillParamName =
                "fillmode_ct" +
                std::to_string(DpctGlobalInfo::getSuffixIndexInRuleThenInc());
            PrefixInsertStr = PrefixInsertStr + "auto " + FillParamName +
                              " = " + ParamsStrsVec[i] + ";" + getNL() +
                              IndentStr;
          } else {
            FillParamName = ParamsStrsVec[i];
          }
          CallExprReplStr = CallExprReplStr + ", " + "(" + FillParamName +
                            "=='L'||" + FillParamName +
                            "=='l') ? mkl::uplo::lower : mkl::uplo::upper";
        }
      } else if (ReplInfo.SideModeIndexInfo == i) {
        Expr::EvalResult ER;
        if (CE->getArg(i)->EvaluateAsInt(ER, *Result.Context) &&
            !CE->getArg(i)->getBeginLoc().isMacroID()) {
          int64_t Value = ER.Val.getInt().getExtValue();
          if (Value == 'L' || Value == 'l') {
            CallExprReplStr = CallExprReplStr + ", mkl::side::left";
          } else {
            CallExprReplStr = CallExprReplStr + ", mkl::side::right";
          }
        } else {
          std::string SideParamName;
          if (CE->getArg(i)->HasSideEffects(DpctGlobalInfo::getContext())) {
            SideParamName =
                "sidemode_ct" +
                std::to_string(DpctGlobalInfo::getSuffixIndexInRuleThenInc());
            PrefixInsertStr = PrefixInsertStr + "auto " + SideParamName +
                              " = " + ParamsStrsVec[i] + ";" + getNL() +
                              IndentStr;
          } else {
            SideParamName = ParamsStrsVec[i];
          }
          CallExprReplStr = CallExprReplStr + ", " + "(" + SideParamName +
                            "=='L'||" + SideParamName +
                            "=='l') ? mkl::side::left : mkl::side::right";
        }
      } else if (ReplInfo.DiagTypeIndexInfo == i) {
        Expr::EvalResult ER;
        if (CE->getArg(i)->EvaluateAsInt(ER, *Result.Context) &&
            !CE->getArg(i)->getBeginLoc().isMacroID()) {
          int64_t Value = ER.Val.getInt().getExtValue();
          if (Value == 'N' || Value == 'n') {
            CallExprReplStr = CallExprReplStr + ", mkl::diag::nonunit";
          } else {
            CallExprReplStr = CallExprReplStr + ", mkl::diag::unit";
          }
        } else {
          std::string DiagParamName;
          if (CE->getArg(i)->HasSideEffects(DpctGlobalInfo::getContext())) {
            DiagParamName =
                "diagtype_ct" +
                std::to_string(DpctGlobalInfo::getSuffixIndexInRuleThenInc());
            PrefixInsertStr = PrefixInsertStr + "auto " + DiagParamName +
                              " = " + ParamsStrsVec[i] + ";" + getNL() +
                              IndentStr;
          } else {
            DiagParamName = ParamsStrsVec[i];
          }
          CallExprReplStr = CallExprReplStr + ", " + "(" + DiagParamName +
                            "=='N'||" + DiagParamName +
                            "=='n') ? mkl::diag::nonunit : mkl::diag::unit";
        }
      } else {
        CallExprReplStr = CallExprReplStr + ", " + ParamsStrsVec[i];
      }
    }

    // All legacy APIs are synchronous
    if (FuncName == "cublasSnrm2" || FuncName == "cublasDnrm2" ||
        FuncName == "cublasScnrm2" || FuncName == "cublasDznrm2" ||
        FuncName == "cublasSdot" || FuncName == "cublasDdot" ||
        FuncName == "cublasCdotu" || FuncName == "cublasCdotc" ||
        FuncName == "cublasZdotu" || FuncName == "cublasZdotc" ||
        FuncName == "cublasIsamax" || FuncName == "cublasIdamax" ||
        FuncName == "cublasIcamax" || FuncName == "cublasIzamax" ||
        FuncName == "cublasIsamin" || FuncName == "cublasIdamin" ||
        FuncName == "cublasIcamin" || FuncName == "cublasIzamin" ||
        FuncName == "cublasSasum" || FuncName == "cublasDasum" ||
        FuncName == "cublasScasum" || FuncName == "cublasDzasum") {
      // APIs which have return value
      std::string ResultTempPtr =
          "res_temp_ptr_ct" +
          std::to_string(DpctGlobalInfo::getSuffixIndexInRuleThenInc());
      std::string ResultTempVal =
          "res_temp_val_ct" +
          std::to_string(DpctGlobalInfo::getSuffixIndexInRuleThenInc());
      std::string ResultTempBuf =
          "res_temp_buf_ct" +
          std::to_string(DpctGlobalInfo::getSuffixIndexInRuleThenInc());
      std::string ResultType =
          ReplInfo.BufferTypeInfo[ReplInfo.BufferTypeInfo.size() - 1];
      std::string ReturnValueParamsStr;
      if (DpctGlobalInfo::getUsmLevel() == UsmLevel::restricted) {
        PrefixInsertStr = PrefixInsertStr + ResultType + "* " + ResultTempPtr +
                          " = " + MapNames::getClNamespace() +
                          "::malloc_shared<" + ResultType +
                          ">(1, dpct::get_default_queue());" + getNL() +
                          IndentStr + CallExprReplStr + ", " + ResultTempPtr +
                          ").wait();" + getNL() + IndentStr;

        ReturnValueParamsStr =
            "(" + ResultTempPtr + "->real(), " + ResultTempPtr + "->imag())";

        if (NeedUseLambda) {
          PrefixInsertStr = PrefixInsertStr + ResultType + " " + ResultTempVal +
                            " = *" + ResultTempPtr + ";" + getNL() + IndentStr +
                            MapNames::getClNamespace() + "::free(" +
                            ResultTempPtr + ", dpct::get_default_queue());" +
                            getNL() + IndentStr;
          ReturnValueParamsStr =
              "(" + ResultTempVal + ".real(), " + ResultTempVal + ".imag())";
        } else {
          SuffixInsertStr = SuffixInsertStr + getNL() + IndentStr +
                            MapNames::getClNamespace() + "::free(" +
                            ResultTempPtr + ", dpct::get_default_queue());";
        }
      } else {
        PrefixInsertStr = PrefixInsertStr + MapNames::getClNamespace() +
                          "::buffer<" + ResultType + "> " + ResultTempBuf +
                          "(" +
                          MapNames::getClNamespace() + "::range<1>(1));" +
                          getNL() + IndentStr + CallExprReplStr + ", " +
                          ResultTempBuf + ");" + getNL() + IndentStr;
        ReturnValueParamsStr =
            "(" + ResultTempBuf + ".get_access<" + MapNames::getClNamespace() +
            "::access::mode::read>()[0].real(), " + ResultTempBuf +
            ".get_access<" + MapNames::getClNamespace() +
            "::access::mode::read>()[0].imag())";
      }

      std::string Repl;
      if (FuncName == "cublasCdotu" || FuncName == "cublasCdotc") {
        Repl = MapNames::getClNamespace() + "::float2" + ReturnValueParamsStr;
      } else if (FuncName == "cublasZdotu" || FuncName == "cublasZdotc") {
        Repl = MapNames::getClNamespace() + "::double2" + ReturnValueParamsStr;
      } else {
        if (DpctGlobalInfo::getUsmLevel() == UsmLevel::restricted) {
          if (NeedUseLambda)
            Repl = ResultTempVal;
          else
            Repl = "*" + ResultTempPtr;
        } else {
          Repl = ResultTempBuf + ".get_access<" + MapNames::getClNamespace() +
                 "::access::mode::read>()[0]";
        }
      }
      if (NeedUseLambda) {
        std::string CallRepl = "return " + Repl;

        insertAroundRange(PrefixInsertLoc, SuffixInsertLoc,
                          std::string("[&](){") + getNL() + IndentStr +
                              PrefixInsertStr,
                          ";" + SuffixInsertStr + getNL() + IndentStr + "}()");
        emplaceTransformation(new ReplaceStmt(CE, CallRepl));
      } else {
        if (DpctGlobalInfo::getUsmLevel() == UsmLevel::none)
          insertAroundRange(PrefixInsertLoc, SuffixInsertLoc,
                            DeclOutOfBrace + "{" + getNL() + IndentStr +
                                PrefixInsertStr,
                            SuffixInsertStr + getNL() + IndentStr + "}");
        else
          insertAroundRange(PrefixInsertLoc, SuffixInsertLoc,
                            DeclOutOfBrace + PrefixInsertStr,
                            std::move(SuffixInsertStr));

        if (IsInitializeVarDecl) {
          auto ParentNodes = (Result.Context)->getParents(*VD);
          const DeclStmt *DS = 0;
          if ((DS = ParentNodes[0].get<DeclStmt>())) {
            emplaceTransformation(
                new ReplaceStmt(DS, VarName + " = " + Repl + ";"));
          } else {
            assert(0 && "Fail to get Var Decl Stmt");
            return;
          }
        } else {
          emplaceTransformation(new ReplaceStmt(CE, Repl));
        }
      }
    } else {
      // APIs which haven't return value
      if (NeedUseLambda) {
        if (DpctGlobalInfo::getUsmLevel() == UsmLevel::restricted) {
          CallExprReplStr = CallExprReplStr + ").wait()";
        } else {
          CallExprReplStr = CallExprReplStr + ")";
        }
        if (CanAvoidUsingLambda) {
          std::string InsertStr;
          if (DpctGlobalInfo::getUsmLevel() == UsmLevel::none)
            InsertStr = DeclOutOfBrace + "{" + getNL() + IndentStr +
                        PrefixInsertStr + CallExprReplStr + ";" +
                        SuffixInsertStr + getNL() + IndentStr + "}" + getNL() +
                        IndentStr;
          else
            InsertStr = DeclOutOfBrace + PrefixInsertStr + CallExprReplStr +
                        ";" + SuffixInsertStr + getNL() + IndentStr;
          emplaceTransformation(
              new InsertText(OuterInsertLoc, std::move(InsertStr)));
          // APIs in this code path haven't return value, so remove the CallExpr
          emplaceTransformation(
              new ReplaceText(FuncNameBegin, FuncCallLength, ""));
        } else {
          emplaceTransformation(
              new ReplaceStmt(CE, std::move(CallExprReplStr)));
          insertAroundRange(PrefixInsertLoc, SuffixInsertLoc,
                            std::string("[&](){") + getNL() + IndentStr +
                                PrefixInsertStr,
                            getNL() + IndentStr + std::string("}()"));
        }
      } else {
        if (DpctGlobalInfo::getUsmLevel() == UsmLevel::restricted) {
          CallExprReplStr = CallExprReplStr + ").wait()";
        } else {
          CallExprReplStr = CallExprReplStr + ")";
        }
        emplaceTransformation(new ReplaceStmt(CE, std::move(CallExprReplStr)));
        if (!PrefixInsertStr.empty()) {
          if (DpctGlobalInfo::getUsmLevel() == UsmLevel::none)
            insertAroundRange(PrefixInsertLoc, SuffixInsertLoc,
                              std::string("{") + getNL() + IndentStr +
                                  PrefixInsertStr,
                              getNL() + IndentStr + std::string("}"));
          else
            insertAroundRange(PrefixInsertLoc, SuffixInsertLoc,
                              std::move(PrefixInsertStr), "");
        }
      }
    }
  } else if (MapNames::BLASFuncWrapperReplInfoMap.find(FuncName) !=
             MapNames::BLASFuncWrapperReplInfoMap.end()) {
    auto ReplInfoPair = MapNames::BLASFuncWrapperReplInfoMap.find(FuncName);
    MapNames::BLASFuncReplInfo ReplInfo = ReplInfoPair->second;
    std::string Replacement = ReplInfo.ReplName;
    BLASEnumInfo EnumInfo(
        ReplInfo.OperationIndexInfo, ReplInfo.FillModeIndexInfo,
        ReplInfo.SideModeIndexInfo, ReplInfo.DiagTypeIndexInfo);
    if (HasDeviceAttr) {
      report(FuncNameBegin, Diagnostics::FUNCTION_CALL_IN_DEVICE, false,
             MapNames::ITFName.at(FuncName), Replacement);
      return;
    }

    StringRef FuncNameRef(FuncName);
    if (FuncNameRef.endswith("getrfBatched")) {
      report(FuncNameBegin, Diagnostics::DIFFERENT_LU_FACTORIZATION, false,
             getStmtSpelling(CE->getArg(4)), Replacement,
             MapNames::ITFName.at(FuncName));
    }

    int ArgNum = CE->getNumArgs();
    for (int i = 0; i < ArgNum; ++i) {
      const CStyleCastExpr *CSCE = nullptr;
      std::string CurrentArgumentRepl;
      if ((CSCE = dyn_cast<CStyleCastExpr>(CE->getArg(i)))) {
        processParamIntCastToBLASEnum(CE->getArg(i), CSCE, i, IndentStr,
                                      EnumInfo, PrefixInsertStr,
                                      CurrentArgumentRepl);
      } else {
        ExprAnalysis EA;
        EA.analyze(CE->getArg(i));
        CurrentArgumentRepl = EA.getReplacedString();
      }
      CallExprArguReplVec.push_back(CurrentArgumentRepl);
    }
    CallExprReplStr = getFinalCallExprStr(Replacement) + CallExprReplStr;

    if (NeedUseLambda) {
      if (PrefixInsertStr == "") {
        // If there is one API call in the migrted code, it is unnecessary to
        // use a lambda expression
        NeedUseLambda = false;
      }
    }

    applyMigrationText(NeedUseLambda, IsMacroArg, CanAvoidBrace,
                       CanAvoidUsingLambda, OriginStmtType, IsAssigned,
                       OuterInsertLoc, PrefixInsertLoc, SuffixInsertLoc,
                       FuncNameBegin, FuncCallEnd, FuncCallLength, IndentStr,
                       PrefixInsertStr, SuffixInsertStr, true, FuncName);
  } else if (FuncName == "cublasCreate_v2" || FuncName == "cublasDestroy_v2" ||
             FuncName == "cublasSetStream_v2" ||
             FuncName == "cublasGetStream_v2" ||
             FuncName == "cublasSetKernelStream") {
    SourceRange SR = getFunctionRange(CE);
    auto Len = SM->getDecomposedLoc(SR.getEnd()).second -
               SM->getDecomposedLoc(SR.getBegin()).second;

    std::string Repl;

    if (FuncName == "cublasCreate_v2") {
      std::string LHS;
      if (isSimpleAddrOf(CE->getArg(0))) {
        LHS = getNameStrRemovedAddrOf(CE->getArg(0));
      } else {
        dpct::ExprAnalysis EA;
        EA.analyze(CE->getArg(0));
        LHS = "*(" + EA.getReplacedString() + ")";
      }
      if (checkWhetherIsDuplicate(CE, false))
        return;
      int Index = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();
      buildTempVariableMap(Index, CE, HelperFuncType::DefaultQueue);
      Repl = LHS + " = &{{NEEDREPLACEQ" + std::to_string(Index) + "}}";
    } else if (FuncName == "cublasDestroy_v2") {
      dpct::ExprAnalysis EA(CE->getArg(0));
      Repl = EA.getReplacedString() + " = nullptr";
    } else if(FuncName == "cublasSetStream_v2") {
      dpct::ExprAnalysis EA0(CE->getArg(0));
      dpct::ExprAnalysis EA1(CE->getArg(1));
      Repl = EA0.getReplacedString() + " = " + EA1.getReplacedString();
    } else if (FuncName == "cublasGetStream_v2") {
      dpct::ExprAnalysis EA0(CE->getArg(0));
      std::string LHS;
      if (isSimpleAddrOf(CE->getArg(1))) {
        LHS = getNameStrRemovedAddrOf(CE->getArg(1));
      } else {
        dpct::ExprAnalysis EA;
        EA.analyze(CE->getArg(1));
        LHS = "*(" + EA.getReplacedString() + ")";
      }
      Repl = LHS + " = " + EA0.getReplacedString();
    } else if (FuncName == "cublasSetKernelStream") {
      dpct::ExprAnalysis EA(CE->getArg(0));
      if (checkWhetherIsDuplicate(CE, false))
        return;
      int Index = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();
      buildTempVariableMap(Index, CE, HelperFuncType::CurrentDevice);
      Repl = "{{NEEDREPLACED" + std::to_string(Index) + "}}.set_saved_queue(" +
             EA.getReplacedString() + ")";
    } else {
      return;
    }

    if (SM->isMacroArgExpansion(CE->getBeginLoc()) &&
        SM->isMacroArgExpansion(CE->getEndLoc())) {
      if (IsAssigned) {
        report(SR.getBegin(), Diagnostics::NOERROR_RETURN_COMMA_OP, false);
        emplaceTransformation(new ReplaceText(
            SR.getBegin(), Len, "(" + Repl + ", 0)", false, FuncName));
      } else {
        emplaceTransformation(new ReplaceText(
            SR.getBegin(), Len, std::move(Repl), false, FuncName));
      }
    } else {
      if (IsAssigned) {
        report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP, false);
        emplaceTransformation(
            new ReplaceStmt(CE, false, FuncName, true, "(" + Repl + ", 0)"));
      } else {
        emplaceTransformation(
            new ReplaceStmt(CE, false, FuncName, true, Repl));
      }
    }
  } else if (FuncName == "cublasInit" || FuncName == "cublasShutdown" ||
             FuncName == "cublasGetError") {
    // Remove these three function calls.
    // TODO: migrate functions when they are in template
    auto Msg = MapNames::RemovedAPIWarningMessage.find(FuncName);
    SourceRange SR = getFunctionRange(CE);
    auto Len = SM->getDecomposedLoc(SR.getEnd()).second -
               SM->getDecomposedLoc(SR.getBegin()).second;
    if (SM->isMacroArgExpansion(CE->getBeginLoc()) &&
        SM->isMacroArgExpansion(CE->getEndLoc())) {
      if (IsAssigned) {
        report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED_0, false,
               MapNames::ITFName.at(FuncName), Msg->second);
        emplaceTransformation(
            new ReplaceText(SR.getBegin(), Len, "0", false, FuncName));
      } else {
        report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED, false,
               MapNames::ITFName.at(FuncName), Msg->second);
        emplaceTransformation(
            new ReplaceText(SR.getBegin(), Len, "0", false, FuncName));
      }
    } else {
      if (IsAssigned) {
        report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED_0, false,
               MapNames::ITFName.at(FuncName), Msg->second);
        emplaceTransformation(new ReplaceStmt(CE, false, FuncName, false, "0"));
      } else {
        report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED, false,
               MapNames::ITFName.at(FuncName), Msg->second);
        emplaceTransformation(new ReplaceStmt(CE, false, FuncName, false, ""));
      }
    }
  } else if (FuncName == "cublasGetPointerMode_v2" ||
             FuncName == "cublasSetPointerMode_v2") {
    std::string Msg = "the function call is redundant in DPC++.";
    SourceRange SR = getFunctionRange(CE);
    auto Len = SM->getDecomposedLoc(SR.getEnd()).second -
               SM->getDecomposedLoc(SR.getBegin()).second;
    if (SM->isMacroArgExpansion(CE->getBeginLoc()) &&
        SM->isMacroArgExpansion(CE->getEndLoc())) {
      if (IsAssigned) {
        report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED_0, false,
               MapNames::ITFName.at(FuncName), Msg);
        emplaceTransformation(
            new ReplaceText(SR.getBegin(), Len, "0", false, FuncName));
      } else {
        report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED, false,
               MapNames::ITFName.at(FuncName), Msg);
        emplaceTransformation(
            new ReplaceText(SR.getBegin(), Len, "0", false, FuncName));
      }
    } else {
      if (IsAssigned) {
        report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED_0, false,
               MapNames::ITFName.at(FuncName), Msg);
        emplaceTransformation(new ReplaceStmt(CE, false, FuncName, true, "0"));
      } else {
        report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED, false,
               MapNames::ITFName.at(FuncName), Msg);
        emplaceTransformation(new ReplaceStmt(CE, false, FuncName, true, ""));
      }
    }
  } else if (FuncName == "cublasSetVector" || FuncName == "cublasGetVector" ||
             FuncName == "cublasSetVectorAsync" ||
             FuncName == "cublasGetVectorAsync") {
    if (HasDeviceAttr) {
      report(CE->getBeginLoc(), Diagnostics::FUNCTION_CALL_IN_DEVICE, false,
             MapNames::ITFName.at(FuncName), "dpct::matrix_mem_copy");
      return;
    }

    std::vector<std::string> ParamsStrsVec =
        getParamsAsStrs(CE, *(Result.Context));
    const Expr *IncxExpr = CE->getArg(3);
    const Expr *IncyExpr = CE->getArg(5);
    Expr::EvalResult IncxExprResult, IncyExprResult;

    if (IncxExpr->EvaluateAsInt(IncxExprResult, *Result.Context) &&
        IncyExpr->EvaluateAsInt(IncyExprResult, *Result.Context)) {
      std::string IncxStr =
          IncxExprResult.Val.getAsString(*Result.Context, IncxExpr->getType());
      std::string IncyStr =
          IncyExprResult.Val.getAsString(*Result.Context, IncyExpr->getType());
      if (IncxStr != IncyStr) {
        report(CE->getBeginLoc(), Diagnostics::POTENTIAL_PERFORMACE_ISSUE,
               false, MapNames::ITFName.at(FuncName),
               "parameter " + ParamsStrsVec[3] +
                   " does not equal to parameter " + ParamsStrsVec[5]);
      } else if ((IncxStr == IncyStr) && (IncxStr != "1")) {
        // incx equals to incy, but does not equal to 1. Performance issue may
        // occur.
        report(CE->getBeginLoc(), Diagnostics::POTENTIAL_PERFORMACE_ISSUE,
               false, MapNames::ITFName.at(FuncName),
               "parameter " + ParamsStrsVec[3] + " equals to parameter " +
                   ParamsStrsVec[5] + " but greater than 1");
      }
    } else {
      report(CE->getBeginLoc(), Diagnostics::POTENTIAL_PERFORMACE_ISSUE, false,
             MapNames::ITFName.at(FuncName),
             "parameter(s) " + ParamsStrsVec[3] + " and/or " +
                 ParamsStrsVec[5] + " could not be evaluated");
    }

    std::string XStr = "(void*)" + getExprString(CE->getArg(2), true);
    std::string YStr = "(void*)" + getExprString(CE->getArg(4), true);
    std::string IncX = getExprString(CE->getArg(3), false);
    std::string IncY = getExprString(CE->getArg(5), false);
    std::string ElemSize = getExprString(CE->getArg(1), false);
    std::string ElemNum = getExprString(CE->getArg(0), false);

    std::string Replacement = "(" + YStr + ", " + XStr + ", " + IncY + ", " +
                              IncX + ", 1, " + ElemNum + ", " + ElemSize;

    if (FuncName == "cublasGetVector" || FuncName == "cublasSetVector") {
      Replacement = "dpct::matrix_mem_copy" + Replacement + ")";
    } else {
      ExprAnalysis EA;
      EA.analyze(CE->getArg(6));
      Replacement = "dpct::matrix_mem_copy" + Replacement +
                    ", dpct::automatic, *" + EA.getReplacedString() + ", true)";
    }
    emplaceTransformation(new ReplaceStmt(CE, std::move(Replacement)));

    if (IsAssigned) {
      report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP, false);
      insertAroundStmt(CE, "(", ", 0)");
    }
  } else if (FuncName == "cublasSetMatrix" || FuncName == "cublasGetMatrix" ||
             FuncName == "cublasSetMatrixAsync" ||
             FuncName == "cublasGetMatrixAsync") {
    if (HasDeviceAttr) {
      report(CE->getBeginLoc(), Diagnostics::FUNCTION_CALL_IN_DEVICE, false,
             MapNames::ITFName.at(FuncName), "dpct::matrix_mem_copy");
      return;
    }

    std::vector<std::string> ParamsStrsVec =
        getParamsAsStrs(CE, *(Result.Context));

    const Expr *LdaExpr = CE->getArg(4);
    const Expr *LdbExpr = CE->getArg(6);
    Expr::EvalResult LdaExprResult, LdbExprResult;
    if (LdaExpr->EvaluateAsInt(LdaExprResult, *Result.Context) &&
        LdbExpr->EvaluateAsInt(LdbExprResult, *Result.Context)) {
      std::string LdaStr =
          LdaExprResult.Val.getAsString(*Result.Context, LdaExpr->getType());
      std::string LdbStr =
          LdbExprResult.Val.getAsString(*Result.Context, LdbExpr->getType());
      if (LdaStr != LdbStr) {
        report(CE->getBeginLoc(), Diagnostics::POTENTIAL_PERFORMACE_ISSUE,
               false, MapNames::ITFName.at(FuncName),
               "parameter " + ParamsStrsVec[4] +
                   " does not equal to parameter " + ParamsStrsVec[6]);
      } else {
        const Expr *RowsExpr = CE->getArg(0);
        Expr::EvalResult RowsExprResult;
        if (RowsExpr->EvaluateAsInt(RowsExprResult, *Result.Context)) {
          std::string RowsStr = RowsExprResult.Val.getAsString(
              *Result.Context, RowsExpr->getType());
          if (std::stoi(LdaStr) > std::stoi(RowsStr)) {
            // lda > rows. Performance issue may occur.
            report(CE->getBeginLoc(), Diagnostics::POTENTIAL_PERFORMACE_ISSUE,
                   false, MapNames::ITFName.at(FuncName),
                   "parameter " + ParamsStrsVec[0] +
                       " is smaller than parameter " + ParamsStrsVec[4]);
          }
        } else {
          // rows cannot be evaluated. Performance issue may occur.
          report(
              CE->getBeginLoc(), Diagnostics::POTENTIAL_PERFORMACE_ISSUE, false,
              MapNames::ITFName.at(FuncName),
              "parameter " + ParamsStrsVec[0] +
                  " could not be evaluated and may be smaller than parameter " +
                  ParamsStrsVec[4]);
        }
      }
    } else {
      report(CE->getBeginLoc(), Diagnostics::POTENTIAL_PERFORMACE_ISSUE, false,
             MapNames::ITFName.at(FuncName),
             "parameter(s) " + ParamsStrsVec[4] + " and/or " +
                 ParamsStrsVec[6] + " could not be evaluated");
    }

    std::string AStr = "(void*)" + getExprString(CE->getArg(3), true);
    std::string BStr = "(void*)" + getExprString(CE->getArg(5), true);
    std::string LdA = getExprString(CE->getArg(4), false);
    std::string LdB = getExprString(CE->getArg(6), false);
    std::string Rows = getExprString(CE->getArg(0), false);
    std::string Cols = getExprString(CE->getArg(1), false);
    std::string ElemSize = getExprString(CE->getArg(2), false);

    std::string Replacement = "(" + BStr + ", " + AStr + ", " + LdB + ", " +
                              LdA + ", " + Rows + ", " + Cols + ", " + ElemSize;

    if (FuncName == "cublasGetMatrix" || FuncName == "cublasSetMatrix") {
      Replacement = "dpct::matrix_mem_copy" + Replacement + ")";
    } else {
      ExprAnalysis EA;
      EA.analyze(CE->getArg(7));
      Replacement = "dpct::matrix_mem_copy" + Replacement +
                    ", dpct::automatic, *" + EA.getReplacedString() + ", true)";
    }
    emplaceTransformation(new ReplaceStmt(CE, std::move(Replacement)));

    if (IsAssigned) {
      report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP, false);
      insertAroundStmt(CE, "(", ", 0)");
    }
  } else if (FuncName == "make_cuComplex" ||
             FuncName == "make_cuDoubleComplex") {
    if (FuncName == "make_cuDoubleComplex")
      emplaceTransformation(new ReplaceCalleeName(
          CE, MapNames::getClNamespace() + "::double2", FuncName));
    else
      emplaceTransformation(new ReplaceCalleeName(
          CE, MapNames::getClNamespace() + "::float2", FuncName));
  } else {
    assert(0 && "Unknown function name");
  }
}

// Check whether the input expression is CallExpr or UnaryExprOrTypeTraitExpr
// (sizeof or alignof) or an identifier or literal
bool BLASFunctionCallRule::isCEOrUETTEOrAnIdentifierOrLiteral(const Expr *E) {
  auto CE = dyn_cast<CallExpr>(E->IgnoreImpCasts());
  if (CE != nullptr) {
    return true;
  }
  auto UETTE = dyn_cast<UnaryExprOrTypeTraitExpr>(E->IgnoreImpCasts());
  if (UETTE != nullptr) {
    return true;
  }
  if (isAnIdentifierOrLiteral(E)) {
    return true;
  }
  return false;
}

// Get the replacement string of the input expression.
// If the AddparenthesisIfNecessary is true, the output string will add "(...)"
// if it is necessary.
std::string BLASFunctionCallRule::getExprString(const Expr *E,
                                    bool AddparenthesisIfNecessary) {
  ExprAnalysis EA;
  EA.analyze(E);
  std::string Res = EA.getReplacedString();
  if (AddparenthesisIfNecessary && !isCEOrUETTEOrAnIdentifierOrLiteral(E)) {
    return "(" + Res + ")";
  } else {
    return Res;
  }
}

bool BLASFunctionCallRule::isReplIndex(int Input,
                                       const std::vector<int> &IndexInfo,
                                       int &IndexTemp) {
  for (int i = 0; i < static_cast<int>(IndexInfo.size()); ++i) {
    if (IndexInfo[i] == Input) {
      IndexTemp = i;
      return true;
    }
  }
  return false;
}

std::vector<std::string>
BLASFunctionCallRule::getParamsAsStrs(const CallExpr *CE,
                                      const ASTContext &Context) {
  std::vector<std::string> ParamsStrVec;
  for (auto Arg : CE->arguments())
    ParamsStrVec.emplace_back(getStmtSpelling(Arg));
  return ParamsStrVec;
}

// sample code looks like:
//   Complex-type res1 = API(...);
//   res2 = API(...);
//
// migrated code looks like:
//   Complex-type res1;
//   {
//   buffer res_buffer;
//   mklAPI(res_buffer);
//   res1 = res_buffer.get_access()[0];
//   }
//   {
//   buffer res_buffer;
//   mklAPI(res_buffer);
//   res2 = res_buffer.get_access()[0];
//   }
//
// If the API return value initializes the var declaration, we need to put the
// var declaration out of the scope and assign it in the scope, otherwise users
// cannot use this var out of the scope.
// So we need to find the Decl node of the var, which is the CallExpr node's
// ancestor.
// The initial node is the matched CallExpr node. Then visit the parent node of
// the current node until the current node is a VarDecl node.
const clang::VarDecl *
BLASFunctionCallRule::getAncestralVarDecl(const clang::CallExpr *CE) {
  auto &Context = dpct::DpctGlobalInfo::getContext();
  auto Parents = Context.getParents(*CE);
  while (Parents.size() == 1) {
    auto *Parent = Parents[0].get<VarDecl>();
    if (Parent) {
      return Parent;
    } else {
      Parents = Context.getParents(Parents[0]);
    }
  }
  return nullptr;
}

std::string BLASFunctionCallRule::processParamIntCastToBLASEnum(
    const Expr *E, const CStyleCastExpr *CSCE, const int DistinctionID,
    const std::string IndentStr, const BLASEnumInfo &EnumInfo,
    std::string &PrefixInsertStr, std::string &CurrentArgumentRepl) {
  std::string DpctTempVarName;
  auto &Context = DpctGlobalInfo::getContext();
  const Expr *SubExpr = CSCE->getSubExpr();
  std::string SubExprStr;
  if (SubExpr->getBeginLoc().isMacroID() && isOuterMostMacro(CSCE)) {
    // when type casting syntax is in a macro,
    // analyze the entire CSCE by ExprAnalysis
    ExprAnalysis SEA;
    SEA.analyze(CSCE);
    SubExprStr = SEA.getReplacedString();

    // Since the type cast in the macro definition need be kept (it may be used
    // in more than one places), so we need add the type cast to int for the
    // current argument.
    CurrentArgumentRepl = "(int)";
  } else {
    // To eliminate the redundant cast of non-macro cases
    SubExprStr = getStmtSpelling(SubExpr);
  }

  int IndexTemp = -1;
  if (isReplIndex(DistinctionID, EnumInfo.OperationIndexInfo, IndexTemp)) {
    Expr::EvalResult ER;
    if (E->EvaluateAsInt(ER, Context) && !E->getBeginLoc().isMacroID()) {
      int64_t Value = ER.Val.getInt().getExtValue();
      if (Value == 0) {
        CurrentArgumentRepl += "mkl::transpose::nontrans";
      } else if (Value == 1) {
        CurrentArgumentRepl += "mkl::transpose::trans";
      } else {
        CurrentArgumentRepl += "mkl::transpose::conjtrans";
      }
    } else {
      if (E->HasSideEffects(DpctGlobalInfo::getContext())) {
        DpctTempVarName =
            "transpose_ct" +
            std::to_string(DpctGlobalInfo::getSuffixIndexInRuleThenInc());
        PrefixInsertStr = PrefixInsertStr + "auto " + DpctTempVarName + " = " +
                          SubExprStr + ";" + getNL() + IndentStr;
        CurrentArgumentRepl +=
            "(int)" + DpctTempVarName +
            "==2 ? mkl::transpose::conjtrans : (mkl::transpose)" +
            DpctTempVarName;
      } else {
        CurrentArgumentRepl +=
            SubExprStr + "==2 ? mkl::transpose::conjtrans : (mkl::transpose)" +
            SubExprStr;
      }
    }
  }
  if (EnumInfo.FillModeIndexInfo == DistinctionID) {
    Expr::EvalResult ER;
    if (E->EvaluateAsInt(ER, Context) && !E->getBeginLoc().isMacroID()) {
      int64_t Value = ER.Val.getInt().getExtValue();
      if (Value == 0) {
        CurrentArgumentRepl += "mkl::uplo::lower";
      } else {
        CurrentArgumentRepl += "mkl::uplo::upper";
      }
    } else {
      CurrentArgumentRepl +=
          SubExprStr + "==0 ? mkl::uplo::lower : mkl::uplo::upper";
    }
  }
  if (EnumInfo.SideModeIndexInfo == DistinctionID) {
    Expr::EvalResult ER;
    if (E->EvaluateAsInt(ER, Context) && !E->getBeginLoc().isMacroID()) {
      int64_t Value = ER.Val.getInt().getExtValue();
      if (Value == 0) {
        CurrentArgumentRepl += "mkl::side::left";
      } else {
        CurrentArgumentRepl += "mkl::side::right";
      }
    } else {
      CurrentArgumentRepl += "(mkl::side)" + SubExprStr;
    }
  }
  if (EnumInfo.DiagTypeIndexInfo == DistinctionID) {
    Expr::EvalResult ER;
    if (E->EvaluateAsInt(ER, Context) && !E->getBeginLoc().isMacroID()) {
      int64_t Value = ER.Val.getInt().getExtValue();
      if (Value == 0) {
        CurrentArgumentRepl += "mkl::diag::nonunit";
      } else {
        CurrentArgumentRepl += "mkl::diag::unit";
      }
    } else {
      CurrentArgumentRepl += "(mkl::diag)" + SubExprStr;
    }
  }

  return DpctTempVarName;
}

REGISTER_RULE(BLASFunctionCallRule)

// Rule for SOLVER enums.
// Migrate SOLVER status values to corresponding int values
// Other SOLVER named values are migrated to corresponding named values
void SOLVEREnumsRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(
      declRefExpr(to(enumConstantDecl(matchesName("CUSOLVER_STATU.*"))))
          .bind("SOLVERStatusConstants"),
      this);
  MF.addMatcher(
      declRefExpr(to(enumConstantDecl(matchesName(
                      "(CUSOLVER_EIG_TYPE.*)|(CUSOLVER_EIG_MODE.*)"))))
          .bind("SLOVERNamedValueConstants"),
      this);
}

void SOLVEREnumsRule::run(const MatchFinder::MatchResult &Result) {
  CHECKPOINT_ASTMATCHER_RUN_ENTRY();
  if (const DeclRefExpr *DE =
          getNodeAsType<DeclRefExpr>(Result, "SOLVERStatusConstants")) {
    auto *EC = cast<EnumConstantDecl>(DE->getDecl());
    emplaceTransformation(new ReplaceStmt(DE, EC->getInitVal().toString(10)));
  }

  if (const DeclRefExpr *DE =
          getNodeAsType<DeclRefExpr>(Result, "SLOVERNamedValueConstants")) {
    auto *EC = cast<EnumConstantDecl>(DE->getDecl());
    std::string Name = EC->getNameAsString();
    auto Search = MapNames::SOLVEREnumsMap.find(Name);
    if (Search == MapNames::SOLVEREnumsMap.end()) {
      llvm::dbgs() << "[" << getName()
                   << "] Unexpected enum variable: " << Name;
      return;
    }
    std::string Replacement = Search->second;
    emplaceTransformation(new ReplaceStmt(DE, std::move(Replacement)));
  }
}

REGISTER_RULE(SOLVEREnumsRule)

void SOLVERFunctionCallRule::registerMatcher(MatchFinder &MF) {
  auto functionName = [&]() {
    return hasAnyName(
        "cusolverDnCreate", "cusolverDnDestroy", "cusolverDnSpotrf_bufferSize",
        "cusolverDnDpotrf_bufferSize", "cusolverDnCpotrf_bufferSize",
        "cusolverDnZpotrf_bufferSize", "cusolverDnSpotri_bufferSize",
        "cusolverDnDpotri_bufferSize", "cusolverDnCpotri_bufferSize",
        "cusolverDnZpotri_bufferSize", "cusolverDnSgetrf_bufferSize",
        "cusolverDnDgetrf_bufferSize", "cusolverDnCgetrf_bufferSize",
        "cusolverDnZgetrf_bufferSize", "cusolverDnSpotrf", "cusolverDnDpotrf",
        "cusolverDnCpotrf", "cusolverDnZpotrf", "cusolverDnSpotrs",
        "cusolverDnDpotrs", "cusolverDnCpotrs", "cusolverDnZpotrs",
        "cusolverDnSpotri", "cusolverDnDpotri", "cusolverDnCpotri",
        "cusolverDnZpotri", "cusolverDnSgetrf", "cusolverDnDgetrf",
        "cusolverDnCgetrf", "cusolverDnZgetrf", "cusolverDnSgetrs",
        "cusolverDnDgetrs", "cusolverDnCgetrs", "cusolverDnZgetrs",
        "cusolverDnSgeqrf_bufferSize", "cusolverDnDgeqrf_bufferSize",
        "cusolverDnCgeqrf_bufferSize", "cusolverDnZgeqrf_bufferSize",
        "cusolverDnSgeqrf", "cusolverDnDgeqrf", "cusolverDnCgeqrf",
        "cusolverDnZgeqrf", "cusolverDnSormqr_bufferSize",
        "cusolverDnDormqr_bufferSize", "cusolverDnSormqr", "cusolverDnDormqr",
        "cusolverDnCunmqr_bufferSize", "cusolverDnZunmqr_bufferSize",
        "cusolverDnCunmqr", "cusolverDnZunmqr", "cusolverDnSorgqr_bufferSize",
        "cusolverDnDorgqr_bufferSize", "cusolverDnCungqr_bufferSize",
        "cusolverDnZungqr_bufferSize", "cusolverDnSorgqr", "cusolverDnDorgqr",
        "cusolverDnCungqr", "cusolverDnZungqr", "cusolverDnSsytrf_bufferSize",
        "cusolverDnDsytrf_bufferSize", "cusolverDnCsytrf_bufferSize",
        "cusolverDnZsytrf_bufferSize", "cusolverDnSsytrf", "cusolverDnDsytrf",
        "cusolverDnCsytrf", "cusolverDnZsytrf", "cusolverDnSgebrd_bufferSize",
        "cusolverDnDgebrd_bufferSize", "cusolverDnCgebrd_bufferSize",
        "cusolverDnZgebrd_bufferSize", "cusolverDnSgebrd", "cusolverDnDgebrd",
        "cusolverDnCgebrd", "cusolverDnZgebrd", "cusolverDnSorgbr_bufferSize",
        "cusolverDnDorgbr_bufferSize", "cusolverDnCungbr_bufferSize",
        "cusolverDnZungbr_bufferSize", "cusolverDnSorgbr", "cusolverDnDorgbr",
        "cusolverDnCungbr", "cusolverDnZungbr", "cusolverDnSsytrd_bufferSize",
        "cusolverDnDsytrd_bufferSize", "cusolverDnChetrd_bufferSize",
        "cusolverDnZhetrd_bufferSize", "cusolverDnSsytrd", "cusolverDnDsytrd",
        "cusolverDnChetrd", "cusolverDnZhetrd", "cusolverDnSormtr_bufferSize",
        "cusolverDnDormtr_bufferSize", "cusolverDnCunmtr_bufferSize",
        "cusolverDnZunmtr_bufferSize", "cusolverDnSormtr", "cusolverDnDormtr",
        "cusolverDnCunmtr", "cusolverDnZunmtr", "cusolverDnSorgtr_bufferSize",
        "cusolverDnDorgtr_bufferSize", "cusolverDnCungtr_bufferSize",
        "cusolverDnZungtr_bufferSize", "cusolverDnSorgtr", "cusolverDnDorgtr",
        "cusolverDnCungtr", "cusolverDnZungtr", "cusolverDnSgesvd_bufferSize",
        "cusolverDnDgesvd_bufferSize", "cusolverDnCgesvd_bufferSize",
        "cusolverDnZgesvd_bufferSize", "cusolverDnSgesvd", "cusolverDnDgesvd",
        "cusolverDnCgesvd", "cusolverDnZgesvd", "cusolverDnSsyevd_bufferSize",
        "cusolverDnDsyevd_bufferSize", "cusolverDnSsyevd_bufferSize",
        "cusolverDnCheevd_bufferSize", "cusolverDnZheevd_bufferSize",
        "cusolverDnDsyevd", "cusolverDnSsyevd", "cusolverDnCheevd",
        "cusolverDnZheevd");
  };

  MF.addMatcher(callExpr(allOf(callee(functionDecl(functionName())),
                               hasAncestor(functionDecl(
                                   anyOf(hasAttr(attr::CUDADevice),
                                         hasAttr(attr::CUDAGlobal))))))
                    .bind("kernelCall"),
                this);

  MF.addMatcher(
      callExpr(
          allOf(callee(functionDecl(functionName())), parentStmt(),
                hasAncestor(functionDecl(unless(allOf(
                    hasAttr(attr::CUDADevice), hasAttr(attr::CUDAGlobal)))))))
          .bind("FunctionCall"),
      this);
  MF.addMatcher(
      callExpr(
          allOf(callee(functionDecl(functionName())), unless(parentStmt()),
                unless(hasParent(varDecl())),
                hasAncestor(functionDecl(unless(allOf(
                    hasAttr(attr::CUDADevice), hasAttr(attr::CUDAGlobal)))))))
          .bind("FunctionCallUsedNotInitializeVarDecl"),
      this);

  MF.addMatcher(
      callExpr(
          allOf(callee(functionDecl(functionName())), hasParent(varDecl()),
                hasAncestor(functionDecl(unless(allOf(
                    hasAttr(attr::CUDADevice), hasAttr(attr::CUDAGlobal)))))))
          .bind("FunctionCallUsedInitializeVarDecl"),
      this);
}

void SOLVERFunctionCallRule::run(const MatchFinder::MatchResult &Result) {
  CHECKPOINT_ASTMATCHER_RUN_ENTRY();
  bool IsAssigned = false;
  bool IsInitializeVarDecl = false;
  bool HasDeviceAttr = false;
  const CallExpr *CE = getNodeAsType<CallExpr>(Result, "kernelCall");
  if (CE) {
    HasDeviceAttr = true;
  } else if (!(CE = getNodeAsType<CallExpr>(Result, "FunctionCall"))) {
    if ((CE = getNodeAsType<CallExpr>(
             Result, "FunctionCallUsedNotInitializeVarDecl"))) {
      IsAssigned = true;
    } else if ((CE = getNodeAsType<CallExpr>(
                    Result, "FunctionCallUsedInitializeVarDecl"))) {
      IsAssigned = true;
      IsInitializeVarDecl = true;
    } else {
      return;
    }
  }

  const SourceManager *SM = Result.SourceManager;
  auto SL = SM->getExpansionLoc(CE->getBeginLoc());
  std::string Key = SM->getFilename(SL).str() +
                    std::to_string(SM->getDecomposedLoc(SL).second);
  DpctGlobalInfo::updateInitSuffixIndexInRule(
      DpctGlobalInfo::getSuffixIndexInitValue(Key));

  // Collect sourceLocations of the function call
  SourceLocation FuncNameBegin(CE->getBeginLoc());
  SourceLocation FuncCallEnd(CE->getEndLoc());

  // Correct sourceLocations for macros
  if (FuncNameBegin.isMacroID())
    FuncNameBegin = SM->getExpansionLoc(FuncNameBegin);
  if (FuncCallEnd.isMacroID())
    FuncCallEnd = SM->getExpansionLoc(FuncCallEnd);
  // Offset 1 is the length of the last token ")"
  FuncCallEnd = FuncCallEnd.getLocWithOffset(1);

  // Collect sourceLocations for creating new scope
  std::string PrefixBeforeScope, PrefixInsertStr, SuffixInsertStr;
  auto SR = getScopeInsertRange(CE, FuncNameBegin, FuncCallEnd);
  SourceLocation StmtBegin = SR.getBegin(), StmtEndAfterSemi = SR.getEnd();
  std::string IndentStr =
      getIndent(StmtBegin, (Result.Context)->getSourceManager()).str();
  Token Tok;
  Lexer::getRawToken(FuncNameBegin, Tok, *SM, LangOptions());
  SourceLocation FuncNameEnd = Tok.getEndLoc();
  auto FuncNameLength =
      SM->getCharacterData(FuncNameEnd) - SM->getCharacterData(FuncNameBegin);

  // Prepare the prefix and the postfix for assignment
  if (!CE->getDirectCallee())
    return;
  std::string FuncName =
      CE->getDirectCallee()->getNameInfo().getName().getAsString();
  std::string AssignPrefix = "";
  std::string AssignPostfix = "";

  if (IsAssigned) {
    AssignPrefix = "(";
    AssignPostfix = ", 0)";
  }

  if (HasDeviceAttr) {
    report(CE->getBeginLoc(), Diagnostics::FUNCTION_CALL_IN_DEVICE, false,
           MapNames::ITFName.at(FuncName), "dpct::dpct_memcpy");
    return;
  }

  const VarDecl *VD = 0;
  if (IsInitializeVarDecl) {
    // Create the prefix for VarDecl before scope and remove the VarDecl inside
    // the scope
    VD = getAncestralVarDecl(CE);
    std::string VarType, VarName;
    if (VD) {
      VarType = VD->getType().getAsString();
      VarName = VD->getNameAsString();
      auto Itr = MapNames::TypeNamesMap.find(VarType);
      if (Itr != MapNames::TypeNamesMap.end())
        VarType = Itr->second;
      PrefixBeforeScope = VarType + " " + VarName + ";" + getNL() + IndentStr +
                          PrefixBeforeScope;
      SourceLocation typeBegin =
          VD->getTypeSourceInfo()->getTypeLoc().getBeginLoc();
      SourceLocation nameBegin = VD->getLocation();
      SourceLocation nameEnd = Lexer::getLocForEndOfToken(
          nameBegin, 0, *SM, Result.Context->getLangOpts());
      auto replLen =
          SM->getCharacterData(nameEnd) - SM->getCharacterData(typeBegin);
      emplaceTransformation(
          new ReplaceText(typeBegin, replLen, std::move(VarName)));
    } else {
      assert(0 && "Fail to get VarDecl information");
      return;
    }
  }

  if (MapNames::SOLVERFuncReplInfoMap.find(FuncName) !=
      MapNames::SOLVERFuncReplInfoMap.end()) {
    // Find replacement string
    auto ReplInfoPair = MapNames::SOLVERFuncReplInfoMap.find(FuncName);
    MapNames::SOLVERFuncReplInfo ReplInfo = ReplInfoPair->second;
    std::string Replacement = ReplInfo.ReplName;

    // Migrate arguments one by one
    int ArgNum = CE->getNumArgs();
    for (int i = 0; i < ArgNum; ++i) {
      int IndexTemp = -1;
      // MyFunction(float* a);
      // In usm: Keep it and type cast if needed.
      // In non-usm: Create a buffer, pass the buffer
      // If type is int: MKL takes int64_t, so need to create a temp
      //                 buffer/variable and copy the value back
      //                 after the function call.
      // Some API migration requires MoveFrom and MoveTo.
      // e.g. move arg#1 to arg#0
      // MyFunction(float* a, float* b);
      // ---> MyFunction(float* b, float*a);
      if (isReplIndex(i, ReplInfo.BufferIndexInfo, IndexTemp)) {
        if (DpctGlobalInfo::getUsmLevel() == UsmLevel::none) {
          std::string BufferDecl;
          std::string BufferName = getBufferNameAndDeclStr(
            CE->getArg(i), *(Result.Context),
            ReplInfo.BufferTypeInfo[IndexTemp], StmtBegin, BufferDecl, i);
          PrefixInsertStr = PrefixInsertStr + BufferDecl;
          if (ReplInfo.BufferTypeInfo[IndexTemp] == "int") {
            PrefixInsertStr =
              PrefixInsertStr + IndentStr + MapNames::getClNamespace() +
              "::buffer<int64_t> "
              "result_temp_buffer" +
              std::to_string(i) + "(" + MapNames::getClNamespace() +
              "::range<1>(1));" + getNL();
            SuffixInsertStr = SuffixInsertStr + BufferName + ".get_access<" +
              MapNames::getClNamespace() +
              "::access::mode::write>()[0] = "
              "(int)result_temp_buffer" +
              std::to_string(i) + ".get_access<" +
              MapNames::getClNamespace() +
              "::access::mode::read>()[0];" +
              getNL() + IndentStr;
            BufferName = "result_temp_buffer" + std::to_string(i);
          }
          bool Moved = false;
          for (int j = 0; j < ReplInfo.MoveFrom.size();j++) {
            if (ReplInfo.MoveFrom[j] == i) {
              Moved = true;
              if (CE->getArg(ReplInfo.MoveTo[j]) > 0) {
                emplaceTransformation(new InsertAfterStmt(
                    CE->getArg(ReplInfo.MoveTo[j] - 1),
                    ", result_temp_buffer" + std::to_string(i)));
              }
              ReplInfo.RedundantIndexInfo.push_back(i);
              break;
            }
          }
          if (!Moved) {
            emplaceTransformation(
              new ReplaceStmt(CE->getArg(i), BufferName));
          }
        } else {
          std::string ArgName = getStmtSpelling(CE->getArg(i));
          if (ReplInfo.BufferTypeInfo[IndexTemp] == "int") {
            PrefixInsertStr = IndentStr + "int64_t result_temp_pointer" +
                              std::to_string(i) + ";" + getNL();
            SuffixInsertStr = SuffixInsertStr + " *" +
                              getStmtSpelling(CE->getArg(i)) +
                              " = result_temp_pointer" + std::to_string(i) +
                              ";" + getNL() + IndentStr;
            ArgName = "&result_temp_pointer" + std::to_string(i);
          }
          bool Moved = false;
          for (int j = 0; j < ReplInfo.MoveFrom.size(); j++) {
            if (ReplInfo.MoveFrom[j] == i) {
              Moved = true;
              if (CE->getArg(ReplInfo.MoveTo[j]) > 0) {
                emplaceTransformation(new InsertAfterStmt(
                    CE->getArg(ReplInfo.MoveTo[j] - 1), ", " + ArgName));
              }
              ReplInfo.RedundantIndexInfo.push_back(i);
              break;
            }
          }
          if (!Moved) {
            std::string TypeStr =
                ReplInfo.BufferTypeInfo[IndexTemp].compare("int")
                    ? "(" + ReplInfo.BufferTypeInfo[IndexTemp] + "*)"
                    : "";
            emplaceTransformation(
                new ReplaceStmt(CE->getArg(i), TypeStr + ArgName));
          }
        }
      }
      // Remove the redundant args including the leading ","
      if (isReplIndex(i, ReplInfo.RedundantIndexInfo, IndexTemp)) {
        SourceLocation RemoveBegin, RemoveEnd;
        if (i == 0) {
          RemoveBegin = CE->getArg(i)->getBeginLoc();
        } else {
          RemoveBegin = CE->getArg(i - 1)->getEndLoc().getLocWithOffset(
              Lexer::MeasureTokenLength(
                  CE->getArg(i - 1)->getEndLoc(), *SM,
                  dpct::DpctGlobalInfo::getContext().getLangOpts()));
        }
        RemoveEnd = CE->getArg(i)->getEndLoc().getLocWithOffset(
            Lexer::MeasureTokenLength(
                CE->getArg(i)->getEndLoc(), *SM,
                dpct::DpctGlobalInfo::getContext().getLangOpts()));
        auto ParameterLength =
            SM->getCharacterData(RemoveEnd) -
            SM->getCharacterData(RemoveBegin);
        emplaceTransformation(
            new ReplaceText(RemoveBegin, ParameterLength, ""));
      }
      // OldFoo(float* out); --> *(out) = NewFoo();
      // In current case, return value is always the last arg
      if (ReplInfo.ReturnValue && i == ArgNum - 1) {
        Replacement = "*(" + getStmtSpelling(CE->getArg(CE->getNumArgs() - 1)) +
                      ") = " + Replacement;
      }
      // The arg#0 is always the handler and will always be migrated to queue.
      if (i == 0) {
        // process handle argument
        emplaceTransformation(new ReplaceStmt(
            CE->getArg(i), "*" + getStmtSpelling(CE->getArg(i))));
      }
    }
    // Declare new args if it is used in MKL
    if (!ReplInfo.MissedArgumentFinalLocation.empty()) {
      std::string ReplStr;
      for (size_t i = 0; i < ReplInfo.MissedArgumentFinalLocation.size(); ++i) {
        if (ReplInfo.MissedArgumentIsBuffer[i]) {
          PrefixInsertStr =
              PrefixInsertStr + IndentStr + MapNames::getClNamespace() +
              "::buffer<" + ReplInfo.MissedArgumentType[i] + "> " +
              ReplInfo.MissedArgumentName[i] + "(" +
              MapNames::getClNamespace() + "::range<1>(1));" + getNL();
        } else {
          PrefixInsertStr = PrefixInsertStr + IndentStr +
                            ReplInfo.MissedArgumentType[i] + " " +
                            ReplInfo.MissedArgumentName[i] + ";" + getNL();
        }
        ReplStr = ReplStr + ", " + ReplInfo.MissedArgumentName[i];
        if (i == ReplInfo.MissedArgumentFinalLocation.size() - 1 ||
            ReplInfo.MissedArgumentInsertBefore[i + 1] !=
                ReplInfo.MissedArgumentInsertBefore[i]) {
          if (ReplInfo.MissedArgumentInsertBefore[i] > 0) {
            emplaceTransformation(new InsertAfterStmt(
              CE->getArg(ReplInfo.MissedArgumentInsertBefore[i] - 1),
              std::move(ReplStr)));
          }
          ReplStr = "";
        }
      }
    }

    // Copy an arg. e.g. copy arg#0 to arg#2
    // OldFoo(int m, int n); --> NewFoo(int m, int n, int m)
    if (!ReplInfo.CopyFrom.empty()) {
      std::string InsStr = "";
      for (size_t i = 0; i < ReplInfo.CopyFrom.size(); ++i) {
        InsStr =
            InsStr + ", " + getStmtSpelling(CE->getArg(ReplInfo.CopyFrom[i]));
        if (i == ReplInfo.CopyTo.size() - 1 ||
            ReplInfo.CopyTo[i + 1] != ReplInfo.CopyTo[i]) {
          emplaceTransformation(new InsertAfterStmt(
              CE->getArg(ReplInfo.CopyTo[i-1]), std::move(InsStr)));
          InsStr = "";
        }
      }
    }
    // Type cast, for enum type migration
    if (!ReplInfo.CastIndexInfo.empty()) {
      for (size_t i = 0; i < ReplInfo.CastIndexInfo.size(); ++i) {
        std::string CastStr = "(" + ReplInfo.CastTypeInfo[i] + ")";
        emplaceTransformation(new InsertBeforeStmt(
            CE->getArg(ReplInfo.CastIndexInfo[i]), std::move(CastStr)));
      }
    }
    // Create scratchpad and scratchpad_size if required in MKL
    if (!ReplInfo.WorkspaceIndexInfo.empty()) {
      std::string BufferSizeArgStr = "";
      for (int i = 0; i < ReplInfo.WorkspaceSizeInfo.size(); ++i) {
        BufferSizeArgStr += i ? " ," : "";
        BufferSizeArgStr +=
            getStmtSpelling(CE->getArg(ReplInfo.WorkspaceSizeInfo[i]));
      }
      std::string ScratchpadSizeNameStr =
          "scratchpad_size_ct" +
          std::to_string(dpct::DpctGlobalInfo::getSuffixIndexInRuleThenInc());
      std::string ScratchpadNameStr =
          "scratchpad_ct" +
          std::to_string(dpct::DpctGlobalInfo::getSuffixIndexInRuleThenInc());
      std::string EventNameStr =
          "event_ct" +
          std::to_string(dpct::DpctGlobalInfo::getSuffixIndexInRuleThenInc());
      std::string WSVectorNameStr =
        "ws_vec_ct" +
        std::to_string(dpct::DpctGlobalInfo::getSuffixIndexInRuleThenInc());
      PrefixInsertStr +=
          IndentStr + "std::int64_t " + ScratchpadSizeNameStr +
          " = " + ReplInfo.WorkspaceSizeFuncName + "(*" + BufferSizeArgStr +
          ");" + getNL();
      std::string BufferTypeStr = "float";
      if (ReplInfo.BufferTypeInfo.size() > 0) {
        BufferTypeStr = ReplInfo.BufferTypeInfo[0];
      }
      if (DpctGlobalInfo::getUsmLevel() == UsmLevel::restricted) {
        DpctGlobalInfo::getInstance().insertHeader(CE->getBeginLoc(), Thread);

        PrefixInsertStr += IndentStr + BufferTypeStr + " *" +
                           ScratchpadNameStr + " = cl::sycl::malloc_device<" +
                           BufferTypeStr + ">(" + ScratchpadSizeNameStr +
                           ", *" + getStmtSpelling(CE->getArg(0)) + ");" + getNL();
        PrefixInsertStr += IndentStr + "cl::sycl::event " +
                           EventNameStr + ";" + getNL();

        Replacement = EventNameStr + " = " + Replacement;

        SuffixInsertStr += "std::vector<void *> " + WSVectorNameStr + "{" +
                           ScratchpadNameStr + "};" + getNL() + IndentStr;
        SuffixInsertStr +=
            "std::thread mem_free_thread(dpct::detail::mem_free, " +
            getStmtSpelling(CE->getArg(0)) + ", " + WSVectorNameStr + ", " +
            EventNameStr + ");" + getNL() + IndentStr;
        SuffixInsertStr +=
            "mem_free_thread.detach();" + std::string(getNL()) + IndentStr;
      } else {
        PrefixInsertStr += IndentStr + "cl::sycl::buffer<" + BufferTypeStr +
                           ", 1> " + ScratchpadNameStr +
                           "{cl::sycl::range<1>(" + ScratchpadSizeNameStr +
                           ")};" + getNL();
      }
      if (ReplInfo.WorkspaceIndexInfo[0] > 0) {
        emplaceTransformation(new InsertAfterStmt(
            CE->getArg(ReplInfo.WorkspaceIndexInfo[0]),
            ", " + ScratchpadNameStr + ", " + ScratchpadSizeNameStr));
      }
    }

    // Create scratchpad_size if only scratchpad_size is required in MKL
    if (!ReplInfo.WSSizeInsertAfter.empty()) {
      std::string BufferSizeArgStr = "";
      for (int i = 0; i < ReplInfo.WSSizeInfo.size(); ++i) {
        BufferSizeArgStr += i ? " ," : "";
        BufferSizeArgStr +=
          getStmtSpelling(CE->getArg(ReplInfo.WSSizeInfo[i]));
      }
      std::string ScratchpadSizeNameStr =
        "scratchpad_size_ct" +
        std::to_string(dpct::DpctGlobalInfo::getSuffixIndexInRuleThenInc());
      PrefixInsertStr +=
        IndentStr + "std::int64_t " + ScratchpadSizeNameStr +
        " = " + ReplInfo.WSSFuncName + "(*" + BufferSizeArgStr +
        ");" + getNL();
      if (ReplInfo.WSSizeInsertAfter[0] > 0) {
        emplaceTransformation(new InsertAfterStmt(
          CE->getArg(ReplInfo.WSSizeInsertAfter[0]),
          ", " + ScratchpadSizeNameStr));
      }
    }

    // Check PrefixInsertStr and SuffixInsertStr to decide whether to add bracket
    std::string PrefixWithBracket = "";
    std::string SuffixWithBracket = "";
    if (!PrefixInsertStr.empty() || !SuffixInsertStr.empty()) {
      PrefixWithBracket = "{" + std::string(getNL()) + PrefixInsertStr + IndentStr;
      SuffixWithBracket = getNL() + IndentStr + SuffixInsertStr + "}";
    }

    std::string ReplaceFuncName = Replacement;
    emplaceTransformation(
        new ReplaceText(FuncNameBegin, FuncNameLength, std::move(Replacement)));
    insertAroundRange(StmtBegin, StmtEndAfterSemi,
                      PrefixBeforeScope + PrefixWithBracket,
                      std::move(SuffixWithBracket));

    StringRef FuncNameRef(FuncName);
    if (FuncNameRef.endswith("getrf")) {
      report(StmtBegin, Diagnostics::DIFFERENT_LU_FACTORIZATION, true,
             getStmtSpelling(CE->getArg(6)), ReplaceFuncName,
             MapNames::ITFName.at(FuncName));
    }
    if (IsAssigned) {
      insertAroundRange(FuncNameBegin, FuncCallEnd,
                        std::move(AssignPrefix), std::move(AssignPostfix));
      report(StmtBegin, Diagnostics::NOERROR_RETURN_COMMA_OP, true);
    }
  } else if (FuncName == "cusolverDnCreate" ||
             FuncName == "cusolverDnDestroy") {
    std::string Repl;
    if (FuncName == "cusolverDnCreate") {
      std::string LHS;
      if (isSimpleAddrOf(CE->getArg(0))) {
        LHS = getNameStrRemovedAddrOf(CE->getArg(0));
      } else {
        dpct::ExprAnalysis EA;
        EA.analyze(CE->getArg(0));
        if (isAnIdentifierOrLiteral(CE->getArg(0)))
          LHS = "*" + EA.getReplacedString();
        else
          LHS = "*(" + EA.getReplacedString() + ")";
      }
      if (checkWhetherIsDuplicate(CE, false))
        return;
      int Index = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();
      buildTempVariableMap(Index, CE, HelperFuncType::DefaultQueue);
      Repl = LHS + " = &{{NEEDREPLACEQ" + std::to_string(Index) + "}}";
    } else if (FuncName == "cusolverDnDestroy") {
      dpct::ExprAnalysis EA(CE->getArg(0));
      Repl = EA.getReplacedString() + " = nullptr";
    } else {
      return;
    }

    if (IsAssigned) {
      report(SM->getExpansionLoc(CE->getBeginLoc()),
             Diagnostics::NOERROR_RETURN_COMMA_OP, false);
      emplaceTransformation(
          new ReplaceStmt(CE, false, FuncName, true, "(" + Repl + ", 0)"));
    } else {
      emplaceTransformation(new ReplaceStmt(CE, false, FuncName, true, Repl));
    }
  }
}

void SOLVERFunctionCallRule::getParameterEnd(
    const SourceLocation &ParameterEnd, SourceLocation &ParameterEndAfterComma,
    const ast_matchers::MatchFinder::MatchResult &Result) {
  Optional<Token> TokSharedPtr;
  TokSharedPtr = Lexer::findNextToken(ParameterEnd, *(Result.SourceManager),
                                      LangOptions());
  Token TokComma = TokSharedPtr.getValue();
  if (TokComma.getKind() == tok::comma) {
    ParameterEndAfterComma = TokComma.getEndLoc();
  } else {
    ParameterEndAfterComma = TokComma.getLocation();
  }
}

bool SOLVERFunctionCallRule::isReplIndex(int Input, std::vector<int> &IndexInfo,
                                         int &IndexTemp) {
  for (int i = 0; i < static_cast<int>(IndexInfo.size()); ++i) {
    if (IndexInfo[i] == Input) {
      IndexTemp = i;
      return true;
    }
  }
  return false;
}

// TODO: Refactoring with BLASFunctionCallRule for removing duplication
std::string SOLVERFunctionCallRule::getBufferNameAndDeclStr(
    const Expr *Arg, const ASTContext &AC, const std::string &TypeAsStr,
    SourceLocation SL, std::string &BufferDecl, int DistinctionID) {

  std::string PointerName = getStmtSpelling(Arg);
  std::string BufferTempName =
      getTempNameForExpr(Arg, true, true) + "buf_ct" +
      std::to_string(dpct::DpctGlobalInfo::getSuffixIndexInRuleThenInc());

  // TODO: reinterpret will copy more data
  BufferDecl = getIndent(SL, AC.getSourceManager()).str() + "auto " +
               BufferTempName + " = dpct::get_buffer<" + TypeAsStr + ">(" +
               PointerName + ");" + getNL();
  return BufferTempName;
}

const clang::VarDecl *
SOLVERFunctionCallRule::getAncestralVarDecl(const clang::CallExpr *CE) {
  auto &Context = dpct::DpctGlobalInfo::getContext();
  auto Parents = Context.getParents(*CE);
  while (Parents.size() == 1) {
    auto *Parent = Parents[0].get<VarDecl>();
    if (Parent) {
      return Parent;
    } else {
      Parents = Context.getParents(Parents[0]);
    }
  }
  return nullptr;
}

REGISTER_RULE(SOLVERFunctionCallRule)

void FunctionCallRule::registerMatcher(MatchFinder &MF) {
  auto functionName = [&]() {
    return hasAnyName(
        "cudaGetDeviceCount", "cudaGetDeviceProperties", "cudaDeviceReset",
        "cudaSetDevice", "cudaDeviceGetAttribute", "cudaDeviceGetP2PAttribute",
        "cudaDeviceGetPCIBusId", "cudaGetDevice", "cudaDeviceSetLimit",
        "cudaGetLastError", "cudaPeekAtLastError", "cudaDeviceSynchronize",
        "cudaThreadSynchronize", "cudaGetErrorString", "cudaGetErrorName",
        "cudaDeviceSetCacheConfig", "cudaDeviceGetCacheConfig", "clock",
        "cudaOccupancyMaxPotentialBlockSize", "cudaThreadSetLimit",
        "cudaFuncSetCacheConfig", "cudaThreadExit", "cudaDeviceGetLimit",
        "cudaDeviceSetSharedMemConfig", "cudaIpcCloseMemHandle",
        "cudaIpcGetEventHandle", "cudaIpcGetMemHandle",
        "cudaIpcOpenEventHandle", "cudaIpcOpenMemHandle", "cudaSetDeviceFlags",
        "cudaDeviceCanAccessPeer", "cudaDeviceDisablePeerAccess",
        "cudaDeviceEnablePeerAccess", "cudaDriverGetVersion",
        "cudaRuntimeGetVersion", "clock64", "__ldg");
  };

  MF.addMatcher(
      callExpr(allOf(callee(functionDecl(functionName())), parentStmt()))
          .bind("FunctionCall"),
      this);
  MF.addMatcher(callExpr(allOf(callee(functionDecl(functionName())),
                               unless(parentStmt())))
                    .bind("FunctionCallUsed"),
                this);
}

void FunctionCallRule::run(const MatchFinder::MatchResult &Result) {
  CHECKPOINT_ASTMATCHER_RUN_ENTRY();
  bool IsAssigned = false;
  const CallExpr *CE = getNodeAsType<CallExpr>(Result, "FunctionCall");
  if (!CE) {
    if (!(CE = getNodeAsType<CallExpr>(Result, "FunctionCallUsed")))
      return;
    IsAssigned = true;
  }
  if (!CE->getDirectCallee())
    return;
  std::string FuncName =
      CE->getDirectCallee()->getNameInfo().getName().getAsString();
  std::string Prefix, Suffix;
  if (IsAssigned) {
    Prefix = "(";
    Suffix = ", 0)";
  }

  if (FuncName == "cudaGetDeviceCount") {
    if (IsAssigned) {
      report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP, false);
    }
    std::string ResultVarName = DereferenceArg(CE->getArg(0), *Result.Context);
    emplaceTransformation(
        new InsertBeforeStmt(CE, Prefix + ResultVarName + " = "));
    emplaceTransformation(new ReplaceStmt(
        CE, "dpct::dev_mgr::instance().device_count()" + Suffix));
  } else if (FuncName == "cudaGetDeviceProperties") {
    if (IsAssigned) {
      report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP, false);
    }
    std::string ResultVarName = DereferenceArg(CE->getArg(0), *Result.Context);
    emplaceTransformation(new ReplaceStmt(
        CE->getCallee(), Prefix + "dpct::dev_mgr::instance().get_device"));
    emplaceTransformation(new RemoveArg(CE, 0));
    emplaceTransformation(new InsertAfterStmt(
        CE, ".get_device_info(" + ResultVarName + ")" + Suffix));
  } else if (FuncName == "cudaDriverGetVersion" ||
             FuncName == "cudaRuntimeGetVersion") {
    if (IsAssigned) {
      report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP, false);
    }
    std::string ResultVarName = DereferenceArg(CE->getArg(0), *Result.Context);
    emplaceTransformation(
        new InsertBeforeStmt(CE, Prefix + ResultVarName + " = "));

    std::string ReplStr = "dpct::get_current_device().get_info<" +
                          MapNames::getClNamespace() +
                          "::info::device::version>()";

    emplaceTransformation(new ReplaceStmt(CE, ReplStr + Suffix));
    report(CE->getBeginLoc(), Warnings::TYPE_MISMATCH, false);
  } else if (FuncName == "cudaDeviceReset" || FuncName == "cudaThreadExit") {
    if (IsAssigned) {
      report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP, false);
    }
    if (checkWhetherIsDuplicate(CE, false))
      return;
    int Index = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();
    buildTempVariableMap(Index, CE, HelperFuncType::CurrentDevice);
    emplaceTransformation(new ReplaceStmt(CE, Prefix + "{{NEEDREPLACED" +
                                                  std::to_string(Index) +
                                                  "}}.reset()" + Suffix));
  } else if (FuncName == "cudaSetDevice") {
    DpctGlobalInfo::setDeviceChangedFlag(true);
    if (IsAssigned) {
      report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP, false);
    }
    emplaceTransformation(new ReplaceStmt(
        CE->getCallee(), Prefix + "dpct::dev_mgr::instance().select_device"));
    if (IsAssigned)
      emplaceTransformation(new InsertAfterStmt(CE, ", 0)"));

  } else if (FuncName == "cudaDeviceGetAttribute") {
    std::string ResultVarName = DereferenceArg(CE->getArg(0), *Result.Context);
    std::string AttributeName = ((const clang::DeclRefExpr *)CE->getArg(1))
                                    ->getNameInfo()
                                    .getName()
                                    .getAsString();
    std::string ReplStr{ResultVarName};
    auto StmtStrArg2 = getStmtSpelling(CE->getArg(2));

    if (AttributeName == "cudaDevAttrComputeMode") {
      report(CE->getBeginLoc(), Diagnostics::COMPUTE_MODE, false);
      ReplStr += " = 1";
    } else {
      auto Search = EnumConstantRule::EnumNamesMap.find(AttributeName);
      if (Search == EnumConstantRule::EnumNamesMap.end()) {
        // TODO report migration error
        return;
      }

      ReplStr += " = dpct::dev_mgr::instance().get_device(";
      ReplStr += StmtStrArg2;
      ReplStr += ").";
      ReplStr += Search->second;
      ReplStr += "()";
    }
    if (IsAssigned)
      ReplStr = "(" + ReplStr + ", 0)";
    emplaceTransformation(new ReplaceStmt(CE, ReplStr));
  } else if (FuncName == "cudaDeviceGetP2PAttribute") {
    std::string ResultVarName = DereferenceArg(CE->getArg(0), *Result.Context);
    emplaceTransformation(new ReplaceStmt(CE, ResultVarName + " = 0"));
    report(CE->getBeginLoc(), Comments::NOTSUPPORTED, "P2P Access", false);
  } else if (FuncName == "cudaDeviceGetPCIBusId") {
    report(CE->getBeginLoc(), Comments::NOTSUPPORTED, "Get PCI BusId", false);
  } else if (FuncName == "cudaGetDevice") {
    std::string ResultVarName = DereferenceArg(CE->getArg(0), *Result.Context);
    emplaceTransformation(new InsertBeforeStmt(CE, ResultVarName + " = "));
    emplaceTransformation(
        new ReplaceStmt(CE, "dpct::dev_mgr::instance().current_device_id()"));
  } else if (FuncName == "cudaDeviceSynchronize" ||
             FuncName == "cudaThreadSynchronize") {
    if (checkWhetherIsDuplicate(CE, false))
      return;
    int Index = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();
    buildTempVariableMap(Index, CE, HelperFuncType::CurrentDevice);
    std::string ReplStr = "{{NEEDREPLACED" + std::to_string(Index) +
                          "}}.queues_wait_and_throw()";
    if (IsAssigned) {
      ReplStr = "(" + ReplStr + ", 0)";
      report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP, false);
    }
    emplaceTransformation(new ReplaceStmt(CE, std::move(ReplStr)));

  } else if (FuncName == "cudaGetLastError" ||
             FuncName == "cudaPeekAtLastError") {
    report(CE->getBeginLoc(),
           Comments::TRNA_WARNING_ERROR_HANDLING_API_REPLACED_0, false,
           MapNames::ITFName.at(FuncName));
    emplaceTransformation(new ReplaceStmt(CE, "0"));
  } else if (FuncName == "cudaGetErrorString" ||
             FuncName == "cudaGetErrorName") {
    // Insert warning messages into the spelling locations in case
    // that these functions are contained in macro definitions
    auto Loc = Result.SourceManager->getSpellingLoc(CE->getBeginLoc());
    report(Loc, Comments::TRNA_WARNING_ERROR_HANDLING_API_COMMENTED, false,
           MapNames::ITFName.at(FuncName));
    emplaceTransformation(
        new InsertBeforeStmt(CE, "\"" + FuncName + " not supported\"/*"));
    emplaceTransformation(new InsertAfterStmt(CE, "*/"));
  } else if (FuncName == "clock" || FuncName == "clock64") {
    report(CE->getBeginLoc(), Diagnostics::API_NOT_MIGRATED_SYCL_UNDEF, false, FuncName);
    // Add '#include <time.h>' directive to the file only once
    auto Loc = CE->getBeginLoc();
    DpctGlobalInfo::getInstance().insertHeader(Loc, Time);
  } else if (FuncName == "cudaDeviceSetLimit" ||
             FuncName == "cudaThreadSetLimit" ||
             FuncName == "cudaDeviceSetCacheConfig" ||
             FuncName == "cudaDeviceGetCacheConfig") {
    auto Msg = MapNames::RemovedAPIWarningMessage.find(FuncName);
    if (IsAssigned) {
      report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED_0, false,
             MapNames::ITFName.at(FuncName), Msg->second);
      emplaceTransformation(new ReplaceStmt(CE, false, FuncName, "0"));
    } else {
      report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED, false,
             MapNames::ITFName.at(FuncName), Msg->second);
      emplaceTransformation(new ReplaceStmt(CE, false, FuncName, ""));
    }
  } else if (FuncName == "cudaFuncSetCacheConfig") {
    report(CE->getBeginLoc(), Diagnostics::NOTSUPPORTED, false,
           MapNames::ITFName.at(FuncName));
  } else if (FuncName == "cudaOccupancyMaxPotentialBlockSize") {
    report(CE->getBeginLoc(), Diagnostics::NOTSUPPORTED, false,
           MapNames::ITFName.at(FuncName));
  } else if (FuncName == "cudaDeviceGetLimit") {
    ExprAnalysis EA;
    EA.analyze(CE->getArg(0));
    auto Arg0Str = EA.getReplacedString();
    std::string ReplStr{"*"};
    ReplStr += Arg0Str;
    ReplStr += " = 0";
    if (IsAssigned) {
      ReplStr = "(" + ReplStr + ", 0)";
      report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP, false);
    }
    emplaceTransformation(new ReplaceStmt(CE, std::move(ReplStr)));
    report(CE->getBeginLoc(), Diagnostics::DEVICE_LIMIT_NOT_SUPPORTED, false);
  } else if (FuncName == "cudaDeviceSetSharedMemConfig") {
    std::string Msg = "DPC++ currently does not support configuring shared "
                      "memory on devices.";
    if (IsAssigned) {
      report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED_0, false,
             MapNames::ITFName.at(FuncName), Msg);
      emplaceTransformation(new ReplaceStmt(CE, false, FuncName, "0"));
    } else {
      report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED, false,
             MapNames::ITFName.at(FuncName), Msg);
      emplaceTransformation(new ReplaceStmt(CE, false, FuncName, ""));
    }
  } else if (FuncName == "cudaSetDeviceFlags") {
    std::string Msg =
        "DPC++ currently does not support setting flags for devices.";
    if (IsAssigned) {
      report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED_0, false,
             MapNames::ITFName.at(FuncName), Msg);
      emplaceTransformation(new ReplaceStmt(CE, false, FuncName, "0"));
    } else {
      report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED, false,
             MapNames::ITFName.at(FuncName), Msg);
      emplaceTransformation(new ReplaceStmt(CE, false, FuncName, ""));
    }
  } else if (FuncName == "cudaDeviceEnablePeerAccess" ||
             FuncName == "cudaDeviceDisablePeerAccess") {
    std::string Msg =
        "DPC++ currently does not support memory access across peer devices.";
    if (IsAssigned) {
      report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED_0, false,
             MapNames::ITFName.at(FuncName), Msg);
      emplaceTransformation(new ReplaceStmt(CE, false, FuncName, "0"));
    } else {
      report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED, false,
             MapNames::ITFName.at(FuncName), Msg);
      emplaceTransformation(new ReplaceStmt(CE, false, FuncName, ""));
    }
  } else if (FuncName == "cudaDeviceCanAccessPeer") {
    ExprAnalysis EA;
    EA.analyze(CE->getArg(0));
    auto Arg0Str = EA.getReplacedString();
    std::string ReplStr{"*"};
    ReplStr += Arg0Str;
    ReplStr += " = 0";
    if (IsAssigned) {
      ReplStr = "(" + ReplStr + ", 0)";
      report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP, false);
    }
    emplaceTransformation(new ReplaceStmt(CE, std::move(ReplStr)));
    report(CE->getBeginLoc(), Diagnostics::EXPLICIT_PEER_ACCESS, false,
           MapNames::ITFName.at(FuncName));
  } else if (FuncName == "cudaIpcGetEventHandle" ||
             FuncName == "cudaIpcOpenEventHandle" ||
             FuncName == "cudaIpcGetMemHandle" ||
             FuncName == "cudaIpcOpenMemHandle" ||
             FuncName == "cudaIpcCloseMemHandle") {
    report(CE->getBeginLoc(), Diagnostics::IPC_NOT_SUPPORTED, false,
           MapNames::ITFName.at(FuncName));
  } else if (FuncName == "__ldg") {
    std::ostringstream OS;
    printDerefOp(OS, CE->getArg(0));
    emplaceTransformation(new ReplaceStmt(CE, OS.str()));
    report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED, false,
           FuncName, "there is no correspoinding API in DPC++.");
  } else {
    llvm::dbgs() << "[" << getName()
                 << "] Unexpected function name: " << FuncName;
    return;
  }
}

REGISTER_RULE(FunctionCallRule)

void EventAPICallRule::registerMatcher(MatchFinder &MF) {
  auto eventAPIName = [&]() {
    return hasAnyName("cudaEventCreate", "cudaEventCreateWithFlags",
                      "cudaEventDestroy", "cudaEventRecord",
                      "cudaEventElapsedTime", "cudaEventSynchronize",
                      "cudaEventQuery");
  };

  MF.addMatcher(
      callExpr(allOf(callee(functionDecl(eventAPIName())), parentStmt()))
          .bind("eventAPICall"),
      this);
  MF.addMatcher(callExpr(allOf(callee(functionDecl(eventAPIName())),
                               unless(parentStmt())))
                    .bind("eventAPICallUsed"),
                this);
}

void EventAPICallRule::run(const MatchFinder::MatchResult &Result) {
  CHECKPOINT_ASTMATCHER_RUN_ENTRY();
  bool IsAssigned = false;
  const CallExpr *CE = getNodeAsType<CallExpr>(Result, "eventAPICall");
  if (!CE) {
    if (!(CE = getNodeAsType<CallExpr>(Result, "eventAPICallUsed")))
      return;
    IsAssigned = true;
  }

  if (!CE->getDirectCallee())
    return;
  std::string FuncName =
      CE->getDirectCallee()->getNameInfo().getName().getAsString();

  if (FuncName == "cudaEventCreate" || FuncName == "cudaEventCreateWithFlags" ||
      FuncName == "cudaEventDestroy") {
    auto Msg = MapNames::RemovedAPIWarningMessage.find(FuncName);
    if (IsAssigned) {
      report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED_0, false,
             MapNames::ITFName.at(FuncName), Msg->second);
      emplaceTransformation(
          new ReplaceStmt(CE, /*IsReplaceCompatibilityAPI*/ false, FuncName,
                          /*IsProcessMacro*/ false, "0"));
    } else {
      report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED, false,
             MapNames::ITFName.at(FuncName), Msg->second);
      emplaceTransformation(
          new ReplaceStmt(CE, /*IsReplaceCompatibilityAPI*/ false, FuncName,
                          /*IsProcessMacro*/ false, ""));
    }
  } else if (FuncName == "cudaEventQuery") {
    ExprAnalysis EA(CE->getArg(0));
    std::string ReplStr = "(int)" + EA.getReplacedString() + ".get_info<" +
                          MapNames::getClNamespace() +
                          "::info::event::command_execution_status>()";
    emplaceTransformation(new ReplaceStmt(CE, false, FuncName, ReplStr));
  } else if (FuncName == "cudaEventRecord") {
    handleEventRecord(CE, Result, IsAssigned);
  } else if (FuncName == "cudaEventElapsedTime") {
    handleEventElapsedTime(CE, Result, IsAssigned);
  } else if (FuncName == "cudaEventSynchronize") {
    std::string ReplStr{getStmtSpelling(CE->getArg(0))};
    ReplStr += ".wait_and_throw()";
    if (IsAssigned) {
      ReplStr = "(" + ReplStr + ", 0)";
      report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP, false);
    }
    emplaceTransformation(new ReplaceStmt(CE, false, FuncName, ReplStr));
  } else {
    llvm::dbgs() << "[" << getName()
                 << "] Unexpected function name: " << FuncName;
    return;
  }
}

// Gets the declared size of the array referred in E, if E is either
// ConstantArrayType or VariableArrayType; otherwise return an empty
// string.
// For example:
// int maxSize = 100;
// int a[10];
// int b[maxSize];
//
// E1(const Expr *)-> a[2]
// E2(const Expr *)-> b[3]
// getArraySize(E1) => 10
// getArraySize(E2) => maxSize
std::string getArrayDeclSize(const Expr *E) {
  const ArrayType *AT = nullptr;
  if (auto ME = dyn_cast<MemberExpr>(E))
    AT = ME->getMemberDecl()->getType()->getAsArrayTypeUnsafe();
  else if (auto DRE = dyn_cast<DeclRefExpr>(E))
    AT = DRE->getDecl()->getType()->getAsArrayTypeUnsafe();

  if (!AT)
    return {};

  if (auto CAT = dyn_cast<ConstantArrayType>(AT))
    return std::to_string(*CAT->getSize().getRawData());
  if (auto VAT = dyn_cast<VariableArrayType>(AT))
    return ExprAnalysis::ref(VAT->getSizeExpr());
  return {};
}

// Returns true if E is array type for a MemberExpr or DeclRefExpr; returns
// false if E is pointer type.
// Requires: E is the base of ArraySubscriptExpr
bool isArrayType(const Expr *E) {
  if (auto ME = dyn_cast<MemberExpr>(E)) {
    auto AT = ME->getMemberDecl()->getType()->getAsArrayTypeUnsafe();
    return AT ? true : false;
  }
  if (auto DRE = dyn_cast<DeclRefExpr>(E)) {
    auto AT = DRE->getDecl()->getType()->getAsArrayTypeUnsafe();
    return AT ? true : false;
  }
  return false;
}

// Get the time point helper variable name for an event. The helper variable is
// declared right after its corresponding event variable.
std::string getTimePointNameForEvent(const Expr *E, bool IsDecl) {
  std::string TimePointName;
  E = E->IgnoreImpCasts();
  if (auto UO = dyn_cast<UnaryOperator>(E))
    return getTimePointNameForEvent(UO->getSubExpr(), IsDecl);
  if (auto ASE = dyn_cast<ArraySubscriptExpr>(E)) {
    auto Base = ASE->getBase()->IgnoreImpCasts();
    if (isArrayType(Base))
      return getTimePointNameForEvent(Base, IsDecl) + "[" +
        (IsDecl ? getArrayDeclSize(Base) : ExprAnalysis::ref(ASE->getIdx())) + "]";
    return getTimePointNameForEvent(Base, IsDecl) + "_" +
      ExprAnalysis::ref(ASE->getIdx());
  }
  if (auto ME = dyn_cast<MemberExpr>(E)) {
    auto Base = ME->getBase()->IgnoreImpCasts();
    return ((IsDecl || ME->isImplicitAccess()) ? "" :
            ExprAnalysis::ref(Base) + (ME->isArrow() ? "->" : ".")) +
      ME->getMemberDecl()->getNameAsString() + getCTFixedSuffix();
  }
  if (auto DRE = dyn_cast<DeclRefExpr>(E))
    return DRE->getDecl()->getNameAsString() + getCTFixedSuffix();
  return TimePointName;
}

// Get the (potentially inner) decl of E for common Expr types, including
// UnaryOperator, ArraySubscriptExpr, MemberExpr and DeclRefExpr; otherwise
// returns nullptr;
const ValueDecl *getDecl(const Expr *E) {
  E = E->IgnoreImpCasts();
  if (auto UO = dyn_cast<UnaryOperator>(E))
    return getDecl(UO->getSubExpr());
  if (auto ASE = dyn_cast<ArraySubscriptExpr>(E))
    return getDecl(ASE->getBase()->IgnoreImpCasts());
  if (auto ME = dyn_cast<MemberExpr>(E))
    return ME->getMemberDecl();
  if (auto DRE = dyn_cast<DeclRefExpr>(E))
    return DRE->getDecl();
  return nullptr;
}

void EventAPICallRule::handleEventRecord(const CallExpr *CE,
                                         const MatchFinder::MatchResult &Result,
                                         bool IsAssigned) {
  report(CE->getBeginLoc(), Diagnostics::TIME_MEASUREMENT_FOUND, false);
  DpctGlobalInfo::getInstance().insertHeader(CE->getBeginLoc(), Chrono);
  std::ostringstream Repl;

  const ValueDecl *MD = getDecl(CE->getArg(0));
  // Insert the helper variable right after the event variables
  static std::set<std::pair<const Decl *, std::string>> DeclDupFilter;
  auto &SM = DpctGlobalInfo::getSourceManager();
  std::string InsertStr = getNL();
  InsertStr += getIndent(MD->getBeginLoc(), SM).str();
  InsertStr += "std::chrono::time_point<std::chrono::high_resolution_clock> ";
  InsertStr += getTimePointNameForEvent(CE->getArg(0), true);
  InsertStr += ";";
  auto Pair = std::make_pair(MD, InsertStr);
  if (DeclDupFilter.find(Pair) == DeclDupFilter.end()) {
    DeclDupFilter.insert(Pair);
    emplaceTransformation(new InsertAfterDecl(MD, std::move(InsertStr)));
  }

  // Replace event recording with std::chrono timing
  Repl << getTimePointNameForEvent(CE->getArg(0), false)
       << " = std::chrono::high_resolution_clock::now()";
  const std::string Name =
      CE->getCalleeDecl()->getAsFunction()->getNameAsString();
  if (IsAssigned) {
    emplaceTransformation(new ReplaceStmt(CE, false, Name, "0"));
    report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_ZERO, false);
    auto OuterStmt = findNearestNonExprNonDeclAncestorStmt(CE);
    Repl << "; ";
    auto IndentLoc = CE->getBeginLoc();
    auto &SM = DpctGlobalInfo::getSourceManager();
    if (IndentLoc.isMacroID())
      IndentLoc = SM.getExpansionLoc(IndentLoc);
    Repl << getNL() << getIndent(IndentLoc, SM).str();
    emplaceTransformation(new InsertBeforeStmt(OuterStmt, std::move(Repl.str()),
                                               /*PairID*/ 0,
                                               /*DoMacroExpansion*/ true));
  } else {
    emplaceTransformation(
        new ReplaceStmt(CE, false, Name, std::move(Repl.str())));
  }
}

void EventAPICallRule::handleEventElapsedTime(
    const CallExpr *CE, const MatchFinder::MatchResult &Result,
    bool IsAssigned) {
  auto StmtStrArg0 = getStmtSpelling(CE->getArg(0));
  auto StmtStrArg1 = getTimePointNameForEvent(CE->getArg(1), false);
  auto StmtStrArg2 = getTimePointNameForEvent(CE->getArg(2), false);
  std::ostringstream Repl;
  std::string Assginee = "*(" + StmtStrArg0 + ")";
  if (auto UO = dyn_cast<UnaryOperator>(CE->getArg(0))) {
    if (UO->getOpcode() == UnaryOperatorKind::UO_AddrOf)
      Assginee = getStmtSpelling(UO->getSubExpr());
  }
  Repl << Assginee << " = std::chrono::duration<float, std::milli>("
       << StmtStrArg2 << " - " << StmtStrArg1 << ").count()";
  if (IsAssigned) {
    std::ostringstream Temp;
    Temp << "(" << Repl.str() << ", 0)";
    Repl = std::move(Temp);
    report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP, false);
  }
  const std::string Name =
      CE->getCalleeDecl()->getAsFunction()->getNameAsString();
  emplaceTransformation(
      new ReplaceStmt(CE, false, Name, std::move(Repl.str())));
  handleTimeMeasurement(CE, Result);
}

bool EventAPICallRule::IsEventArgArraySubscriptExpr(const Expr *E) {
  E = E->IgnoreImpCasts();
  if (auto UO = dyn_cast<UnaryOperator>(E))
    return IsEventArgArraySubscriptExpr(UO->getSubExpr());
  if (auto PE = dyn_cast<ParenExpr>(E))
    return IsEventArgArraySubscriptExpr(PE->getSubExpr());
  if (auto ASE = dyn_cast<ArraySubscriptExpr>(E))
    return true;
  return false;
}

const Expr *EventAPICallRule::findNextRecordedEvent(const Stmt *Parent,
                                                    unsigned KCallLoc) {
  for (auto Iter = Parent->child_begin(); Iter != Parent->child_end(); ++Iter) {
    if (auto CE = dyn_cast<CallExpr>(*Iter)) {
      if (CE->getBeginLoc().getRawEncoding() > KCallLoc &&
          CE->getDirectCallee()->getName() == "cudaEventRecord")
        return CE->getArg(0);
    }
  }
  return nullptr;
}

void EventAPICallRule::handleTimeMeasurement(
    const CallExpr *CE, const MatchFinder::MatchResult &Result) {
  auto CELoc = CE->getBeginLoc().getRawEncoding();
  auto Parents = Result.Context->getParents(*CE);
  if(Parents.size() < 1)
    return;
  auto *Parent = Parents[0].get<Stmt>();
  if (!Parent) {
    return;
  }
  const Stmt *RecordBegin = nullptr, *RecordEnd = nullptr;
  auto EventArg = CE->getArg(0);
  if (IsEventArgArraySubscriptExpr(EventArg)) {
    // If the event arg is a ArraySubscriptExpr, mark all kernels in the current
    // function to wait
    RecordBegin = *Parent->child_begin();
    for (auto Iter = Parent->child_begin(); Iter != Parent->child_end(); ++Iter)
      RecordEnd = *Iter;
  } else {
    // Find the last Event record call on start and stop
    for (auto Iter = Parent->child_begin(); Iter != Parent->child_end(); ++Iter) {
      if (Iter->getBeginLoc().getRawEncoding() > CELoc)
        continue;

      if (const CallExpr *RecordCall = dyn_cast<CallExpr>(*Iter)) {
        if (!RecordCall->getDirectCallee())
          continue;
        std::string RecordFuncName =
            RecordCall->getDirectCallee()->getNameInfo().getName().getAsString();
        // Find the last call of Event Record on start and stop before
        // calculate the time elapsed
        if (RecordFuncName == "cudaEventRecord") {
          auto Arg0 = getStmtSpelling(RecordCall->getArg(0));
          if (Arg0 == getStmtSpelling(CE->getArg(1)))
            RecordBegin = RecordCall;
          else if (Arg0 == getStmtSpelling(CE->getArg(2)))
            RecordEnd = RecordCall;
        }
      }
    }
  }
  if (!RecordBegin || !RecordEnd)
    return;

  // Find the kernel calls between start and stop
  auto RecordBeginLoc = RecordBegin->getBeginLoc().getRawEncoding();
  auto RecordEndLoc = RecordEnd->getBeginLoc().getRawEncoding();
  for (auto Iter = Parent->child_begin(); Iter != Parent->child_end(); ++Iter) {
    const CUDAKernelCallExpr *KCall = nullptr;
    if (auto *Expr = dyn_cast<ExprWithCleanups>(*Iter)) {
      auto *SubExpr = Expr->getSubExpr();
      if (auto *Call = dyn_cast<CUDAKernelCallExpr>(SubExpr))
        KCall = Call;
    } else if (auto *Call = dyn_cast<CUDAKernelCallExpr>(*Iter)) {
      KCall = Call;
    }

    if (KCall) {
      auto KCallLoc = KCall->getBeginLoc().getRawEncoding();
      // Only the kernel calls between begin and end are set to be synced
      if (KCallLoc > RecordBeginLoc && KCallLoc < RecordEndLoc) {
        auto K = DpctGlobalInfo::getInstance().insertKernelCallExpr(KCall);
        auto EventExpr = findNextRecordedEvent(Parent, KCallLoc);
        if (!EventExpr)
          EventExpr = CE->getArg(2);
        K->setEvent(ExprAnalysis::ref(EventExpr));
        K->setSync();
      }
    }
  }
}

REGISTER_RULE(EventAPICallRule)

void StreamAPICallRule::registerMatcher(MatchFinder &MF) {
  auto streamFunctionName = [&]() {
    return hasAnyName("cudaStreamCreate", "cudaStreamCreateWithFlags",
                      "cudaStreamCreateWithPriority", "cudaStreamDestroy",
                      "cudaStreamSynchronize", "cudaStreamGetPriority",
                      "cudaStreamGetFlags", "cudaDeviceGetStreamPriorityRange",
                      "cudaStreamAttachMemAsync", "cudaStreamBeginCapture",
                      "cudaStreamEndCapture", "cudaStreamIsCapturing",
                      "cudaStreamQuery", "cudaStreamWaitEvent",
                      "cudaStreamAddCallback");
  };

  MF.addMatcher(
      callExpr(allOf(callee(functionDecl(streamFunctionName())), parentStmt()))
          .bind("streamAPICall"),
      this);
  MF.addMatcher(callExpr(allOf(callee(functionDecl(streamFunctionName())),
                               unless(parentStmt())))
                    .bind("streamAPICallUsed"),
                this);
}

std::string getNewQueue(int Index) {
  extern bool AsyncHandlerFlag;
  std::string Result;
  llvm::raw_string_ostream OS(Result);
  printPartialArguments(OS << "{{NEEDREPLACED" << std::to_string(Index)
                           << "}}.create_queue(",
                        AsyncHandlerFlag ? 1 : 0, "true")
      << ")";
  return OS.str();
}

void StreamAPICallRule::run(const MatchFinder::MatchResult &Result) {
  CHECKPOINT_ASTMATCHER_RUN_ENTRY();
  bool IsAssigned = false;
  const CallExpr *CE = getNodeAsType<CallExpr>(Result, "streamAPICall");
  if (!CE) {
    if (!(CE = getNodeAsType<CallExpr>(Result, "streamAPICallUsed")))
      return;
    IsAssigned = true;
  }

  if (!CE->getDirectCallee())
    return;
  std::string FuncName =
      CE->getDirectCallee()->getNameInfo().getName().getAsString();

  if (FuncName == "cudaStreamCreate" ||
      FuncName == "cudaStreamCreateWithFlags" ||
      FuncName == "cudaStreamCreateWithPriority") {
    std::string ReplStr;
    auto StmtStr0 = getStmtSpelling(CE->getArg(0));
    // TODO: simplify expression
    if (StmtStr0[0] == '&')
      ReplStr = StmtStr0.substr(1);
    else
      ReplStr = "*(" + StmtStr0 + ")";

    if (checkWhetherIsDuplicate(CE, false))
      return;
    int Index = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();
    buildTempVariableMap(Index, CE, HelperFuncType::CurrentDevice);
    ReplStr += " = " + getNewQueue(Index);
    if (IsAssigned) {
      ReplStr = "(" + ReplStr + ", 0)";
      report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP, false);
    }
    emplaceTransformation(new ReplaceStmt(CE, false, FuncName, ReplStr));
    if (FuncName == "cudaStreamCreateWithFlags" ||
        FuncName == "cudaStreamCreateWithPriority") {
      report(CE->getBeginLoc(), Diagnostics::QUEUE_CREATED_IGNORING_OPTIONS,
             false);
    }
  } else if (FuncName == "cudaStreamDestroy") {
    auto StmtStr0 = getStmtSpelling(CE->getArg(0));
    if (checkWhetherIsDuplicate(CE, false))
      return;
    int Index = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();
    buildTempVariableMap(Index, CE, HelperFuncType::CurrentDevice);
    auto ReplStr = "{{NEEDREPLACED" + std::to_string(Index) +
                   "}}.destroy_queue(" + StmtStr0 + ")";
    if (IsAssigned) {
      ReplStr = "(" + ReplStr + ", 0)";
      report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP, false);
    }
    emplaceTransformation(new ReplaceStmt(CE, false, FuncName, ReplStr));
  } else if (FuncName == "cudaStreamSynchronize") {
    auto StmtStr = getStmtSpelling(CE->getArg(0));
    std::string ReplStr;
    if (StmtStr == "0" || StmtStr == "cudaStreamDefault" ||
        StmtStr == "cudaStreamPerThread" || StmtStr == "cudaStreamLegacy") {
      if (checkWhetherIsDuplicate(CE, false))
        return;
      int Index = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();
      buildTempVariableMap(Index, CE, HelperFuncType::DefaultQueue);
      ReplStr = "{{NEEDREPLACEQ" + std::to_string(Index) + "}}.";
    } else {
      ReplStr = StmtStr + "->";
    }
    ReplStr += "wait()";
    const std::string Name =
        CE->getCalleeDecl()->getAsFunction()->getNameAsString();
    if (IsAssigned) {
      ReplStr = "(" + ReplStr + ", 0)";
      report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP, false);
    }
    emplaceTransformation(new ReplaceStmt(CE, false, Name, ReplStr));
  } else if (FuncName == "cudaStreamGetFlags" ||
             FuncName == "cudaStreamGetPriority") {
    report(CE->getBeginLoc(), Diagnostics::STREAM_FLAG_PRIORITY_NOT_SUPPORTED,
           false);
    auto StmtStr1 = getStmtSpelling(CE->getArg(1));
    std::string ReplStr{"*("};
    ReplStr += StmtStr1;
    ReplStr += ") = 0";
    if (IsAssigned) {
      ReplStr = "(" + ReplStr + ", 0)";
      report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP, false);
    }
    const std::string Name =
        CE->getCalleeDecl()->getAsFunction()->getNameAsString();
    emplaceTransformation(new ReplaceStmt(CE, false, Name, ReplStr));
  } else if (FuncName == "cudaDeviceGetStreamPriorityRange") {
    report(CE->getBeginLoc(), Diagnostics::STREAM_FLAG_PRIORITY_NOT_SUPPORTED,
           false);
    auto StmtStr0 = getStmtSpelling(CE->getArg(0));
    auto StmtStr1 = getStmtSpelling(CE->getArg(1));
    std::string ReplStr{"*("};
    ReplStr += StmtStr0;
    ReplStr += ") = 0, *(";
    ReplStr += StmtStr1;
    ReplStr += ") = 0";
    if (IsAssigned) {
      ReplStr = "(" + ReplStr + ", 0)";
      report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP, false);
    }
    const std::string Name =
        CE->getCalleeDecl()->getAsFunction()->getNameAsString();
    emplaceTransformation(new ReplaceStmt(CE, false, Name, ReplStr));
  } else if (FuncName == "cudaStreamAttachMemAsync" ||
             FuncName == "cudaStreamBeginCapture" ||
             FuncName == "cudaStreamEndCapture" ||
             FuncName == "cudaStreamIsCapturing" ||
             FuncName == "cudaStreamQuery") {
    auto Msg = MapNames::RemovedAPIWarningMessage.find(FuncName);
    if (IsAssigned) {
      report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED_0, false,
             MapNames::ITFName.at(FuncName), Msg->second);
      emplaceTransformation(new ReplaceStmt(CE, false, FuncName, "0"));
    } else {
      report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED, false,
             MapNames::ITFName.at(FuncName), Msg->second);
      emplaceTransformation(new ReplaceStmt(CE, false, FuncName, ""));
    }
  } else if (FuncName == "cudaStreamWaitEvent") {
    auto StmtStr1 = getStmtSpelling(CE->getArg(1));
    std::string ReplStr = StmtStr1 + ".wait()";
    if (IsAssigned) {
      ReplStr = "(" + ReplStr + ", 0)";
      report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP, false);
    }
    emplaceTransformation(
        new ReplaceStmt(CE, false, FuncName, std::move(ReplStr)));
  } else if (FuncName == "cudaStreamAddCallback") {
    auto StmtStr0 = getStmtSpelling(CE->getArg(0));
    auto StmtStr1 = getStmtSpelling(CE->getArg(1));
    auto StmtStr2 = getStmtSpelling(CE->getArg(2));
    std::string ReplStr{"std::async([&]() { "};
    ReplStr += StmtStr0;
    ReplStr += "->wait(); ";
    ReplStr += StmtStr1;
    ReplStr += "(";
    ReplStr += StmtStr0;
    ReplStr += ", 0, ";
    ReplStr += StmtStr2;
    ReplStr += "); ";
    ReplStr += "})";
    if (IsAssigned) {
      ReplStr = "(" + ReplStr + ", 0)";
      report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP, false);
    }
    emplaceTransformation(new ReplaceStmt(CE, false, FuncName, ReplStr));
    DpctGlobalInfo::getInstance().insertHeader(CE->getBeginLoc(), Future);
  } else {
    llvm::dbgs() << "[" << getName()
                 << "] Unexpected function name: " << FuncName;
    return;
  }
}

REGISTER_RULE(StreamAPICallRule)

// kernel call information collection
void KernelCallRule::registerMatcher(ast_matchers::MatchFinder &MF) {
  MF.addMatcher(
      cudaKernelCallExpr(hasAncestor(functionDecl().bind("callContext")))
          .bind("kernelCall"),
      this);

  MF.addMatcher(callExpr(callee(functionDecl(hasName("cudaLaunchKernel"))))
                    .bind("launch"),
                this);
}

void KernelCallRule::run(const ast_matchers::MatchFinder::MatchResult &Result) {
  CHECKPOINT_ASTMATCHER_RUN_ENTRY();
  if (auto KCall =
          getAssistNodeAsType<CUDAKernelCallExpr>(Result, "kernelCall")) {
    auto FD = getAssistNodeAsType<FunctionDecl>(Result, "callContext");
    const auto &SM = (*Result.Context).getSourceManager();

    if (SM.isMacroArgExpansion(KCall->getCallee()->getBeginLoc())) {
      // Report warning message
      report(KCall->getBeginLoc(), Diagnostics::KERNEL_CALLEE_MACRO_ARG, false);
    }

    // Remove KCall in the original location
    emplaceTransformation(new ReplaceStmt(KCall, ""));
    removeTrailingSemicolon(KCall, Result);

    // Add kernel call to map,
    // will do code generation in Global.buildReplacements();
    if (!FD->isTemplateInstantiation())
      DpctGlobalInfo::getInstance().insertKernelCallExpr(KCall);

    if (!FD)
      return;

    // Filter out compiler generated methods
    if (const CXXMethodDecl *CXXMDecl = dyn_cast<CXXMethodDecl>(FD)) {
      if (!CXXMDecl->isUserProvided()) {
        return;
      }
    }

    auto BodySLoc = FD->getBody()->getSourceRange().getBegin().getRawEncoding();
    if (Insertions.find(BodySLoc) != Insertions.end())
      return;

    Insertions.insert(BodySLoc);
  } else if (auto LaunchKernelCall =
                 getNodeAsType<CallExpr>(Result, "launch")) {
    if (DpctGlobalInfo::getInstance().buildLaunchKernelInfo(LaunchKernelCall)) {
      emplaceTransformation(new ReplaceStmt(LaunchKernelCall, true,
                                            std::string("cudaLaunchKernel"),
                                            true, false, ""));
      removeTrailingSemicolon(LaunchKernelCall, Result);
    } else {
      report(LaunchKernelCall->getBeginLoc(), Diagnostics::NOTSUPPORTED, false);
    }
  }
}

// Find and remove the semicolon after the kernel call
void KernelCallRule::removeTrailingSemicolon(
    const CallExpr *KCall,
    const ast_matchers::MatchFinder::MatchResult &Result) {
  const auto &SM = (*Result.Context).getSourceManager();
  auto KELoc = KCall->getEndLoc();
  if (KELoc.isMacroID() && !isOuterMostMacro(KCall)) {
    KELoc = SM.getImmediateSpellingLoc(KELoc);
  }
  auto Tok = Lexer::findNextToken(KELoc, SM, LangOptions()).getValue();
  if(Tok.is(tok::TokenKind::semi))
      emplaceTransformation(new ReplaceToken(Tok.getLocation(), ""));
}

REGISTER_RULE(KernelCallRule)

// __device__ function call information collection
void DeviceFunctionCallRule::registerMatcher(ast_matchers::MatchFinder &MF) {
  MF.addMatcher(
      callExpr(hasAncestor(functionDecl(anyOf(hasAttr(attr::CUDADevice),
                                              hasAttr(attr::CUDAGlobal)))
                               .bind("funcDecl")))
          .bind("callExpr"),
      this);

  MF.addMatcher(
      functionDecl(anyOf(hasAttr(attr::CUDADevice), hasAttr(attr::CUDAGlobal)))
          .bind("funcDecl"),
      this);

  MF.addMatcher(
      callExpr(hasAncestor(functionDecl(anyOf(hasAttr(attr::CUDADevice),
                                              hasAttr(attr::CUDAGlobal)),
                                        unless(hasAttr(attr::CUDAHost)))
                               .bind("PrintfInfuncDecl")),
               callee(functionDecl(hasName("printf"))))
          .bind("PrintfExpr"),
      this);

  MF.addMatcher(
      callExpr(hasAncestor(functionDecl(hasAttr(attr::CUDADevice),
                                        hasAttr(attr::CUDAHost))),
               callee(functionDecl(hasName("printf"))))
          .bind("PrintfExprForReport"),
      this);
}

void DeviceFunctionCallRule::run(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  CHECKPOINT_ASTMATCHER_RUN_ENTRY();
  auto CE = getAssistNodeAsType<CallExpr>(Result, "callExpr");
  auto FD = getAssistNodeAsType<FunctionDecl>(Result, "funcDecl");
  if (FD) {
    auto FuncInfo = DeviceFunctionDecl::LinkRedecls(FD);
    if (CE) {
      FuncInfo->addCallee(CE);
    }
  }

  CE = getAssistNodeAsType<CallExpr>(Result, "PrintfExpr");
  FD = getAssistNodeAsType<FunctionDecl>(Result, "PrintfInfuncDecl");
  if (CE && FD) {
    auto FuncInfo = DeviceFunctionDecl::LinkRedecls(FD);
    std::string ReplacedStmt;
    llvm::raw_string_ostream OS(ReplacedStmt);
    OS << DpctGlobalInfo::getStreamName() << " << ";
    CE->getArg(0)->printPretty(OS, nullptr,
                                Result.Context->getPrintingPolicy());
    emplaceTransformation(new ReplaceStmt(CE, std::move(OS.str())));
    if (CE->getNumArgs() > 1 ||
        CE->getArg(0)->IgnoreImplicitAsWritten()->getStmtClass() !=
            Stmt::StringLiteralClass)
      report(CE->getBeginLoc(), Warnings::PRINTF_FUNC_MIGRATION_WARNING,
              false);
    FuncInfo->setStream();
  }

  CE = getAssistNodeAsType<CallExpr>(Result, "PrintfExprForReport");
  if(CE) {
    report(CE->getBeginLoc(), Warnings::PRINTF_FUNC_NOT_SUPPORT, false);
  }
}

REGISTER_RULE(DeviceFunctionCallRule)

/// __constant__/__shared__/__device__ var information collection
void MemVarRule::registerMatcher(MatchFinder &MF) {
  auto DeclMatcher =
      varDecl(anyOf(hasAttr(attr::CUDAConstant), hasAttr(attr::CUDADevice),
                    hasAttr(attr::CUDAShared), hasAttr(attr::CUDAManaged)),
              unless(hasAnyName("threadIdx", "blockDim", "blockIdx", "gridDim",
                                "warpSize")));
  MF.addMatcher(DeclMatcher.bind("var"), this);
  MF.addMatcher(
      declRefExpr(anyOf(hasParent(implicitCastExpr(
                                      unless(hasParent(arraySubscriptExpr())))
                                      .bind("impl")),
                        anything()),
                  to(DeclMatcher.bind("var")),
                  hasAncestor(functionDecl().bind("func")))
          .bind("used"),
      this);

  MF.addMatcher(varDecl(hasParent(translationUnitDecl())).bind("hostGlobalVar"),
                this);

}

void MemVarRule::processDeref(const Stmt *S, ASTContext &Context) {
  auto Parents = Context.getParents(*S);
  if (Parents.size()) {
    auto &Parent = Parents[0];
    if (auto ICE = Parent.get<ImplicitCastExpr>()) {
      processDeref(ICE, Context);
    } else if (auto ME = Parent.get<MemberExpr>()) {
      emplaceTransformation(new ReplaceToken(ME->getOperatorLoc(), "->"));
    } else if (auto CDSME = Parent.get<CXXDependentScopeMemberExpr>()) {
      emplaceTransformation(new ReplaceToken(CDSME->getOperatorLoc(), "->"));
    } else if (Parent.get<BinaryOperator>() || Parent.get<CallExpr>() ||
               Parent.get<CXXConstructExpr>() || Parent.get<ParenExpr>()) {
      emplaceTransformation(new InsertBeforeStmt(S, "*"));

    } else if (auto UO = Parent.get<UnaryOperator>()) {
      if (UO->getOpcode() == UnaryOperatorKind::UO_AddrOf) {
        emplaceTransformation(new ReplaceToken(UO->getOperatorLoc(), ""));
      } else {
        insertAroundStmt(S, "(*", ")");
      }
    } else {
      insertAroundStmt(S, "(*", ")");
    }
  }
}

void MemVarRule::run(const MatchFinder::MatchResult &Result) {
  CHECKPOINT_ASTMATCHER_RUN_ENTRY();
  if (auto MemVar = getNodeAsType<VarDecl>(Result, "var")) {
    emplaceTransformation(ReplaceVarDecl::getVarDeclReplacement(
        MemVar,
        MemVarInfo::buildMemVarInfo(MemVar)->getDeclarationReplacement()));
  }
  auto MemVarRef = getNodeAsType<DeclRefExpr>(Result, "used");
  auto Func = getAssistNodeAsType<FunctionDecl>(Result, "func");
  DpctGlobalInfo &Global = DpctGlobalInfo::getInstance();
  if (MemVarRef && Func) {
    auto VD = dyn_cast<VarDecl>(MemVarRef->getDecl());
    if (VD == nullptr)
      return;

    auto Var = Global.findMemVarInfo(VD);
    if (Func->hasAttr<CUDAGlobalAttr>() ||
        (Func->hasAttr<CUDADeviceAttr>() && !Func->hasAttr<CUDAHostAttr>())) {
      if (Var)
        DeviceFunctionDecl::LinkRedecls(Func)->addVar(Var);
      if (!VD->getType()->isArrayType() && !VD->hasAttr<CUDAConstantAttr>()) {
        processDeref(MemVarRef, *Result.Context);
      }
    } else {
      if (Var && !VD->getType()->isArrayType() &&
          VD->hasAttr<CUDAManagedAttr>()) {
        emplaceTransformation(new InsertAfterStmt(MemVarRef, "[0]"));
      }
    }
  }

  if (auto VD = getNodeAsType<VarDecl>(Result, "hostGlobalVar")) {
    auto VarName = VD->getNameAsString();
    auto TypeName = VD->getType().getAsString();
    bool IsHost =
        !(VD->hasAttr<CUDAConstantAttr>() || VD->hasAttr<CUDADeviceAttr>() ||
          VD->hasAttr<CUDASharedAttr>());
    if(IsHost)
      dpct::DpctGlobalInfo::getGlobalVarNameSet().insert(VarName);
  }
}

REGISTER_RULE(MemVarRule)

std::string MemoryMigrationRule::getTypeStrRemovedAddrOf(const Expr *E,
                                                         bool isCOCE) {
  QualType QT;
  if (isCOCE) {
    auto COCE = dyn_cast<CXXOperatorCallExpr>(E);
    if (!COCE) {
      return "";
    }
    QT = COCE->getArg(0)->getType();
  } else {
    auto UO = dyn_cast<UnaryOperator>(E);
    if (!UO) {
      return "";
    }
    QT = UO->getSubExpr()->getType();
  }
  std::string ReplType = DpctGlobalInfo::getReplacedTypeName(QT);
  return ReplType;
}

/// Get the assigned part of the malloc function call.
/// \param [in] E The expression need to be analyzed.
/// \param [in] Arg0Str The original string of the first argument of the malloc.
/// e.g.:
/// origin code:
///   int2 const * d_data;
///   cudaMalloc((void **)&d_data, sizeof(int2));
/// This function will return a string "d_data = (sycl::int2 const *)"
/// In this example, \param E is "&d_data", \param Arg0Str is "(void **)&d_data"
std::string MemoryMigrationRule::getAssignedStr(const Expr *E,
                                                const std::string &Arg0Str) {
  std::ostringstream Repl;
  if (isSimpleAddrOf(E)) {
    Repl << getNameStrRemovedAddrOf(E);
    Repl << " = (" << getTypeStrRemovedAddrOf(E) << ")";
  } else if (isCOCESimpleAddrOf(E)) {
    Repl << getNameStrRemovedAddrOf(E, true);
    Repl << " = (" << getTypeStrRemovedAddrOf(E, true) << ")";
  } else {
    Repl << "*" << Arg0Str;
    auto QT = E->getType().getTypePtr()->getPointeeType();
    std::string ReplType = DpctGlobalInfo::getReplacedTypeName(QT);
    Repl << " = (" << ReplType << ")";
  }
  return Repl.str();
}

const ArraySubscriptExpr *
MemoryMigrationRule::getArraySubscriptExpr(const Expr *E) {
  if (const auto MTE = dyn_cast<MaterializeTemporaryExpr>(E)) {
    if (auto TE = MTE->getSubExpr()) {
      if (auto UO = dyn_cast<UnaryOperator>(TE)) {
        if (auto Arg = dyn_cast<ArraySubscriptExpr>(UO->getSubExpr())) {
          return Arg;
        }
      }
    }
  }
  return nullptr;
}

const Expr *MemoryMigrationRule::getUnaryOperatorExpr(const Expr *E) {
  if (const auto MTE = dyn_cast<MaterializeTemporaryExpr>(E)) {
    if (auto TE = MTE->getSubExpr()) {
      if (auto UO = dyn_cast<UnaryOperator>(TE)) {
        return UO->getSubExpr();
      }
    }
  }
  return nullptr;
}

llvm::raw_ostream &printMemcpy3DParmsName(llvm::raw_ostream &OS,
                                          StringRef BaseName,
                                          StringRef MemberName) {
  return OS << BaseName << "_" << MemberName << getCTFixedSuffix();
}

void MemoryMigrationRule::replaceMemAPIArg(
    const Expr *E, const ast_matchers::MatchFinder::MatchResult &Result,
    std::string OffsetFromBaseStr) {

  auto ASE = getArraySubscriptExpr(E);
  const clang::Expr *BASE = nullptr;
  if (ASE) {
    BASE = ASE->getBase();
  }

  auto UO = getUnaryOperatorExpr(E);
  std::shared_ptr<clang::dpct::MemVarInfo> VI = nullptr;
  if (BASE &&
      (VI = DpctGlobalInfo::getInstance().findMemVarInfo(getVarDecl(BASE)))) {
    // Migrate the expr such as "&const_angle[3]" to
    // const_angle.get_ptr() + sizeof(TYPE) * (3)",
    // and "&const_angle[3]" to "const_angle.get_ptr()".
    std::string VarName = VI->getName();
    VarName += ".get_ptr()";

    bool IsOffsetNeeded = true;
    Expr::EvalResult ER;

    if (ASE->getIdx()->EvaluateAsInt(ER, *Result.Context)) {
      auto ExprValue = ER.Val.getAsString(*Result.Context, ASE->getType());
      if (ExprValue == "0") {
        IsOffsetNeeded = false;
      }
    }

    if (IsOffsetNeeded) {
      std::string Type = ASE->getType().getAsString();
      ExprAnalysis EA;
      EA.analyze(ASE->getIdx());
      auto StmtStrArg = EA.getReplacedString();
      std::string Offset = " + sizeof(" + Type + ") * (" + StmtStrArg + ")";
      VarName += Offset;
    }

    if (!OffsetFromBaseStr.empty()) {
      VarName = "(char *)(" + VarName + ") + " + OffsetFromBaseStr;
    }
    emplaceTransformation(
        new ReplaceToken(E->getBeginLoc(), E->getEndLoc(), std::move(VarName)));
  } else if (UO && (VI = DpctGlobalInfo::getInstance().findMemVarInfo(
                        getVarDecl(UO)))) {
    // Migrate the expr such as "&const_one" to "const_one.get_ptr()".
    std::string VarName = VI->getName();
    VarName += ".get_ptr()";

    if (!OffsetFromBaseStr.empty()) {
      VarName = "(char *)(" + VarName + ") + " + OffsetFromBaseStr;
    }
    emplaceTransformation(
        new ReplaceToken(E->getBeginLoc(), E->getEndLoc(), std::move(VarName)));
  } else if (VI = DpctGlobalInfo::getInstance().findMemVarInfo(getVarDecl(E))) {
    // Migrate the expr such as "const_one" to "const_one.get_ptr()".
    std::string VarName = VI->getName();
    VarName += ".get_ptr()";

    if (!OffsetFromBaseStr.empty()) {
      VarName = "(char *)(" + VarName + ") + " + OffsetFromBaseStr;
    }
    emplaceTransformation(
        new ReplaceToken(E->getBeginLoc(), E->getEndLoc(), std::move(VarName)));
  }
}

// Incase the previous arg is another macro or function-like macro,
// we take 1 token(the last token of Expr, because of the design of Clang's
// EndLoc) after getExpansionRange().getEnd() as the real end location. If the
// call expr is in a function-like macro or nested macros, to get the correct
// loc of the previous arg, we need to use getImmediateSpellingLoc step by
// step until reaching a FileID or a non-function-like macro. E.g.
// MACRO_A(MACRO_B(callexpr(arg1, arg2, arg3)));
// When we try to remove arg3, Begin should be at the end of arg2.
// However, the expansionLoc of Begin is at the beginning of MACRO_A.
// After 1st time of Begin=SM.getImmediateSpellingLoc(Begin),
// Begin is at the beginning of MACRO_B.
// After 2nd time of Begin=SM.getImmediateSpellingLoc(Begin),
// Begin is at the beginning of arg2.
// CANNOT use SM.getSpellingLoc because arg2 might be a simple macro,
// and SM.getSpellingLoc will return the macro definition in this case.
CharSourceRange getAccurateExpansionRange(SourceLocation Loc,
                                          const SourceManager &SM) {
  while (Loc.isMacroID() && !SM.isAtStartOfImmediateMacroExpansion(Loc)) {
    auto ISL = SM.getImmediateSpellingLoc(Loc);
    if (!DpctGlobalInfo::isInRoot(
            SM.getFilename(SM.getExpansionLoc(ISL)).str()))
      break;
    Loc = ISL;
  }
  return SM.getExpansionRange(Loc);
}

/// Get the accurate begin expansion location of argument \p ArgIndex of call \p
/// C.
SourceLocation getArgEndLocation(const CallExpr *C, size_t ArgIndex,
                                 const SourceManager &SM) {
  auto Loc =
      getAccurateExpansionRange(C->getArg(ArgIndex)->getEndLoc(), SM).getEnd();
  return Loc.getLocWithOffset(Lexer::MeasureTokenLength(
      SM.getExpansionLoc(Loc), SM,
      dpct::DpctGlobalInfo::getContext().getLangOpts()));
}

/// Get the accurate end expansion location of argument \p ArgIndex of call \p
/// C.
SourceLocation getArgBeginLocation(const CallExpr *C, size_t ArgIndex,
                                   const SourceManager &SM) {
  return getAccurateExpansionRange(C->getArg(ArgIndex)->getBeginLoc(), SM)
      .getBegin();
}

TextModification *replaceText(SourceLocation Begin, SourceLocation End,
                              std::string &&Str, const SourceManager &SM) {
  auto Length = SM.getFileOffset(End) - SM.getFileOffset(Begin);
  if (Length > 0) {
    return new ReplaceText(Begin, Length, std::move(Str));
  }
  return nullptr;
}

/// Return a TextModication that removes nth argument of the CallExpr,
/// together with the preceding comma.
TextModification *removeArg(const CallExpr *C, unsigned n,
                            const SourceManager &SM) {
  if (C->getNumArgs() <= n)
    return nullptr;
  if (C->getArg(n)->isDefaultArgument())
    return nullptr;

  SourceLocation Begin, End;
  if (n) {
    Begin = getArgEndLocation(C, n - 1, SM);
    End = getArgEndLocation(C, n, SM);
  } else {
    Begin = getArgBeginLocation(C, n, SM);
    if (C->getNumArgs() > 1)
      End = getArgBeginLocation(C, n + 1, SM);
    else
      End = getArgEndLocation(C, n, SM);
  }

  return replaceText(Begin, End, "", SM);
}

bool MemoryMigrationRule::canUseTemplateStyleMigration(
    const Expr *AllocatedExpr, const Expr *SizeExpr, std::string &ReplType,
    std::string &ReplSize) {
  const Expr *AE = nullptr;
  if (auto CSCE = dyn_cast<CStyleCastExpr>(AllocatedExpr)) {
    AE = CSCE->getSubExpr()->IgnoreImplicitAsWritten();
  } else {
    AE = AllocatedExpr;
  }

  QualType DerefQT = AE->getType();
  if (DerefQT->isPointerType()) {
    DerefQT = DerefQT->getPointeeType();
    if (DerefQT->isPointerType()) {
      DerefQT = DerefQT->getPointeeType();
    } else {
      return false;
    }
  } else {
    return false;
  }

  std::string TypeStr = DpctGlobalInfo::getReplacedTypeName(DerefQT);
  // ReplType will be used as the template arguement in memory API.
  ReplType = TypeStr;

  auto BO = dyn_cast<BinaryOperator>(SizeExpr);
  if (BO && BO->getOpcode() == BinaryOperatorKind::BO_Mul) {
    SourceLocation RemoveBegin, RemoveEnd;
    if (isSameSizeofTypeWithTypeStr(BO->getLHS(), TypeStr)) {
      // case 1: sizeof(b) * a
      RemoveBegin = BO->getBeginLoc();
      RemoveEnd = BO->getOperatorLoc();
    } else if (isSameSizeofTypeWithTypeStr(BO->getRHS(), TypeStr)) {
      // case 2: a * sizeof(b)
      RemoveBegin = BO->getOperatorLoc();
      RemoveEnd = BO->getEndLoc();
    } else {
      return false;
    }

    RemoveBegin =
        DpctGlobalInfo::getSourceManager().getExpansionLoc(RemoveBegin);
    RemoveEnd = DpctGlobalInfo::getSourceManager().getExpansionLoc(RemoveEnd);
    RemoveEnd = RemoveEnd.getLocWithOffset(
        Lexer::MeasureTokenLength(RemoveEnd, DpctGlobalInfo::getSourceManager(),
                                  DpctGlobalInfo::getContext().getLangOpts()));
    emplaceTransformation(replaceText(RemoveBegin, RemoveEnd, "",
                                      DpctGlobalInfo::getSourceManager()));
    return true;
  } else {
    // case 3: sizeof(b)
    if (isSameSizeofTypeWithTypeStr(SizeExpr, TypeStr)) {
      emplaceTransformation(new ReplaceStmt(SizeExpr, "1"));
      return true;
    }
  }
  return false;
}

/// Transform cudaMallocxxx() to xxx = mallocxxx();
void MemoryMigrationRule::mallocMigrationWithTransformation(
    SourceManager &SM, const CallExpr *C, const std::string &CallName,
    std::string &&ReplaceName, const std::string &PaddingArgs,
    bool NeedTypeCast, size_t AllocatedArgIndex, size_t SizeArgIndex) {
  std::string ReplSize, ReplType;
  if (DpctGlobalInfo::getUsmLevel() == UsmLevel::restricted &&
      CallName != "cudaMallocArray" && CallName != "cudaMalloc3DArray" &&
      CallName != "cublasAlloc" &&
      canUseTemplateStyleMigration(C->getArg(AllocatedArgIndex),
                                   C->getArg(SizeArgIndex), ReplType,
                                   ReplSize)) {
    emplaceTransformation(
        new InsertBeforeStmt(C,
                             getTransformedMallocPrefixStr(
                                 C->getArg(AllocatedArgIndex), NeedTypeCast, true),
                             InsertPosition::InsertPositionRight));
    emplaceTransformation(
        new ReplaceCalleeName(C, ReplaceName + "<" + ReplType + ">", CallName));
  } else {
    emplaceTransformation(
        new InsertBeforeStmt(C,
                             getTransformedMallocPrefixStr(
                                 C->getArg(AllocatedArgIndex), NeedTypeCast),
                             InsertPosition::InsertPositionRight));
    emplaceTransformation(
        new ReplaceCalleeName(C, std::move(ReplaceName), CallName));
  }
  emplaceTransformation(removeArg(C, AllocatedArgIndex, SM));
  if (!PaddingArgs.empty())
    emplaceTransformation(
        new InsertText(C->getRParenLoc(), ", " + PaddingArgs));
}

/// e.g., for int *a and cudaMalloc(&a, size), print "a = ".
/// If \p DerefType is not null, assign a string "int *".
void printDerefOp(std::ostream &OS, const Expr *E, std::string *DerefType) {
  E = E->IgnoreImplicitAsWritten();
  bool NeedDerefOp = true;
  if (auto UO = dyn_cast<UnaryOperator>(E)) {
    if (UO->getOpcode() == clang::UO_AddrOf) {
      E = UO->getSubExpr()->IgnoreImplicitAsWritten();
      NeedDerefOp = false;
    }
  } else if (auto COCE = dyn_cast<CXXOperatorCallExpr>(E)) {
    if (COCE->getOperator() == clang::OO_Amp && COCE->getNumArgs() == 1) {
      E = COCE->getArg(0)->IgnoreImplicitAsWritten();
      NeedDerefOp = false;
    }
  }

  std::unique_ptr<ParensPrinter<std::ostream>> PP;
  if (NeedDerefOp) {
    OS << "*";
    switch (E->getStmtClass()) {
    case Stmt::DeclRefExprClass:
    case Stmt::MemberExprClass:
    case Stmt::ParenExprClass:
    case Stmt::CallExprClass:
      break;
    default:
      PP = std::make_unique<ParensPrinter<std::ostream>>(OS);
      break;
    }
  }
  ExprAnalysis EA(E);
  EA.analyze();
  OS << EA.getReplacedString();

  if (DerefType) {
    QualType DerefQT = E->getType();
    if (NeedDerefOp)
      DerefQT = DerefQT->getPointeeType();
    *DerefType = DpctGlobalInfo::getReplacedTypeName(DerefQT);
  }
}

/// e.g., for int *a and cudaMalloc(&a, size), return "a = (int *)".
/// If \p NeedTypeCast is false, return "a = ";
/// If \p TemplateStyle is true, \p NeedTypeCast will be specified as false always
std::string
MemoryMigrationRule::getTransformedMallocPrefixStr(const Expr *MallocOutArg,
                                                   bool NeedTypeCast,
                                                   bool TemplateStyle) {
  if (TemplateStyle)
    NeedTypeCast = false;
  std::ostringstream OS;
  std::string CastTypeName;
  MallocOutArg = MallocOutArg->IgnoreImplicitAsWritten();
  if (auto CSCE = dyn_cast<CStyleCastExpr>(MallocOutArg)) {
    MallocOutArg = CSCE->getSubExpr()->IgnoreImplicitAsWritten();
    if (!TemplateStyle)
      NeedTypeCast = true;
  }
  printDerefOp(OS, MallocOutArg, NeedTypeCast ? &CastTypeName : nullptr);

  OS << " = ";
  if (!CastTypeName.empty())
    OS << "(" << CastTypeName << ")";

  return OS.str();
}

/// Common migration for cudaMallocArray and cudaMalloc3DArray.
void MemoryMigrationRule::mallocArrayMigration(const CallExpr *C,
                                               const std::string &Name,
                                               size_t FlagIndex,
                                               SourceManager &SM) {
  mallocMigrationWithTransformation(SM, C, Name, "new dpct::image_matrix", "",
                                    false);
  emplaceTransformation(removeArg(C, FlagIndex, SM));

  std::ostringstream OS;
  printDerefOp(OS, C->getArg(1));
  emplaceTransformation(new ReplaceStmt(C->getArg(1), OS.str()));
}

void MemoryMigrationRule::mallocMigration(
    const MatchFinder::MatchResult &Result, const CallExpr *C,
    const UnresolvedLookupExpr *ULExpr, bool IsAssigned) {
  std::string Name;
  if (ULExpr) {
    Name = ULExpr->getName().getAsString();
  } else {
    Name = C->getCalleeDecl()->getAsFunction()->getNameAsString();
  }

  if (checkWhetherIsDuplicate(C, false))
    return;
  int Index = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();

  if (Name == "cudaMalloc") {
    DpctGlobalInfo::getInstance().insertCudaMalloc(C);
    if (USMLevel == UsmLevel::restricted) {
      buildTempVariableMap(Index, C, HelperFuncType::DefaultQueue);
      mallocMigrationWithTransformation(
          *Result.SourceManager, C, Name,
          MapNames::getClNamespace() + "::malloc_device",
          "{{NEEDREPLACEQ" + std::to_string(Index) + "}}");
    } else {
      emplaceTransformation(
          new ReplaceCalleeName(C, "dpct::dpct_malloc", Name));
    }
  } else if (Name == "cudaHostAlloc" || Name == "cudaMallocHost") {
    std::string ReplaceName;
    if (USMLevel == UsmLevel::restricted) {
      buildTempVariableMap(Index, C, HelperFuncType::DefaultQueue);
      mallocMigrationWithTransformation(
          *Result.SourceManager, C, Name,
          MapNames::getClNamespace() + "::malloc_host",
          "{{NEEDREPLACEQ" + std::to_string(Index) + "}}");
    }else{
      mallocMigrationWithTransformation(*Result.SourceManager, C, Name,
                                        "malloc");
    }
    emplaceTransformation(removeArg(C, 2, *Result.SourceManager));
  } else if (Name == "cudaMallocManaged") {
    if (USMLevel == UsmLevel::restricted) {
      buildTempVariableMap(Index, C, HelperFuncType::DefaultQueue);
      mallocMigrationWithTransformation(
          *Result.SourceManager, C, Name,
          MapNames::getClNamespace() + "::malloc_shared",
          "{{NEEDREPLACEQ" + std::to_string(Index) + "}}");
      emplaceTransformation(removeArg(C, 2, *Result.SourceManager));
    } else {
      // Report unsupported warnings
      report(C->getBeginLoc(), Diagnostics::NOTSUPPORTED, false,
             MapNames::ITFName.at(Name));
    }
  } else if (Name == "cublasAlloc") {
    // TODO: migrate functions when they are in template
    // TODO: migrate functions when they are in macro body
    emplaceTransformation(
        replaceText(getArgEndLocation(C, 0, *Result.SourceManager),
                    getArgBeginLocation(C, 1, *Result.SourceManager), "*",
                    *Result.SourceManager));
    insertAroundStmt(C->getArg(0), "(", ")");
    insertAroundStmt(C->getArg(1), "(", ")");
    DpctGlobalInfo::getInstance().insertCublasAlloc(C);
    emplaceTransformation(removeArg(C, 2, *Result.SourceManager));
    if (USMLevel == UsmLevel::restricted) {
      buildTempVariableMap(Index, C, HelperFuncType::DefaultQueue);
      mallocMigrationWithTransformation(
          *Result.SourceManager, C, Name,
          MapNames::getClNamespace() + "::malloc_device",
          "{{NEEDREPLACEQ" + std::to_string(Index) + "}}", true, 2);
    } else {
      ExprAnalysis EA(C->getArg(2));
      EA.analyze();
      emplaceTransformation(
          new ReplaceCalleeName(C, "dpct::dpct_malloc", Name));
      emplaceTransformation(
          new InsertBeforeStmt(C->getArg(0), EA.getReplacedString() + ", ",
                               InsertPosition::InsertPositionAlwaysLeft));
    }
  } else if (Name == "cudaMallocPitch" || Name == "cudaMalloc3D") {
    emplaceTransformation(new ReplaceCalleeName(C, "dpct::dpct_malloc", Name));
  } else if (Name == "cudaMalloc3DArray") {
    mallocArrayMigration(C, Name, 3, *Result.SourceManager);
  } else if (Name == "cudaMallocArray") {
    mallocArrayMigration(C, Name, 4, *Result.SourceManager);
    static std::string SizeClassName =
        DpctGlobalInfo::getCtadClass(MapNames::getClNamespace() + "::range", 2);
    if (C->getArg(3)->isDefaultArgument())
      aggregateArgsToCtor(C, SizeClassName, 2, 2, ", 0", *Result.SourceManager);
    else
      aggregateArgsToCtor(C, SizeClassName, 2, 3, "", *Result.SourceManager);
  }
}

void MemoryMigrationRule::memcpyMigration(
    const MatchFinder::MatchResult &Result, const CallExpr *C,
    const UnresolvedLookupExpr *ULExpr, bool IsAssigned) {
  std::string Name;
  if (ULExpr) {
    Name = ULExpr->getName().getAsString();
  } else {
    Name = C->getCalleeDecl()->getAsFunction()->getNameAsString();
  }

  std::string ReplaceStr;
  StringRef NameRef(Name);
  bool IsAsync = NameRef.endswith("Async");
  if (IsAsync) {
    NameRef = NameRef.drop_back(5 /* len of "Async" */);
    ReplaceStr = "dpct::async_dpct_memcpy";
  } else {
    ReplaceStr = "dpct::dpct_memcpy";
  }

  if (NameRef == "cudaMemcpy2D") {
    handleDirection(C, 6);
    handleAsync(C, 7, Result);
  } else if (NameRef == "cudaMemcpy3D") {
    handleAsync(C, 1, Result);
    if (auto UO =
            dyn_cast<UnaryOperator>(C->getArg(0)->IgnoreImplicitAsWritten())) {
      if (auto DRE = dyn_cast<DeclRefExpr>(
              UO->getSubExpr()->IgnoreImplicitAsWritten())) {
        emplaceTransformation(new ReplaceStmt(
            C->getArg(0), MemoryDataTypeRule::getMemcpy3DArguments(
                              DRE->getDecl()->getName())));
      }
    }
  } else if (NameRef == "cudaMemcpy") {
    handleDirection(C, 3);
    replaceMemAPIArg(C->getArg(0), Result);
    replaceMemAPIArg(C->getArg(1), Result);
    if (USMLevel == UsmLevel::restricted) {
      emplaceTransformation(removeArg(C, 3, *Result.SourceManager));
      if (IsAsync) {
        emplaceTransformation(removeArg(C, 4, *Result.SourceManager));
      } else {
        emplaceTransformation(new InsertAfterStmt(C, ".wait()"));
      }
      std::string AsyncQueue;
      if (C->getNumArgs() > 4 && !C->getArg(4)->isDefaultArgument()) {
        if (!isPredefinedStreamHandle(C->getArg(4)))
          AsyncQueue = ExprAnalysis::ref(C->getArg(4));
      }
      if (AsyncQueue.empty()) {
        if (checkWhetherIsDuplicate(C, false))
          return;
        int Index = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();
        buildTempVariableMap(Index, C, HelperFuncType::DefaultQueue);
        ReplaceStr = "{{NEEDREPLACEQ" + std::to_string(Index) + "}}.memcpy";
      } else {
        ReplaceStr = AsyncQueue + "->memcpy";
      }
    } else {
      handleAsync(C, 4, Result);
    }
  }

  if (ULExpr)
    emplaceTransformation(new ReplaceToken(
        ULExpr->getBeginLoc(), ULExpr->getEndLoc(), std::move(ReplaceStr)));
  else
    emplaceTransformation(
        new ReplaceCalleeName(C, std::move(ReplaceStr), Name));
}

void MemoryMigrationRule::arrayMigration(
    const ast_matchers::MatchFinder::MatchResult &Result, const CallExpr *C,
    const UnresolvedLookupExpr *ULExpr, bool IsAssigned) {
  std::string Name;
  if (ULExpr) {
    Name = ULExpr->getName().getAsString();
  } else {
    Name = C->getCalleeDecl()->getAsFunction()->getNameAsString();
  }

  std::string ReplaceStr;
  StringRef NameRef(Name);
  bool IsAsync = NameRef.endswith("Async");
  if (IsAsync) {
    NameRef = NameRef.drop_back(5 /* len of "Async" */);
    ReplaceStr = "dpct::async_dpct_memcpy";
  } else {
    ReplaceStr = "dpct::dpct_memcpy";
  }

  auto&SM = *Result.SourceManager;
  if (NameRef == "cudaMemcpy2DArrayToArray") {
    insertToPitchedData(C, 0);
    aggregate3DVectorClassCtor(C, "id", 1, "0", SM);
    insertToPitchedData(C, 3);
    aggregate3DVectorClassCtor(C, "id", 4, "0", SM);
    aggregate3DVectorClassCtor(C, "range", 6, "1", SM);
    emplaceTransformation(removeArg(C, 8, SM));
  } else if (NameRef == "cudaMemcpy2DFromArray") {
    handleAsync(C, 8, Result);
    emplaceTransformation(removeArg(C, 7, *Result.SourceManager));
    aggregatePitchedData(C, 0, 1, SM);
    insertZeroOffset(C, 2);
    insertToPitchedData(C, 2);
    aggregate3DVectorClassCtor(C, "id", 3, "0", SM);
    aggregate3DVectorClassCtor(C, "range", 5, "1", SM);
  } else if (NameRef == "cudaMemcpy2DToArray") {
    handleAsync(C, 8, Result);
    emplaceTransformation(removeArg(C, 7, *Result.SourceManager));
    insertToPitchedData(C, 0);
    aggregate3DVectorClassCtor(C, "id", 1, "0", SM);
    aggregatePitchedData(C, 3, 4, SM);
    insertZeroOffset(C, 5);
    aggregate3DVectorClassCtor(C, "range", 5, "1", SM);
  } else if (NameRef == "cudaMemcpyArrayToArray") {
    insertToPitchedData(C, 0);
    aggregate3DVectorClassCtor(C, "id", 1, "0", SM);
    insertToPitchedData(C, 3);
    aggregate3DVectorClassCtor(C, "id", 4, "0", SM);
    aggregate3DVectorClassCtor(C, "range", 6, "1", SM, 1);
    emplaceTransformation(removeArg(C, 7, SM));
  } else if (NameRef == "cudaMemcpyFromArray") {
    handleAsync(C, 6, Result);
    emplaceTransformation(removeArg(C, 5, SM));
    aggregatePitchedData(C, 0, 4, SM, true);
    insertZeroOffset(C, 1);
    insertToPitchedData(C, 1);
    aggregate3DVectorClassCtor(C, "id", 2, "0", SM);
    aggregate3DVectorClassCtor(C, "range", 4, "1", SM, 1);
  } else if (NameRef == "cudaMemcpyToArray") {
    handleAsync(C, 6, Result);
    emplaceTransformation(removeArg(C, 5, SM));
    insertToPitchedData(C, 0);
    aggregate3DVectorClassCtor(C, "id", 1, "0", SM);
    aggregatePitchedData(C, 3, 4, SM, true);
    insertZeroOffset(C, 4);
    aggregate3DVectorClassCtor(C, "range", 4, "1", SM, 1);
  }

  if (ULExpr)
    emplaceTransformation(new ReplaceToken(
        ULExpr->getBeginLoc(), ULExpr->getEndLoc(), std::move(ReplaceStr)));
  else
    emplaceTransformation(
        new ReplaceCalleeName(C, std::move(ReplaceStr), Name));
}

void MemoryMigrationRule::memcpySymbolMigration(
    const MatchFinder::MatchResult &Result, const CallExpr *C,
    const UnresolvedLookupExpr *ULExpr, bool IsAssigned) {
  std::string DirectionName;
  // Currently, if memory API occurs in a template, we will migrate the API call
  // under the undeclared decl AST node and the explicit specialization AST
  // node. The API call in explicit specialization is same as without template.
  // But if the API has non-specified default parameters and it is in an
  // undeclared decl, these default parameters will not be counted into the
  // number of call arguments. So, we need check the argument number before get
  // it.
  if (C->getNumArgs() >= 5 && !C->getArg(4)->isDefaultArgument()) {
    const Expr *Direction = C->getArg(4);
    const DeclRefExpr *DD = dyn_cast_or_null<DeclRefExpr>(Direction);
    if (DD && isa<EnumConstantDecl>(DD->getDecl())) {
      DirectionName = DD->getNameInfo().getName().getAsString();
      auto Search = EnumConstantRule::EnumNamesMap.find(DirectionName);
      if(Search == EnumConstantRule::EnumNamesMap.end())
        return;
      Direction = nullptr;
      DirectionName = "dpct::" + Search->second;
    }
  }

  DpctGlobalInfo &Global = DpctGlobalInfo::getInstance();
  auto MallocInfo = Global.findCudaMalloc(C->getArg(1));
  auto VD = CudaMallocInfo::getDecl(C->getArg(0));
  if (MallocInfo && VD) {
    if (auto Var = Global.findMemVarInfo(VD)) {
      emplaceTransformation(new ReplaceStmt(
          C, Var->getName() + ".assign(" +
                 MallocInfo->getAssignArgs(Var->getType()->getBaseName()) +
                 ")"));
      return;
    }
  }

  std::string Name;
  if (ULExpr) {
    Name = ULExpr->getName().getAsString();
  } else {
    Name = C->getCalleeDecl()->getAsFunction()->getNameAsString();
  }

  std::string ReplaceStr;
  if (checkWhetherIsDuplicate(C, false))
    return;
  int Index = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();
  if (Name == "cudaMemcpyToSymbol" || Name == "cudaMemcpyFromSymbol") {
    if (USMLevel == UsmLevel::restricted) {
      buildTempVariableMap(Index, C, HelperFuncType::DefaultQueue);
      ReplaceStr = "{{NEEDREPLACEQ" + std::to_string(Index) + "}}.memcpy";
    } else {
      ReplaceStr = "dpct::dpct_memcpy";
    }
  } else {
    if (USMLevel == UsmLevel::restricted) {
      if (C->getNumArgs() == 6 && !C->getArg(5)->isDefaultArgument()) {
        const Expr *Stream = C->getArg(5);
        if (Stream) {
          if (isPredefinedStreamHandle(C->getArg(5))) {
            buildTempVariableMap(Index, C, HelperFuncType::DefaultQueue);
            ReplaceStr = "{{NEEDREPLACEQ" + std::to_string(Index) + "}}.memcpy";
          } else {
            auto StreamStr = ExprAnalysis::ref(Stream);
            ReplaceStr = StreamStr + "->memcpy";
          }
        }
      } else {
        buildTempVariableMap(Index, C, HelperFuncType::DefaultQueue);
        ReplaceStr = "{{NEEDREPLACEQ" + std::to_string(Index) + "}}.memcpy";
      }
    } else
      ReplaceStr = "dpct::async_dpct_memcpy";
  }

  if (ULExpr) {
    emplaceTransformation(new ReplaceToken(
        ULExpr->getBeginLoc(), ULExpr->getEndLoc(), std::move(ReplaceStr)));
  } else {
    emplaceTransformation(
        new ReplaceCalleeName(C, std::move(ReplaceStr), Name));
  }

  ExprAnalysis EA;
  std::string OffsetFromBaseStr;
  if (C->getNumArgs() >= 4 && !C->getArg(3)->isDefaultArgument()) {
    EA.analyze(C->getArg(3));
    OffsetFromBaseStr = EA.getReplacedString();
  } else {
    OffsetFromBaseStr = "0";
  }

  if ((Name == "cudaMemcpyToSymbol" || Name == "cudaMemcpyToSymbolAsync") &&
      OffsetFromBaseStr != "0") {
    replaceMemAPIArg(C->getArg(0), Result, OffsetFromBaseStr);
  } else {
    replaceMemAPIArg(C->getArg(0), Result);
  }

  if ((Name == "cudaMemcpyFromSymbol" || Name == "cudaMemcpyFromSymbolAsync") &&
      OffsetFromBaseStr != "0") {
    replaceMemAPIArg(C->getArg(1), Result, OffsetFromBaseStr);
  } else {
    replaceMemAPIArg(C->getArg(1), Result);
  }

  // Remove C->getArg(3)
  if (C->getNumArgs() >= 4 && !C->getArg(3)->isDefaultArgument()) {
    if (auto TM = removeArg(C, 3, *Result.SourceManager))
      emplaceTransformation(TM);
  }

  if (C->getNumArgs() >= 5 && !C->getArg(4)->isDefaultArgument()) {
    emplaceTransformation(
        new ReplaceStmt(C->getArg(4), std::move(DirectionName)));
  }

  // Async
  if (Name == "cudaMemcpyToSymbolAsync" ||
      Name == "cudaMemcpyFromSymbolAsync") {
    if (C->getNumArgs() == 6 && !C->getArg(4)->isDefaultArgument()) {
      if (USMLevel == UsmLevel::restricted) {
        if (auto TM = removeArg(C, 4, *Result.SourceManager))
          emplaceTransformation(TM);
        if (!C->getArg(5)->isDefaultArgument()) {
          if (auto TM = removeArg(C, 5, *Result.SourceManager))
            emplaceTransformation(TM);
        }
      } else {
        handleAsync(C, 5, Result);
      }
    } else if (C->getNumArgs() == 5 && !C->getArg(4)->isDefaultArgument()) {
      if (USMLevel == UsmLevel::restricted) {
        if (auto TM = removeArg(C, 4, *Result.SourceManager))
          emplaceTransformation(TM);
      }
    }
  } else {
    if (USMLevel == UsmLevel::restricted) {
      if (C->getNumArgs() == 5 && !C->getArg(4)->isDefaultArgument()) {
        if (auto TM = removeArg(C, 4, *Result.SourceManager))
          emplaceTransformation(TM);
      }
      emplaceTransformation(new InsertAfterStmt(C, ".wait()"));
    }
  }
}

void MemoryMigrationRule::freeMigration(const MatchFinder::MatchResult &Result,
                                        const CallExpr *C,
                                        const UnresolvedLookupExpr *ULExpr,
                                        bool IsAssigned) {

  std::string Name;
  if (ULExpr) {
    Name = ULExpr->getName().getAsString();
  } else {
    Name = C->getCalleeDecl()->getAsFunction()->getNameAsString();
  }
  if (checkWhetherIsDuplicate(C, false))
    return;
  int Index = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();
  if (Name == "cudaFree" || Name == "cublasFree") {
    if (USMLevel == UsmLevel::restricted) {
      ExprAnalysis EA;
      EA.analyze(C->getArg(0));
      std::ostringstream Repl;
      buildTempVariableMap(Index, C, HelperFuncType::DefaultQueue);
      Repl << MapNames::getClNamespace() + "::free(" << EA.getReplacedString()
           << ", {{NEEDREPLACEQ" + std::to_string(Index) + "}})";
      emplaceTransformation(new ReplaceStmt(C, std::move(Repl.str())));
    } else {
      emplaceTransformation(new ReplaceCalleeName(C, "dpct::dpct_free", Name));
    }
  } else if (Name == "cudaFreeHost") {
    if (USMLevel == UsmLevel::restricted) {
      ExprAnalysis EA;
      EA.analyze(C->getArg(0));
      std::ostringstream Repl;
      buildTempVariableMap(Index, C, HelperFuncType::DefaultQueue);
      Repl << MapNames::getClNamespace() + "::free(" << EA.getReplacedString()
           << ", {{NEEDREPLACEQ" + std::to_string(Index) + "}})";
      emplaceTransformation(new ReplaceStmt(C, std::move(Repl.str())));
    } else {
      emplaceTransformation(new ReplaceCalleeName(C, "free", Name));
    }
  } else if (Name == "cudaFreeArray") {
    ExprAnalysis EA(C->getArg(0));
    EA.analyze();
    emplaceTransformation(
        new ReplaceStmt(C, "delete " + EA.getReplacedString()));
  }
}

void MemoryMigrationRule::memsetMigration(
    const MatchFinder::MatchResult &Result, const CallExpr *C,
    const UnresolvedLookupExpr *ULExpr, bool IsAssigned) {
  std::string Name;
  if (ULExpr) {
    Name = ULExpr->getName().getAsString();
  } else {
    Name = C->getCalleeDecl()->getAsFunction()->getNameAsString();
  }

  std::string ReplaceStr;
  StringRef NameRef(Name);
  bool IsAsync = NameRef.endswith("Async");
  if (IsAsync) {
    NameRef = NameRef.drop_back(5 /* len of "Async" */);
    ReplaceStr = "dpct::async_dpct_memset";
  } else {
    ReplaceStr = "dpct::dpct_memset";
  }

  if (NameRef == "cudaMemset2D") {
    handleAsync(C, 5, Result);
  } else if (NameRef == "cudaMemset3D") {
    handleAsync(C, 3, Result);
  } else if (NameRef == "cudaMemset") {
    replaceMemAPIArg(C->getArg(0), Result);
    if (USMLevel == UsmLevel::restricted) {
      if (IsAsync) {
        emplaceTransformation(removeArg(C, 3, *Result.SourceManager));
      } else {
        emplaceTransformation(new InsertAfterStmt(C, ".wait()"));
      }
      std::string AsyncQueue;
      if (C->getNumArgs() > 3 && !C->getArg(3)->isDefaultArgument()) {
        if (!isPredefinedStreamHandle(C->getArg(3)))
          AsyncQueue = ExprAnalysis::ref(C->getArg(3));
      }
      if (AsyncQueue.empty()) {
        if (checkWhetherIsDuplicate(C, false))
          return;
        int Index = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();
        buildTempVariableMap(Index, C, HelperFuncType::DefaultQueue);
        ReplaceStr = "{{NEEDREPLACEQ" + std::to_string(Index) + "}}.memset";
      } else {
        ReplaceStr = AsyncQueue + "->memset";
      }
    } else {
      handleAsync(C, 3, Result);
    }
  }

  emplaceTransformation(new ReplaceCalleeName(C, std::move(ReplaceStr), Name));
}

void MemoryMigrationRule::getSymbolSizeMigration(
    const ast_matchers::MatchFinder::MatchResult &Result, const CallExpr *C,
    const UnresolvedLookupExpr *ULExpr, bool IsAssigned) {
  // Here only handle ordinary variable name reference, for accessing the
  // size of something residing on the device directly from host side should
  // not be possible.
  std::string Replacement;
  ExprAnalysis EA;
  EA.analyze(C->getArg(0));
  auto StmtStrArg0 = EA.getReplacedString();
  EA.analyze(C->getArg(1));
  auto StmtStrArg1 = EA.getReplacedString();
  if (isSimpleAddrOf(C->getArg(0))) {
    Replacement = getNameStrRemovedAddrOf(C->getArg(0)) + " = " + StmtStrArg1 +
                  ".get_size()";
  } else {
    Replacement =
        "*(" + StmtStrArg0 + ")" + " = " + StmtStrArg1 + ".get_size()";
  }
  emplaceTransformation(new ReplaceStmt(C, std::move(Replacement)));
}

void MemoryMigrationRule::prefetchMigration(
    const ast_matchers::MatchFinder::MatchResult &Result, const CallExpr *C,
    const UnresolvedLookupExpr *ULExpr, bool IsAssigned) {
  if (USMLevel == UsmLevel::restricted) {
    const SourceManager *SM = Result.SourceManager;
    std::string Replacement;
    ExprAnalysis EA;
    EA.analyze(C->getArg(0));
    auto StmtStrArg0 = EA.getReplacedString();
    EA.analyze(C->getArg(1));
    auto StmtStrArg1 = EA.getReplacedString();
    EA.analyze(C->getArg(2));
    auto StmtStrArg2 = EA.getReplacedString();
    std::string StmtStrArg3;
    if (C->getNumArgs() == 4 && !C->getArg(3)->isDefaultArgument()) {
      if (!isPredefinedStreamHandle(C->getArg(3)))
        StmtStrArg3 = ExprAnalysis::ref(C->getArg(3));
    } else {
      StmtStrArg3 = "0";
    }

    // In clang "define NULL __null"
    if (StmtStrArg3 == "0"|| StmtStrArg3 == "") {
      Replacement = "dpct::dev_mgr::instance().get_device(" + StmtStrArg2 +
                    ").default_queue().prefetch(" + StmtStrArg0 + "," +
                    StmtStrArg1 + ")";
    } else {
      if (SM->getCharacterData(C->getArg(3)->getBeginLoc()) -
              SM->getCharacterData(C->getArg(3)->getEndLoc()) ==
          0) {
        Replacement =
            StmtStrArg3 + "->prefetch(" + StmtStrArg0 + "," + StmtStrArg1 + ")";
      } else {
        Replacement = "(" + StmtStrArg3 + ")->prefetch(" + StmtStrArg0 + "," +
                      StmtStrArg1 + ")";
      }
    }
    emplaceTransformation(new ReplaceStmt(C, std::move(Replacement)));
  } else {
    report(C->getBeginLoc(), Diagnostics::NOTSUPPORTED, false,
           "cudaMemPrefetchAsync");
  }
}

void MemoryMigrationRule::miscMigration(const MatchFinder::MatchResult &Result,
                                        const CallExpr *C,
                                        const UnresolvedLookupExpr *ULExpr,
                                        bool IsAssigned) {
  std::string Name;
  if (ULExpr) {
    Name = ULExpr->getName().getAsString();
  } else {
    Name = C->getCalleeDecl()->getAsFunction()->getNameAsString();
  }

  if (Name == "cudaHostGetDevicePointer") {
    if (USMLevel == UsmLevel::restricted) {
      std::ostringstream Repl;
      ExprAnalysis EA;
      EA.analyze(C->getArg(0));
      auto Arg0Str = EA.getReplacedString();
      EA.analyze(C->getArg(1));
      auto Arg1Str = EA.getReplacedString();
      Repl << "*(" << Arg0Str << ") = " << Arg1Str;
      emplaceTransformation(new ReplaceStmt(C, std::move(Repl.str())));
    } else {
      report(C->getBeginLoc(), Diagnostics::NOTSUPPORTED, false,
             MapNames::ITFName.at(Name));
    }
  } else if (Name == "cudaHostRegister" || Name == "cudaHostUnregister") {
    auto Msg = MapNames::RemovedAPIWarningMessage.find(Name);
    if (IsAssigned) {
      report(C->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED_0, false,
             MapNames::ITFName.at(Name), Msg->second);
      emplaceTransformation(new ReplaceStmt(C, false, Name, "0"));
    } else {
      report(C->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED, false,
             MapNames::ITFName.at(Name), Msg->second);
      emplaceTransformation(new ReplaceStmt(C, false, Name, ""));
    }
  } else if (Name == "make_cudaExtent" || Name == "make_cudaPos") {
    std::string CtorName;
    llvm::raw_string_ostream OS(CtorName);
    DpctGlobalInfo::printCtadClass(
        OS,
        buildString(MapNames::getClNamespace(),
                    "::", (Name == "make_cudaPos") ? "id" : "range"),
        3);
    emplaceTransformation(new ReplaceCalleeName(C, std::move(OS.str()), Name));
  } else if (Name == "cudaGetChannelDesc") {
    std::ostringstream OS;
    printDerefOp(OS, C->getArg(0));
    OS << " = " << ExprAnalysis::ref(C->getArg(1)) << "->get_channel()";
    emplaceTransformation(new ReplaceStmt(C, OS.str()));
  }
}

void MemoryMigrationRule::cudaArrayGetInfo(const MatchFinder::MatchResult &Result,
                                           const CallExpr *C,
                                           const UnresolvedLookupExpr *ULExpr,
                                           bool IsAssigned) {
  std::string IndentStr = getIndent(C->getBeginLoc(), *Result.SourceManager).str();
  if (IsAssigned)
    IndentStr += "  ";
  std::ostringstream OS;
  std::string Arg3Str = ExprAnalysis::ref(C->getArg(3));
  printDerefOp(OS, C->getArg(0));
  OS << " = " << Arg3Str << "->get_channel();" << getNL() << IndentStr;
  printDerefOp(OS, C->getArg(1));
  OS << " = " << Arg3Str << "->get_range();" << getNL() << IndentStr;
  printDerefOp(OS, C->getArg(2));
  OS << " = 0";
  emplaceTransformation(new ReplaceStmt(C, OS.str()));
}

void MemoryMigrationRule::cudaHostGetFlags(const MatchFinder::MatchResult &Result,
                                           const CallExpr *C,
                                           const UnresolvedLookupExpr *ULExpr,
                                           bool IsAssigned) {
  std::ostringstream OS;
  printDerefOp(OS, C->getArg(0));
  OS << " = 0";
  emplaceTransformation(new ReplaceStmt(C, OS.str()));
}

void MemoryMigrationRule::cudaMemAdvise(const MatchFinder::MatchResult &Result,
                                        const CallExpr *C,
                                        const UnresolvedLookupExpr *ULExpr,
                                        bool IsAssigned) {
  // Do nothing if USM is disabled
  if (USMLevel == UsmLevel::none) {
    report(C->getBeginLoc(), Diagnostics::NOTSUPPORTED, false, "cudaMemAdvise");
    return;
  }
  static std::map<std::string, std::string> AdviceMapping{
      {"cudaMemAdviseSetReadMostly", "PI_MEM_ADVICE_SET_READ_MOSTLY"},
      {"cudaMemAdviseUnsetReadMostly", "PI_MEM_ADVICE_CLEAR_READ_MOSTLY"},
      {"cudaMemAdviseSetPreferredLocation", "PI_MEM_ADVICE_SET_PREFERRED_LOCATION"},
      {"cudaMemAdviseUnsetPreferredLocation", "PI_MEM_ADVICE_CLEAR_PREFERRED_LOCATION"},
      {"cudaMemAdviseSetAccessedBy", "PI_MEM_ADVICE_SET_ACCESSED_BY"},
      {"cudaMemAdviseUnsetAccessedBy", "PI_MEM_ADVICE_CLEAR_ACCESSED_BY"}};

  auto Arg0Str = ExprAnalysis::ref(C->getArg(0));
  auto Arg1Str = ExprAnalysis::ref(C->getArg(1));
  auto Arg2Str = ExprAnalysis::ref(C->getArg(2));
  auto Arg3Str = ExprAnalysis::ref(C->getArg(3));
  auto It = AdviceMapping.find(Arg2Str);
  if (It != AdviceMapping.end()) {
    Arg2Str = It->second;
  } else {
    // Simplify casts of IntegerLiterals to enums, e.g. cudaMemoryAdvise(1),
    // (cudaMemoryAdvise)1, static_cast<cudaMemoryAdvise>(1) can be migrated to
    // PI_MEM_ADVICE_SET_READ_MOSTLY.
    Expr::EvalResult ER;
    const Expr *SubE = nullptr;
    if (auto CSCE = dyn_cast<CStyleCastExpr>(C->getArg(2))) {
      SubE = CSCE->getSubExpr();
    } else if (auto CXXSCE = dyn_cast<CXXStaticCastExpr>(C->getArg(2))) {
      SubE = CXXSCE->getSubExpr();
    } else if (auto CXXFCE = dyn_cast<CXXFunctionalCastExpr>(C->getArg(2))) {
      SubE = CXXFCE->getSubExpr();
    }
    if (SubE && SubE->EvaluateAsInt(ER, *Result.Context)) {
      static std::vector<std::string> Advices{
          "PI_MEM_ADVICE_SET_READ_MOSTLY", "PI_MEM_ADVICE_CLEAR_READ_MOSTLY",
          "PI_MEM_ADVICE_SET_PREFERRED_LOCATION", "PI_MEM_ADVICE_CLEAR_PREFERRED_LOCATION",
          "PI_MEM_ADVICE_SET_ACCESSED_BY", "PI_MEM_ADVICE_CLEAR_ACCESSED_BY"};
      Arg2Str = Advices[ER.Val.getInt().getExtValue() - 1];
    } else {
      Arg2Str = "pi_mem_advice(" + Arg2Str + " - 1)";
    }
  }

  std::ostringstream OS;
  if (getStmtSpelling(C->getArg(3)) == "cudaCpuDeviceId") {
    OS << "dpct::cpu_device().default_queue().mem_advise(" << Arg0Str
       << ", " << Arg1Str << ", " << Arg2Str << ")";
    emplaceTransformation(new ReplaceStmt(C, OS.str()));
    return;
  }
  OS << "dpct::get_device(" << Arg3Str
     << ").default_queue().mem_advise(" << Arg0Str << ", " << Arg1Str
     << ", " << Arg2Str <<  ")";
  emplaceTransformation(new ReplaceStmt(C, OS.str()));
}

// Memory migration rules live here.
void MemoryMigrationRule::registerMatcher(MatchFinder &MF) {
  auto memoryAPI = [&]() {
    return hasAnyName(
        "cudaMalloc", "cudaMemcpy", "cudaMemcpyAsync", "cudaMemcpyToSymbol",
        "cudaMemcpyToSymbolAsync", "cudaMemcpyFromSymbol",
        "cudaMemcpyFromSymbolAsync", "cudaFree", "cudaMemset",
        "cudaMemsetAsync", "cublasFree", "cublasAlloc", "cudaGetSymbolAddress",
        "cudaFreeHost", "cudaHostAlloc", "cudaHostGetDevicePointer",
        "cudaHostRegister", "cudaHostUnregister", "cudaMallocHost",
        "cudaMallocManaged", "cudaGetSymbolSize", "cudaMemPrefetchAsync",
        "cudaMalloc3D", "cudaMallocPitch", "cudaMemset2D", "cudaMemset3D",
        "cudaMemset2DAsync", "cudaMemset3DAsync", "cudaMemcpy2D",
        "cudaMemcpy3D", "cudaMemcpy2DAsync", "cudaMemcpy3DAsync",
        "cudaMemcpy2DArrayToArray", "cudaMemcpy2DToArray",
        "cudaMemcpy2DToArrayAsync", "cudaMemcpy2DFromArray",
        "cudaMemcpy2DFromArrayAsync", "cudaMemcpyArrayToArray",
        "cudaMemcpyToArray", "cudaMemcpyToArrayAsync", "cudaMemcpyFromArray",
        "cudaMemcpyFromArrayAsync", "cudaMallocArray", "cudaMalloc3DArray",
        "cudaFreeArray", "cudaArrayGetInfo", "cudaHostGetFlags",
        "cudaMemAdvise", "cudaGetChannelDesc");
  };

  MF.addMatcher(callExpr(allOf(callee(functionDecl(memoryAPI())), parentStmt()))
                    .bind("call"),
                this);

  MF.addMatcher(
      callExpr(allOf(callee(functionDecl(memoryAPI())), unless(parentStmt())))
          .bind("callUsed"),
      this);

  MF.addMatcher(
      unresolvedLookupExpr(
          hasAnyDeclaration(namedDecl(memoryAPI())),
          hasParent(callExpr(unless(parentStmt())).bind("callExprUsed")))
          .bind("unresolvedCallUsed"),
      this);

  MF.addMatcher(
      unresolvedLookupExpr(hasAnyDeclaration(namedDecl(memoryAPI())),
                           hasParent(callExpr(parentStmt()).bind("callExpr")))
          .bind("unresolvedCall"),
      this);
}

void MemoryMigrationRule::run(const MatchFinder::MatchResult &Result) {
  CHECKPOINT_ASTMATCHER_RUN_ENTRY();
  auto MigrateCallExpr = [&](const CallExpr *C, const bool IsAssigned,
                             const UnresolvedLookupExpr *ULExpr = NULL) {
    if (!C)
      return;

    std::string Name;
    if (ULExpr && C) {
      Name = ULExpr->getName().getAsString();
    } else {
      Name = C->getCalleeDecl()->getAsFunction()->getNameAsString();
    }
    if(MigrationDispatcher.find(Name) == MigrationDispatcher.end())
        return;

    // If there is a malloc function call in a template function, and the
    // template function is implicitly instantiated with two types. Then there
    // will be three FunctionDecl nodes in the AST. We should do replacement on
    // the FunctionDecl node which is not implicitly instantiated.
    auto &Context = dpct::DpctGlobalInfo::getContext();
    auto Parents = Context.getParents(*C);
    while (Parents.size() == 1) {
      auto *Parent = Parents[0].get<FunctionDecl>();
      if (Parent) {
        if (Parent->getTemplateSpecializationKind() ==
                TSK_ExplicitSpecialization ||
            Parent->getTemplateSpecializationKind() == TSK_Undeclared)
          break;
        else
          return;
      } else {
        Parents = Context.getParents(Parents[0]);
      }
    }

    MigrationDispatcher.at(Name)(Result, C, ULExpr, IsAssigned);

    // if API is removed, then no need to add (*, 0)
    // Currently, there are only cudaHostRegister and cudaHostUnregister
    if (IsAssigned && Name.compare("cudaHostRegister") &&
        Name.compare("cudaHostUnregister") && Name.compare("cudaMemAdvise") &&
        Name.compare("cudaArrayGetInfo")) {
      report(C->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP, false);
      insertAroundStmt(C, "(", ", 0)");
    } else if (IsAssigned && !Name.compare("cudaMemAdvise") &&
               USMLevel != UsmLevel::none) {
      report(C->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP, false);
      insertAroundStmt(C, "(", ", 0)");
    } else if (IsAssigned && !Name.compare("cudaArrayGetInfo")) {
      report(C->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP, false);
      std::string IndentStr = getIndent(C->getBeginLoc(), *Result.SourceManager).str();
      IndentStr += "  ";
      std::string PreStr{"([&](){"};
      PreStr += getNL();
      PreStr += IndentStr;
      std::string PostStr{";"};
      PostStr += getNL();
      PostStr += IndentStr;
      PostStr += "}(), 0)";
      insertAroundStmt(C, std::move(PreStr), std::move(PostStr));
    }
  };
  MigrateCallExpr(getAssistNodeAsType<CallExpr>(Result, "call"),
                  /* IsAssigned */ false);
  MigrateCallExpr(getAssistNodeAsType<CallExpr>(Result, "callUsed"),
                  /* IsAssigned */ true);
  MigrateCallExpr(
      getAssistNodeAsType<CallExpr>(Result, "callExprUsed"),
      /* IsAssigned */ true,
      getAssistNodeAsType<UnresolvedLookupExpr>(Result, "unresolvedCallUsed"));

  MigrateCallExpr(
      getAssistNodeAsType<CallExpr>(Result, "callExpr"),
      /* IsAssigned */ false,
      getAssistNodeAsType<UnresolvedLookupExpr>(Result, "unresolvedCall"));
}

void MemoryMigrationRule::getSymbolAddressMigration(
    const ast_matchers::MatchFinder::MatchResult &Result, const CallExpr *C,
    const UnresolvedLookupExpr *ULExpr, bool IsAssigned) {
  // Here only handle ordinary variable name reference, for accessing the
  // address of something residing on the device directly from host side should
  // not be possible.
  std::string Replacement;
  ExprAnalysis EA;
  EA.analyze(C->getArg(0));
  auto StmtStrArg0 = EA.getReplacedString();
  EA.analyze(C->getArg(1));
  auto StmtStrArg1 = EA.getReplacedString();
  Replacement = "*(" + StmtStrArg0 + ")" + " = " + StmtStrArg1 + ".get_ptr()";
  emplaceTransformation(new ReplaceStmt(C, std::move(Replacement)));
}

MemoryMigrationRule::MemoryMigrationRule() {
  SetRuleProperty(ApplyToCudaFile | ApplyToCppFile);
  std::map<
      std::string,
      std::function<void(
          MemoryMigrationRule *, const ast_matchers::MatchFinder::MatchResult &,
          const CallExpr *, const UnresolvedLookupExpr *, bool)>>
      Dispatcher{
          {"cudaMalloc", &MemoryMigrationRule::mallocMigration},
          {"cudaHostAlloc", &MemoryMigrationRule::mallocMigration},
          {"cudaMallocHost", &MemoryMigrationRule::mallocMigration},
          {"cudaMallocManaged", &MemoryMigrationRule::mallocMigration},
          {"cublasAlloc", &MemoryMigrationRule::mallocMigration},
          {"cudaMallocPitch", &MemoryMigrationRule::mallocMigration},
          {"cudaMalloc3D", &MemoryMigrationRule::mallocMigration},
          {"cudaMallocArray", &MemoryMigrationRule::mallocMigration},
          {"cudaMalloc3DArray", &MemoryMigrationRule::mallocMigration},
          {"cudaMemcpy", &MemoryMigrationRule::memcpyMigration},
          {"cudaMemcpyAsync", &MemoryMigrationRule::memcpyMigration},
          {"cudaMemcpyToSymbol", &MemoryMigrationRule::memcpySymbolMigration},
          {"cudaMemcpyToSymbolAsync",
           &MemoryMigrationRule::memcpySymbolMigration},
          {"cudaMemcpyFromSymbol", &MemoryMigrationRule::memcpySymbolMigration},
          {"cudaMemcpyFromSymbolAsync",
           &MemoryMigrationRule::memcpySymbolMigration},
          {"cudaMemcpy2D", &MemoryMigrationRule::memcpyMigration},
          {"cudaMemcpy3D", &MemoryMigrationRule::memcpyMigration},
          {"cudaMemcpy2DAsync", &MemoryMigrationRule::memcpyMigration},
          {"cudaMemcpy3DAsync", &MemoryMigrationRule::memcpyMigration},
          {"cudaMemcpy2DArrayToArray", &MemoryMigrationRule::arrayMigration},
          {"cudaMemcpy2DFromArray", &MemoryMigrationRule::arrayMigration},
          {"cudaMemcpy2DFromArrayAsync", &MemoryMigrationRule::arrayMigration},
          {"cudaMemcpy2DToArray", &MemoryMigrationRule::arrayMigration},
          {"cudaMemcpy2DToArrayAsync", &MemoryMigrationRule::arrayMigration},
          {"cudaMemcpyArrayToArray", &MemoryMigrationRule::arrayMigration},
          {"cudaMemcpyToArray", &MemoryMigrationRule::arrayMigration},
          {"cudaMemcpyToArrayAsync", &MemoryMigrationRule::arrayMigration},
          {"cudaMemcpyFromArray", &MemoryMigrationRule::arrayMigration},
          {"cudaMemcpyFromArrayAsync", &MemoryMigrationRule::arrayMigration},
          {"cudaFree", &MemoryMigrationRule::freeMigration},
          {"cudaFreeArray", &MemoryMigrationRule::freeMigration},
          {"cudaFreeHost", &MemoryMigrationRule::freeMigration},
          {"cublasFree", &MemoryMigrationRule::freeMigration},
          {"cudaMemset", &MemoryMigrationRule::memsetMigration},
          {"cudaMemsetAsync", &MemoryMigrationRule::memsetMigration},
          {"cudaMemset2D", &MemoryMigrationRule::memsetMigration},
          {"cudaMemset2DAsync", &MemoryMigrationRule::memsetMigration},
          {"cudaMemset3D", &MemoryMigrationRule::memsetMigration},
          {"cudaMemset3DAsync", &MemoryMigrationRule::memsetMigration},
          {"cudaGetSymbolAddress",
           &MemoryMigrationRule::getSymbolAddressMigration},
          {"cudaGetSymbolSize", &MemoryMigrationRule::getSymbolSizeMigration},
          {"cudaHostGetDevicePointer", &MemoryMigrationRule::miscMigration},
          {"cudaHostRegister", &MemoryMigrationRule::miscMigration},
          {"cudaHostUnregister", &MemoryMigrationRule::miscMigration},
          {"cudaMemPrefetchAsync", &MemoryMigrationRule::prefetchMigration},
          {"cudaArrayGetInfo", &MemoryMigrationRule::cudaArrayGetInfo},
          {"cudaHostGetFlags", &MemoryMigrationRule::cudaHostGetFlags},
          {"cudaMemAdvise", &MemoryMigrationRule::cudaMemAdvise},
          {"cudaGetChannelDesc", &MemoryMigrationRule::miscMigration}};

  for (auto &P : Dispatcher)
    MigrationDispatcher[P.first] =
        std::bind(P.second, this, std::placeholders::_1, std::placeholders::_2,
                  std::placeholders::_3, std::placeholders::_4);
}

/// Convert a raw pointer argument and a pitch argument to a dpct::pitched_data
/// constructor. If \p ExcludeSizeArg is true, the argument represent the pitch
/// size will not be included in the constructor.
/// e.g. (...data, pitch, ...) => (...dpct::pitched_data(data, pitch, pitch, 1),
/// ...).
/// If \p ExcludeSizeArg is true, e.g. (...data, ..., pitch, ...) =>
/// (...dpct::pitched_data(data, pitch, pitch, 1), ..., pitch, ...)
void MemoryMigrationRule::aggregatePitchedData(const CallExpr *C,
                                               size_t DataArgIndex,
                                               size_t SizeArgIndex,
                                               SourceManager &SM,
                                               bool ExcludeSizeArg) {
  if (C->getNumArgs() <= DataArgIndex || C->getNumArgs() <= SizeArgIndex)
    return;
  size_t EndArgIndex = SizeArgIndex;
  std::string PaddingArgs, SizeArg;
  llvm::raw_string_ostream PaddingOS(PaddingArgs);
  ArgumentAnalysis A(C->getArg(SizeArgIndex), false);
  A.analyze();
  SizeArg = A.getReplacedString();
  if (ExcludeSizeArg) {
    PaddingOS << ", " << SizeArg;
    EndArgIndex = DataArgIndex;
  }
  PaddingOS << ", " << SizeArg << ", 1";
  aggregateArgsToCtor(C, "dpct::pitched_data", DataArgIndex, EndArgIndex,
                      PaddingOS.str(), SM);
}

/// Convert several arguments to a constructor of class \p ClassName.
/// e.g. (...width, height, ...) => (...sycl::range<3>(width, height, 1), ...)
void MemoryMigrationRule::aggregateArgsToCtor(
    const CallExpr *C, const std::string &ClassName, size_t StartArgIndex,
    size_t EndArgIndex, const std::string &PaddingArgs, SourceManager &SM) {
  insertAroundRange(getArgBeginLocation(C, StartArgIndex, SM),
                    getArgEndLocation(C, EndArgIndex, SM), ClassName + "(",
                    PaddingArgs + ")");
}

/// Convert several arguments to a 3D vector constructor, like id<3> or
/// range<3>.
/// e.g. (...width, height, ...) => (...sycl::range<3>(width, height, 1), ...)
void MemoryMigrationRule::aggregate3DVectorClassCtor(
    const CallExpr *C, StringRef ClassName, size_t StartArgIndex,
    StringRef DefaultValue, SourceManager &SM, size_t ArgsNum) {
  if (C->getNumArgs() <= StartArgIndex + ArgsNum - 1)
    return;
  std::string Class, Padding;
  llvm::raw_string_ostream ClassOS(Class), PaddingOS(Padding);
  ClassOS << MapNames::getClNamespace() << "::";
  DpctGlobalInfo::printCtadClass(ClassOS, ClassName, 3);
  for (size_t i = 0; i < 3 - ArgsNum; ++i) {
    PaddingOS << ", " << DefaultValue;
  }
  aggregateArgsToCtor(C, ClassOS.str(), StartArgIndex,
                      StartArgIndex + ArgsNum - 1, PaddingOS.str(), SM);
}

void MemoryMigrationRule::handleDirection(const CallExpr *C, unsigned i) {
  if (C->getNumArgs() > i && !C->getArg(i)->isDefaultArgument()) {
    if (auto DRE = dyn_cast<DeclRefExpr>(C->getArg(i))) {
      if (auto Enum = dyn_cast<EnumConstantDecl>(DRE->getDecl())) {
        auto &ReplaceDirection = MapNames::findReplacedName(
            EnumConstantRule::EnumNamesMap, Enum->getName().str());
        if (!ReplaceDirection.empty()) {
          emplaceTransformation(
              new ReplaceStmt(DRE, "dpct::" + ReplaceDirection));
        }
      }
    }
  }
}

void
MemoryMigrationRule::handleAsync(const CallExpr *C, unsigned i,
                                 const MatchFinder::MatchResult &Result) {
  if (C->getNumArgs() > i && !C->getArg(i)->isDefaultArgument()) {
    auto StreamExpr = C->getArg(i)->IgnoreImplicitAsWritten();
    emplaceTransformation(new InsertBeforeStmt(StreamExpr, "*"));
    if (auto IL = dyn_cast<IntegerLiteral>(StreamExpr)) {
      if (IL->getValue().getZExtValue() == 0) {
        emplaceTransformation(removeArg(C, i, *Result.SourceManager));
        return;
      } else {
        emplaceTransformation(
            new InsertBeforeStmt(StreamExpr, "(cl::sycl::queue *)"));
      }
    } else if (isPredefinedStreamHandle(StreamExpr)) {
      emplaceTransformation(removeArg(C, i, *Result.SourceManager));
      return;
    } else if (!isa<DeclRefExpr>(StreamExpr)) {
      insertAroundStmt(StreamExpr, "(", ")");
    }
  }
}

REGISTER_RULE(MemoryMigrationRule)

void MemoryDataTypeRule::emplaceMemcpy3DDeclarations(const VarDecl *VD) {
  if (DpctGlobalInfo::isCommentsEnabled()) {
    emplaceTransformation(ReplaceVarDecl::getVarDeclReplacement(
        VD, "// These variables are defined for 3d matrix memory copy."));
  }
  emplaceParamDecl(VD, "dpct::pitched_data", false, "from_data", "to_data");
  emplaceParamDecl(VD, getCtadType("id"), true, "from_pos", "to_pos");
  emplaceParamDecl(VD, getCtadType("range"), true, "size");
  emplaceParamDecl(VD, "dpct::memcpy_direction", false, "direction");
}

std::string MemoryDataTypeRule::getMemcpy3DArguments(StringRef BaseName) {
  std::string Result;
  llvm::raw_string_ostream OS(Result);
  printParamName(OS, BaseName, "to_data") << ", ";
  printParamName(OS, BaseName, "to_pos") << ", ";
  printParamName(OS, BaseName, "from_data") << ", ";
  printParamName(OS, BaseName, "from_pos") << ", ";
  printParamName(OS, BaseName, "size") << ", ";
  printParamName(OS, BaseName, "direction");
  return OS.str();
}

void MemoryDataTypeRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(
      varDecl(hasType(recordDecl(hasName("cudaMemcpy3DParms")))).bind("decl"),
      this);
  MF.addMatcher(memberExpr(hasObjectExpression(declRefExpr(hasType(
                               recordDecl(hasName("cudaMemcpy3DParms"))))))
                    .bind("parmsMember"),
                this);
  MF.addMatcher(memberExpr(hasObjectExpression(hasType(recordDecl(hasAnyName(
                               "cudaExtent", "cudaPos", "cudaPitchedPtr")))))
                    .bind("otherMember"),
                this);
  MF.addMatcher(
      callExpr(callee(functionDecl(hasAnyName("make_cudaExtent", "make_cudaPos",
                                              "make_cudaPitchedPtr"))))
          .bind("makeData"),
      this);
}

void MemoryDataTypeRule::run(const MatchFinder::MatchResult &Result) {
  CHECKPOINT_ASTMATCHER_RUN_ENTRY();
  if (auto VD = getNodeAsType<VarDecl>(Result, "decl")) {
    emplaceMemcpy3DDeclarations(VD);
  } else if (auto ME = getNodeAsType<MemberExpr>(Result, "parmsMember")) {
    if (auto BO = DpctGlobalInfo::findAncestor<BinaryOperator>(ME)) {
      if (BO->getOpcode() == BinaryOperatorKind::BO_Assign &&
          ME == BO->getLHS()) {
        if (DpctGlobalInfo::getUnqualifiedTypeName(ME->getType()) ==
            "cudaArray_t") {
          emplaceTransformation(
              new InsertAfterStmt(BO->getRHS(), "->to_pitched_data()"));
        }
      }
    }
    if (auto DRE =
            dyn_cast<DeclRefExpr>(ME->getBase()->IgnoreImplicitAsWritten())) {
      emplaceTransformation(new ReplaceStmt(
          ME, getMemcpy3DMemberName(DRE->getDecl()->getName(),
                                    ME->getMemberDecl()->getName().str())));
    }
  } else if (auto CE = getNodeAsType<CallExpr>(Result, "makeData")) {
    if (auto FD = CE->getDirectCallee()) {
      auto Name = FD->getName();
      std::string ReplaceName;
      if (Name == "make_cudaExtent") {
        ReplaceName = DpctGlobalInfo::getCtadClass(
            MapNames::getClNamespace() + "::range", 3);
      } else if (Name == "make_cudaPos") {
        ReplaceName = DpctGlobalInfo::getCtadClass(
            MapNames::getClNamespace() + "::id", 3);
      } else if (Name == "make_cudaPitchedPtr") {
        ReplaceName = "dpct::pitched_data";
      } else {
        DpctDiags() << "Unexpected function name [" << Name
                    << "] in MemoryDataTypeRule";
        return;
      }
      emplaceTransformation(
          new ReplaceCalleeName(CE, std::move(ReplaceName), Name.str()));
    }
  } else if (auto M = getNodeAsType<MemberExpr>(Result, "otherMember")) {
    auto BaseName =
        DpctGlobalInfo::getUnqualifiedTypeName(M->getBase()->getType());
    auto MemberName = M->getMemberDecl()->getName();
    if (BaseName == "cudaPos") {
      auto &Replace = MapNames::findReplacedName(MapNames::Dim3MemberNamesMap,
                                                 MemberName.str());
      if (!Replace.empty())
        emplaceTransformation(new ReplaceToken(
            M->getOperatorLoc(), M->getEndLoc(), std::string(Replace)));
    } else if (BaseName == "cudaExtent") {
      auto &Replace =
          MapNames::findReplacedName(ExtentMemberNames, MemberName.str());
      if (!Replace.empty())
        emplaceTransformation(new ReplaceToken(
            M->getOperatorLoc(), M->getEndLoc(), std::string(Replace)));
    } else if (BaseName == "cudaPitchedPtr") {
      auto &Replace =
          MapNames::findReplacedName(PitchMemberNames, MemberName.str());
      if (!Replace.empty())
        emplaceTransformation(
            new ReplaceToken(M->getMemberLoc(), std::string(Replace)));
    }
  }
}

REGISTER_RULE(MemoryDataTypeRule)

void UnnamedTypesRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(typedefDecl(hasDescendant(loc(recordType(hasDeclaration(
                    cxxRecordDecl(unless(anyOf(has(cxxRecordDecl(isImplicit())),
                                               isImplicit())),
                                  hasDefinition())
                        .bind("unnamedType")))))),
                this);
}

void UnnamedTypesRule::run(const MatchFinder::MatchResult &Result) {
  CHECKPOINT_ASTMATCHER_RUN_ENTRY();
  auto D = getNodeAsType<CXXRecordDecl>(Result, "unnamedType");
  if (D && D->getName().empty())
    emplaceTransformation(new InsertClassName(D));
}

REGISTER_RULE(UnnamedTypesRule)

void GuessIndentWidthRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(
      functionDecl(allOf(hasParent(translationUnitDecl()),
                         hasBody(compoundStmt(unless(anyOf(
                             statementCountIs(0), statementCountIs(1)))))))
          .bind("FunctionDecl"),
      this);
  MF.addMatcher(
      cxxMethodDecl(hasParent(cxxRecordDecl(hasParent(translationUnitDecl()))))
          .bind("CXXMethodDecl"),
      this);
  MF.addMatcher(
      fieldDecl(hasParent(cxxRecordDecl(hasParent(translationUnitDecl()))))
          .bind("FieldDecl"),
      this);
}

void GuessIndentWidthRule::run(const MatchFinder::MatchResult &Result) {
  CHECKPOINT_ASTMATCHER_RUN_ENTRY();
  if (DpctGlobalInfo::getGuessIndentWidthMatcherFlag())
    return;
  SourceManager &SM = DpctGlobalInfo::getSourceManager();
  // Case 1:
  // TranslationUnitDecl
  // `-FunctionDecl
  //   `-CompoundStmt
  //     |-Stmt_1
  //     |-Stmt_2
  //     ...
  //     |-Stmt_n-1
  //     `-Stmt_n
  // The stmt in the compound stmt should >= 2, then we use the indent of the
  // first stmt as IndentWidth.
  auto FD = getNodeAsType<FunctionDecl>(Result, "FunctionDecl");
  if (FD) {
    CompoundStmt *CS = nullptr;
    Stmt *S = nullptr;
    if ((CS = dyn_cast<CompoundStmt>(FD->getBody())) &&
        (!CS->children().empty()) && (S = *(CS->children().begin()))) {
      DpctGlobalInfo::setIndentWidth(
          getIndent(SM.getExpansionLoc(S->getBeginLoc()), SM).size());
      DpctGlobalInfo::setGuessIndentWidthMatcherFlag(true);
      return;
    }
  }

  // Case 2:
  // TranslationUnitDecl
  // `-CXXRecordDecl
  //   |-CXXRecordDecl
  //   `-CXXMethodDecl
  // Use the indent of the CXXMethodDecl as the IndentWidth.
  auto CMD = getNodeAsType<CXXMethodDecl>(Result, "CXXMethodDecl");
  if (CMD) {
    DpctGlobalInfo::setIndentWidth(
        getIndent(SM.getExpansionLoc(CMD->getBeginLoc()), SM).size());
    DpctGlobalInfo::setGuessIndentWidthMatcherFlag(true);
    return;
  }

  // Case 3:
  // TranslationUnitDecl
  // `-CXXRecordDecl
  //   |-CXXRecordDecl
  //   `-FieldDecl
  // Use the indent of the FieldDecl as the IndentWidth.
  auto FieldD = getNodeAsType<FieldDecl>(Result, "FieldDecl");
  if (FieldD) {
    DpctGlobalInfo::setIndentWidth(
        getIndent(SM.getExpansionLoc(FieldD->getBeginLoc()), SM).size());
    DpctGlobalInfo::setGuessIndentWidthMatcherFlag(true);
    return;
  }
}

REGISTER_RULE(GuessIndentWidthRule)

void MathFunctionsRule::registerMatcher(MatchFinder &MF) {
  std::vector<std::string> MathFunctions = {
#define ENTRY_RENAMED(SOURCEAPINAME, TARGETAPINAME) SOURCEAPINAME,
#define ENTRY_RENAMED_SINGLE(SOURCEAPINAME, TARGETAPINAME) SOURCEAPINAME,
#define ENTRY_RENAMED_DOUBLE(SOURCEAPINAME, TARGETAPINAME) SOURCEAPINAME,
#define ENTRY_EMULATED(SOURCEAPINAME, TARGETAPINAME) SOURCEAPINAME,
#define ENTRY_OPERATOR(APINAME, OPKIND) APINAME,
#define ENTRY_TYPECAST(APINAME) APINAME,
#define ENTRY_UNSUPPORTED(APINAME) APINAME,
#include "APINamesMath.inc"
#undef ENTRY_RENAMED
#undef ENTRY_RENAMED_SINGLE
#undef ENTRY_RENAMED_DOUBLE
#undef ENTRY_EMULATED
#undef ENTRY_OPERATOR
#undef ENTRY_TYPECAST
#undef ENTRY_UNSUPPORTED
  };

  MF.addMatcher(
      callExpr(callee(functionDecl(
                   internal::Matcher<NamedDecl>(
                       new internal::HasNameMatcher(MathFunctions)),
                   anyOf(unless(hasDeclContext(namespaceDecl(anything()))),
                         hasDeclContext(namespaceDecl(hasName("std")))))),
               unless(hasAncestor(
                   cxxConstructExpr(hasType(typedefDecl(hasName("dim3")))))))
          .bind("math"),
      this);
}

void MathFunctionsRule::run(const MatchFinder::MatchResult &Result) {
  CHECKPOINT_ASTMATCHER_RUN_ENTRY();
  if (auto CE = getNodeAsType<CallExpr>(Result, "math")) {
    // Make sure all args in CE are not straddle nodes.
    // (partially in function-like macro)
    int ArgNum = CE->getNumArgs();
    bool HasStraddleArg = false;
    for (int i = 0; i < ArgNum; ++i) {
      auto AE = CE->getArg(i);
      ExprSpellingStatus SpellingStatus;
      if (isExprStraddle(AE, &SpellingStatus)) {
        HasStraddleArg = true;
        break;
      }
    }
    if (!HasStraddleArg) {
      ExprAnalysis EA(CE);
      emplaceTransformation(EA.getReplacement());
    }
  }
}

REGISTER_RULE(MathFunctionsRule)

void WarpFunctionsRule::registerMatcher(MatchFinder &MF) {
  std::vector<std::string> WarpFunctions = {
#define ENTRY_WARP(SOURCEAPINAME, TARGETAPINAME) SOURCEAPINAME,
#include "APINamesWarp.inc"
#undef ENTRY_WARP
  };

  MF.addMatcher(callExpr(callee(functionDecl(internal::Matcher<NamedDecl>(
                             new internal::HasNameMatcher(WarpFunctions)))),
                         hasAncestor(functionDecl().bind("ancestor")))
                    .bind("warp"),
                this);
}

void WarpFunctionsRule::run(const MatchFinder::MatchResult &Result) {
  CHECKPOINT_ASTMATCHER_RUN_ENTRY();
  if (auto FD = getAssistNodeAsType<FunctionDecl>(Result, "ancestor"))
    DeviceFunctionDecl::LinkRedecls(FD)->setItem();

  if (auto CE = getNodeAsType<CallExpr>(Result, "warp")) {
    ExprAnalysis EA(CE);
    emplaceTransformation(EA.getReplacement());
  }
}
REGISTER_RULE(WarpFunctionsRule)

void SyncThreadsRule::registerMatcher(MatchFinder &MF) {
  auto SyncAPI = [&]() {
    return hasAnyName("__syncthreads", "this_thread_block", "sync",
                      "__threadfence_block", "__syncthreads_and",
                      "__syncthreads_or");
  };
  MF.addMatcher(callExpr(allOf(callee(functionDecl(SyncAPI())), parentStmt(),
                         hasAncestor(functionDecl().bind("FuncDecl"))))
                    .bind("SyncFuncCall"),
                this);
  MF.addMatcher(callExpr(allOf(callee(functionDecl(SyncAPI())), unless(parentStmt()),
                         hasAncestor(functionDecl().bind("FuncDeclUsed"))))
                    .bind("SyncFuncCallUsed"),
                this);
}

void SyncThreadsRule::run(const MatchFinder::MatchResult &Result) {
  CHECKPOINT_ASTMATCHER_RUN_ENTRY();
  bool IsAssigned = false;
  const CallExpr *CE = getNodeAsType<CallExpr>(Result, "SyncFuncCall");
  const FunctionDecl *FD = getAssistNodeAsType<FunctionDecl>(Result, "FuncDecl");
  if (!CE) {
    if (!(CE = getNodeAsType<CallExpr>(Result, "SyncFuncCallUsed")))
      return;
    FD = getAssistNodeAsType<FunctionDecl>(Result, "FuncDeclUsed");
    IsAssigned = true;
  }
  if (FD)
    DeviceFunctionDecl::LinkRedecls(FD)->setItem();

  std::string FuncName =
      CE->getDirectCallee()->getNameInfo().getName().getAsString();
  if (FuncName == "__syncthreads" || FuncName == "sync") {
    std::string Replacement = getItemName() + ".barrier()";
    emplaceTransformation(new ReplaceStmt(CE, std::move(Replacement)));
  } else if (FuncName == "this_thread_block") {
    if (auto P = getAncestorDeclStmt(CE)) {
      std::string ReplStr{"sycl::group<3> "};
      for (auto It = P->decl_begin(); It != P->decl_end(); ++It) {
        auto VD = dyn_cast<VarDecl>(*It);
        if (It != P->decl_begin())
          ReplStr += ", ";
        auto &SM = DpctGlobalInfo::getSourceManager();
        if (VD->getLocation().isMacroID() && SM.isMacroArgExpansion(VD->getLocation())) {
          auto VDBegin = SM.getImmediateExpansionRange(VD->getLocation()).getBegin();
          VDBegin = SM.getSpellingLoc(VDBegin);
          Token T;
          Lexer::getRawToken(VDBegin, T, SM, Result.Context->getLangOpts());
          ReplStr += T.getRawIdentifier().str();
        } else {
          ReplStr += VD->getName();
        }
        ReplStr += " = ";
        ReplStr += DpctGlobalInfo::getItemName() + ".get_group()";
      }
      ReplStr += ";";
      emplaceTransformation(new ReplaceStmt(P, std::move(ReplStr)));
    } else {
      emplaceTransformation(new ReplaceStmt(CE, ""));
    }
  } else if (FuncName == "__threadfence_block") {
    std::string ReplStr = DpctGlobalInfo::getItemName() + ".mem_fence()";
    emplaceTransformation(new ReplaceStmt(CE, std::move(ReplStr)));
  } else if (FuncName == "__syncthreads_and" || FuncName == "__syncthreads_or") {
    std::string ReplStr;
    if (IsAssigned) {
      ReplStr = "(";
      ReplStr += DpctGlobalInfo::getItemName() + ".barrier(), ";
    } else {
      ReplStr += DpctGlobalInfo::getItemName() + ".barrier();" + getNL();
      ReplStr += getIndent(CE->getBeginLoc(), *Result.SourceManager).str();
    }
    if (FuncName == "__syncthreads_and")
      ReplStr += "sycl::intel::all_of(";
    else
      ReplStr += "sycl::intel::any_of(";
    ReplStr += getItemName();
    ReplStr += ".get_group(), ";
    ReplStr += ExprAnalysis::ref(CE->getArg(0));
    ReplStr += ")";
    if (IsAssigned)
      ReplStr += ")";
    emplaceTransformation(new ReplaceStmt(CE, std::move(ReplStr)));
  }
}

REGISTER_RULE(SyncThreadsRule)

void KernelFunctionInfoRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(
      varDecl(hasType(recordDecl(hasName("cudaFuncAttributes")))).bind("decl"),
      this);
  MF.addMatcher(
      callExpr(callee(functionDecl(hasAnyName("cudaFuncGetAttributes"))))
          .bind("call"),
      this);
  MF.addMatcher(memberExpr(hasObjectExpression(hasType(
                               recordDecl(hasName("cudaFuncAttributes")))))
                    .bind("member"),
                this);
}

void KernelFunctionInfoRule::run(const MatchFinder::MatchResult &Result) {
  CHECKPOINT_ASTMATCHER_RUN_ENTRY();
  if (auto V = getNodeAsType<VarDecl>(Result, "decl"))
    emplaceTransformation(
        new ReplaceTypeInDecl(V, "dpct::kernel_function_info"));
  else if (auto C = getNodeAsType<CallExpr>(Result, "call")) {
    emplaceTransformation(
        new ReplaceToken(C->getBeginLoc(), "(dpct::get_kernel_function_info"));
    emplaceTransformation(new InsertAfterStmt(C, ", 0)"));
    auto FuncArg = C->getArg(1);
    emplaceTransformation(new InsertBeforeStmt(FuncArg, "(const void *)"));
  } else if (auto M = getNodeAsType<MemberExpr>(Result, "member")) {
    auto MemberName = M->getMemberNameInfo();
    auto NameMap = AttributesNamesMap.find(MemberName.getAsString());
    if (NameMap != AttributesNamesMap.end())
      emplaceTransformation(new ReplaceToken(MemberName.getBeginLoc(),
                                             std::string(NameMap->second)));
  }
}

REGISTER_RULE(KernelFunctionInfoRule)

void RecognizeAPINameRule::registerMatcher(MatchFinder &MF) {
  std::vector<std::string> AllAPINames = MigrationStatistics::GetAllAPINames();
  MF.addMatcher(
      callExpr(allOf(callee(functionDecl(internal::Matcher<NamedDecl>(
                         new internal::HasNameMatcher(AllAPINames)))),
                     unless(hasAncestor(cudaKernelCallExpr())),
                     unless(callee(hasDeclContext(namedDecl(hasName("std")))))))
          .bind("APINamesUsed"),
      this);
  MF.addMatcher(
      callExpr(allOf(callee(functionDecl(matchesName("(nccl.*)|(cudnn.*)"))),
                     unless(callee(functionDecl(internal::Matcher<NamedDecl>(
                         new internal::HasNameMatcher(AllAPINames))))),
                     unless(hasAncestor(cudaKernelCallExpr())),
                     unless(callee(hasDeclContext(namedDecl(hasName("std")))))))
          .bind("ManualMigrateAPI"),
      this);
}

const std::string
RecognizeAPINameRule::GetFunctionSignature(const FunctionDecl *Func) {

  std::string Buf;
  llvm::raw_string_ostream OS(Buf);
  OS << Func->getReturnType().getAsString() << " "
     << Func->getQualifiedNameAsString() << "(";

  for (unsigned int Index = 0; Index < Func->getNumParams(); Index++) {
    if (Index > 0) {
      OS << ",";
    }
    OS << QualType::getAsString(Func->parameters()[Index]->getType().split(),
                                PrintingPolicy{{}})
       << " " << Func->parameters()[Index]->getQualifiedNameAsString();
  }
  OS << ")";
  return OS.str();
}

void RecognizeAPINameRule::run(const MatchFinder::MatchResult &Result) {
  CHECKPOINT_ASTMATCHER_RUN_ENTRY();
  const CallExpr *C = getNodeAsType<CallExpr>(Result, "APINamesUsed");
  if (!C) {
    C = getNodeAsType<CallExpr>(Result, "ManualMigrateAPI");
    if (!C) {
      return;
    }
  }
  std::string Namespace;
  const NamedDecl *ND = dyn_cast<NamedDecl>(C->getCalleeDecl());
  if (ND) {
    const auto *NSD = dyn_cast<NamespaceDecl>(ND->getDeclContext());
    if (NSD && !NSD->isInlineNamespace()) {
      Namespace = NSD->getName().str();
    }
  }

  std::string APIName = C->getCalleeDecl()->getAsFunction()->getNameAsString();

  if (!Namespace.empty() && Namespace == "thrust") {
    APIName = Namespace + "::" + APIName;
  }

  SrcAPIStaticsMap[GetFunctionSignature(C->getCalleeDecl()->getAsFunction())]++;
  if (APIName.size() >= 4 && APIName.substr(0, 4) == "nccl") {
    auto D = C->getCalleeDecl();
    if (D) {
      auto FilePath = DpctGlobalInfo::getSourceManager()
                          .getFilename(D->getBeginLoc())
                          .str();
      if (DpctGlobalInfo::isInRoot(FilePath)) {
        return;
      }
    }
    report(C->getBeginLoc(), Diagnostics::MANUAL_MIGRATION_LIBRARY, false,
           "Intel(R) oneAPI Collective Communications Library");
  } else if (APIName.size() >= 5 && APIName.substr(0, 5) == "cudnn") {
    auto D = C->getCalleeDecl();
    if (D) {
      auto FilePath = DpctGlobalInfo::getSourceManager()
                          .getFilename(D->getBeginLoc())
                          .str();
      if (DpctGlobalInfo::isInRoot(FilePath)) {
        return;
      }
    }
    report(C->getBeginLoc(), Diagnostics::MANUAL_MIGRATION_LIBRARY, false,
           "Intel(R) oneAPI Deep Neural Network Library (oneDNN)");
  } else if (!MigrationStatistics::IsMigrated(APIName)) {
    GAnalytics(GetFunctionSignature(C->getCalleeDecl()->getAsFunction()));
    const SourceManager &SM = (*Result.Context).getSourceManager();
    const SourceLocation FileLoc = SM.getFileLoc(C->getBeginLoc());

    std::string SLStr = FileLoc.printToString(SM);

    std::size_t PosCol = SLStr.rfind(':');
    std::size_t PosRow = SLStr.rfind(':', PosCol - 1);
    std::string FileName = SLStr.substr(0, PosRow);
    LOCStaticsMap[FileName][2]++;
    report(C->getBeginLoc(), Diagnostics::API_NOT_MIGRATED, false,
           MapNames::ITFName.at(APIName.c_str()));
  }
}

REGISTER_RULE(RecognizeAPINameRule)

const BinaryOperator *TextureRule::getParentAsAssignedBO(const Expr *E,
                                                         ASTContext &Context) {
  auto Parents = Context.getParents(*E);
  if (Parents.size() > 0)
    return getAssignedBO(Parents[0].get<Expr>(), Context);
  return nullptr;
}

// Return the binary operator if E is the lhs of an assign experssion, otherwise
// nullptr.
const BinaryOperator *TextureRule::getAssignedBO(const Expr *E,
                                                 ASTContext &Context) {
  if (dyn_cast<MemberExpr>(E)) {
    // Continue finding parents when E is MemberExpr.
    return getParentAsAssignedBO(E, Context);
  } else if (auto ICE = dyn_cast<ImplicitCastExpr>(E)) {
    // Stop finding parents and return nullptr when E is ImplicitCastExpr,
    // except for ArrayToPointerDecay cast.
    if (ICE->getCastKind() == CK_ArrayToPointerDecay) {
      return getParentAsAssignedBO(E, Context);
    }
  } else if (auto ASE = dyn_cast<ArraySubscriptExpr>(E)) {
    // Continue finding parents when E is ArraySubscriptExpr, and remove
    // subscript operator anyway for texture object's member.
    emplaceTransformation(new ReplaceToken(
        Lexer::getLocForEndOfToken(ASE->getLHS()->getEndLoc(), 0,
                                   Context.getSourceManager(),
                                   Context.getLangOpts()),
        ASE->getRBracketLoc(), ""));
    return getParentAsAssignedBO(E, Context);
  } else if (auto BO = dyn_cast<BinaryOperator>(E)) {
    // If E is BinaryOperator, return E only when it is assign expression,
    // otherwise return nullptr.
    if (BO->getOpcode() == BO_Assign)
      return BO;
  }
  return nullptr;
}

void TextureRule::registerMatcher(MatchFinder &MF) {
  auto DeclMatcher = varDecl(hasType(templateSpecializationType(
      hasDeclaration(classTemplateSpecializationDecl(hasName("texture"))))));
  // Match texture object's declaration
  MF.addMatcher(DeclMatcher.bind("texDecl"), this);
  MF.addMatcher(
      declRefExpr(
          hasDeclaration(DeclMatcher.bind("texDecl")),
          anyOf(hasAncestor(functionDecl(anyOf(hasAttr(attr::CUDADevice),
                                               hasAttr(attr::CUDAGlobal)))
                                .bind("texFunc")),
                // Match the __global__/__device__ functions inside which
                // texture object is referenced
                anything()) // Make this matcher available whether it has
                            // ancestors as before
          )
          .bind("tex"),
      this);
  MF.addMatcher(typeLoc(loc(qualType(hasDeclaration(
                            typedefDecl(hasName("cudaTextureObject_t"))))))
                    .bind("texObj"),
                this);
  MF.addMatcher(memberExpr(hasObjectExpression(hasType(namedDecl(hasAnyName(
                               "cudaChannelFormatDesc", "cudaTextureDesc",
                               "cudaResourceDesc", "textureReference")))))
                    .bind("texMember"),
                this);
  MF.addMatcher(typeLoc(loc(qualType(hasDeclaration(namedDecl(hasAnyName(
                            "cudaChannelFormatDesc", "cudaTextureDesc",
                            "cudaResourceDesc", "cudaArray", "cudaArray_t"))))))
                    .bind("texType"),
                this);

  MF.addMatcher(
      declRefExpr(to(enumConstantDecl(hasType(enumDecl(hasAnyName(
                      "cudaTextureAddressMode", "cudaTextureFilterMode",
                      "cudaChannelFormatKind", "cudaResourceType"))))))
          .bind("texEnum"),
      this);
  MF.addMatcher(
      callExpr(
          callee(functionDecl(hasAnyName(
              "cudaCreateChannelDesc", "cudaCreateChannelDescHalf",
              "cudaUnbindTexture", "cudaBindTextureToArray", "cudaBindTexture",
              "cudaBindTexture2D", "tex1D", "tex2D", "tex3D", "tex1Dfetch",
              "tex1DLayered", "tex2DLayered", "cudaCreateTextureObject",
              "cudaDestroyTextureObject", "cudaGetTextureObjectResourceDesc",
              "cudaGetTextureObjectTextureDesc",
              "cudaGetTextureObjectResourceViewDesc"))))
          .bind("call"),
      this);
}

void TextureRule::run(const MatchFinder::MatchResult &Result) {
  CHECKPOINT_ASTMATCHER_RUN_ENTRY();
  if (auto VD = getAssistNodeAsType<VarDecl>(Result, "texDecl")) {
    auto Tex = DpctGlobalInfo::getInstance().insertTextureInfo(VD);
    emplaceTransformation(new ReplaceVarDecl(VD, Tex->getHostDeclString()));
    if (auto FD = getAssistNodeAsType<FunctionDecl>(Result, "texFunc")) {
      DeviceFunctionDecl::LinkRedecls(FD)->addTexture(Tex);
    }
  } else if (auto ME = getNodeAsType<MemberExpr>(Result, "texMember")) {
    auto BaseTy = DpctGlobalInfo::getUnqualifiedTypeName(
        ME->getBase()->getType().getUnqualifiedType(), *Result.Context);
    auto MemberName = ME->getMemberNameInfo().getAsString();
    if (BaseTy == "cudaResourceDesc") {
      if (MemberName == "res") {
        emplaceTransformation(new ReplaceToken(ME->getMemberLoc(), ""));
        replaceResourceDataExpr(getParentMemberExpr(ME), *Result.Context);
      } else if (MemberName == "resType") {
        emplaceTransformation(new RenameFieldInMemberExpr(ME, "type"));
      }
    } else if (BaseTy == "cudaChannelFormatDesc") {
      if (ME->getMemberNameInfo().getAsString() == "f") {
        emplaceTransformation(new RenameFieldInMemberExpr(ME, "type"));
      } else if (auto BO = getParentAsAssignedBO(ME, *Result.Context)) {
        static std::map<std::string, std::string> ChannelOrderMap = {
            {"x", "1"}, {"y", "2"}, {"z", "3"}, {"w", "4"}};
        emplaceTransformation(new RenameFieldInMemberExpr(
            ME,
            buildString("set_channel_size(",
                        ChannelOrderMap[ME->getMemberNameInfo().getAsString()],
                        ", ", ExprAnalysis::ref(BO->getRHS()), ")")));
        emplaceTransformation(new ReplaceToken(
            Lexer::getLocForEndOfToken(BO->getLHS()->getEndLoc(), 0,
                                       *Result.SourceManager,
                                       Result.Context->getLangOpts()),
            BO->getRHS()->getEndLoc(), ""));
      } else {
        emplaceTransformation(
            new RenameFieldInMemberExpr(ME, "get_channel_size()"));
      }
    } else {
      auto Field = ME->getMemberNameInfo().getAsString();
      auto ReplField = MapNames::findReplacedName(TextureMemberNames, Field);
      if (ReplField.empty()) {
        return report(ME->getBeginLoc(), Diagnostics::NOTSUPPORTED, false,
                      Field);
      }
      emplaceTransformation(new RenameFieldInMemberExpr(ME, ReplField + "()"));
      if (Field == "addressMode") {
        if (auto A = DpctGlobalInfo::findAncestor<ArraySubscriptExpr>(ME)) {
          emplaceTransformation(new ReplaceToken(
              Lexer::getLocForEndOfToken(A->getLHS()->getEndLoc(), 0,
                                         *Result.SourceManager,
                                         Result.Context->getLangOpts()),
              A->getRBracketLoc(), ""));
        }
      }
    }
  } else if (auto TL = getNodeAsType<TypeLoc>(Result, "texType")) {
    const std::string &ReplType = MapNames::findReplacedName(
        MapNames::TypeNamesMap,
        DpctGlobalInfo::getUnqualifiedTypeName(TL->getType(), *Result.Context));
    if (!ReplType.empty())
      emplaceTransformation(new ReplaceToken(TL->getBeginLoc(), TL->getEndLoc(),
                                             std::string(ReplType)));
  } else if (auto CE = getNodeAsType<CallExpr>(Result, "call")) {
    ExprAnalysis A;
    A.analyze(CE);
    emplaceTransformation(A.getReplacement());
  } else if (auto DRE = getNodeAsType<DeclRefExpr>(Result, "texEnum")) {
    if (auto ECD = dyn_cast<EnumConstantDecl>(DRE->getDecl())) {
      std::string EnumName = ECD->getName().str();
      if (MapNames::replaceName(EnumConstantRule::EnumNamesMap, EnumName)) {
        emplaceTransformation(new ReplaceStmt(DRE, EnumName));
      } else {
        report(DRE->getBeginLoc(), Diagnostics::NOTSUPPORTED, false, EnumName);
      }
    }
  } else if (auto TL = getNodeAsType<TypeLoc>(Result, "texObj")) {
    if (auto FD = DpctGlobalInfo::getParentFunction(TL)) {
      if (!FD->hasAttr<CUDAGlobalAttr>() && !FD->hasAttr<CUDADeviceAttr>()) {
        emplaceTransformation(new ReplaceToken(
            TL->getBeginLoc(), TL->getEndLoc(), "dpct::image_base_p"));
      }
    }
  }
}

void TextureRule::replaceResourceDataExpr(const MemberExpr *ME,
                                          const ASTContext &Context) {
  if (!ME)
    return;
  auto TopMember = getParentMemberExpr(ME);
  if (!TopMember)
    return;
  auto ResName = ME->getMemberNameInfo().getAsString();
  if (ResName == "array") {
    emplaceTransformation(
        new ReplaceToken(ME->getOperatorLoc(), TopMember->getEndLoc(), "data"));
    if (auto BO = DpctGlobalInfo::findAncestor<BinaryOperator>(TopMember)) {
      if (BO->getRHS()->IgnoreImplicitAsWritten() == TopMember) {
        emplaceTransformation(
            new InsertBeforeStmt(TopMember, "(dpct::image_matrix_p)"));
      }
    }
  } else if (ResName == "linear") {
    emplaceTransformation(new ReplaceToken(
      ME->getOperatorLoc(), TopMember->getEndLoc(),
      std::string(MapNames::findReplacedName(
        LinearResourceTypeNames,
        TopMember->getMemberNameInfo().getAsString()))));
  } else if (ResName == "pitch2D") {
    emplaceTransformation(new ReplaceToken(
      ME->getOperatorLoc(), TopMember->getEndLoc(),
      std::string(MapNames::findReplacedName(
        Pitched2DResourceTypeNames,
        TopMember->getMemberNameInfo().getAsString()))));
  } else {
    report(ME->getBeginLoc(), Diagnostics::NOTSUPPORTED, false,
           ME->getMemberDecl()->getName());
  }
}

REGISTER_RULE(TextureRule)

void CXXNewExprRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(cxxNewExpr().bind("newExpr"), this);
}

void CXXNewExprRule::run(const ast_matchers::MatchFinder::MatchResult &Result) {
  CHECKPOINT_ASTMATCHER_RUN_ENTRY();
  if (auto CNE = getAssistNodeAsType<CXXNewExpr>(Result, "newExpr")) {
    // E.g. new cudaEvent_t *()
    Token Tok;
    auto LOpts = Result.Context->getLangOpts();
    SourceManager *SM = Result.SourceManager;
    auto BeginLoc =
        CNE->getAllocatedTypeSourceInfo()->getTypeLoc().getBeginLoc();
    Lexer::getRawToken(BeginLoc, Tok, *SM, LOpts, true);
    if (Tok.isAnyIdentifier()) {
      std::string Str = MapNames::findReplacedName(MapNames::TypeNamesMap,
                                                  Tok.getRawIdentifier().str());
      SourceManager &SM = DpctGlobalInfo::getSourceManager();
      BeginLoc = SM.getExpansionLoc(BeginLoc);
      if (!Str.empty()) {
        emplaceTransformation(new ReplaceToken(BeginLoc, std::move(Str)));
        return;
      }
    }

    // E.g. #define NEW_STREAM new cudaStream_t
    //      stream = NEW_STREAM;
    auto TypeName = CNE->getAllocatedType().getAsString();
    auto ReplName = std::string(
        MapNames::findReplacedName(MapNames::TypeNamesMap, TypeName));
    if (!ReplName.empty()) {
      auto BeginLoc =
          CNE->getAllocatedTypeSourceInfo()->getTypeLoc().getBeginLoc();
      emplaceTransformation(new ReplaceToken(BeginLoc, std::move(ReplName)));
    }
  }
}

REGISTER_RULE(CXXNewExprRule)

void NamespaceRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(usingDirectiveDecl().bind("usingDirective"), this);
  MF.addMatcher(namespaceAliasDecl().bind("namespaceAlias"), this);
}

void NamespaceRule::run(const MatchFinder::MatchResult &Result) {
  CHECKPOINT_ASTMATCHER_RUN_ENTRY();
  if (auto UDD = getAssistNodeAsType<UsingDirectiveDecl>(Result, "usingDirective")) {
    if (UDD->getNominatedNamespace()->getNameAsString() == "cooperative_groups")
      emplaceTransformation(new ReplaceDecl(UDD, ""));
  } else if (auto NAD = getAssistNodeAsType<NamespaceAliasDecl>(Result, "namespaceAlias")) {
    if (NAD->getNamespace()->getNameAsString() == "cooperative_groups")
      emplaceTransformation(new ReplaceDecl(NAD, ""));
  }
}

REGISTER_RULE(NamespaceRule)

void RemoveBaseClassRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(cxxRecordDecl(isDirectlyDerivedFrom(hasAnyName(
                                  "unary_function", "binary_function")))
                    .bind("derivedFrom"),
                this);
}

void RemoveBaseClassRule::run(const MatchFinder::MatchResult &Result) {
  CHECKPOINT_ASTMATCHER_RUN_ENTRY();
  auto SM = Result.SourceManager;
  auto LOpts = Result.Context->getLangOpts();
  auto findColon = [&](SourceRange SR) {
    Token Tok;
    auto E = SR.getEnd();
    SourceLocation Loc = SR.getBegin();
    Lexer::getRawToken(Loc, Tok, *SM, LOpts, true);
    bool ColonFound = false;
    while (Loc <= E) {
      if (Tok.is(tok::TokenKind::colon)) {
        ColonFound = true;
        break;
      }
      Tok = Lexer::findNextToken(Tok.getLocation(), *SM, LOpts).getValue();
      Loc = Tok.getLocation();
    }
    if (ColonFound)
      return Loc;
    else
      return SourceLocation();
  };

  auto getBaseDecl = [](QualType QT) {
    const Type *T = QT.getTypePtr();
    const NamedDecl *ND = nullptr;
    if (const auto *E = dyn_cast<ElaboratedType>(T)) {
      T = E->desugar().getTypePtr();
      if (const auto *TT = dyn_cast<TemplateSpecializationType>(T))
        ND = TT->getTemplateName().getAsTemplateDecl();
    } else
      ND = T->getAsCXXRecordDecl();
    return ND;
  };

  if (auto D = getNodeAsType<CXXRecordDecl>(Result, "derivedFrom")) {
    if (D->getNumBases() != 1)
      return;
    auto SR =
        SourceRange(D->getInnerLocStart(), D->getBraceRange().getBegin());
    auto ColonLoc = findColon(SR);
    if (ColonLoc.isValid()) {
      auto QT = D->bases().begin()->getType();
      const NamedDecl *BaseDecl = getBaseDecl(QT);
      if (BaseDecl) {
        auto BaseName = BaseDecl->getDeclName().getAsString();
        auto ThrustName = "thrust::" + BaseName;
        auto StdName = "std::" + BaseName;
        report(ColonLoc, Diagnostics::DEPRECATED_BASE_CLASS, false, ThrustName, StdName);
        auto Len = SM->getFileOffset(D->getBraceRange().getBegin()) -
          SM->getFileOffset(ColonLoc);
        emplaceTransformation(new ReplaceText(ColonLoc, Len, ""));
      }
    }
  }
}

REGISTER_RULE(RemoveBaseClassRule)

void ThrustVarRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(declRefExpr(to(varDecl(hasName("seq")).bind("varDecl")))
                    .bind("declRefExpr"),
                this);
}

void ThrustVarRule::run(const MatchFinder::MatchResult &Result) {
  CHECKPOINT_ASTMATCHER_RUN_ENTRY();
  if (auto DRE = getNodeAsType<DeclRefExpr>(Result, "declRefExpr")) {
    auto VD = getAssistNodeAsType<VarDecl>(Result, "varDecl", false);

    if (DRE->hasQualifier()) {

      auto Namespace = DRE->getQualifierLoc()
                           .getNestedNameSpecifier()
                           ->getAsNamespace()
                           ->getNameAsString();

      if (Namespace != "thrust")
        return;

      const std::string ThrustVarName = Namespace + "::" + VD->getName().str();

      std::string Replacement =
          MapNames::findReplacedName(MapNames::TypeNamesMap, ThrustVarName);

      if (!Replacement.empty()) {
        emplaceTransformation(new ReplaceToken(
            DRE->getBeginLoc(), DRE->getEndLoc(), std::move(Replacement)));
      }
    }
  }
}

REGISTER_RULE(ThrustVarRule)

void PreDefinedStreamHandleRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(integerLiteral(equals(0)).bind("stream"), this);
  MF.addMatcher(parenExpr(has(cStyleCastExpr(has(
                              integerLiteral(anyOf(equals(1), equals(2)))))))
                    .bind("stream"),
                this);
}

void PreDefinedStreamHandleRule::run(const MatchFinder::MatchResult &Result) {
  CHECKPOINT_ASTMATCHER_RUN_ENTRY();
  if (auto E = getNodeAsType<Expr>(Result, "stream")) {
    std::string Str = getStmtSpelling(E);
    if (Str == "cudaStreamDefault" || Str == "cudaStreamLegacy" ||
        Str == "cudaStreamPerThread") {
      auto &SM = DpctGlobalInfo::getSourceManager();
      auto Begin = getAccurateExpansionRange(E->getBeginLoc(), SM).getBegin();
      unsigned int Length = Lexer::MeasureTokenLength(
          Begin, SM, DpctGlobalInfo::getContext().getLangOpts());
      if (checkWhetherIsDuplicate(E, false))
        return;
      int Index = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();
      buildTempVariableMap(Index, E, HelperFuncType::DefaultQueue);
      emplaceTransformation(new ReplaceText(
          Begin, Length, "&{{NEEDREPLACEQ" + std::to_string(Index) + "}}"));
    }
  }
}

REGISTER_RULE(PreDefinedStreamHandleRule)

void ASTTraversalManager::matchAST(ASTContext &Context, TransformSetTy &TS,
                                   StmtStringMap &SSM) {
  this->Context = &Context;
  for (auto &I : Storage) {
    I->registerMatcher(Matchers);
    if (auto TR = dyn_cast<MigrationRule>(&*I)) {
      TR->TM = this;
      TR->setTransformSet(TS);
    }
  }

  DebugInfo::printMigrationRules(Storage);

  Matchers.matchAST(Context);

  DebugInfo::printMatchedRules(Storage);
  CHECKPOINT_ASTMATCHER_RUN_EXIT();
}

void ASTTraversalManager::emplaceAllRules(int SourceFileFlag) {
  std::vector<std::vector<std::string>> Rules;

  for (auto &F : ASTTraversalMetaInfo::getConstructorTable()) {

    auto RuleObj = (MigrationRule *)F.second();
    CommonRuleProperty RuleProperty = RuleObj->GetRuleProperty();

    auto RType = RuleProperty.RType;
    auto RulesDependon = RuleProperty.RulesDependon;

    if (RType & SourceFileFlag) {
      std::string CurrentRuleName = ASTTraversalMetaInfo::getName(F.first);
      std::vector<std::string> Vec;
      Vec.push_back(CurrentRuleName);
      for (auto const &RuleName : RulesDependon) {
        Vec.push_back(RuleName);
      }
      Rules.push_back(Vec);
    }
  }

  std::vector<std::string> SortedRules = ruleTopoSort(Rules);

  for (std::vector<std::string>::reverse_iterator it = SortedRules.rbegin();
       it != SortedRules.rend(); it++) {
    auto *ID = ASTTraversalMetaInfo::getID(*it);
    if (!ID) {
      llvm::errs() << "[ERROR] Rule\"" << *it << "\" not found\n";
      std::exit(MigrationError);
    }
    emplaceMigrationRule(ID);
  }
}

const CompilerInstance &MigrationRule::getCompilerInstance() { return TM->CI; }
