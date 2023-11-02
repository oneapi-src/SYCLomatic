//===--------------- SaveNewFiles.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SaveNewFiles.h"
#include "AnalysisInfo.h"
#include "CrashRecovery.h"
#include "Diagnostics.h"
#include "ExternalReplacement.h"
#include "GenMakefile.h"
#include "Statics.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include <algorithm>

#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/Refactoring.h"

#include "llvm/Support/raw_os_ostream.h"

#include "Utility.h"
#include "PatternRewriter.h"
#include "llvm/Support/raw_os_ostream.h"
#include <cassert>
#include <fstream>
using namespace clang::dpct;
using namespace llvm;
namespace path = llvm::sys::path;
namespace fs = llvm::sys::fs;

namespace clang {
namespace tooling {
std::string getFormatSearchPath();
}
} // namespace clang

extern std::map<std::string, uint64_t> ErrorCnt;

static bool formatFile(StringRef FileName,
                       const std::vector<clang::tooling::Range> &Ranges,
                       clang::tooling::Replacements &FormatChanges) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> ErrorOrMemoryBuffer =
      MemoryBuffer::getFileAsStream(FileName);
  if (std::error_code EC = ErrorOrMemoryBuffer.getError()) {
    return false;
  }

  std::unique_ptr<llvm::MemoryBuffer> FileBuffer =
      std::move(ErrorOrMemoryBuffer.get());
  if (FileBuffer->getBufferSize() == 0)
    return false;

  clang::format::FormattingAttemptStatus Status;
  clang::format::FormatStyle Style = DpctGlobalInfo::getCodeFormatStyle();

  if (clang::format::BlockLevelFormatFlag) {
    if (clang::dpct::DpctGlobalInfo::getFormatRange() ==
        clang::format::FormatRange::migrated) {
      Style.IndentWidth = clang::dpct::DpctGlobalInfo::getKCIndentWidth();
    }
  } else {
    if (clang::dpct::DpctGlobalInfo::getFormatRange() ==
            clang::format::FormatRange::migrated &&
        clang::dpct::DpctGlobalInfo::getGuessIndentWidthMatcherFlag()) {
      Style.IndentWidth = clang::dpct::DpctGlobalInfo::getIndentWidth();
    }
  }

  // Here need new SourceManager. Because SourceManager caches the file buffer,
  // if we use a common SourceManager, the second time format will still act on
  // the first input (the original output of dpct without format), then the
  // result is wrong.
  clang::LangOptions DefaultLangOptions;
  IntrusiveRefCntPtr<clang::DiagnosticOptions> DiagOpts =
      new clang::DiagnosticOptions();
  clang::TextDiagnosticPrinter DiagnosticPrinter(llvm::errs(), &*DiagOpts);
  clang::DiagnosticsEngine Diagnostics(
      IntrusiveRefCntPtr<clang::DiagnosticIDs>(new clang::DiagnosticIDs()),
      &*DiagOpts, &DiagnosticPrinter, false);

  clang::FileSystemOptions FSO;
  FSO.WorkingDir = ".";
  clang::FileManager FM(FSO, nullptr);
  clang::SourceManager SM(Diagnostics, FM, false);
  clang::Rewriter Rewrite(SM, clang::LangOptions());
  if (DpctGlobalInfo::getFormatRange() == clang::format::FormatRange::all) {
    std::vector<clang::tooling::Range> AllLineRanges;
    AllLineRanges.push_back(clang::tooling::Range(
        /*Offest*/ 0, /*Length*/ FileBuffer.get()->getBufferSize()));
    FormatChanges = reformat(Style, FileBuffer->getBuffer(), AllLineRanges,
                             FileName, &Status);
  } else {
    // only format migrated lines
    FormatChanges =
        reformat(Style, FileBuffer->getBuffer(), Ranges, FileName, &Status);
  }

  clang::tooling::applyAllReplacements(FormatChanges, Rewrite);
  Rewrite.overwriteChangedFiles();
  return true;
}

// TODO: it's global variable, refine in future
std::map<std::string, bool> IncludeFileMap;

bool rewriteDir(SmallString<512> &FilePath, const StringRef InRoot,
                const StringRef OutRoot) {
  assert(isCanonical(InRoot) && "InRoot must be a canonical path.");
  assert(isCanonical(FilePath) && "FilePath must be a canonical path.");

  SmallString<512> InRootAbs;
  SmallString<512> OutRootAbs;
  SmallString<512> FilePathAbs;
  std::error_code EC;
  bool InRootAbsValid = true;
  EC = llvm::sys::fs::real_path(InRoot, InRootAbs, true);
  if ((bool)EC) {
    InRootAbsValid = false;
  }
  bool OutRootAbsValid = true;
  EC = llvm::sys::fs::real_path(OutRoot, OutRootAbs, true);
  if ((bool)EC) {
    OutRootAbsValid = false;
  }
  bool FilePathAbsValid = true;
  EC = llvm::sys::fs::real_path(FilePath, FilePathAbs, true);
  if ((bool)EC) {
    FilePathAbsValid = false;
  }

#if defined(_WIN64)
  std::string Filename = sys::path::filename(FilePath).str();
  std::string LocalFilePath = StringRef(FilePath).lower();
  std::string LocalInRoot =
      InRootAbsValid ? InRootAbs.str().lower() : InRoot.lower();
  std::string LocalOutRoot =
      OutRootAbsValid ? OutRootAbs.str().lower() : OutRoot.lower();
#elif defined(__linux__)
  std::string LocalFilePath =
      FilePathAbsValid ? FilePathAbs.c_str() : StringRef(FilePath).str();
  std::string LocalInRoot = InRootAbsValid ? InRootAbs.c_str() : InRoot.str();
  std::string LocalOutRoot =
      OutRootAbsValid ? OutRootAbs.c_str() : OutRoot.str();
#else
#error Only support windows and Linux.
#endif

  if (!isChildPath(LocalInRoot, LocalFilePath, false) ||
      DpctGlobalInfo::isExcluded(LocalFilePath, false)) {
    // Skip rewriting file path if LocalFilePath is not child of LocalInRoot
    // E.g,
    //  LocalFilePath : /path/to/inc/util.cuh
    //    LocalInRoot : /path/to/inroot
    //   LocalOutRoot : /path/to/outroot
    //  AnalysisScope : /path/to
    return false;
  }
  auto PathDiff =
      std::mismatch(path::begin(LocalFilePath), path::end(LocalFilePath),
                    path::begin(LocalInRoot));
  SmallString<512> NewFilePath = StringRef(LocalOutRoot);
  path::append(NewFilePath, PathDiff.first, path::end(LocalFilePath));

#if defined(_WIN64)
  sys::path::remove_filename(NewFilePath);
  sys::path::append(NewFilePath, Filename);
#endif

  FilePath = NewFilePath;
  return true;
}

void rewriteFileName(SmallString<512> &FileName) {
  rewriteFileName(FileName, FileName);
}

void rewriteFileName(llvm::SmallString<512> &FileName,
                     llvm::StringRef FullPathName) {
  const auto Extension = path::extension(FileName);
  SourceProcessType FileType = GetSourceFileType(FullPathName);
  // If user does not specify which extension need be changed, we change all the
  // SPT_CudaSource, SPT_CppSource and SPT_CudaHeader files.
  if (DpctGlobalInfo::getChangeExtensions().empty() ||
      DpctGlobalInfo::getChangeExtensions().count(Extension.str())) {
    if (FileType & SPT_CudaSource)
      path::replace_extension(FileName, "dp.cpp");
    else if (FileType & SPT_CppSource)
      path::replace_extension(FileName, Extension + ".dp.cpp");
    else if (FileType & SPT_CudaHeader)
      path::replace_extension(FileName, "dp.hpp");
  }
}

static std::vector<std::string> FilesNotInCompilationDB;

void processallOptionAction(StringRef InRoot, StringRef OutRoot) {

  for (const auto &File : FilesNotInCompilationDB) {

    if (IncludeFileMap.find(File) != IncludeFileMap.end()) {
      // Skip the files parsed by dpct parser.
      continue;
    }

    std::ifstream In(File);
    SmallString<512> OutputFile = llvm::StringRef(File);
    if (!rewriteDir(OutputFile, InRoot, OutRoot)) {
      continue;
    }
    auto Parent = path::parent_path(OutputFile);
    std::error_code EC;
    EC = fs::create_directories(Parent);
    if ((bool)EC) {
      std::string ErrMsg =
          "[ERROR] Create Directory : " + std::string(Parent.str()) +
          " fail: " + EC.message() + "\n";
      PrintMsg(ErrMsg);
    }

    std::ofstream Out(OutputFile.c_str());
    if (Out.fail()) {
      std::string ErrMsg =
          "[ERROR] Create file : " + std::string(OutputFile.c_str()) +
          " failure!\n";
      PrintMsg(ErrMsg);
    }
    Out << In.rdbuf();
    Out.close();
    In.close();
  }
}

void processAllFiles(StringRef InRoot, StringRef OutRoot,
                     std::vector<std::string> &FilesNotProcessed) {
  std::error_code EC;
  for (fs::recursive_directory_iterator Iter(Twine(InRoot), EC), End;
       Iter != End; Iter.increment(EC)) {
    if ((bool)EC) {
      std::string ErrMsg = "[ERROR] Access : " + std::string(InRoot.str()) +
                           " fail: " + EC.message() + "\n";
      PrintMsg(ErrMsg);
    }

    auto FilePath = Iter->path();

    // Skip output directory if it is in the in-root directory.
    if (isChildOrSamePath(OutRoot.str(), FilePath))
      continue;

    bool IsHidden = false;
    for (path::const_iterator PI = path::begin(FilePath),
                              PE = path::end(FilePath);
         PI != PE; ++PI) {
      StringRef Comp = *PI;
      if (Comp.startswith(".")) {
        IsHidden = true;
        break;
      }
    }

    // Skip hidden folder or file whose name begins with ".".
    if (IsHidden) {
      continue;
    }

    if (Iter->type() == fs::file_type::regular_file) {
      SmallString<512> OutputFile = llvm::StringRef(FilePath);
      if (!rewriteDir(OutputFile, InRoot, OutRoot)) {
        continue;
      }
      if (IncludeFileMap.find(FilePath) != IncludeFileMap.end()) {
        // Skip the files processed by the first loop of
        // calling processFiles() in Tooling.cpp::ClangTool::run().
        continue;
      } else {
        if (DpctGlobalInfo::isExcluded(FilePath, false)) {
          continue;
        }
        if (GetSourceFileType(FilePath) & SPT_CudaSource) {
          // Only migrates isolated CUDA source files.
          FilesNotProcessed.push_back(FilePath);
        } else {
          // Collect the rest files which are not in the compilation database or
          // included by main source file in the compilation database.
          FilesNotInCompilationDB.push_back(FilePath);
        }
      }

    } else if (Iter->type() == fs::file_type::directory_file) {
      const auto Path = Iter->path();
      SmallString<512> OutDirectory = llvm::StringRef(Path);
      if (!rewriteDir(OutDirectory, InRoot, OutRoot)) {
        continue;
      }

      if (fs::exists(OutDirectory))
        continue;

      std::error_code EC;
      EC = fs::create_directories(OutDirectory);
      if ((bool)EC) {
        std::string ErrMsg =
            "[ERROR] Create Directory : " + std::string(OutDirectory.str()) +
            " fail: " + EC.message() + "\n";
        PrintMsg(ErrMsg);
      }
    }
  }
}

extern llvm::cl::opt<std::string> BuildScriptFile;
extern llvm::cl::opt<bool> GenBuildScript;

static void getMainSrcFilesRepls(
    std::vector<clang::tooling::Replacement> &MainSrcFilesRepls) {
  auto &FileRelpsMap = DpctGlobalInfo::getFileRelpsMap();
  for (const auto &Entry : FileRelpsMap)
    for (const auto &Repl : Entry.second)
      MainSrcFilesRepls.push_back(Repl);
}
static void getMainSrcFilesDigest(
    std::vector<std::pair<std::string, std::string>> &MainSrcFilesDigest) {
  auto &DigestMap = DpctGlobalInfo::getDigestMap();
  for (const auto &Entry : DigestMap)
    MainSrcFilesDigest.push_back(std::make_pair(Entry.first, Entry.second));
}

static void saveUpdatedMigrationDataIntoYAML(
    std::vector<clang::tooling::Replacement> &MainSrcFilesRepls,
    std::vector<std::pair<std::string, std::string>> &MainSrcFilesDigest,
    std::string YamlFile, std::string SrcFile,
    std::unordered_map<std::string, bool> &MainSrcFileMap) {
  // Save history repls to yaml file.
  auto &FileRelpsMap = DpctGlobalInfo::getFileRelpsMap();
  for (const auto &Entry : FileRelpsMap) {
    if (MainSrcFileMap[Entry.first])
      continue;
    for (const auto &Repl : Entry.second) {
      MainSrcFilesRepls.push_back(Repl);
    }
  }

  // Save history main src file and its content md5 hash to yaml file.
  auto &DigestMap = DpctGlobalInfo::getDigestMap();
  for (const auto &Entry : DigestMap) {
    if (!MainSrcFileMap[Entry.first]) {
      MainSrcFilesDigest.push_back(std::make_pair(Entry.first, Entry.second));
    }
  }

  if (!MainSrcFilesRepls.empty() || !MainSrcFilesDigest.empty() ||
      !CompileCmdsPerTarget.empty()) {
    save2Yaml(YamlFile, SrcFile, MainSrcFilesRepls, MainSrcFilesDigest,
              CompileCmdsPerTarget);
  }
}

void applyPatternRewriter(const std::string &InputString,
                          llvm::raw_os_ostream &Stream) {
  std::string LineEndingString;
  // pattern_rewriter require the input file to be LF
  bool IsCRLF = fixLineEndings(InputString, LineEndingString);
  for (const auto &PR : MapNames::PatternRewriters) {
    LineEndingString = applyPatternRewriter(PR, LineEndingString);
  }
  // Restore line ending for the formator
  if (IsCRLF) {
    std::stringstream ResultStream;
    std::vector<std::string> SplitedStr = split(LineEndingString, '\n');
    for (auto &SS : SplitedStr) {
      ResultStream << SS << "\r\n";
    }
    Stream << llvm::StringRef(ResultStream.str().c_str());
  } else {
    Stream << llvm::StringRef(LineEndingString.c_str());
  }
}

/// Apply all generated replacements, and immediately save the results to files
/// in output directory.
///
/// \returns 0 upon success. Non-zero upon failure.
/// Prerequisite: InRoot and OutRoot are both absolute paths
int saveNewFiles(clang::tooling::RefactoringTool &Tool, StringRef InRoot,
                 StringRef OutRoot) {
  assert(isCanonical(InRoot) && "InRoot must be a canonical path.");
  using namespace clang;
  volatile ProcessStatus status = MigrationSucceeded;
  // Set up Rewriter.
  LangOptions DefaultLangOptions;
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
  TextDiagnosticPrinter DiagnosticPrinter(llvm::errs(), &*DiagOpts);
  DiagnosticsEngine Diagnostics(
      IntrusiveRefCntPtr<DiagnosticIDs>(new DiagnosticIDs()), &*DiagOpts,
      &DiagnosticPrinter, false);
  SourceManager Sources(Diagnostics, Tool.getFiles());
  Rewriter Rewrite(Sources, DefaultLangOptions);
  extern bool ProcessAllFlag;
  SmallString<512> OutPath;

  // The variable defined here assists to merge history records.
  std::unordered_map<std::string /*FileName*/,
                     bool /*false:Not processed in current migration*/>
      MainSrcFileMap;

  std::string YamlFile =
      OutRoot.str() + "/" + DpctGlobalInfo::getYamlFileName();
  std::string SrcFile = "MainSrcFiles_placehold";

  if (clang::dpct::DpctGlobalInfo::isIncMigration()) {
    auto PreTU = clang::dpct::DpctGlobalInfo::getMainSourceYamlTUR();
    for (const auto &Repl : PreTU->Replacements) {
      auto &FileRelpsMap = DpctGlobalInfo::getFileRelpsMap();
      FileRelpsMap[Repl.getFilePath().str()].push_back(Repl);
    }
    for (const auto &FileDigest : PreTU->MainSourceFilesDigest) {
      auto &DigestMap = DpctGlobalInfo::getDigestMap();
      DigestMap[FileDigest.first] = FileDigest.second;

      // Mark all the main src files loaded from yaml file are not processed
      // in current migration.
      MainSrcFileMap[FileDigest.first] = false;
    }

    for (const auto &Entry : PreTU->CompileTargets)
      CompileCmdsPerTarget[Entry.first] = Entry.second;
  }

  std::vector<clang::tooling::Replacement> MainSrcFilesRepls;
  std::vector<std::pair<std::string, std::string>> MainSrcFilesDigest;

  if (Tool.getReplacements().empty()) {
    // There are no rules applying on the *.cpp files,
    // dpct just do nothing with them.
    status = MigrationNoCodeChangeHappen;

    getMainSrcFilesRepls(MainSrcFilesRepls);
    getMainSrcFilesDigest(MainSrcFilesDigest);
  } else {
    std::unordered_map<std::string, std::vector<clang::tooling::Range>>
        FileRangesMap;
    std::unordered_map<std::string, std::vector<clang::tooling::Range>>
        FileBlockLevelFormatRangesMap;
    // There are matching rules for *.cpp files, *.cu files, also header files
    // included, migrate these files into *.dp.cpp files.
    auto GroupResult = groupReplacementsByFile(
        Rewrite.getSourceMgr().getFileManager(), Tool.getReplacements());
    for (auto &Entry : GroupResult) {
#if defined(_WIN32)
      OutPath =
          StringRef(DpctGlobalInfo::removeSymlinks(
                        Rewrite.getSourceMgr().getFileManager(), Entry.first))
              .lower();
#else
      OutPath = StringRef(DpctGlobalInfo::removeSymlinks(
          Rewrite.getSourceMgr().getFileManager(), Entry.first));
#endif
      makeCanonical(OutPath);
      bool HasRealReplacements = true;
      auto Repls = Entry.second;

      if (Repls.size() == 1) {
        auto Repl = *Repls.begin();
        if (Repl.getLength() == 0 && Repl.getReplacementText().empty())
          HasRealReplacements = false;
      }
      auto Find = IncludeFileMap.find(OutPath.c_str());
      if (HasRealReplacements && Find != IncludeFileMap.end()) {
        IncludeFileMap[OutPath.c_str()] = true;
      }

      // This operation won't fail; it already succeeded once during argument
      // validation.
      makeCanonical(OutPath);
      rewriteFileName(OutPath);
      if (!rewriteDir(OutPath, InRoot, OutRoot)) {
        continue;
      }

      std::error_code EC;
      EC = fs::create_directories(path::parent_path(OutPath));
      if ((bool)EC) {
        std::string ErrMsg =
            "[ERROR] Create file : " + std::string(OutPath.str()) +
            " fail: " + EC.message() + "\n";
        status = MigrationSaveOutFail;
        PrintMsg(ErrMsg);
        return status;
      }
      // std::ios::binary prevents ofstream::operator<< from converting \n to
      // \r\n on windows.
      std::ofstream File(OutPath.str().str(), std::ios::binary);
      llvm::raw_os_ostream Stream(File);
      if (!File) {
        std::string ErrMsg =
            "[ERROR] Create file: " + std::string(OutPath.str()) + " fail.\n";
        PrintMsg(ErrMsg);
        status = MigrationSaveOutFail;
        return status;
      }

      // For header file, as it can be included from different file, it needs
      // merge the migration triggered by each including.
      // For main file, as it can be compiled or preprocessed with different
      // macro defined, it also needs merge the migration triggered by each
      // command.
      SourceProcessType FileType = GetSourceFileType(Entry.first);
      if (FileType & (SPT_CppHeader | SPT_CudaHeader)) {
        mergeExternalReps(Entry.first, OutPath.str().str(), Entry.second);
      } else {

        auto Hash = llvm::sys::fs::md5_contents(Entry.first);
        MainSrcFilesDigest.push_back(
            std::make_pair(Entry.first, Hash->digest().c_str()));

        bool IsMainSrcFileChanged = false;
        std::string FilePath = Entry.first;

        auto &DigestMap = DpctGlobalInfo::getDigestMap();
        auto DigestIter = DigestMap.find(Entry.first);
        if (DigestIter != DigestMap.end()) {
          auto Digest = llvm::sys::fs::md5_contents(Entry.first);
          if (DigestIter->second != Digest->digest().c_str())
            IsMainSrcFileChanged = true;
        }

        auto &FileRelpsMap = dpct::DpctGlobalInfo::getFileRelpsMap();
        auto Iter = FileRelpsMap.find(Entry.first);
        if (Iter != FileRelpsMap.end() && !IsMainSrcFileChanged) {
          const auto &PreRepls = Iter->second;
          mergeAndUniqueReps(Entry.second, PreRepls);
        }

        // Mark current migrating main src file processed.
        MainSrcFileMap[Entry.first] = true;

        for (const auto &Repl : Entry.second) {
          MainSrcFilesRepls.push_back(Repl);
        }
      }

      std::vector<clang::tooling::Range> Ranges;
      Ranges = calculateRangesWithFormatFlag(Entry.second);
      FileRangesMap.insert(std::make_pair(OutPath.str().str(), Ranges));

      std::vector<clang::tooling::Range> BlockLevelFormatRanges;
      BlockLevelFormatRanges =
          calculateRangesWithBlockLevelFormatFlag(Entry.second);
      FileBlockLevelFormatRangesMap.insert(
          std::make_pair(OutPath.str().str(), BlockLevelFormatRanges));

      tooling::applyAllReplacements(Entry.second, Rewrite);

      llvm::Expected<FileEntryRef> Result =
          Tool.getFiles().getFileRef(Entry.first);

      if (auto E = Result.takeError()) {
        continue;
      }

      if (MapNames::PatternRewriters.empty()) {
        Rewrite
            .getEditBuffer(Sources.getOrCreateFileID(
                *Result, clang::SrcMgr::C_User /*normal user code*/))
            .write(Stream);
      } else {
        std::string OutputString;
        llvm::raw_string_ostream RSW(OutputString);
        Rewrite
            .getEditBuffer(Sources.getOrCreateFileID(
                *Result, clang::SrcMgr::C_User /*normal user code*/))
            .write(RSW);
        applyPatternRewriter(OutputString, Stream);
      }
    }

    // Print the in-root path and the number of processed files
    size_t ProcessedFileNumber;
    if (ProcessAllFlag) {
      ProcessedFileNumber = IncludeFileMap.size();
    } else {
      ProcessedFileNumber = GroupResult.size();
    }
    std::string ReportMsg = "Processed " + std::to_string(ProcessedFileNumber) +
                            " file(s) in -in-root folder \"" + InRoot.str() +
                            "\"";
    std::string ErrorFileMsg;
    int ErrNum = 0;
    for (const auto &KV : ErrorCnt) {
      if (KV.second != 0) {
        ErrNum++;
        ErrorFileMsg += "  " + KV.first + ": ";
        if (KV.second & 0xffffffff) {
          ErrorFileMsg +=
              std::to_string(KV.second & 0xffffffff) + " parsing error(s)";
        }
        if ((KV.second & 0xffffffff) && ((KV.second >> 32) & 0xffffffff))
          ErrorFileMsg += ", ";
        if ((KV.second >> 32) & 0xffffffff) {
          ErrorFileMsg += std::to_string((KV.second >> 32) & 0xffffffff) +
                          " segmentation fault(s) ";
        }
        ErrorFileMsg += "\n";
      }
    }
    if (ErrNum) {
      ReportMsg += ", " + std::to_string(ErrNum) + " file(s) with error(s):\n";
      ReportMsg += ErrorFileMsg;
    } else {
      ReportMsg += "\n";
    }

    ReportMsg += "\n";
    ReportMsg += DiagRef;

    PrintMsg(ReportMsg);

    runWithCrashGuard(
        [&]() {
          if (DpctGlobalInfo::getFormatRange() !=
              clang::format::FormatRange::none) {
            clang::format::setFormatRangeGetterHandler(
                clang::dpct::DpctGlobalInfo::getFormatRange);
            bool FormatResult = true;
            for (const auto &Iter : FileRangesMap) {
              clang::tooling::Replacements FormatChanges;
              FormatResult =
                  formatFile(Iter.first, Iter.second, FormatChanges) &&
                  FormatResult;

              // If range is "all", one file only need to be formatted once.
              if (DpctGlobalInfo::getFormatRange() ==
                  clang::format::FormatRange::all)
                continue;

              auto BlockLevelFormatIter =
                  FileBlockLevelFormatRangesMap.find(Iter.first);
              if (BlockLevelFormatIter != FileBlockLevelFormatRangesMap.end()) {
                clang::format::BlockLevelFormatFlag = true;

                std::vector<clang::tooling::Range>
                    BlockLevelFormatRangeAfterFisrtFormat =
                        calculateUpdatedRanges(FormatChanges,
                                               BlockLevelFormatIter->second);
                FormatResult = formatFile(BlockLevelFormatIter->first,
                                          BlockLevelFormatRangeAfterFisrtFormat,
                                          FormatChanges) &&
                               FormatResult;

                clang::format::BlockLevelFormatFlag = false;
              }
            }
            if (!FormatResult) {
              PrintMsg("[Warning] Error happened while formatting. Generating "
                       "unformatted code.\n");
            }
          }
        },
        "Error: dpct internal error. Formatting of the code skipped. Migration "
        "continues.\n");
  }

  // The necessary header files which have no replacements will be copied to
  // "-out-root" directory.
  for (const auto &Entry : IncludeFileMap) {
    SmallString<512> FilePath = StringRef(Entry.first);
    if (!Entry.second) {
      makeCanonical(FilePath);
      bool IsExcluded = DpctGlobalInfo::isExcluded(FilePath.str().str(), false);
      if (IsExcluded) {
        continue;
      }
      // Always migrate *.cuh files to *.dp.hpp files,
      // Always migrate *.cu files to *.dp.cpp files.
      SourceProcessType FileType = GetSourceFileType(FilePath.str());
      if (FileType & SPT_CudaHeader) {
        path::replace_extension(FilePath, "dp.hpp");
      } else if (FileType & SPT_CudaSource) {
        path::replace_extension(FilePath, "dp.cpp");
      }

      if (!rewriteDir(FilePath, InRoot, OutRoot)) {
        continue;
      }
      if (fs::exists(FilePath)) {
        // A header file with this name already exists.
        llvm::errs() << "File '" << FilePath
                     << "' already exists; skipping it.\n";
        continue;
      }

      std::error_code EC;
      EC = fs::create_directories(path::parent_path(FilePath));
      if ((bool)EC) {
        std::string ErrMsg =
            "[ERROR] Create file: " + std::string(FilePath.str()) +
            " fail: " + EC.message() + "\n";
        status = MigrationSaveOutFail;
        PrintMsg(ErrMsg);
        return status;
      }
      // std::ios::binary prevents ofstream::operator<< from converting \n to
      // \r\n on windows.
      std::ofstream File(FilePath.str().str(), std::ios::binary);

      if (!File) {
        std::string ErrMsg =
            "[ERROR] Create file: " + std::string(FilePath.str()) +
            " failed.\n";
        status = MigrationSaveOutFail;
        PrintMsg(ErrMsg);
        return status;
      }
      llvm::raw_os_ostream Stream(File);
      std::string OutputString;
      llvm::raw_string_ostream RSW(OutputString);
      llvm::Expected<FileEntryRef> Result =
          Tool.getFiles().getFileRef(Entry.first);
      if (auto E = Result.takeError()) {
        continue;
      }
      if (MapNames::PatternRewriters.empty()) {
        Rewrite
            .getEditBuffer(Sources.getOrCreateFileID(
                *Result, clang::SrcMgr::C_User /*normal user code*/))
            .write(Stream);
      } else {
        std::string OutputString;
        llvm::raw_string_ostream RSW(OutputString);
        Rewrite
            .getEditBuffer(Sources.getOrCreateFileID(
                *Result, clang::SrcMgr::C_User /*normal user code*/))
            .write(RSW);
        applyPatternRewriter(OutputString, Stream);
      }
    }
  }

  std::string ScriptFineName = "Makefile.dpct";
  if (!BuildScriptFile.empty())
    ScriptFineName = BuildScriptFile;
  if (GenBuildScript)
    genBuildScript(Tool, InRoot, OutRoot, ScriptFineName);

  saveUpdatedMigrationDataIntoYAML(MainSrcFilesRepls, MainSrcFilesDigest,
                                   YamlFile, SrcFile, MainSrcFileMap);

  processallOptionAction(InRoot, OutRoot);

  return status;
}

void loadYAMLIntoFileInfo(std::string Path) {
  SmallString<512> SourceFilePath(Path);

  SourceFilePath = StringRef(
      DpctGlobalInfo::removeSymlinks(DpctGlobalInfo::getFileManager(), Path));
  makeCanonical(SourceFilePath);

  std::string OriginPath = SourceFilePath.str().str();
  rewriteFileName(SourceFilePath);
  if (!rewriteDir(SourceFilePath, DpctGlobalInfo::getInRoot(),
                  DpctGlobalInfo::getOutRoot())) {
    return;
  }

  std::string YamlFilePath = SourceFilePath.str().str() + ".yaml";
  auto PreTU = std::make_shared<clang::tooling::TranslationUnitReplacements>();
  if (fs::exists(YamlFilePath)) {
    if (clang::dpct::DpctGlobalInfo::isIncMigration()) {
      if (loadFromYaml(YamlFilePath, *PreTU) == 0) {
        DpctGlobalInfo::getInstance().insertReplInfoFromYAMLToFileInfo(
            OriginPath, PreTU);
      } else {
        llvm::errs() << getLoadYamlFailWarning(YamlFilePath);
      }
    }
  }
}
