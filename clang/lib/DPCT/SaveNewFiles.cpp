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
#include "PatternRewriter.h"
#include "Statics.h"
#include "TextModification.h"
#include "Utility.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include <algorithm>

#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/LangOptions.h"
#include "clang/DPCT/DpctOptions.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Rewrite/Core/Rewriter.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_os_ostream.h"

#include <algorithm>
#include <cassert>
#include <fstream>
#include <queue>

using namespace clang::dpct;
using namespace llvm;
namespace path = llvm::sys::path;
namespace fs = llvm::sys::fs;

namespace clang {
namespace tooling {
UnifiedPath getFormatSearchPath();
} // namespace tooling
} // namespace clang

extern std::map<std::string, uint64_t> ErrorCnt;

static bool formatFile(const clang::tooling::UnifiedPath &FileName,
                       const std::vector<clang::tooling::Range> &Ranges,
                       clang::tooling::Replacements &FormatChanges) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> ErrorOrMemoryBuffer =
      MemoryBuffer::getFileAsStream(FileName.getCanonicalPath());
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
                             FileName.getCanonicalPath(), &Status);
  } else {
    // only format migrated lines
    FormatChanges = reformat(Style, FileBuffer->getBuffer(), Ranges,
                             FileName.getCanonicalPath(), &Status);
  }

  clang::tooling::applyAllReplacements(FormatChanges, Rewrite);
  Rewrite.overwriteChangedFiles();
  return true;
}

// TODO: it's global variable, refine in future
std::map<clang::tooling::UnifiedPath, bool> IncludeFileMap;

bool rewriteDir(std::string &FilePath,
                const clang::tooling::UnifiedPath &InRoot,
                const clang::tooling::UnifiedPath &OutRoot) {
#if defined(_WIN64)
  std::string Filename = sys::path::filename(FilePath).str();
#endif

  if (!isChildPath(InRoot, FilePath) || DpctGlobalInfo::isExcluded(FilePath)) {
    // Skip rewriting file path if FilePath is not child of InRoot
    // E.g,
    //  FilePath : /path/to/inc/util.cuh
    //    InRoot : /path/to/inroot
    //   OutRoot : /path/to/outroot
    //  AnalysisScope : /path/to
    return false;
  }
  auto PathDiff = std::mismatch(path::begin(FilePath), path::end(FilePath),
                                path::begin(InRoot.getCanonicalPath()));
  SmallString<512> NewFilePath = OutRoot.getCanonicalPath();
  path::append(NewFilePath, PathDiff.first, path::end(FilePath));
#if defined(_WIN64)
  sys::path::remove_filename(NewFilePath);
  sys::path::append(NewFilePath, Filename);
#endif

  FilePath = NewFilePath.str();
  return true;
}

bool rewriteAbsoluteDir(clang::tooling::UnifiedPath &FilePath,
                        const clang::tooling::UnifiedPath &InRoot,
                        const clang::tooling::UnifiedPath &OutRoot) {
  std::string FilePathStr = FilePath.getAbsolutePath().str();
  bool Result = rewriteDir(FilePathStr, InRoot, OutRoot);
  FilePath = FilePathStr;
  return Result;
}
bool rewriteCanonicalDir(clang::tooling::UnifiedPath &FilePath,
                         const clang::tooling::UnifiedPath &InRoot,
                         const clang::tooling::UnifiedPath &OutRoot) {
  std::string FilePathStr = FilePath.getCanonicalPath().str();
  bool Result = rewriteDir(FilePathStr, InRoot, OutRoot);
  FilePath = FilePathStr;
  return Result;
}

void rewriteFileName(clang::tooling::UnifiedPath &FileName) {
  rewriteFileName(FileName, FileName);
}

void rewriteFileName(clang::tooling::UnifiedPath &FileName,
                     const clang::tooling::UnifiedPath &FullPathName) {
  std::string FilePath = FileName.getPath().str();
  rewriteFileName(FilePath, FullPathName.getPath().str());
  FileName = FilePath;
}

void rewriteFileName(std::string &FileName, const std::string &FullPathName) {
  SmallString<512> CanonicalPathStr(FullPathName);
  const auto Extension = path::extension(CanonicalPathStr);
  SourceProcessType FileType = GetSourceFileType(FullPathName);
  // If user does not specify which extension need be changed, we change all the
  // SPT_CudaSource, SPT_CppSource and SPT_CudaHeader files.
  if (DpctGlobalInfo::getChangeExtensions().empty() ||
      DpctGlobalInfo::getChangeExtensions().count(Extension.str())) {
    if (FileType & SPT_CudaSource) {
      path::replace_extension(CanonicalPathStr,
                              DpctGlobalInfo::getSYCLSourceExtension());
    } else if (FileType & SPT_CppSource) {
      if (Extension == ".c") {
        if (auto FileInfo = DpctGlobalInfo::getInstance().findFile(FileName)) {
          if (FileInfo->hasCUDASyntax()) {
            path::replace_extension(
                CanonicalPathStr,
                Extension + DpctGlobalInfo::getSYCLSourceExtension());
          }
        }
      } else {
        path::replace_extension(CanonicalPathStr,
                                Extension +
                                    DpctGlobalInfo::getSYCLSourceExtension());
      }
    } else if (FileType & SPT_CudaHeader) {
      path::replace_extension(CanonicalPathStr,
                              DpctGlobalInfo::getSYCLHeaderExtension());
    }
  }
  FileName = CanonicalPathStr.c_str();
}

static std::vector<std::string> FilesNotInCompilationDB;

std::map<std::string, std::string> OutFilePath2InFilePath;

static bool checkOverwriteAndWarn(StringRef OutFilePath, StringRef InFilePath) {
  auto SrcFilePath = OutFilePath2InFilePath.find(OutFilePath.str());

  bool Overwrites = false;
  // Make sure that the output file corresponds to a single and unique input
  // file.
  if (SrcFilePath != OutFilePath2InFilePath.end() &&
      SrcFilePath->second != InFilePath) {
    llvm::errs() << "[WARNING]: The output file of '" << InFilePath << "' and '"
                 << SrcFilePath->second << "' have same name '" << OutFilePath
                 << "'. To avoid overwrite, the migration of '" << InFilePath
                 << "' is skipped. Please change the output file extension "
                    "with option '--sycl-file-extension'."
                 << getNL();
    Overwrites = true;
  }
  return Overwrites;
}

// Search all the files through Pattern, and copy the files to the OutRoot
// directory.
void copyFileToOutRoot(clang::tooling::UnifiedPath &InRoot,
                       clang::tooling::UnifiedPath &OutRoot,
                       const std::string &Pattern) {
  std::error_code EC;
  for (fs::recursive_directory_iterator Iter(Twine(InRoot.getPath()), EC), End;
       Iter != End; Iter.increment(EC)) {
    if ((bool)EC) {
      std::string ErrMsg = "[ERROR] Access : " + std::string(InRoot.getPath()) +
                           " fail: " + EC.message() + "\n";
      PrintMsg(ErrMsg);
      continue;
    }
    if (Iter->type() == fs::file_type::regular_file) {
      std::string FileName = llvm::sys::path::filename(Iter->path()).str();
      std::transform(FileName.begin(), FileName.end(), FileName.begin(),
                     ::toUpper);
      if (FileName.find(Pattern) != std::string::npos) {
        tooling::UnifiedPath FilePath = Iter->path();
        std::string CanFilePath = FilePath.getCanonicalPath().str();

        if (CanFilePath.find("_codepin_") != std::string::npos) {
          continue;
        }
        rewriteDir(CanFilePath, InRoot, OutRoot);
        createDirectories(path::parent_path(CanFilePath));
        if (llvm::sys::fs::exists(CanFilePath)) {
          std::string ErrMsg = "[Warning] The " + CanFilePath +
                               " is exists. Skip copy the file.\n";
          PrintMsg(ErrMsg);
          continue;
        }
        fs::copy_file(Iter->path(), CanFilePath);
      }
    }
  }
}
void processallOptionAction(clang::tooling::UnifiedPath &InRoot,
                            clang::tooling::UnifiedPath &OutRoot,
                            bool IsForSYCL) {
  extern DpctOption<clang::dpct::opt, bool> ProcessAll;
  if (ProcessAll) {
    std::error_code EC;
    for (fs::recursive_directory_iterator Iter(Twine(InRoot.getPath()), EC),
         End;
         Iter != End; Iter.increment(EC)) {
      if ((bool)EC) {
        std::string ErrMsg =
            "[ERROR] Access : " + std::string(InRoot.getPath()) +
            " fail: " + EC.message() + "\n";
        PrintMsg(ErrMsg);
        continue;
      }
      if (Iter->type() == fs::file_type::symlink_file) {
        tooling::UnifiedPath FilePath = Iter->path();
        std::string TargetPath = FilePath.getCanonicalPath().str();
        std::string LinkPath = FilePath.getAbsolutePath().str();
        if (IncludeFileMap[FilePath]) {
          rewriteFileName(TargetPath, TargetPath);
          rewriteFileName(LinkPath, LinkPath);
        }
        rewriteDir(TargetPath, InRoot, OutRoot);
        rewriteDir(LinkPath, InRoot, OutRoot);
        if (!sys::fs::exists(LinkPath)) {
          std::error_code EC = sys::fs::create_link(TargetPath, LinkPath);
          if (EC) {
            std::string ErrMsg = "Error creating symlink: " + EC.message() +
                                 ". The target path is " + TargetPath +
                                 ". And the link path is " + LinkPath + "\n";
            PrintMsg(ErrMsg);
            continue;
          }
        }
      }
    }
  }
  for (const auto &File : FilesNotInCompilationDB) {
    if (IncludeFileMap.find(File) != IncludeFileMap.end()) {
      // Skip the files parsed by dpct parser.
      continue;
    }

    std::ifstream In(File);
    clang::tooling::UnifiedPath OutputFile = File;
    if (!rewriteCanonicalDir(OutputFile, InRoot, OutRoot)) {
      continue;
    }

    // Check for another file with SYCL extension. For example
    // In in-root we have
    //  * src.cpp
    //  * src.cu
    // After migration we will end up replacing src.cpp with migrated src.cu
    // when the --sycl-file-extension is cpp
    // In such a case warn the user.

    // Make sure that the output file corresponds to a single and unique input
    // file.
    if (IsForSYCL &&
        checkOverwriteAndWarn(OutputFile.getCanonicalPath(), File)) {
      continue;
    }

    createDirectories(path::parent_path(OutputFile.getCanonicalPath()));
    clang::dpct::RawFDOStream Out(OutputFile.getCanonicalPath().str());
    std::stringstream buffer;
    buffer << In.rdbuf();
    Out << buffer.str();
    In.close();

    if (IsForSYCL) {
      OutFilePath2InFilePath[OutputFile.getCanonicalPath().str()] = File;
    }
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
      if (Comp.starts_with(".")) {
        IsHidden = true;
        break;
      }
    }

    // Skip hidden folder or file whose name begins with ".".
    if (IsHidden) {
      continue;
    }

    if (Iter->type() == fs::file_type::regular_file) {
      clang::tooling::UnifiedPath OutputFile = FilePath;
      if (!rewriteCanonicalDir(OutputFile, InRoot, OutRoot)) {
        continue;
      }
      if (IncludeFileMap.find(FilePath) != IncludeFileMap.end()) {
        // Skip the files processed by the first loop of
        // calling processFiles() in Tooling.cpp::ClangTool::run().
        continue;
      } else {
        if (DpctGlobalInfo::isExcluded(FilePath)) {
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
      clang::tooling::UnifiedPath OutDirectory = Path;
      if (!rewriteCanonicalDir(OutDirectory, InRoot, OutRoot)) {
        continue;
      }

      if (fs::exists(OutDirectory.getCanonicalPath()))
        continue;

      createDirectories(OutDirectory.getCanonicalPath());
    }
  }
}

extern DpctOption<dpct::opt, std::string> BuildScriptFile;
extern DpctOption<dpct::opt, bool> GenBuildScript;

static void getMainSrcFilesRepls(
    std::vector<clang::tooling::Replacement> &MainSrcFilesRepls) {
  auto &FileRelpsMap = DpctGlobalInfo::getFileRelpsMap();
  for (const auto &Entry : FileRelpsMap)
    for (const auto &Repl : Entry.second)
      MainSrcFilesRepls.push_back(Repl);
}
static void getMainSrcFilesDigest(
    std::vector<clang::tooling::MainSourceFileInfo> &MainSrcFilesDigest) {
  auto &DigestMap = DpctGlobalInfo::getDigestMap();
  for (const auto &Entry : DigestMap)
    MainSrcFilesDigest.push_back(Entry.second);
}

static void saveUpdatedMigrationDataIntoYAML(
    std::vector<clang::tooling::Replacement> &MainSrcFilesRepls,
    std::vector<clang::tooling::MainSourceFileInfo> &MainSrcFilesDigest,
    clang::tooling::UnifiedPath YamlFile, clang::tooling::UnifiedPath SrcFile,
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
      MainSrcFilesDigest.push_back(Entry.second);
    }
  }

  if (!MainSrcFilesRepls.empty() || !MainSrcFilesDigest.empty() ||
      !CompileCmdsPerTarget.empty()) {
    save2Yaml(YamlFile, SrcFile, MainSrcFilesRepls, MainSrcFilesDigest,
              CompileCmdsPerTarget);
  }
}

void applyPatternRewriter(const std::string &InputString,
                          llvm::raw_fd_ostream &Stream) {
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

int writeReplacementsToFiles(
    ReplTy &Replset, Rewriter &Rewrite, const std::string &Folder,
    clang::tooling::UnifiedPath &InRoot,
    std::vector<clang::tooling::MainSourceFileInfo> &MainSrcFilesDigest,
    std::unordered_map<std::string, bool> &MainSrcFileMap,
    std::vector<clang::tooling::Replacement> &MainSrcFilesRepls,
    std::unordered_map<clang::tooling::UnifiedPath,
                       std::vector<clang::tooling::Range>> &FileRangesMap,
    std::unordered_map<clang::tooling::UnifiedPath,
                       std::vector<clang::tooling::Range>>
        &FileBlockLevelFormatRangesMap,
    clang::dpct::ReplacementType IsForCodePin = RT_ForSYCLMigration) {

  volatile ProcessStatus status = MigrationSucceeded;
  clang::tooling::UnifiedPath OutPath;

  for (auto &Entry : Replset) {
    OutPath = StringRef(DpctGlobalInfo::removeSymlinks(
        Rewrite.getSourceMgr().getFileManager(), Entry.first));
    bool HasRealReplacements = true;
    auto Repls = Entry.second;

    if (Repls.size() == 1) {
      auto Repl = *Repls.begin();
      if (Repl.getLength() == 0 && Repl.getReplacementText().empty())
        HasRealReplacements = false;
    }

    if (IsForCodePin != clang::dpct::RT_CUDAWithCodePin) {
      auto Find = IncludeFileMap.find(OutPath);
      if (HasRealReplacements && Find != IncludeFileMap.end()) {
        IncludeFileMap[OutPath] = true;
      }

      // This operation won't fail; it already succeeded once during argument
      // validation.

      rewriteFileName(OutPath);
    }

    if (!rewriteCanonicalDir(OutPath, InRoot, Folder)) {
      continue;
    }

    // Check for another file with SYCL extension. For example
    // In in-root we have
    //  * src.cpp
    //  * src.cu
    // After migration we will end up replacing src.cpp with migrated src.cu
    // when the --sycl-file-extension is cpp
    // In such a case warn the user.
    if (checkOverwriteAndWarn(OutPath.getCanonicalPath(), Entry.first))
      continue;
    createDirectories(path::parent_path(OutPath.getCanonicalPath()));
    dpct::RawFDOStream OutStream(OutPath.getCanonicalPath());

    // For header file, as it can be included from different file, it needs
    // merge the migration triggered by each including.
    // For main file, as it can be compiled or preprocessed with different
    // macro defined, it also needs merge the migration triggered by each
    // command.
    if (IsForCodePin != clang::dpct::RT_CUDAWithCodePin) {
      SourceProcessType FileType = GetSourceFileType(Entry.first);
      if (FileType & (SPT_CppHeader | SPT_CudaHeader)) {
        mergeExternalReps(Entry.first, OutPath, Entry.second);
      } else {
        auto Hash = llvm::sys::fs::md5_contents(Entry.first);

        bool HasCUDASyntax = false;
        if (auto FileInfo =
                dpct::DpctGlobalInfo::getInstance().findFile(Entry.first)) {
          if (FileInfo->hasCUDASyntax()) {
            HasCUDASyntax = true;
          }
        }

        MainSrcFilesDigest.push_back(clang::tooling::MainSourceFileInfo(
            Entry.first, Hash->digest().c_str(), HasCUDASyntax));

        bool IsMainSrcFileChanged = false;
        std::string FilePath = Entry.first;

        auto &DigestMap = DpctGlobalInfo::getDigestMap();
        auto DigestIter = DigestMap.find(Entry.first);
        if (DigestIter != DigestMap.end()) {
          auto Digest = llvm::sys::fs::md5_contents(Entry.first);
          if (DigestIter->second.Digest != Digest->digest().c_str())
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
    }

    std::vector<clang::tooling::Range> Ranges;
    Ranges = calculateRangesWithFormatFlag(Entry.second);
    FileRangesMap.insert(std::make_pair(OutPath, Ranges));

    std::vector<clang::tooling::Range> BlockLevelFormatRanges;
    BlockLevelFormatRanges =
        calculateRangesWithBlockLevelFormatFlag(Entry.second);
    FileBlockLevelFormatRangesMap.insert(
        std::make_pair(OutPath, BlockLevelFormatRanges));
    tooling::applyAllReplacements(Replset[Entry.first], Rewrite);

    llvm::Expected<FileEntryRef> Result =
        Rewrite.getSourceMgr().getFileManager().getFileRef(Entry.first);

    if (auto E = Result.takeError()) {
      continue;
    }

    // Do not apply PatternRewriters for CodePin CUDA debug file
    if (MapNames::PatternRewriters.empty() ||
        IsForCodePin == clang::dpct::RT_CUDAWithCodePin) {
      Rewrite
          .getEditBuffer(Rewrite.getSourceMgr().getOrCreateFileID(
              *Result, clang::SrcMgr::C_User /*normal user code*/))
          .write(OutStream);
    } else {
      std::string OutputString;
      llvm::raw_string_ostream RSW(OutputString);
      Rewrite
          .getEditBuffer(Rewrite.getSourceMgr().getOrCreateFileID(
              *Result, clang::SrcMgr::C_User /*normal user code*/))
          .write(RSW);
      applyPatternRewriter(OutputString, OutStream);
    }
    // We have written a migrated file; Update the output file path info
    OutFilePath2InFilePath[OutPath.getCanonicalPath().str()] = Entry.first;
  }
  return status;
}

std::vector<std::string> codePinTypeTopologicalSort(
    std::vector<std::pair<std::string, std::vector<std::string>>> &Graph) {
  std::map<std::string, int> indegree;
  std::map<std::string, std::vector<std::string>> adj;
  std::vector<std::string> result;

  for (auto &node : Graph) {
    std::string u = node.first;
    std::vector<std::string> &edges = node.second;
    if (indegree.find(u) == indegree.end()) {
      indegree[u] = 0;
    }
    for (auto &v : edges) {
      adj[u].push_back(v);
      indegree[v]++;
    }
  }

  std::queue<std::string> q;
  for (auto &pair : indegree) {
    if (pair.second == 0) {
      q.push(pair.first);
    }
  }

  while (!q.empty()) {
    std::string u = q.front();
    q.pop();
    result.push_back(u);
    for (auto &v : adj[u]) {
      indegree[v]--;
      if (indegree[v] == 0) {
        q.push(v);
      }
    }
  }
  if (result.size() != indegree.size()) {
    return {};
  }
  return result;
}

std::string getCodePinPostfixName(MemberOrBaseInfoForCodePin &Info,
                                  bool IsForCUDADebug) {
  std::string TypeName;
  if (IsForCUDADebug) {
    TypeName = Info.TypeNameInCuda;
  } else {
    TypeName = Info.TypeNameInSycl;
  }
  if (Info.UserDefinedTypeFlag) {
    auto LPos = TypeName.find('<');
    if (LPos != std::string::npos) {
      TypeName.insert(LPos, "_codepin");
    } else {
      auto BPos = TypeName.find('[');
      if (BPos != std::string::npos) {
        TypeName.insert(BPos, "_codepin");
      } else {
        TypeName.append("_codepin");
      }
    }
  }
  for (int i = 0; i < Info.PointerDepth; i++) {
    TypeName += "*";
  }
  return TypeName;
}

void genCodePinDecl(dpct::RawFDOStream &RS, std::vector<std::string> &SortedVec,
                    bool IsForwardDecl, bool IsForCUDADebug) {
  int SortedVecSize = SortedVec.size();
  for (int j = SortedVecSize - 1; j >= 0; --j) {
    VarInfoForCodePin Info;
    bool IsTemplate = false;
    auto &CodePinTypeVec = dpct::DpctGlobalInfo::getCodePinTypeInfoVec();
    auto &CodePinTemplateTypeVec =
        dpct::DpctGlobalInfo::getCodePinTemplateTypeInfoVec();
    auto Iter = std::find_if(
        CodePinTypeVec.begin(), CodePinTypeVec.end(),
        [&](const auto &pair) { return pair.second.HashKey == SortedVec[j]; });
    if (Iter != CodePinTypeVec.end()) {
      if (Iter->second.TemplateFlag) {
        Iter =
            std::find_if(CodePinTemplateTypeVec.begin(),
                         CodePinTemplateTypeVec.end(), [&](const auto &pair) {
                           return pair.second.HashKey == SortedVec[j];
                         });
        if (Iter != CodePinTemplateTypeVec.end()) {
          Info = Iter->second;
          IsTemplate = true;
        } else {
          continue;
        }
      } else {
        Info = Iter->second;
      }
    } else {
      continue;
    }
    if (!Info.IsValid) {
      continue;
    }
    int NamespaceSize = IsForwardDecl ? Info.Namespaces.size() : 0;
    if (NamespaceSize) {
      for (int i = 0; i < NamespaceSize; i++) {
        RS << "namespace " << Info.Namespaces[i] << "{" << getNL();
      }
    }
    if (IsTemplate) {
      RS << "template<";
      int TemplateArgsNum = Info.TemplateArgs.size();
      for (int i = 0; i < TemplateArgsNum; i++) {
        if (i != 0) {
          RS << ", ";
        }
        RS << Info.TemplateArgs[i];
      }
      RS << ">" << getNL();
    }
    RS << Info.VarRecordType;
    if (IsForwardDecl) {
      if (Info.IsTypeDef) {
        RS << " " << Info.OrgTypeName << ";" << getNL();
        RS << "using " << Info.VarNameWithoutScopeAndTemplateArgs << " = "
           << Info.OrgTypeName << ";" << getNL();
      } else {
        RS << " " << Info.VarNameWithoutScopeAndTemplateArgs << ";" << getNL();
      }
    } else {
      RS << " " << Info.VarNameWithoutScopeAndTemplateArgs << "_codepin";
      if (!Info.Bases.empty()) {
        RS << " : ";
        int BaseSize = Info.Bases.size();
        for (int i = 0; i < BaseSize; ++i) {
          if (i != 0) {
            RS << ", ";
          }
          RS << "public "
             << getCodePinPostfixName(Info.Bases[i], IsForCUDADebug);
        }
      }
      RS << " {" << getNL() << "public:" << getNL();
      int MemberSize = Info.Members.size();
      for (int i = 0; i < MemberSize; ++i) {
        if (Info.Members[i].IsBaseMember) {
          continue;
        }
        RS << "  " << getCodePinPostfixName(Info.Members[i], IsForCUDADebug)
           << " " << Info.Members[i].MemberName;
        for (auto &D : Info.Members[i].Dims) {
          RS << "[" << std::to_string(D) << "]";
        }
        RS << ";" << getNL();
      }
      RS << "};" << getNL();
    }
    if (NamespaceSize) {
      for (int i = 0; i < NamespaceSize; i++) {
        RS << "}" << getNL();
      }
    }
    RS << getNL();
  }
}

void genCodePinDumpFunc(dpct::RawFDOStream &RS, bool IsForCUDADebug) {
  auto &Vec = dpct::DpctGlobalInfo::getCodePinTypeInfoVec();
  std::vector<std::string> SortedVec =
      codePinTypeTopologicalSort(DpctGlobalInfo::getCodePinDumpFuncDepsVec());
  int SortedVecSize = SortedVec.size();
  for (int j = SortedVecSize - 1; j >= 0; --j) {
    auto Iter = std::find_if(Vec.begin(), Vec.end(), [&](const auto &pair) {
      return pair.first == SortedVec[j];
    });
    if (Iter == Vec.end()) {
      continue;
    }
    if (!Iter->second.IsValid) {
      continue;
    }
    auto &Name = Iter->first;
    auto &Info = Iter->second;
    std::string CodepinTypeName = Info.VarNameWithoutScopeAndTemplateArgs +
                                  "_codepin" + Info.TemplateInstArgs;
    RS << "template <> class data_ser<" << CodepinTypeName << "> {" << getNL()
       << "public:" << getNL()
       << "  static void dump(dpctexp::codepin::detail::json_stringstream "
          "&ss, "
       << CodepinTypeName << " &value," << getNL()
       << "                   dpctexp::codepin::queue_t queue) {" << getNL();
    RS << "    auto arr = ss.array();" << getNL();

    if (int MemberNum = Info.Members.size()) {
      for (int i = 0; i < MemberNum; i++) {
        RS << "    {" << getNL() << "      auto obj" << i << " = arr.object();"
           << getNL() << "      obj" << i << ".key(\""
           << Info.Members[i].MemberName << "\");" << getNL()
           << "      auto value" << i << " = obj" << i
           << ".value<dpctexp::codepin::detail::json_stringstream::json_obj>("
              ");"
           << getNL() << "      dpctexp::codepin::detail::data_ser<"
           << getCodePinPostfixName(Info.Members[i], IsForCUDADebug);
        for (auto &D : Info.Members[i].Dims) {
          RS << "[" << std::to_string(D) << "]";
        }
        RS << ">::print_type_name(value" << i << ");" << getNL() << "      obj"
           << i << ".key(\"Data\");" << getNL()
           << "      dpctexp::codepin::detail::data_ser<"
           << getCodePinPostfixName(Info.Members[i], IsForCUDADebug);
        for (auto &D : Info.Members[i].Dims) {
          RS << "[" << std::to_string(D) << "]";
        }
        RS << ">::dump(ss, value." << Info.Members[i].MemberName << ", queue);"
           << getNL() << "    }" << getNL();
      }
    }
    RS << getNL() << "  }" << getNL();
    RS << "  static void print_type_name(json_stringstream::json_obj &obj) {"
       << getNL() << "    obj.key(\"Type\");" << getNL() << "    obj.value(\""
       << CodepinTypeName << "\");" << getNL() << "  }" << getNL() << "};"
       << getNL() << getNL();
    if (Info.TopTypeFlag) {
      RS << "template <> class data_ser<" << Name << "> {" << getNL()
         << "public:" << getNL()
         << "  static void dump(dpctexp::codepin::detail::json_stringstream "
            "&ss, "
         << Name << " &value," << getNL()
         << "                   dpctexp::codepin::queue_t queue) {" << getNL();
      RS << "    " + CodepinTypeName << "& temp = reinterpret_cast<"
         << CodepinTypeName << "&>(value);" << getNL();
      RS << "    dpctexp::codepin::detail::data_ser<" << CodepinTypeName
         << ">::dump(ss, temp, queue);" << getNL() << "  }" << getNL()
         << "  static void print_type_name(json_stringstream::json_obj &obj) {"
         << getNL() << "    obj.key(\"Type\");" << getNL() << "    obj.value(\""
         << Name << "\");" << getNL() << "  }" << getNL() << "};" << getNL()
         << getNL();
    }
  }
}

void genCodePinHeader(dpct::RawFDOStream &RS, bool IsForCUDADebug) {
  std::vector<std::string> SortedVec =
      codePinTypeTopologicalSort(DpctGlobalInfo::getCodePinTypeDepsVec());
  RS << "// This file is auto-generated to support the instrument API of "
        "CodePin."
     << getNL();
  RS << "#ifndef __DPCT_CODEPIN_AUTOGEN_UTIL__" << getNL()
     << "#define __DPCT_CODEPIN_AUTOGEN_UTIL__" << getNL() << getNL();
  genCodePinDecl(RS, SortedVec, true, IsForCUDADebug);
  RS << getNL() << "namespace dpct {" << getNL() << "namespace experimental {"
     << getNL() << "namespace codepin {" << getNL() << "namespace detail {"
     << getNL();
  genCodePinDecl(RS, SortedVec, false, IsForCUDADebug);
  genCodePinDumpFunc(RS, IsForCUDADebug);
  RS << "}" << getNL() << "}" << getNL() << "}" << getNL() << "}" << getNL();
  RS << "#endif" << getNL();
}

/// Apply all generated replacements, and immediately save the results to files
/// in output directory.
///
/// \returns 0 upon success. Non-zero upon failure.
/// Prerequisite: InRoot and OutRoot are both absolute paths
int saveNewFiles(clang::tooling::RefactoringTool &Tool,
                 clang::tooling::UnifiedPath InRoot,
                 clang::tooling::UnifiedPath OutRoot,
                 clang::tooling::UnifiedPath CUDAMigratedOutRoot,
                 ReplTy &ReplCUDA, ReplTy &ReplSYCL) {
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
  Rewriter DebugCUDARewrite(Sources, DefaultLangOptions);
  extern DpctOption<clang::dpct::opt, bool> ProcessAll;

  // The variable defined here assists to merge history records.
  std::unordered_map<std::string /*FileName*/,
                     bool /*false:Not processed in current migration*/>
      MainSrcFileMap;

  std::string SrcFile = "MainSrcFiles_placehold";
  std::string YamlFile = appendPath(OutRoot.getCanonicalPath().str(),
                                    DpctGlobalInfo::getYamlFileName());
  if (clang::dpct::DpctGlobalInfo::isIncMigration()) {
    auto PreTU = clang::dpct::DpctGlobalInfo::getMainSourceYamlTUR();
    for (const auto &Repl : PreTU->Replacements) {
      auto &FileRelpsMap = DpctGlobalInfo::getFileRelpsMap();
      FileRelpsMap[Repl.getFilePath().str()].push_back(Repl);
    }
    for (const auto &FileDigest : PreTU->MainSourceFilesDigest) {
      auto &DigestMap = DpctGlobalInfo::getDigestMap();
      DigestMap[FileDigest.MainSourceFile] = FileDigest;

      // Mark all the main src files loaded from yaml file are not processed
      // in current migration.
      MainSrcFileMap[FileDigest.MainSourceFile] = false;
    }

    for (const auto &Entry : PreTU->CompileTargets)
      CompileCmdsPerTarget[Entry.first] = Entry.second;
  }

  std::vector<clang::tooling::Replacement> MainSrcFilesRepls;
  std::vector<clang::tooling::MainSourceFileInfo> MainSrcFilesDigest;

  if (ReplSYCL.empty()) {
    // There are no rules applying on the *.cpp files,
    // dpct just do nothing with them.
    status = MigrationNoCodeChangeHappen;

    getMainSrcFilesRepls(MainSrcFilesRepls);
    getMainSrcFilesDigest(MainSrcFilesDigest);
  } else {
    std::unordered_map<clang::tooling::UnifiedPath,
                       std::vector<clang::tooling::Range>>
        FileRangesMap;
    std::unordered_map<clang::tooling::UnifiedPath,
                       std::vector<clang::tooling::Range>>
        FileBlockLevelFormatRangesMap;
    // There are matching rules for *.cpp files, *.cu files, also header files
    // included, migrate these files into *.dp.cpp files.
    auto GroupResult = groupReplacementsByFile(
        Rewrite.getSourceMgr().getFileManager(), ReplSYCL);
    if (auto RewriteStatus = writeReplacementsToFiles(
            ReplSYCL, Rewrite, OutRoot.getCanonicalPath().str(), InRoot,
            MainSrcFilesDigest, MainSrcFileMap, MainSrcFilesRepls,
            FileRangesMap, FileBlockLevelFormatRangesMap,
            clang::dpct::RT_ForSYCLMigration))
      return RewriteStatus;
    if (DpctGlobalInfo::isCodePinEnabled()) {
      if (auto RewriteStatus = writeReplacementsToFiles(
              ReplCUDA, DebugCUDARewrite,
              CUDAMigratedOutRoot.getCanonicalPath().str(), InRoot,
              MainSrcFilesDigest, MainSrcFileMap, MainSrcFilesRepls,
              FileRangesMap, FileBlockLevelFormatRangesMap,
              clang::dpct::RT_CUDAWithCodePin))
        return RewriteStatus;
    }
    // Print the in-root path and the number of processed files
    size_t ProcessedFileNumber;
    if (ProcessAll) {
      ProcessedFileNumber = IncludeFileMap.size();
    } else {
      ProcessedFileNumber = GroupResult.size();
    }
    std::string ReportMsg = "Processed " + std::to_string(ProcessedFileNumber) +
                            " file(s) in -in-root folder \"" +
                            InRoot.getCanonicalPath().str() + "\"";
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
    // Generated SYCL file in outroot. E.g., /path/to/outroot/a.dp.cpp
    clang::tooling::UnifiedPath FilePath = Entry.first;
    // Generated CUDA file in outroot_debug. E.g., /path/to/outroot_debug/a.cu
    clang::tooling::UnifiedPath DebugFilePath = Entry.first;
    // Original CUDA file in inroot. E.g., /path/to/inroot/a.cu
    clang::tooling::UnifiedPath OriginalFilePath = Entry.first;
    if (!Entry.second) {
      bool IsExcluded = DpctGlobalInfo::isExcluded(FilePath);
      if (IsExcluded) {
        continue;
      }
      // Always migrate *.cuh files to *.dp.hpp files,
      // Always migrate *.cu files to *.dp.cpp files.
      SourceProcessType FileType = GetSourceFileType(FilePath);
      SmallString<512> TempFilePath(FilePath.getCanonicalPath());
      if (FileType & SPT_CudaHeader) {
        path::replace_extension(TempFilePath,
                                DpctGlobalInfo::getSYCLHeaderExtension());
      } else if (FileType & SPT_CudaSource) {
        path::replace_extension(TempFilePath,
                                DpctGlobalInfo::getSYCLSourceExtension());
      }
      FilePath = TempFilePath;

      if (!rewriteCanonicalDir(FilePath, InRoot, OutRoot)) {
        continue;
      }

      if (dpct::DpctGlobalInfo::isCodePinEnabled() &&
          !rewriteCanonicalDir(DebugFilePath, InRoot, CUDAMigratedOutRoot)) {
        continue;
      }

      // Check for another file with SYCL extension. For example
      // In in-root we have
      //  * src.cpp
      //  * src.cu
      // After migration we will end up replacing src.cpp with migrated src.cu
      // when the --sycl-file-extension is cpp
      // In such a case warn the user.
      if (checkOverwriteAndWarn(FilePath.getCanonicalPath(),
                                Entry.first.getCanonicalPath()))
        continue;

      // If the file needs no replacement and it already exist, don't
      // make any changes
      if (fs::exists(FilePath.getCanonicalPath())) {
        // A header file with this name already exists.
        llvm::errs() << "File '" << FilePath
                     << "' already exists; skipping it.\n";
        continue;
      }

      createDirectories(path::parent_path(FilePath.getCanonicalPath()));
      dpct::RawFDOStream Stream(FilePath.getCanonicalPath());

      llvm::Expected<FileEntryRef> Result =
          Tool.getFiles().getFileRef(Entry.first.getCanonicalPath());
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

      // This will help us to detect the same output filename
      // for two different input files
      OutFilePath2InFilePath[FilePath.getCanonicalPath().str()] =
          Entry.first.getCanonicalPath().str();

      if (dpct::DpctGlobalInfo::isCodePinEnabled()) {
        // Copy non-replacement CUDA files into debug folder
        fs::copy_file(OriginalFilePath.getCanonicalPath(),
                      DebugFilePath.getCanonicalPath());
      }
    }
  }

  std::string ScriptFineName = "Makefile.dpct";
  if (!BuildScriptFile.empty())
    ScriptFineName = BuildScriptFile;
  if (GenBuildScript) {
    genBuildScript(Tool, InRoot, OutRoot, ScriptFineName);
  }
  saveUpdatedMigrationDataIntoYAML(MainSrcFilesRepls, MainSrcFilesDigest,
                                   YamlFile, SrcFile, MainSrcFileMap);
  if (dpct::DpctGlobalInfo::isCodePinEnabled()) {
    copyFileToOutRoot(InRoot, CUDAMigratedOutRoot, "MAKEFILE");
    copyFileToOutRoot(InRoot, CUDAMigratedOutRoot, "CMAKELISTS.TXT");
    copyFileToOutRoot(InRoot, CUDAMigratedOutRoot, ".CMAKE");
    std::string SchemaPathCUDA = CUDAMigratedOutRoot.getCanonicalPath().str() +
                                 "/codepin_autogen_util.hpp";
    std::string SchemaPathSYCL =
        OutRoot.getCanonicalPath().str() + "/codepin_autogen_util.hpp";
    std::error_code EC;
    createDirectories(path::parent_path(SchemaPathCUDA));
    createDirectories(path::parent_path(SchemaPathSYCL));

    dpct::RawFDOStream SchemaStreamCUDA(SchemaPathCUDA);
    genCodePinHeader(SchemaStreamCUDA, true);

    clang::dpct::RawFDOStream SchemaStreamSYCL(SchemaPathSYCL);
    genCodePinHeader(SchemaStreamSYCL, false);

    processallOptionAction(InRoot, CUDAMigratedOutRoot, false);
  }
  processallOptionAction(InRoot, OutRoot, true);

  return status;
}

void loadYAMLIntoFileInfo(clang::tooling::UnifiedPath Path) {
  clang::tooling::UnifiedPath OriginPath = Path;
  rewriteFileName(Path);
  if (!rewriteCanonicalDir(Path, DpctGlobalInfo::getInRoot(),
                           DpctGlobalInfo::getOutRoot())) {
    return;
  }

  clang::tooling::UnifiedPath YamlFilePath = Path.getCanonicalPath() + ".yaml";
  auto PreTU = std::make_shared<clang::tooling::TranslationUnitReplacements>();
  if (fs::exists(YamlFilePath.getCanonicalPath())) {
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
