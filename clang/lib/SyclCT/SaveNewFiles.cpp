//===--- SaveNewFiles.cpp --------------------------------*- C++ -*---===//
//
// Copyright (C) 2018 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

#include "SaveNewFiles.h"
#include "ExternalReplacement.h"

#include "llvm/ADT/SmallString.h"
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
#include "llvm/Support/raw_os_ostream.h"
#include <cassert>
#include <fstream>

using namespace llvm;
namespace path = llvm::sys::path;
namespace fs = llvm::sys::fs;

// TODO: it's global variable,  refine in future.
std::map<std::string, bool> IncludeFileMap;

static void rewriteDir(SmallString<256> &FilePath, const StringRef InRoot,
                       const StringRef OutRoot) {
  assert(isCanonical(InRoot) && "InRoot must be a canonical path.");
  assert(isCanonical(FilePath) && "FilePath must be a canonical path.");
  auto PathDiff =
      mismatch(path::begin(FilePath), path::end(FilePath), path::begin(InRoot));
  SmallString<256> NewFilePath = OutRoot;
  path::append(NewFilePath, PathDiff.first, path::end(FilePath));
  FilePath = NewFilePath;
}

static void rewriteFileName(SmallString<256> &FilePath) {
  SourceProcessType FileType = GetSourceFileType(FilePath.str());

  if (FileType & TypeCudaSource) {
    path::replace_extension(FilePath, "sycl.cpp");
  } else if (FileType & TypeCppSource) {
    // to avoid conflict in the case that xxx.cu xxx.cpp show up in the same
    // folder
    path::replace_extension(FilePath, "cc_sycl.cpp");
  } else if (FileType & TypeCudaHeader) {
    path::replace_extension(FilePath, "sycl.hpp");
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
  ProcessStatus status = MigrationSucceeded;
  // Set up Rewriter.
  LangOptions DefaultLangOptions;
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
  TextDiagnosticPrinter DiagnosticPrinter(llvm::errs(), &*DiagOpts);
  DiagnosticsEngine Diagnostics(
      IntrusiveRefCntPtr<DiagnosticIDs>(new DiagnosticIDs()), &*DiagOpts,
      &DiagnosticPrinter, false);
  SourceManager Sources(Diagnostics, Tool.getFiles());
  Rewriter Rewrite(Sources, DefaultLangOptions);

  SmallString<256> OutPath;

  bool AppliedAll = true;
  if (Tool.getReplacements().empty()) {
    // There are no rules applying on the *.cpp files,
    // cyclct just do nothing with them.
    status = MigrationNoCodeChangeHappen;
  } else {
    // There are matching rules for *.cpp files ,*.cu files, also header files
    // included, migrate these files into *.sycl.cpp files.
    for (auto &Entry : groupReplacementsByFile(
             Rewrite.getSourceMgr().getFileManager(), Tool.getReplacements())) {

      OutPath = StringRef(Entry.first);
      makeCanonical(OutPath);
      auto Find = IncludeFileMap.find(OutPath.c_str());
      if (Find != IncludeFileMap.end()) {
        IncludeFileMap[OutPath.c_str()] = true;
      }

      // This operation won't fail; it already succeeded once during argument
      // validation.
      makeCanonical(OutPath);
      rewriteDir(OutPath, InRoot, OutRoot);

      rewriteFileName(OutPath);

      // for headfile, as it can be included from differnt file, it need
      // merge the migration triggered by each including.
      if (OutPath.back() == 'h') {
        // note the replacement of Entry.second are updated by this call.
        mergeExternalReps(std::string(OutPath.str()), Entry.second);
      }

      fs::create_directories(path::parent_path(OutPath));
      std::ofstream File(OutPath.str());
      llvm::raw_os_ostream Stream(File);

      AppliedAll =
          tooling::applyAllReplacements(Entry.second, Rewrite) || AppliedAll;
      Rewrite
          .getEditBuffer(Sources.getOrCreateFileID(
              Tool.getFiles().getFile(Entry.first),
              clang::SrcMgr::C_User /*normal user code*/))
          .write(Stream);
    }
  }

  // The necessary header files which have no no replacements will be copied to
  // "-out-root" directory
  for (const auto &Entry : IncludeFileMap) {
    SmallString<256> FilePath = StringRef(Entry.first);
    if (!Entry.second) {
      makeCanonical(FilePath);
      rewriteDir(FilePath, InRoot, OutRoot);
      if (fs::exists(FilePath)) {
        // A header file with this name already exists.
        llvm::errs() << "File '" << FilePath
                     << "' already exists; skipping it.\n";
        AppliedAll = false;
        continue;
      }

      fs::create_directories(path::parent_path(FilePath));
      std::ofstream File(FilePath.str());
      llvm::raw_os_ostream Stream(File);

      Rewrite
          .getEditBuffer(Sources.getOrCreateFileID(
              Tool.getFiles().getFile(Entry.first),
              clang::SrcMgr::C_User /*normal user code*/))
          .write(Stream);
    }
  }

  if (!AppliedAll) {
    llvm::errs() << "Skipped some replacements.\n";
    status = MigrationSkipped;
  }
  return status;
}
