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

static void rewriteDir(SmallString<256> &FilePath, const StringRef InRoot,
                       const StringRef OutRoot) {
  auto PathDiff =
      mismatch(path::begin(FilePath), path::end(FilePath), path::begin(InRoot));
  SmallString<256> NewFilePath = OutRoot;
  path::append(NewFilePath, PathDiff.first, path::end(FilePath));
  FilePath = NewFilePath;
}

// Prerequisite: InFilePath contains a .xxx extension
static void rewriteFileName(SmallString<256> &FilePath) {
  auto Extension = path::extension(FilePath);
  if (Extension == ".cu") {
    path::replace_extension(FilePath, "sycl.cpp");
  } else if (Extension == ".cuh") {
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
  for (const auto &Entry : groupReplacementsByFile(
           Rewrite.getSourceMgr().getFileManager(), Tool.getReplacements())) {
    OutPath = StringRef{Entry.first};
    // This operation won't fail; it already succeeded once during argument
    // validation.
    makeCanonical(OutPath);
    rewriteDir(OutPath, InRoot, OutRoot);
    rewriteFileName(OutPath);

    if (OutPath.back() == 'h' && fs::exists(OutPath)) {
      // A header file with this name already exists.
      // For now we do no merging and do not handle this case.
      // TODO: Implement strategy to handle translated headers that might
      // differ due to their point of inclusion
      llvm::errs() << "File '" << OutPath << "' already exists; skipping it.\n";
      AppliedAll = false;
      continue;
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

  if (!AppliedAll) {
    llvm::errs() << "Skipped some replacements.\n";
    return 1;
  }
  return 0;
}
