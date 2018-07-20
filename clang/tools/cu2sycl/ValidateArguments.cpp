//===--- ValidateArguments.cpp -------------------------*- C++ -*---===//
//
// Copyright (C) 2018 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===---------------------------------------------------------------===//

#include "ValidateArguments.h"
#include "Utility.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include <algorithm>
#include <cassert>

using namespace llvm;
using namespace std;
namespace path = llvm::sys::path;
namespace fs = llvm::sys::fs;

// Set OutRoot to the current working directory.
static bool getDefaultOutRoot(std::string &OutRootPar) {
  SmallString<256> OutRoot;
  if (fs::current_path(OutRoot) != std::error_code()) {
    llvm::errs() << "Could not get current path.\n";
    return false;
  }

  OutRootPar.assign(begin(OutRoot), end(OutRoot));
  return true;
}

// Set InRoot to the directory of the only input source file.
static bool getDefaultInRoot(std::string &InRootPar,
                             const vector<string> &SourceFiles) {
  if (SourceFiles.size() != 1) {
    llvm::errs() << "-in-root was not specified; only one input file allowed "
                    "in this mode.\n";
    return false;
  }

  SmallString<256> InRoot = StringRef{SourceFiles.front()};
  // Remove the last component from path.
  path::remove_filename(InRoot);
  if (!makeCanonical(InRoot))
    return false;

  InRootPar.assign(begin(InRoot), end(InRoot));
  return true;
}

bool makeCanonicalOrSetDefaults(string &InRoot, string &OutRoot,
                                const std::vector<std::string> SourceFiles) {
  if (OutRoot.empty()) {
    if (!getDefaultOutRoot(OutRoot))
      return false;
  } else if (!makeCanonical(OutRoot)) {
    return false;
  }

  if (InRoot.empty()) {
    if (!getDefaultInRoot(InRoot, SourceFiles))
      return false;
  } else if (!makeCanonical(InRoot)) {
    return false;
  }
  if (fs::get_file_type(InRoot) != fs::file_type::directory_file) {
    llvm::errs() << "Error: '" << InRoot << "' is not a directory.\n";
    return false;
  }

  return true;
}

bool isChildPath(const std::string &Root, const std::string &Child) {
  auto Diff = mismatch(path::begin(Root), path::end(Root), path::begin(Child));
  return Diff.first == path::end(Root);
}

// Make sure all files have an extension and are under InRoot.
//
// TODO: Produce diagnostics with llvm machinery
bool validatePaths(const std::string &InRoot,
                   const vector<string> &SourceFiles) {
  assert(isCanonical(InRoot) && "InRoot must be a canonical path.");
  bool Ok = true;
  for (const auto &FilePath : SourceFiles) {
    auto AbsPath = FilePath;
    if (!makeCanonical(AbsPath)) {
      Ok = false;
      continue;
    }

    if (!isChildPath(InRoot, AbsPath)) {
      Ok = false;
      llvm::errs() << "Error: File '" << AbsPath
                   << "' is not under the specified input root directory '"
                   << InRoot << "'\n";
    }

    if (!path::has_extension(AbsPath)) {
      Ok = false;
      llvm::errs() << "Error: File '" << AbsPath
                   << "' does not have an extension.\n";
    }
  }

  return Ok;
}
