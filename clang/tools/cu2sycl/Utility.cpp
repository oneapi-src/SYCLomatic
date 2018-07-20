//===--- Utility.cpp -----------------------------------*- C++ -*---===//
//
// Copyright (C) 2018 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===---------------------------------------------------------------===//

#include "Utility.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include <algorithm>

using namespace llvm;
using namespace std;

namespace path = llvm::sys::path;
namespace fs = llvm::sys::fs;

bool makeCanonical(SmallVectorImpl<char> &Path) {
  if (fs::make_absolute(Path) != std::error_code()) {
    llvm::errs() << "Could not get absolute path from '" << Path << "'\n ";
    return false;
  }
  path::remove_dots(Path, /* remove_dot_dot= */ true);
  return true;
}

bool makeCanonical(string &PathPar) {
  SmallString<256> Path = StringRef{PathPar};
  if (!makeCanonical(Path))
    return false;
  PathPar.assign(begin(Path), end(Path));
  return true;
}

bool isCanonical(const string &Path) {
  bool HasNoDots = all_of(path::begin(Path), path::end(Path),
                          [](StringRef e) { return e != "." && e != ".."; });
  return HasNoDots && path::is_absolute(Path);
}
