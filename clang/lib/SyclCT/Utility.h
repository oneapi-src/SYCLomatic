//===--- Utility.h -------------------------------------*- C++ -*---===//
//
// Copyright (C) 2018 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===---------------------------------------------------------------===//

#ifndef SYCLCT_UTILITY_H
#define SYCLCT_UTILITY_H

#include <string>

namespace llvm {
template <typename T> class SmallVectorImpl;
class StringRef;
} // namespace llvm

namespace clang {
class SourceManager;
class SourceLocation;
} // namespace clang

bool makeCanonical(llvm::SmallVectorImpl<char> &Path);
bool makeCanonical(std::string &Path);
bool isCanonical(llvm::StringRef Path);

// Returns true if Root is a real path-prefix of Child
// /x/y and /x/y/z -> true
// /x/y and /x/y   -> false
// /x/y and /x/yy/ -> false (not a prefix in terms of a path)
bool isChildPath(const std::string &Root, const std::string &Child);

// Returns the character sequence ("\n" or "\r\n") used to represent new line
// in the source line containing Loc.
const char *getNL(clang::SourceLocation Loc, const clang::SourceManager &SM);

// Returns the character sequence indenting the source line containing Loc.
llvm::StringRef getIndent(clang::SourceLocation Loc,
                          const clang::SourceManager &SM);

#endif // SYCLCT_UTILITY_H
