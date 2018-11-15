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

#include <ios>
#include <iostream>
#include <list>
#include <map>
#include <sstream>
#include <stack>
#include <string>
#include <vector>

namespace llvm {
template <typename T> class SmallVectorImpl;
class StringRef;
} // namespace llvm

namespace clang {
class SourceManager;
class SourceLocation;
class Stmt;
class ASTContext;
} // namespace clang

// classes for keeping track of Stmt->String mappings
struct StmtStringPair {
  const clang::Stmt *StmtVal;
  std::string Str;
};

class StmtStringMap {
  typedef std::map<const clang::Stmt *, std::string> MapTy;

public:
  void insert(StmtStringPair &SSP) {
    Map.insert(std::make_pair(SSP.StmtVal, SSP.Str));
  }
  std::string lookup(const clang::Stmt *S) {
    MapTy::iterator It = Map.find(S);
    if (It != Map.end()) {
      return It->second;
    }
    return "";
  }

private:
  MapTy Map;
};

bool makeCanonical(llvm::SmallVectorImpl<char> &Path);
bool makeCanonical(std::string &Path);
bool isCanonical(llvm::StringRef Path);

// Returns true if Root is a real path-prefix of Child
// /x/y and /x/y/z -> true
// /x/y and /x/y   -> false
// /x/y and /x/yy/ -> false (not a prefix in terms of a path)
bool isChildPath(const std::string &Root, const std::string &Child);

// Returns true if Root is a real same with Child
// /x/y and /x/y/z -> false
// /x/y and /x/y   -> true
bool isSamePath(const std::string &Root, const std::string &Child);

// Returns the character sequence ("\n" or "\r\n") used to represent new line
// in the source line containing Loc.
const char *getNL(clang::SourceLocation Loc, const clang::SourceManager &SM);

// Returns the character sequence indenting the source line containing Loc.
llvm::StringRef getIndent(clang::SourceLocation Loc,
                          const clang::SourceManager &SM);

// Get the Stmt spelling
std::string getStmtSpelling(const clang::Stmt *E,
                            const clang::ASTContext &Context);
// Get the Stmt spelling with the existing transforms applied
std::string getStmtSpellingWithTransforms(const clang::Stmt *S,
                                          const clang::ASTContext &Context,
                                          StmtStringMap *SSM);

template <typename T> std::string getHashAsString(const T &Val) {
  std::stringstream Stream;
  Stream << std::hex << std::hash<T>()(Val);
  return Stream.str();
}

enum SourceProcessType {
  // flag for *.cu
  TypeCudaSource = 1,

  // flag for *.cuh
  TypeCudaHeader = 2,

  // flag for *.cpp, *.cxx, *.cc
  TypeCppSource = 4,

  // flag for *.hpp, *.hxx *.h
  TypeCppHeader = 8,
};

SourceProcessType GetSourceFileType(llvm::StringRef SourcePath);

// For exmaple we have four rules named "A", "B", "C" and "D"
// "A" depends on "B" and "C"
// "B" depends on "D"
// "C" doesn't depend on anyone
// "D" depends on "C"
// Using topological sorting, it should output A -> B -> D -> C,
// and all rule dependencies will be met.
std::vector<std::string>
ruleTopoSort(std::vector<std::vector<std::string>> &TableRules);

#endif // SYCLCT_UTILITY_H
