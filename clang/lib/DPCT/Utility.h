//===--- Utility.h -------------------------------------*- C++ -*---===//
//
// Copyright (C) 2018 - 2019 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===---------------------------------------------------------------===//

#ifndef DPCT_UTILITY_H
#define DPCT_UTILITY_H

#include <functional>
#include <ios>
#include <iostream>
#include <list>
#include <map>
#include <unordered_map>
#include <memory>
#include <sstream>
#include <stack>
#include <string>
#include <utility>
#include <vector>

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
namespace path = llvm::sys::path;

namespace llvm {
template <typename T> class SmallVectorImpl;
class StringRef;
} // namespace llvm

namespace clang {
class SourceManager;
class SourceLocation;
class SourceRange;
class Stmt;
class CompoundStmt;
class ASTContext;
class ValueDecl;
class DeclRefExpr;
class Expr;
class MemberExpr;
class FunctionDecl;
} // namespace clang

// classes for keeping track of Stmt->String mappings
using StmtStringPair = std::pair<const clang::Stmt *, std::string>;

class StmtStringMap {
  typedef std::map<const clang::Stmt *, std::string> MapTy;

public:
  void insert(const StmtStringPair &SSP) { Map.insert(SSP); }
  std::string lookup(const clang::Stmt *S) const {
    auto It = Map.find(S);
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

/// Check \param Child is whether the child path of \param RootAbs
/// \param [in] RootAbs An absolute path as reference.
/// \param [in] Child A path to be checked.
/// \return true: child path, false: not child path
/// /x/y and /x/y/z -> true
/// /x/y and /x/y   -> false
/// /x/y and /x/yy/ -> false (not a prefix in terms of a path)
inline bool isChildPath(const std::string &RootAbs, const std::string &Child) {
  // 1st make Child as absolute path, then do compare.
  llvm::SmallString<256> ChildAbs;
  std::error_code EC;
  bool InChildAbsValid = true;

  EC = llvm::sys::fs::real_path(Child, ChildAbs);
  if ((bool)EC) {
    InChildAbsValid = false;
  }

#if defined(_WIN64)
  std::string LocalRoot = llvm::StringRef(RootAbs).lower();
  std::string LocalChild = InChildAbsValid ? ChildAbs.str().lower() : llvm::StringRef(Child).lower();
#elif defined(__linux__)
  std::string LocalRoot = RootAbs.c_str();
  std::string LocalChild = InChildAbsValid ? ChildAbs.c_str() : Child;
#else
#error Only support windows and Linux.
#endif

  auto Diff = mismatch(path::begin(LocalRoot), path::end(LocalRoot),
                       path::begin(LocalChild));
  // LocalRoot is not considered prefix of LocalChild if they are equal.
  return Diff.first == path::end(LocalRoot) &&
         Diff.second != path::end(LocalChild);
}

/// Check \param Child is whether have the same path of \param RootAbs
/// \param [in] RootAbs An absolute path as reference.
/// \param [in] Child A path to be checked.
/// \return true: same path, false: different path
/// /x/y and /x/y/z -> false
/// /x/y and /x/y   -> true
inline bool isSamePath(const std::string &RootAbs, const std::string &Child) {
  // 1st make Child as absolute path, then do compare.
  llvm::SmallString<256> ChildAbs;
  std::error_code EC;
  bool InChildAbsValid = true;
  EC = llvm::sys::fs::real_path(Child, ChildAbs);
  if ((bool)EC) {
    InChildAbsValid = false;
  }
#if defined(_WIN64)
  std::string LocalRoot = llvm::StringRef(RootAbs).lower();
  std::string LocalChild = InChildAbsValid ? ChildAbs.str().lower() : Child;
#elif defined(__linux__)
  std::string LocalRoot = RootAbs.c_str();
  std::string LocalChild = InChildAbsValid ? ChildAbs.c_str() : Child;
#else
#error Only support windows and Linux.
#endif
  auto Diff = mismatch(path::begin(LocalRoot), path::end(LocalRoot),
                       path::begin(LocalChild));
  return Diff.first == path::end(LocalRoot) &&
         Diff.second == path::end(LocalChild);
}

extern std::unordered_map<std::string, bool> ChildOrSameCache;

/// Check \param Child is whether the child or same path of \param RootAbs
/// \param [in] RootAbs An absolute path as reference.
/// \param [in] Child A path to be checked.
/// \return true: child path, false: not child path
inline bool isChildOrSamePath(const std::string &RootAbs,
                            const std::string &Child) {
  if (Child.empty()) {
    return false;
  }
  auto Iter = ChildOrSameCache.find(RootAbs + Child);
  if (Iter != ChildOrSameCache.end()) {
    return Iter->second;
  }
  // 1st make Child as absolute path, then do compare.
  llvm::SmallString<256> ChildAbs;
  std::error_code EC;
  bool InChildAbsValid = true;
  EC = llvm::sys::fs::real_path(Child, ChildAbs);
  if ((bool)EC) {
    InChildAbsValid = false;
  }
#if defined(_WIN64)
  std::string LocalRoot = llvm::StringRef(RootAbs).lower();
  std::string LocalChild = InChildAbsValid ? ChildAbs.str().lower() : Child;
#elif defined(__linux__)
  std::string LocalRoot = RootAbs.c_str();
  std::string LocalChild = InChildAbsValid ? ChildAbs.c_str() : Child;
#else
#error Only support windows and Linux.
#endif
  auto Diff = mismatch(path::begin(LocalRoot), path::end(LocalRoot),
                       path::begin(LocalChild));
  bool Ret = Diff.first == path::end(LocalRoot);
  ChildOrSameCache.insert(make_pair(RootAbs + Child, Ret));
  return Ret;
}

/// Returns character sequence ("\n") on Linux, return  ("\r\n") on Windows.
const char *getNL(void);

/// Returns the character sequence ("\n" or "\r\n") used to represent new line
/// in the source line containing Loc.
const char *getNL(clang::SourceLocation Loc, const clang::SourceManager &SM);

/// Returns the character sequence indenting the source line containing Loc.
llvm::StringRef getIndent(clang::SourceLocation Loc,
                          const clang::SourceManager &SM);

/// Get the Stmt spelling
std::string getStmtSpelling(const clang::Stmt *E,
                            const clang::ASTContext &Context);

template <typename T> std::string getHashAsString(const T &Val) {
  std::stringstream Stream;
  Stream << std::hex << std::hash<T>()(Val);
  return Stream.str();
}

/// Get the fixed suffix of compatibility tool
inline std::string getCTFixedSuffix() { return "_ct1"; }

/// Get the declaration of a statement
template <typename T> const T *getDecl(const clang::Stmt *E);

// TODO:  implement one of this for each source language.
enum SourceProcessType {
  // flag for *.cu
  TypeCudaSource = 1,

  // flag for *.cuh
  TypeCudaHeader = 2,

  // flag for *.cpp, *.cxx, *.cc, *.c, *.C
  TypeCppSource = 4,

  // flag for *.hpp, *.hxx *.h
  TypeCppHeader = 8,
};

SourceProcessType GetSourceFileType(llvm::StringRef SourcePath);

/// Topology sort for rules
/// For exmaple we have four rules named "A", "B", "C" and "D"
/// "A" depends on "B" and "C"
/// "B" depends on "D"
/// "C" doesn't depend on anyone
/// "D" depends on "C"
/// Using topological sorting, it should output A -> B -> D -> C,
/// and all rule dependencies will be met.
std::vector<std::string>
ruleTopoSort(std::vector<std::vector<std::string>> &TableRules);

const std::string &getFmtEndStatement(void);
const std::string &getFmtStatementIndent(std::string &BaseIndent);

const std::string &getFmtEndArg(void);
const std::string &getFmtArgIndent(std::string &BaseIndent);

/// Split a string into a vector of strings with a specific delimiter
/// \param [in] Str The string to be splited
/// \param [in] Delim The delimiter
std::vector<std::string> split(const std::string &Str, char Delim);

/// Determines if a string starts with a prefix
/// \param [in] Str The target string
/// \param [in] Prefix The prefix
bool startsWith(const std::string &Str, const std::string &Prefix);

/// Determines if a string starts with a char
/// \param [in] Str The target string
/// \param [in] C The char
bool startsWith(const std::string &Str, char C);

/// Determines if a string ends with a suffix
/// \param [in] Str The target string
/// \param [in] Prefix The prefix
bool endsWith(const std::string &Str, const std::string &Suffix);

/// Determines if a string ends with a char
/// \param [in] Str The target string
/// \param [in] C The char
bool endsWith(const std::string &Str, char C);

const clang::CompoundStmt *findImmediateBlock(const clang::Stmt *S);
const clang::CompoundStmt *findImmediateBlock(const clang::ValueDecl *D);
bool isInSameScope(const clang::Stmt *S, const clang::ValueDecl *D);
const clang::DeclRefExpr *getInnerValueDecl(const clang::Expr *Arg);
const clang::Stmt *getParentStmt(const clang::Stmt *S);
bool IsSingleLineStatement(const clang::Stmt *S);
clang::SourceRange getScopeInsertRange(const clang::MemberExpr *ME);
clang::SourceRange
getScopeInsertRange(const clang::Expr *CE,
                    const clang::SourceLocation &FuncNameBegin,
                    const clang::SourceLocation &FuncCallEnd);
const clang::Stmt *findNearestNonExprNonDeclAncestorStmt(const clang::Expr *E);
std::string getCanonicalPath(clang::SourceLocation Loc);
bool containOnlyDigits(const std::string &str);
void replaceSubStr(std::string &Str, const std::string &SubStr,
                   const std::string &Repl);
void replaceSubStrAll(std::string &Str, const std::string &SubStr,
                   const std::string &Repl);
#endif // DPCT_UTILITY_H
