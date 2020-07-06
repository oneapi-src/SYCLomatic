//===--- Utility.h -------------------------------------*- C++ -*---===//
//
// Copyright (C) 2018 - 2020 Intel Corporation. All rights reserved.
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
#include <memory>
#include <sstream>
#include <stack>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "clang/AST/Attr.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "clang/Frontend/CompilerInstance.h"
namespace path = llvm::sys::path;

namespace llvm {
class StringRef;
} // namespace llvm

namespace clang {
class SourceManager;
class SourceLocation;
class SourceRange;
class Stmt;
class Expr;
class CompoundStmt;
class ASTContext;
class ValueDecl;
class DeclRefExpr;
class Expr;
class MemberExpr;
class FunctionDecl;
class CallExpr;
class Token;
class LangOptions;
class DynTypedNode;

namespace tooling {
class Range;
class Replacements;
} // namespace tooling
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

template <class StreamT> class ParensPrinter {
  StreamT &OS;

public:
  ParensPrinter(StreamT &OS) : OS(OS) { OS << "("; }
  ~ParensPrinter() { OS << ")"; }
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
inline bool isChildPath(const std::string &RootAbs, const std::string &Child,
                        bool IsChildAbs = true) {
  // 1st make Child as absolute path, then do compare.
  llvm::SmallString<256> ChildAbs;
  std::error_code EC;
  bool InChildAbsValid = true;

  if (IsChildAbs) {
    EC = llvm::sys::fs::real_path(Child, ChildAbs);
    if ((bool)EC) {
      InChildAbsValid = false;
    }
  } else {
    ChildAbs = Child;
  }

#if defined(_WIN64)
  std::string LocalRoot = llvm::StringRef(RootAbs).lower();
  std::string LocalChild =
      InChildAbsValid ? ChildAbs.str().lower() : llvm::StringRef(Child).lower();
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

int getCurrnetColumn(clang::SourceLocation Loc, const clang::SourceManager &SM);

/// Returns the character sequence indenting the source line containing Loc.
llvm::StringRef getIndent(clang::SourceLocation Loc,
                          const clang::SourceManager &SM);

/// Get the Stmt spelling
std::string getStmtSpelling(const clang::Stmt *E);

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
bool callingFuncHasDeviceAttr(const clang::CallExpr *CE);
const clang::FunctionDecl *getImmediateOuterFuncDecl(const clang::Stmt *S);
bool isInSameScope(const clang::Stmt *S, const clang::ValueDecl *D);
const clang::DeclRefExpr *getInnerValueDecl(const clang::Expr *Arg);
const clang::Stmt *getParentStmt(clang::DynTypedNode Node);
const clang::Stmt *getParentStmt(const clang::Stmt *S);
const clang::Stmt *getParentStmt(const clang::Decl *D);
const clang::Decl *getParentDecl(const clang::Decl *D);
const clang::Stmt *getNonImplicitCastParentStmt(const clang::Stmt *S);
const clang::DeclStmt *getAncestorDeclStmt(const clang::Expr *E);
const std::shared_ptr<clang::DynTypedNode>
getParentNode(const std::shared_ptr<clang::DynTypedNode> N);
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
bool isArgUsedAsLvalueUntil(const clang::DeclRefExpr *Arg,
                            const clang::Stmt *S);
unsigned int getLenIncludingTrailingSpaces(clang::SourceRange Range,
                                           const clang::SourceManager &SM);
std::vector<const clang::Stmt *>
getConditionNode(clang::DynTypedNode Node);
std::vector<const clang::Stmt *> getConditionExpr(clang::DynTypedNode Node);
bool isConditionOfFlowControl(const clang::CallExpr *CE,
                              std::string &OriginStmtType,
                              bool &CanAvoidUsingLambda,
                              clang::SourceLocation &SL);
bool isConditionOfFlowControl(const clang::Expr *E,
                              bool OnlyCheckConditionExpr = false);
std::string getBufferNameAndDeclStr(const std::string &PointerName,
                                    const std::string &TypeAsStr,
                                    const std::string &IndentStr,
                                    std::string &BufferDecl);
std::string getBufferNameAndDeclStr(const clang::Expr *Arg,
                                    const std::string &TypeAsStr,
                                    const std::string &IndentStr,
                                    std::string &BufferDecl);
void VarReferencedInFD(const clang::Stmt *S, const clang::ValueDecl *VD,
                         std::vector<const clang::DeclRefExpr *> &Result);
int getLengthOfSpacesToEndl(const char *CharData);

template <class StreamTy>
StreamTy &printPartialArguments(StreamTy &Stream, size_t PrintingArgsNum) {
  return Stream;
}
template <class StreamTy, class FirstArg, class... RestArgs>
StreamTy &printPartialArguments(StreamTy &Stream, size_t PrintingArgsNum,
                                FirstArg &&First, RestArgs &&... Rest) {
  if (PrintingArgsNum) {
    Stream << std::forward<FirstArg>(First);
    if (--PrintingArgsNum) {
      Stream << ", ";
    }
    return printPartialArguments(Stream, PrintingArgsNum,
                                 std::forward<RestArgs>(Rest)...);
  }
  return Stream;
}
template <class StreamTy, class... Args>
StreamTy &printArguments(StreamTy &Stream, Args &&... Arguments) {
  return printPartialArguments(Stream, sizeof...(Args),
                               std::forward<Args>(Arguments)...);
}
void printDerefOp(std::ostream &OS, const clang::Expr *E,
                  std::string *DerefType = nullptr);
bool isInSameLine(clang::SourceLocation A, clang::SourceLocation B,
                  const clang::SourceManager &SM, bool &Invalid);
clang::SourceRange getFunctionRange(const clang::CallExpr *CE);
std::vector<clang::tooling::Range>
calculateRangesWithFormatFlag(
    const clang::tooling::Replacements &Replaces);

bool isAssigned(const clang::Stmt *S);

std::string getTempNameForExpr(const clang::Expr *E, bool HandleLiteral = false,
                               bool KeepLastUnderline = true,
                               bool IsInMacroDefine = false);
bool isOuterMostMacro(const clang::Stmt *E);
bool isInsideFunctionLikeMacro(
    const clang::SourceLocation BeginLoc, const clang::SourceLocation EndLoc,
    const std::shared_ptr<clang::DynTypedNode> Parent);
enum ExprSpellingStatus {
  NoType = 0,
  IsDefine = 1,
  IsExpansion = 2
};
bool isExprStraddle(const clang::Stmt *S, ExprSpellingStatus *SpellingStatus);
bool isSimpleAddrOf(const clang::Expr *E);
bool isCOCESimpleAddrOf(const clang::Expr *E);
std::string getNameStrRemovedAddrOf(const clang::Expr *E, bool isCOCE = false);
std::string getDrefName(const clang::Expr *E);
std::vector<const clang::DeclaratorDecl *>
getSiblingDecls(const clang::DeclaratorDecl *DD);
std::string deducePointerType(const clang::DeclaratorDecl *DD,
                                      std::string TypeName);
bool isAnIdentifierOrLiteral(const clang::Expr *E);
bool isSameSizeofTypeWithTypeStr(const clang::Expr *E,
                                 const std::string &TypeStr);
std::string addIndirectionIfNecessary(const clang::Expr *E);
bool isInReturnStmt(const clang::Expr *E,
                    clang::SourceLocation &OuterInsertLoc);
std::string getHashStrFromLoc(clang::SourceLocation Loc);
const clang::FunctionDecl *getFunctionDecl(const clang::Stmt *S);
const clang::CXXRecordDecl *getParentRecordDecl(const clang::ValueDecl *DD);
bool IsTypeChangedToPointer(const clang::DeclRefExpr * DRE);
clang::SourceLocation getBeginLocOfPreviousEmptyMacro(clang::SourceLocation Loc);
clang::SourceLocation getEndLocOfFollowingEmptyMacro(clang::SourceLocation Loc);

std::string getNestedNameSpecifierString(const clang::NestedNameSpecifier *);
std::string getNestedNameSpecifierString(const clang::NestedNameSpecifierLoc &);

bool needExtraParens(const clang::Expr *);
#endif // DPCT_UTILITY_H
