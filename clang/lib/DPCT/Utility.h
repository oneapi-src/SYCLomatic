//===--------------- Utility.h --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

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
#include <unordered_set>
#include <utility>
#include <vector>

#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/ParentMapContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Frontend/CompilerInstance.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
namespace path = llvm::sys::path;

using namespace clang;
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
class Decl;
class Expr;
class MemberExpr;
class FunctionDecl;
class CallExpr;
class Token;
class LangOptions;
class DynTypedNode;

namespace dpct {
enum class FFTTypeEnum;
class DeviceFunctionInfo;
enum class HelperFileEnum : unsigned int;
struct HelperFunc;
} // namespace dpct

namespace tooling {
class Range;
class Replacements;
} // namespace tooling
} // namespace clang

extern bool IsUsingDefaultOutRoot;
void removeDefaultOutRootFolder(const std::string &DefaultOutRoot);
void dpctExit(int ExitCode, bool NeedCleanUp = true);

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

template <class StreamT> class CurlyBracketsPrinter {
  StreamT &OS;

public:
  CurlyBracketsPrinter(StreamT &OS) : OS(OS) { OS << "{"; }
  ~CurlyBracketsPrinter() { OS << "}"; }
};

bool makeCanonical(llvm::SmallVectorImpl<char> &Path);
bool makeCanonical(std::string &Path);
bool isCanonical(llvm::StringRef Path);

extern std::unordered_map<std::string, llvm::SmallString<256>> RealPathCache;
extern std::unordered_map<std::string, bool> ChildPathCache;
extern std::unordered_map<std::string, bool> IsDirectoryCache;

/// Check \param FilePath is whether a directory path
/// \param [in] FilePath is a file path.
/// \return true: directory path, false: not directory path.
inline bool isDirectory(const std::string &FilePath) {
  const auto &Key = FilePath;
  auto Iter = IsDirectoryCache.find(Key);
  if (Iter != IsDirectoryCache.end()) {
    return Iter->second;
  } else {
    auto Ret = llvm::sys::fs::is_directory(FilePath);
    IsDirectoryCache[Key] = Ret;
    return Ret;
  }
}

/// Check \param Child is whether the child path of \param RootAbs
/// \param [in] RootAbs An absolute path as reference.
/// \param [in] Child A path to be checked.
/// \return true: child path, false: not child path
/// /x/y and /x/y/z -> true
/// /x/y and /x/y   -> false
/// /x/y and /x/yy/ -> false (not a prefix in terms of a path)
inline bool isChildPath(const std::string &RootAbs, const std::string &Child,
                        bool IsChildRelative = true) {
  // 1st make Child as absolute path, then do compare.
  llvm::SmallString<256> ChildAbs;
  std::error_code EC;
  bool InChildAbsValid = true;

  if (IsChildRelative) {
    auto &RealPath = RealPathCache[Child];
    if (!RealPath.empty()) {
      ChildAbs = RealPath;
    } else {
      EC = llvm::sys::fs::real_path(Child, ChildAbs, true);
      if ((bool)EC) {
        InChildAbsValid = false;
      } else {
        RealPathCache[Child] = ChildAbs;
      }
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
  auto Key = LocalRoot + ":" + LocalChild;
  auto Iter = ChildPathCache.find(Key);
  if (Iter != ChildPathCache.end()) {
    return Iter->second;
  }

  auto Diff = std::mismatch(path::begin(LocalRoot), path::end(LocalRoot),
                            path::begin(LocalChild));
  // LocalRoot is not considered prefix of LocalChild if they are equal.
  bool Ret = Diff.first == path::end(LocalRoot) &&
             Diff.second != path::end(LocalChild);
  ChildPathCache[Key] = Ret;
  return Ret;
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

  auto Key = RootAbs + ":" + Child;
  auto Iter = ChildOrSameCache.find(Key);
  if (Iter != ChildOrSameCache.end()) {
    return Iter->second;
  }
  // 1st make Child as absolute path, then do compare.
  llvm::SmallString<256> ChildAbs;
  std::error_code EC;
  bool InChildAbsValid = true;

  auto &RealPath = RealPathCache[Child];
  if (!RealPath.empty()) {
    ChildAbs = RealPath;
  } else {
    EC = llvm::sys::fs::real_path(Child, ChildAbs, true);
    if ((bool)EC) {
      InChildAbsValid = false;
    } else {
      RealPath = ChildAbs;
    }
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
  auto Diff = std::mismatch(path::begin(LocalRoot), path::end(LocalRoot),
                            path::begin(LocalChild));
  bool Ret = Diff.first == path::end(LocalRoot);
  ChildOrSameCache[Key] = Ret;
  return Ret;
}

/// Returns character sequence ("\n") on Linux, return ("\r\n") on Windows.
const char *getNL(void);

/// Returns the character sequence ("\n" or "\r\n") used to represent new line
/// in the source line containing Loc.
const char *getNL(clang::SourceLocation Loc, const clang::SourceManager &SM);

unsigned int getOffsetOfLineBegin(clang::SourceLocation Loc,
                                  const clang::SourceManager &SM);
int getCurrnetColumn(clang::SourceLocation Loc, const clang::SourceManager &SM);

/// Returns the character sequence indenting the source line containing Loc.
llvm::StringRef getIndent(clang::SourceLocation Loc,
                          const clang::SourceManager &SM);

clang::SourceRange getStmtExpansionSourceRange(const clang::Stmt *S);
clang::SourceRange getStmtSpellingSourceRange(const clang::Stmt *S);
clang::SourceRange getSpellingSourceRange(clang::SourceLocation L1,
                                          clang::SourceLocation L2);
size_t calculateExpansionLevel(clang::SourceLocation Loc);
/// Get the Stmt spelling
std::string getStmtSpelling(const clang::Stmt *E,
                            clang::SourceRange Parent = clang::SourceRange());
std::string getStmtSpelling(clang::SourceRange SR,
                            clang::SourceRange Parent = clang::SourceRange());

template <typename T> std::string getHashAsString(const T &Val) {
  std::stringstream Stream;
  Stream << std::hex << std::hash<T>()(Val);
  return Stream.str();
}

/// Get the fixed suffix of the tool
inline std::string getCTFixedSuffix() { return "_ct1"; }

/// Get the declaration of a statement
template <typename T> const T *getDecl(const clang::Stmt *E);

// TODO:  implement one of this for each source language.
enum SourceProcessType {
  // flag for *.cu
  SPT_CudaSource = 1,

  // flag for *.cuh
  SPT_CudaHeader = 2,

  // flag for *.cpp, *.cxx, *.cc, *.c, *.C
  SPT_CppSource = 4,

  // flag for *.hpp, *.hxx *.h
  SPT_CppHeader = 8,
};

SourceProcessType GetSourceFileType(llvm::StringRef SourcePath);

const std::string &getFmtEndStatement(void);
const std::string &getFmtStatementIndent(std::string &BaseIndent);

const std::string &getFmtEndArg(void);
const std::string &getFmtArgIndent(std::string &BaseIndent);

/// Split a string into a vector of strings with a specific delimiter
/// \param [in] Str The string to be split
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
const clang::CUDAKernelCallExpr *getParentKernelCall(const clang::Expr *E);
bool isInSameScope(const clang::Stmt *S, const clang::ValueDecl *D);
const clang::DeclRefExpr *getInnerValueDecl(const clang::Expr *Arg);
const clang::Stmt *getParentStmt(clang::DynTypedNode Node);
const clang::Stmt *getParentStmt(const clang::Stmt *S, bool SkipNonWritten = false);
const clang::Stmt *getParentStmt(const clang::Decl *D);
const clang::Decl *getParentDecl(const clang::Decl *D);
const clang::Stmt *getNonImplicitCastParentStmt(const clang::Stmt *S);
const clang::Stmt *
getNonImplicitCastNonParenExprParentStmt(const clang::Stmt *S);
const clang::DeclStmt *getAncestorDeclStmt(const clang::Expr *E);
const std::shared_ptr<clang::DynTypedNode>
getParentNode(const std::shared_ptr<clang::DynTypedNode> N);
const std::shared_ptr<clang::DynTypedNode>
getNonImplicitCastParentNode(const std::shared_ptr<clang::DynTypedNode> N);
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
std::vector<const clang::Stmt *> getConditionNode(clang::DynTypedNode Node);
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
                                FirstArg &&First, RestArgs &&...Rest) {
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
StreamTy &printArguments(StreamTy &Stream, Args &&...Arguments) {
  return printPartialArguments(Stream, sizeof...(Args),
                               std::forward<Args>(Arguments)...);
}
void printDerefOp(std::ostream &OS, const clang::Expr *E,
                  std::string *DerefType = nullptr);
bool isInSameLine(clang::SourceLocation A, clang::SourceLocation B,
                  const clang::SourceManager &SM, bool &Invalid);
clang::SourceRange getFunctionRange(const clang::CallExpr *CE);
std::vector<clang::tooling::Range>
calculateRangesWithFormatFlag(const clang::tooling::Replacements &Replaces);
std::vector<clang::tooling::Range> calculateRangesWithBlockLevelFormatFlag(
    const clang::tooling::Replacements &Replaces);
std::vector<clang::tooling::Range>
calculateUpdatedRanges(const clang::tooling::Replacements &Repls,
                       const std::vector<clang::tooling::Range> &Ranges);

bool isAssigned(const clang::Stmt *S);
bool isInRetStmt(const clang::Stmt *S);
std::string getTempNameForExpr(const clang::Expr *E, bool HandleLiteral = false,
                               bool KeepLastUnderline = true,
                               bool IsInMacroDefine = false,
                               clang::SourceLocation CallBegin = clang::SourceLocation(),
                               clang::SourceLocation CallEnd = clang::SourceLocation());
bool isOuterMostMacro(const clang::Stmt *E);
bool isSameLocation(const clang::SourceLocation L1,
                    const clang::SourceLocation L2);
bool isInsideFunctionLikeMacro(
    const clang::SourceLocation BeginLoc, const clang::SourceLocation EndLoc,
    const std::shared_ptr<clang::DynTypedNode> Parent);
bool isLocationStraddle(clang::SourceLocation Begin, clang::SourceLocation End);
bool isExprStraddle(const clang::Stmt *S);
bool isContainMacro(const clang::Expr *E);
std::string getDrefName(const clang::Expr *E);
std::vector<const clang::DeclaratorDecl *>
getAllDecls(const clang::DeclaratorDecl *DD);
std::string deducePointerType(const clang::DeclaratorDecl *DD,
                              std::string TypeName);
bool isAnIdentifierOrLiteral(const clang::Expr *E);
bool isSameSizeofTypeWithTypeStr(const clang::Expr *E,
                                 const std::string &TypeStr);
bool isInReturnStmt(const clang::Expr *E,
                    clang::SourceLocation &OuterInsertLoc);
std::string getHashStrFromLoc(clang::SourceLocation Loc);
const clang::FunctionDecl *getFunctionDecl(const clang::Stmt *S);
const clang::CXXRecordDecl *getParentRecordDecl(const clang::ValueDecl *DD);
bool IsTypeChangedToPointer(const clang::DeclRefExpr *DRE);
clang::SourceLocation
getBeginLocOfPreviousEmptyMacro(clang::SourceLocation Loc);
unsigned int getEndLocOfFollowingEmptyMacro(clang::SourceLocation Loc);

std::string getNestedNameSpecifierString(const clang::NestedNameSpecifier *);
std::string getNestedNameSpecifierString(const clang::NestedNameSpecifierLoc &);

bool needExtraParens(const clang::Expr *);
std::pair<clang::SourceLocation, clang::SourceLocation>
getTheOneBeforeLastImmediateExapansion(const clang::SourceLocation Begin,
                                       const clang::SourceLocation End);
std::pair<clang::SourceLocation, clang::SourceLocation>
getTheLastCompleteImmediateRange(clang::SourceLocation BeginLoc,
                                 clang::SourceLocation EndLoc);
bool isInRange(clang::SourceLocation PB, clang::SourceLocation PE,
               clang::SourceLocation Loc);
bool isInRange(clang::SourceLocation PB, clang::SourceLocation PE,
               clang::StringRef FilePath, size_t Offset);
clang::SourceLocation getLocInRange(clang::SourceLocation Loc,
                                    clang::SourceRange Range);
std::pair<clang::SourceLocation, clang::SourceLocation>
getRangeInRange(const clang::Stmt *E, clang::SourceLocation RangeBegin,
                clang::SourceLocation RangeEnd, bool IncludeLastToken = true);
std::pair<clang::SourceLocation, clang::SourceLocation>
getRangeInRange(clang::SourceRange Range, clang::SourceLocation RangeBegin,
                clang::SourceLocation RangeEnd, bool IncludeLastToken = true);
unsigned int calculateIndentWidth(const clang::CUDAKernelCallExpr *Node,
                                  clang::SourceLocation SL, bool &Flag);
bool isIncludedFile(const std::string &CurrentFile,
                    const std::string &CheckingFile);
clang::SourceRange getRangeInsideFuncLikeMacro(const clang::Stmt *S);
std::string getCombinedStrFromLoc(const clang::SourceLocation Loc);

/// For types like curandState, the template argument of the migrated type
/// cannot be decided at this time. It is known after AST traversal. So here we
/// need use placeholder and replace the placeholder in
/// ExtReplacements::emplaceIntoReplSet
std::string getFinalCastTypeNameStr(std::string CastTypeName);
bool isLexicallyInLocalScope(const clang::Decl *);
const clang::DeclaratorDecl *getHandleVar(const clang::Expr *Arg);
bool checkPointerInStructRecursively(const clang::DeclRefExpr *DRE);
bool checkPointerInStructRecursively(const clang::RecordDecl *);
clang::RecordDecl *getRecordDecl(clang::QualType QT);
clang::SourceLocation
getImmSpellingLocRecursive(const clang::SourceLocation Loc);
clang::dpct::FFTTypeEnum getFFTTypeFromValue(std::int64_t Value);
std::string getPrecAndDomainStrFromValue(std::int64_t Value);
std::string getPrecAndDomainStrFromExecFuncName(std::string ExecFuncName);
bool getTypeRange(const clang::VarDecl *PVD, clang::SourceRange &SR);
llvm::StringRef getCalleeName(const clang::CallExpr *CE);
clang::SourceRange getDefinitionRange(clang::SourceLocation Begin,
                                      clang::SourceLocation End);
bool isLocInSameMacroArg(clang::SourceLocation Begin,
                         clang::SourceLocation End);
const clang::CompoundStmt *
findTheOuterMostCompoundStmtUntilMeetControlFlowNodes(
    const clang::CallExpr *CE);
bool isInMacroDefinition(clang::SourceLocation BeginLoc,
                         clang::SourceLocation EndLoc);
bool isPartOfMacroDef(clang::SourceLocation BeginLoc,
                      clang::SourceLocation EndLoc);
void constructUnionFindSetRecursively(
    std::shared_ptr<clang::dpct::DeviceFunctionInfo> DFIPtr);
// Determine if S is a statement inside
// a if/while/do while/for statement.
template <typename NodeTy>
bool isInCtrlFlowStmt(const clang::Stmt *S, const NodeTy *Root,
                      clang::ASTContext &Context) {
  auto ParentStmt = getParentStmt(S);
  if (!ParentStmt)
    return false;

  auto Parents = Context.getParents(*S);

  if (Parents.size() < 1)
    return false;
  const NodeTy *Parent = Parents[0].get<NodeTy>();
  auto ParentStmtClass = ParentStmt->getStmtClass();
  bool Ret = ParentStmtClass == clang::Stmt::StmtClass::IfStmtClass ||
             ParentStmtClass == clang::Stmt::StmtClass::WhileStmtClass ||
             ParentStmtClass == clang::Stmt::StmtClass::DoStmtClass ||
             ParentStmtClass == clang::Stmt::StmtClass::ForStmtClass;
  if (Ret)
    return true;
  else if (Parent != Root)
    return isInCtrlFlowStmt(ParentStmt, Root, Context);
  else
    return false;
}
void getShareAttrRecursive(const clang::Expr *Expr, bool &HasSharedAttr,
                           bool &NeedReport);
enum class LocalVarAddrSpaceEnum { AS_CannotDeduce, AS_IsPrivate, AS_IsGlobal };
void checkIsPrivateVar(const clang::Expr *Expr, LocalVarAddrSpaceEnum &Result);
bool isModifiedRef(const clang::DeclRefExpr *DRE);
bool isDefaultStream(const clang::Expr *StreamArg);
const clang::NamedDecl *getNamedDecl(const clang::Type *TypePtr);
bool isTypeInAnalysisScope(const clang::Type *TypePtr);
void findAssignments(const clang::DeclaratorDecl *HandleDecl,
                     const clang::CompoundStmt *CS,
                     std::vector<const clang::DeclRefExpr *> &Refs);
llvm::SmallVector<clang::ast_matchers::BoundNodes, 1U>
findDREInScope(const clang::Stmt *Scope,
               const std::vector<std::string> &IgnoreTypes = {});
void findDREs(const Expr *E, std::set<const clang::DeclRefExpr *> &DRESet,
              bool &HasCallExpr,
              const std::vector<std::string> &IgnoreTypes = {});

enum class MemcpyOrderAnalysisNodeKind {
  MOANK_Memcpy = 0,
  MOANK_MemcpyInFlowControl,
  MOANK_OtherCallExpr,
  MOANK_KernelCallExpr,
  MOANK_SpecialCallExpr
};
bool canOmitMemcpyWait(const clang::CallExpr *CE);
bool checkIfContainSizeofTypeRecursively(
    const clang::Expr *E, const clang::Expr *&ExprContainSizeofType);
bool containSizeOfType(const clang::Expr *E);
bool maybeDependentCubType(const clang::TypeSourceInfo *TInfo);
bool isCubVar(const clang::VarDecl *VD);
void findAllVarRef(const clang::DeclRefExpr *DRE,
                   std::vector<const clang::DeclRefExpr *> &RefMatchResult,
                   bool IsGlobalScopeAllowed = false);
bool isExprUsed(const clang::Expr *E, bool &Result);
const std::string &getItemName();
bool isUserDefinedDecl(const clang::Decl *D);
void insertHeaderForTypeRule(std::string, clang::SourceLocation);
std::string getRemovedAPIWarningMessage(std::string FuncName);
std::string getBaseTypeStr(const clang::CallExpr *CE);
std::string getParamTypeStr(const clang::CallExpr *CE, unsigned int Idx);
std::string getArgTypeStr(const clang::CallExpr *CE, unsigned int Idx);
std::string getFunctionName(const clang::FunctionDecl *Node);
std::string getFunctionName(const clang::UnresolvedLookupExpr *Node);
std::string getFunctionName(const clang::FunctionTemplateDecl *Node);
bool isLambda(const clang::FunctionDecl *FD);
const clang::LambdaExpr *
getImmediateOuterLambdaExpr(const clang::FunctionDecl *FuncDecl);
bool typeIsPostfix(clang::QualType QT);
bool isPointerHostAccessOnly(const clang::ValueDecl* VD);
std::string getBaseTypeRemoveTemplateArguments(const clang::MemberExpr* ME);
bool containIterationSpaceBuiltinVar(const clang::Stmt *Node);
bool containBuiltinWarpSize(const clang::Stmt *Node);
bool isCapturedByLambda(const clang::TypeLoc *TL);
std::string getAddressSpace(const clang::CallExpr *C, int ArgIdx);
std::string getNameSpace(const NamespaceDecl *NSD);
bool isFromCUDA(const Decl *D);
namespace clang {
namespace dpct {
std::string getDpctVersionStr();
enum class HelperFeatureEnum : unsigned int {
  device_ext,
  none,
};
void requestFeature(HelperFeatureEnum Feature);
void requestHelperFeatureForEnumNames(const std::string Name);
void requestHelperFeatureForTypeNames(const std::string Name);

class PairedPrinter {
  StringRef Postfix;
  llvm::raw_ostream &OS;

public:
  PairedPrinter(llvm::raw_ostream &Stream, StringRef PrefixString,
                StringRef PosfixString, bool ShouldPrint = true)
      : OS(Stream) {
    if (ShouldPrint) {
      OS << PrefixString;
      Postfix = PosfixString;
    }
  }
  ~PairedPrinter() { OS << Postfix; }
};

std::error_code real_path(const Twine &path, SmallVectorImpl<char> &output,
                          bool expand_tilde = false);
} // namespace dpct
namespace ast_matchers {
AST_MATCHER_P(DeclRefExpr, isDeclSameAs, const VarDecl *, TargetVD) {
  const DeclRefExpr *DRE = &Node;
  return DRE->getDecl() == TargetVD;
}
} // namespace ast_matchers
} // namespace clang
#endif // DPCT_UTILITY_H
