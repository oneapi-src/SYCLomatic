//===--------------- Utility.h --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DPCT_UTILITY_H
#define DPCT_UTILITY_H

#include <fstream>
#include <functional>
#include <ios>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <sstream>
#include <stack>
#include <string>
#include <system_error>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "ErrorHandle/Error.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/ParentMapContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
namespace path = llvm::sys::path;

using namespace clang;
// ------------Utility: Path ------------------------------------------------//
extern std::unordered_map<std::string, bool> IsDirectoryCache;
extern std::unordered_map<std::string, bool> ChildPathCache;
extern std::unordered_map<std::string, bool> ChildOrSameCache;

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

namespace tooling {
class Range;
class Replacements;
} // namespace tooling
} // namespace clang


namespace clang {
namespace dpct {

class DeviceFunctionInfo;
class MigrationRule;

enum class MemcpyOrderAnalysisNodeKind {
  MOANK_Memcpy = 0,
  MOANK_MemcpyInFlowControl,
  MOANK_OtherCallExpr,
  MOANK_KernelCallExpr,
  MOANK_SpecialCallExpr
};
// ------------Utility: source type------------------------------------------//
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
SourceProcessType GetSourceFileType(const clang::tooling::UnifiedPath &SourcePath);

// ------------Utility: printer----------------------------------------------//
template <class StreamT> class ParensPrinter {
  StreamT &OS;
public:
  ParensPrinter(StreamT &OS) : OS(OS) { OS << "("; }
  ~ParensPrinter() { OS << ")"; }
};

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
void PrintFullTemplateName(raw_ostream &OS, const PrintingPolicy &Policy, TemplateName Name);
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

// ------------Utility: string processing------------------------------------//
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

bool containOnlyDigits(const std::string &str);
void replaceSubStr(std::string &Str, const std::string &SubStr,
                   const std::string &Repl);
void replaceSubStrAll(std::string &Str, const std::string &SubStr,
                      const std::string &Repl);

/// Check \param FilePath is whether a directory path
/// \param [in] FilePath is a file path.
/// \return true: directory path, false: not directory path.
inline bool isDirectory(clang::tooling::UnifiedPath FilePath) {
  std::string CanonicalFilePath = !FilePath.getCanonicalPath().empty()
                                      ? FilePath.getCanonicalPath().str()
                                      : FilePath.getPath().str();
  auto Iter = IsDirectoryCache.find(CanonicalFilePath);
  if (Iter != IsDirectoryCache.end()) {
    return Iter->second;
  } else {
    auto Ret = llvm::sys::fs::is_directory(CanonicalFilePath);
    IsDirectoryCache[CanonicalFilePath] = Ret;
    return Ret;
  }
}

// ------------Utility: path processing--------------------------------------//
/// Check \param Child is whether the child path of \param Root
/// \param [in] RootAbs A path as reference.
/// \param [in] Child A path to be checked.
/// \return true: child path, false: not child path
/// /x/y and /x/y/z -> true
/// /x/y and /x/y   -> false
/// /x/y and /x/yy/ -> false (not a prefix in terms of a path)
inline bool isChildPath(const clang::tooling::UnifiedPath &Root,
                        const clang::tooling::UnifiedPath &Child) {
  if (Child.getPath().empty()) {
    return false;
  }

  std::string RootCanonicalPath = !Root.getCanonicalPath().empty()
                                      ? Root.getCanonicalPath().str()
                                      : Root.getPath().str();
  std::string ChildCanonicalPath = !Child.getCanonicalPath().empty()
                                       ? Child.getCanonicalPath().str()
                                       : Child.getPath().str();
  auto Key = RootCanonicalPath + ":" + ChildCanonicalPath;
  auto Iter = ChildPathCache.find(Key);
  if (Iter != ChildPathCache.end()) {
    return Iter->second;
  }

  auto Diff = std::mismatch(path::begin(RootCanonicalPath),
                            path::end(RootCanonicalPath),
                            path::begin(ChildCanonicalPath));
  // RootCanonicalPath is not considered prefix of ChildCanonicalPath if they
  // are equal.
  bool Ret = Diff.first == path::end(RootCanonicalPath) &&
             Diff.second != path::end(ChildCanonicalPath);
  ChildPathCache[Key] = Ret;
  return Ret;
}

/// Check \param Child is whether the child or same path of \param Root
/// \param [in] Root A path as reference.
/// \param [in] Child A path to be checked.
/// \return true: child path, false: not child path
inline bool isChildOrSamePath(clang::tooling::UnifiedPath Root,
                              clang::tooling::UnifiedPath Child) {
  if (Child.getPath().empty()) {
    return false;
  }

  std::string RootCanonicalPath = !Root.getCanonicalPath().empty()
                                      ? Root.getCanonicalPath().str()
                                      : Root.getPath().str();
  std::string ChildCanonicalPath = !Child.getCanonicalPath().empty()
                                       ? Child.getCanonicalPath().str()
                                       : Child.getPath().str();
  auto Key = RootCanonicalPath + ":" + ChildCanonicalPath;
  auto Iter = ChildOrSameCache.find(Key);
  if (Iter != ChildOrSameCache.end()) {
    return Iter->second;
  }

  auto Diff = std::mismatch(path::begin(RootCanonicalPath),
                            path::end(RootCanonicalPath),
                            path::begin(ChildCanonicalPath));
  bool Ret = Diff.first == path::end(RootCanonicalPath);
  ChildOrSameCache[Key] = Ret;
  return Ret;
}
std::string getCanonicalPath(clang::SourceLocation Loc);
std::string appendPath(const std::string &P1, const std::string &P2);

// ------------Utility: Format ----------------------------------------------//
/// Returns the character sequence indenting the source line containing Loc.
llvm::StringRef getIndent(clang::SourceLocation Loc,
                          const clang::SourceManager &SM);
unsigned int calculateIndentWidth(const clang::CUDAKernelCallExpr *Node,
                                  clang::SourceLocation SL, bool &Flag);
int getLengthOfSpacesToEndl(const char *CharData);
std::vector<clang::tooling::Range>
calculateRangesWithFormatFlag(const clang::tooling::Replacements &Replaces);
std::vector<clang::tooling::Range> calculateRangesWithBlockLevelFormatFlag(
    const clang::tooling::Replacements &Replaces);

// ------------Utility: Macro process----------------------------------------//
clang::SourceRange getStmtExpansionSourceRange(const clang::Stmt *S);
size_t calculateExpansionLevel(clang::SourceLocation Loc);
std::string getStmtSpelling(const clang::Stmt *E,
                            clang::SourceRange Parent = clang::SourceRange());
std::string getStmtSpelling(clang::SourceRange SR,
                            clang::SourceRange Parent = clang::SourceRange());
clang::SourceLocation
getBeginLocOfPreviousEmptyMacro(clang::SourceLocation Loc);
unsigned int getEndLocOfFollowingEmptyMacro(clang::SourceLocation Loc);
clang::SourceRange getRangeInsideFuncLikeMacro(const clang::Stmt *S);
clang::SourceRange getDefinitionRange(clang::SourceLocation Begin,
                                      clang::SourceLocation End);
std::pair<clang::SourceLocation, clang::SourceLocation>
getTheOneBeforeLastImmediateExapansion(const clang::SourceLocation Begin,
                                       const clang::SourceLocation End);
std::pair<clang::SourceLocation, clang::SourceLocation>
getTheLastCompleteImmediateRange(clang::SourceLocation BeginLoc,
                                 clang::SourceLocation EndLoc);
clang::SourceLocation getLocInRange(clang::SourceLocation Loc,
                                    clang::SourceRange Range);
std::pair<clang::SourceLocation, clang::SourceLocation>
getRangeInRange(const clang::Stmt *E, clang::SourceLocation RangeBegin,
                clang::SourceLocation RangeEnd, bool IncludeLastToken = true);
std::pair<clang::SourceLocation, clang::SourceLocation>
getRangeInRange(clang::SourceRange Range, clang::SourceLocation RangeBegin,
                clang::SourceLocation RangeEnd, bool IncludeLastToken = true);
std::string getStringInRange(clang::SourceRange Range,
                             clang::SourceLocation RangeBegin,
                             clang::SourceLocation RangeEnd,
                             bool IncludeLastToken = true);
clang::SourceLocation
getImmSpellingLocRecursive(const clang::SourceLocation Loc);

//checkers
bool isInMacroDefinition(clang::SourceLocation BeginLoc,
                         clang::SourceLocation EndLoc);
bool isPartOfMacroDef(clang::SourceLocation BeginLoc,
                      clang::SourceLocation EndLoc);
bool isOuterMostMacro(const clang::Stmt *E);
bool isInsideFunctionLikeMacro(
    const clang::SourceLocation BeginLoc, const clang::SourceLocation EndLoc,
    const std::shared_ptr<clang::DynTypedNode> Parent);
bool isContainMacro(const clang::Expr *E);
bool isLocInSameMacroArg(clang::SourceLocation Begin,
                         clang::SourceLocation End);
bool isLocationStraddle(clang::SourceLocation Begin, clang::SourceLocation End);
bool isExprStraddle(const clang::Stmt *S);

// ------------Utility:  AST Traversal---------------------------------------//
const clang::CompoundStmt *findImmediateBlock(const clang::Stmt *S);
const clang::CompoundStmt *findImmediateBlock(const clang::ValueDecl *D);
const clang::FunctionDecl *getImmediateOuterFuncDecl(const clang::Stmt *S);
const clang::CUDAKernelCallExpr *getParentKernelCall(const clang::Expr *E);
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
const clang::DynTypedNode
findNearestNonExprNonDeclAncestorNode(const clang::DynTypedNode &N);
template <typename T>
const clang::Stmt *findNearestNonExprNonDeclAncestorStmt(const T *SD) {
  return findNearestNonExprNonDeclAncestorNode(clang::DynTypedNode::create(*SD))
      .template get<clang::Stmt>();
}
std::vector<const clang::Stmt *> getConditionNode(clang::DynTypedNode Node);
std::vector<const clang::Stmt *> getConditionExpr(clang::DynTypedNode Node);
void getVarReferencedInFD(const clang::Stmt *S, const clang::ValueDecl *VD,
                       std::vector<const clang::DeclRefExpr *> &Result);
std::string getDrefName(const clang::Expr *E);
std::vector<const clang::DeclaratorDecl *>
getAllDecls(const clang::DeclaratorDecl *DD);
const clang::FunctionDecl *getFunctionDecl(const clang::Stmt *S);
const clang::CXXRecordDecl *getParentRecordDecl(const clang::ValueDecl *DD);
const clang::DeclaratorDecl *getHandleVar(const clang::Expr *Arg);
clang::RecordDecl *getRecordDecl(clang::QualType QT);
llvm::StringRef getCalleeName(const clang::CallExpr *CE);
const clang::CompoundStmt *
findTheOuterMostCompoundStmtUntilMeetControlFlowNodes(
    const clang::CallExpr *CE);
const clang::NamedDecl *getNamedDecl(const clang::Type *TypePtr);
const clang::LambdaExpr *
getImmediateOuterLambdaExpr(const clang::FunctionDecl *FuncDecl);
const Expr *getAddressedRef(const Expr *E, bool IsCheckFunctionDecl = true,
                            const FunctionDecl **FuncDecl = nullptr);
const clang::FunctionDecl *findTheOuterMostFunctionDecl(const clang::Decl *D);

// Source Range & location, offset.
clang::SourceRange getScopeInsertRange(const clang::MemberExpr *ME);
clang::SourceRange
getScopeInsertRange(const clang::Expr *CE,
                    const clang::SourceLocation &FuncNameBegin,
                    const clang::SourceLocation &FuncCallEnd);
clang::SourceRange getFunctionRange(const clang::CallExpr *CE);
unsigned int getLenIncludingTrailingSpaces(clang::SourceRange Range,
                                           const clang::SourceManager &SM);
unsigned int getOffsetOfLineBegin(clang::SourceLocation Loc,
                                  const clang::SourceManager &SM);
int getCurrnetColumn(clang::SourceLocation Loc, const clang::SourceManager &SM);

// match AST
std::set<const clang::DeclRefExpr *>
matchTargetDREInScope(const clang::VarDecl *TargetDecl,
                      const clang::Stmt *Range);
void findDREs(const Expr *E, std::set<const clang::DeclRefExpr *> &DRESet,
              bool &HasCallExpr,
              const std::vector<std::string> &IgnoreTypes = {});
void findAllVarRef(const clang::DeclRefExpr *DRE,
                   std::vector<const clang::DeclRefExpr *> &RefMatchResult,
                   bool IsGlobalScopeAllowed = false);
llvm::SmallVector<clang::ast_matchers::BoundNodes, 1U>
findDREInScope(const clang::Stmt *Scope,
               const std::vector<std::string> &IgnoreTypes = {});
void findAssignments(const clang::DeclaratorDecl *HandleDecl,
                     const clang::CompoundStmt *CS,
                     std::vector<const clang::DeclRefExpr *> &Refs);

//  Gen Migration code
std::string getBufferNameAndDeclStr(const std::string &PointerName,
                                    const std::string &TypeAsStr,
                                    const std::string &IndentStr,
                                    std::string &BufferDecl);
std::string getBufferNameAndDeclStr(const clang::Expr *Arg,
                                    const std::string &TypeAsStr,
                                    const std::string &IndentStr,
                                    std::string &BufferDecl);
std::string getTempNameForExpr(const clang::Expr *E, bool HandleLiteral = false,
                               bool KeepLastUnderline = true,
                               bool IsInMacroDefine = false,
                               clang::SourceLocation CallBegin = clang::SourceLocation(),
                               clang::SourceLocation CallEnd = clang::SourceLocation());
std::string getNestedNameSpecifierString(const clang::NestedNameSpecifier *);
std::string getNestedNameSpecifierString(const clang::NestedNameSpecifierLoc &);
std::string getCombinedStrFromLoc(const clang::SourceLocation Loc);
std::string getRemovedAPIWarningMessage(std::string FuncName);
std::string getBaseTypeStr(const clang::CallExpr *CE);
std::string getParamTypeStr(const clang::CallExpr *CE, unsigned int Idx);
std::string getArgTypeStr(const clang::CallExpr *CE, unsigned int Idx);
std::string getFunctionName(const clang::FunctionDecl *Node);
std::string getFunctionName(const clang::UnresolvedLookupExpr *Node);
std::string getFunctionName(const clang::FunctionTemplateDecl *Node);
/// For types like curandState, the template argument of the migrated type
/// cannot be decided at this time. It is known after AST traversal. So here we
/// need use placeholder and replace the placeholder in
/// ExtReplacements::emplaceIntoReplSet
std::string getFinalCastTypeNameStr(std::string CastTypeName);
std::string getRecordTypeStr(const CXXRecordDecl *RD);
std::string getBaseTypeRemoveTemplateArguments(const clang::MemberExpr* ME);
std::string getNameSpace(const NamespaceDecl *NSD);
void getNameSpace(const NamespaceDecl *NSD,
                  std::vector<std::string> &Namespaces);
std::string getTemplateArgumentAsString(const clang::TemplateArgument &Arg,
                                        const clang::ASTContext &Ctx);
// utils
bool needExtraParens(const clang::Expr *);
bool needExtraParensInMemberExpr(const clang::Expr *);
bool getTypeRange(const clang::VarDecl *PVD, clang::SourceRange &SR);

bool checkPointerInStructRecursively(const clang::DeclRefExpr *DRE);
bool checkPointerInStructRecursively(const clang::RecordDecl *);
void constructUnionFindSetRecursively(
    std::shared_ptr<clang::dpct::DeviceFunctionInfo> DFIPtr);
void getShareAttrRecursive(const clang::Expr *Expr, bool &HasSharedAttr,
                           bool &NeedReport);
enum class LocalVarAddrSpaceEnum { AS_CannotDeduce, AS_IsPrivate, AS_IsGlobal };
void checkIsPrivateVar(const clang::Expr *Expr, LocalVarAddrSpaceEnum &Result);

// Checker:
bool isInSameScope(const clang::Stmt *S, const clang::ValueDecl *D);
bool IsSingleLineStatement(const clang::Stmt *S);
bool isArgUsedAsLvalueUntil(const clang::DeclRefExpr *Arg,
                            const clang::Stmt *S);
bool isConditionOfFlowControl(const clang::CallExpr *CE,
                              std::string &OriginStmtType,
                              bool &CanAvoidUsingLambda,
                              clang::SourceLocation &SL);
bool isConditionOfFlowControl(const clang::Expr *E,
                              bool OnlyCheckConditionExpr = false);
bool isInSameLine(clang::SourceLocation A, clang::SourceLocation B,
                  const clang::SourceManager &SM, bool &Invalid);
bool isSameLocation(const clang::SourceLocation L1,
                    const clang::SourceLocation L2);
bool isAssigned(const clang::Stmt *S);
bool isInRetStmt(const clang::Stmt *S);
bool isAnIdentifierOrLiteral(const clang::Expr *E);
bool isSameSizeofTypeWithTypeStr(const clang::Expr *E,
                                 const std::string &TypeStr);
bool isInReturnStmt(const clang::Expr *E,
                    clang::SourceLocation &OuterInsertLoc);
bool IsTypeChangedToPointer(const clang::DeclRefExpr *DRE);

bool isInRange(clang::SourceLocation PB, clang::SourceLocation PE,
               clang::SourceLocation Loc);
bool isInRange(clang::SourceLocation PB, clang::SourceLocation PE,
               clang::StringRef FilePath, size_t Offset);
bool isIncludedFile(const clang::tooling::UnifiedPath &CurrentFile,
                    const clang::tooling::UnifiedPath &CheckingFile);
bool isLexicallyInLocalScope(const clang::Decl *);
/// Determine if S is a statement inside a if/while/do while/for statement.
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
bool isModifiedRef(const clang::DeclRefExpr *DRE);
bool isDefaultStream(const clang::Expr *StreamArg);
bool isRedeclInCUDAHeader(const clang::TypedefType *T);
bool isTypeInAnalysisScope(const clang::Type *TypePtr);
bool isCubVar(const clang::VarDecl *VD);
bool isCubTempStorageType(QualType T);
bool isCubCollectiveRecordType(QualType T);
bool isExprUsed(const clang::Expr *E, bool &Result);
bool isUserDefinedDecl(const clang::Decl *D);
bool isLambda(const clang::FunctionDecl *FD);
bool isPointerHostAccessOnly(const clang::ValueDecl* VD);
bool isCapturedByLambda(const clang::TypeLoc *TL);
bool isTypePostfix(clang::QualType QT);
bool isFromCUDA(const Decl *D);
bool containIterationSpaceBuiltinVar(const clang::Stmt *Node);
bool containBuiltinWarpSize(const clang::Stmt *Node);
/// @brief Check if an argument is initialized.
/// @param Arg Function call argument (may be an expression).
/// @param DeclsRequireInit UnInitialized VarDecl(s).
/// @return  1: Initialized
///          0: Not initialized.
///         -1: Cannot deduce.
int isArgumentInitialized(
    const clang::Expr *Arg,
    std::vector<const clang::VarDecl *> &DeclsRequireInit);
bool isDeviceCopyable(QualType Type, clang::dpct::MigrationRule *Rule);
bool canOmitMemcpyWait(const clang::CallExpr *CE);
bool checkIfContainSizeofTypeRecursively(
    const clang::Expr *E, const clang::Expr *&ExprContainSizeofType);
bool containSizeOfType(const clang::Expr *E);

// ------------Utility:  Misc------------------------------------------------//
template <typename T> std::string getHashAsString(const T &Val) {
  std::stringstream Stream;
  Stream << std::hex << std::hash<T>()(Val);
  return Stream.str();
}
std::string getHashStrFromLoc(clang::SourceLocation Loc);
std::string getStrFromLoc(clang::SourceLocation Loc);

void insertHeaderForTypeRule(std::string, clang::SourceLocation);

/// Returns character sequence ("\n") on Linux, return ("\r\n") on Windows.
const char *getNL(bool AddBackSlash = false);
/// Returns the character sequence ("\n" or "\r\n") used to represent new line
/// in the source line containing Loc.
const char *getNL(clang::SourceLocation Loc, const clang::SourceManager &SM);

/// Get the fixed suffix of the tool
inline std::string getCTFixedSuffix() { return "_ct1"; }
const std::string &getItemName();

// ------------Utility:  version---------------------------------------------//
std::string getDpctVersionStr();
std::string GetPython();

// ------------Utility:  Helper function-------------------------------------//
enum class HelperFeatureEnum : unsigned int {
  device_ext,
  none,
};
void requestFeature(HelperFeatureEnum Feature);
void requestHelperFeatureForEnumNames(const std::string Name);
void requestHelperFeatureForTypeNames(const std::string Name);


// ------------Utility:  file function---------------------------------------//
void writeDataToFile(const std::string &FileName, const std::string &Data);
void appendDataToFile(const std::string &FileName, const std::string &Data);
void createDirectories(const clang::tooling::UnifiedPath &FilePath,
                       bool IgnoreExisting = true);
void PrintMsg(const std::string &Msg, bool IsPrintOnNormal = true);
class RawFDOStream : public llvm::raw_fd_ostream {
  StringRef FileName;
  std::error_code EC;

public:
  RawFDOStream(StringRef FileName);
  RawFDOStream(StringRef FileName, llvm::sys::fs::OpenFlags OF);

  template <class T> RawFDOStream &operator<<(T &&var) {
    llvm::raw_fd_ostream::operator<<(std::forward<T>(var));
    if ((bool)this->error()) {
      std::string ErrMsg =
          "[ERROR] Write data to " + FileName.str() + " Fail!\n";
      dpct::PrintMsg(ErrMsg);
      dpctExit(MigrationErrorCannotWrite);
    }
    return *this;
  }
  ~RawFDOStream();
};

} // namespace dpct

// ------------Utility:  AST Matcher-----------------------------------------//
namespace ast_matchers {
AST_MATCHER_P(DeclRefExpr, isDeclSameAs, const VarDecl *, TargetVD) {
  const DeclRefExpr *DRE = &Node;
  return DRE->getDecl() == TargetVD;
}

AST_MATCHER_P(QualType, typeContainsString, std::string, substring) {
  return Node.getAsString().find(substring) != std::string::npos;
}

} // namespace ast_matchers
} // namespace clang
#endif // DPCT_UTILITY_H
