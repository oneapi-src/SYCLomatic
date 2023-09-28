//===--------------- Utility.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Utility.h"
#include "ASTTraversal.h"
#include "AnalysisInfo.h"
#include "Config.h"
#include "DNNAPIMigration.h"
#include "ExprAnalysis.h"
#include "MapNames.h"
#include "SaveNewFiles.h"
#include "Statics.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTTypeTraits.h"
#include "clang/AST/ExprCXX.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"
#include "clang/Tooling/Core/Replacement.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include <algorithm>
#include <fstream>

using namespace llvm;
using namespace clang;
using namespace std;

namespace path = llvm::sys::path;
namespace fs = llvm::sys::fs;

extern std::string DpctInstallPath; // Installation directory for this tool
bool IsUsingDefaultOutRoot = false;

void removeDefaultOutRootFolder(const std::string &DefaultOutRoot) {
  if (isDirectory(DefaultOutRoot)) {
    std::error_code EC;
    llvm::sys::fs::directory_iterator Iter(DefaultOutRoot, EC);
    if ((bool)EC)
      return;
    llvm::sys::fs::directory_iterator End;
    if (Iter == End) {
      // This folder is empty, then remove it.
      llvm::sys::fs::remove_directories(DefaultOutRoot, false);
    }
  }
}

void dpctExit(int ExitCode, bool NeedCleanUp) {
  if (IsUsingDefaultOutRoot && NeedCleanUp) {
    removeDefaultOutRootFolder(dpct::DpctGlobalInfo::getOutRoot());
  }
  std::exit(ExitCode);
}

bool makeCanonical(SmallVectorImpl<char> &Path) {
  if (fs::make_absolute(Path) != std::error_code()) {
    llvm::errs() << "Could not get absolute path from '" << Path << "'\n ";
    return false;
  }
  path::native(Path);
  path::remove_dots(Path, /* remove_dot_dot= */ true);
  return true;
}

bool makeCanonical(string &PathPar) {
  SmallString<256> Path = StringRef(PathPar);
  if (!makeCanonical(Path))
    return false;
  PathPar.assign(begin(Path), end(Path));
  return true;
}

bool isCanonical(StringRef Path) {
  bool HasNoDots = all_of(path::begin(Path), path::end(Path),
                          [](StringRef e) { return e != "." && e != ".."; });
  return HasNoDots && path::is_absolute(Path);
}

const char *getNL(void) {
#if defined(__linux__)
  return "\n";
#elif defined(_WIN64)
  return "\r\n";
#else
#error Only support windows and Linux.
#endif
}

const char *getNL(SourceLocation Loc, const SourceManager &SM) {
  auto LocInfo = SM.getDecomposedLoc(Loc);
  auto Buffer = SM.getBufferData(LocInfo.first);
  Buffer = Buffer.data() + LocInfo.second;
  // Search for both to avoid searching till end of file.
  auto pos = Buffer.find_first_of("\r\n");
  if (pos == StringRef::npos || Buffer[pos] == '\n')
    return "\n";
  else
    return "\r\n";
}

// index base is zero. (The offset of the file begin is 0.)
unsigned int getOffsetOfLineBegin(SourceLocation Loc, const SourceManager &SM) {
  auto LocInfo = SM.getDecomposedLoc(Loc);
  auto Buffer = SM.getBufferData(LocInfo.first);
  // Find last line end.
  auto Begin = Buffer.find_last_of('\n', LocInfo.second);
  if (Begin == StringRef::npos) {
    // We're at the beginning of the file.
    Begin = 0;
  }
  return Begin + 1;
}

// index base is one. (The column of the line begin is 1.)
int getCurrnetColumn(SourceLocation Loc, const SourceManager &SM) {
  auto LocInfo = SM.getDecomposedLoc(Loc);
  return LocInfo.second - getOffsetOfLineBegin(Loc, SM) + 1;
}

StringRef getIndent(SourceLocation Loc, const SourceManager &SM) {
  auto LocInfo = SM.getDecomposedLoc(Loc);
  auto Buffer = SM.getBufferData(LocInfo.first);
  // Find start of indentation.
  auto begin = Buffer.find_last_of('\n', LocInfo.second);
  if (begin != StringRef::npos) {
    ++begin;
  } else {
    // We're at the beginning of the file.
    begin = 0;
  }
  auto end = Buffer.find_if([](char c) { return !isspace(c); }, begin);
  return Buffer.substr(begin, end - begin);
}

SourceRange getRangeInsideFuncLikeMacro(const Stmt *S) {
  auto &SM = dpct::DpctGlobalInfo::getSourceManager();
  SourceLocation BeginLoc, EndLoc;
  BeginLoc = S->getBeginLoc();
  EndLoc = S->getEndLoc();

  if (S->getBeginLoc().isMacroID() && !isOuterMostMacro(S)) {
    BeginLoc = SM.getImmediateSpellingLoc(S->getBeginLoc());
    EndLoc = SM.getImmediateSpellingLoc(S->getEndLoc());
    // If the call expr is in a function-like macro or nested macros, to get the
    // correct loc of the previous arg, we need to use getImmediateSpellingLoc
    // step by step until reaching a FileID or a non-function-like macro. E.g.
    // MACRO_A(MACRO_B(callexpr(arg1, arg2, arg3)));
    // When we try to remove arg3, Begin should be at the end of arg2.
    // However, the expansionLoc of Begin is at the beginning of MACRO_A.
    // After 1st time of Begin=SM.getImmediateSpellingLoc(Begin),
    // Begin is at the beginning of MACRO_B.
    // After 2nd time of Begin=SM.getImmediateSpellingLoc(Begin),
    // Begin is at the beginning of arg2.
    // CANNOT use SM.getSpellingLoc because arg2 might be a simple macro,
    // and SM.getSpellingLoc will return the macro definition in this case.
    while (BeginLoc.isMacroID() &&
           !SM.isAtStartOfImmediateMacroExpansion(BeginLoc)) {
      auto ISL = SM.getImmediateSpellingLoc(BeginLoc);
      if (!dpct::DpctGlobalInfo::isInAnalysisScope(
              SM.getFilename(SM.getExpansionLoc(ISL)).str()))
        break;
      BeginLoc = SM.getImmediateSpellingLoc(BeginLoc);
      EndLoc = SM.getImmediateSpellingLoc(EndLoc);
    }
  }
  return SourceRange(BeginLoc, EndLoc);
}

SourceRange getStmtExpansionSourceRange(const Stmt *S) {
  auto &SM = dpct::DpctGlobalInfo::getSourceManager();
  SourceLocation BeginLoc, EndLoc;
  auto Range = getRangeInsideFuncLikeMacro(S);
  if (Range.getBegin().isMacroID() && Range.getEnd().isMacroID() &&
      isInRange(SM.getExpansionRange(Range.getBegin()).getBegin(),
                SM.getExpansionRange(Range.getBegin()).getEnd(),
                SM.getSpellingLoc(Range.getBegin())) &&
      isInRange(SM.getExpansionRange(Range.getBegin()).getBegin(),
                SM.getExpansionRange(Range.getBegin()).getEnd(),
                SM.getSpellingLoc(Range.getEnd()))) {
    // MACRO(callExpr())
    BeginLoc = SM.getSpellingLoc(Range.getBegin());
    EndLoc = SM.getSpellingLoc(Range.getEnd());
  } else {
    BeginLoc = SM.getExpansionRange(Range.getBegin()).getBegin();
    EndLoc = SM.getExpansionRange(Range.getEnd()).getEnd();
  }
  return SourceRange(BeginLoc, EndLoc);
}

size_t calculateExpansionLevel(const SourceLocation Loc, bool IsBegin) {
  if (Loc.isFileID())
    return 0;
  auto &SM = dpct::DpctGlobalInfo::getSourceManager();
  auto ExpanLoc = Loc;
  size_t Count = 0;
  while (ExpanLoc.isMacroID()) {
    Count++;
    if (IsBegin) {
      ExpanLoc = SM.getImmediateExpansionRange(ExpanLoc).getBegin();
    } else {
      ExpanLoc = SM.getImmediateExpansionRange(ExpanLoc).getEnd();
    }
  }
  return Count;
}

// Get textual representation of the Stmt.
std::string getStmtSpelling(const Stmt *S, SourceRange ParentRange) {
  if (!S)
    return "";
  return getStmtSpelling(S->getSourceRange(), ParentRange);
}

std::string getStmtSpelling(clang::SourceRange SR, SourceRange ParentRange) {
  auto &SM = dpct::DpctGlobalInfo::getSourceManager();
  SourceLocation BeginLoc, EndLoc;

  // Get the definition range of the parent
  ParentRange =
      getDefinitionRange(ParentRange.getBegin(), ParentRange.getEnd());
  ParentRange.setEnd(
      ParentRange.getEnd().getLocWithOffset(Lexer::MeasureTokenLength(
          ParentRange.getEnd(), SM,
          dpct::DpctGlobalInfo::getContext().getLangOpts())));

  int Length = 0;

  auto ParentRangeSize = SM.getFileOffset(ParentRange.getEnd()) -
                         SM.getFileOffset(ParentRange.getBegin());
  if (ParentRangeSize <= 0) {
    // if ParentRange is invalid, getDefinitionRange is the best we can have
    auto DRange = getDefinitionRange(SR.getBegin(), SR.getEnd());
    BeginLoc = DRange.getBegin();
    EndLoc = DRange.getEnd();
    Length = SM.getFileOffset(EndLoc) - SM.getFileOffset(BeginLoc) +
             Lexer::MeasureTokenLength(
                 EndLoc, SM, dpct::DpctGlobalInfo::getContext().getLangOpts());
  } else {
    // if ParentRange is valid, find the expansion location in the ParentRange
    auto Range =
        getRangeInRange(SR, ParentRange.getBegin(), ParentRange.getEnd());
    BeginLoc = Range.first;
    EndLoc = Range.second;
    Length = SM.getFileOffset(EndLoc) - SM.getFileOffset(BeginLoc);
  }

  if (Length <= 0)
    return "";
  return std::string(SM.getCharacterData(BeginLoc), Length);
   
}

SourceProcessType GetSourceFileType(llvm::StringRef SourcePath) {
  SmallString<256> FilePath = SourcePath;
  auto Extension = path::extension(FilePath);

  if (Extension == ".cu") {
    return SPT_CudaSource;
  }
  if (Extension == ".cuh") {
    return SPT_CudaHeader;
  }
  // the database check and including check need before the extension check.
  // Because the header file "xxx.cc" without CUDA syntax will not change file
  // name, but the "include" statement will change file name when this check is
  // after the extension check.
  // clang-format off
  // For unknown file extensions, determine the file type according to:
  // A. If it shows up in the compilation database as single migration
  //    file, then treat it as a main source file.
  // B. If it is included by another source file, then treat it as a header
  //    file.
  // C. If both A and B hold, then default to A.
  // clang-format on
  auto &FileSetInDB = dpct::DpctGlobalInfo::getFileSetInCompiationDB();
  if (FileSetInDB.find(SourcePath.str()) != end(FileSetInDB)) {
    return SPT_CppSource;
  }
  auto &IncludingFileSet = dpct::DpctGlobalInfo::getIncludingFileSet();
  if (IncludingFileSet.find(SourcePath.str()) != end(IncludingFileSet)) {
    return SPT_CppHeader;
  }
  if (Extension == ".hpp" || Extension == ".hxx" || Extension == ".h" ||
      Extension == ".hh" || Extension == ".inl" || Extension == ".inc" ||
      Extension == ".INL" || Extension == ".INC" || Extension == ".TPP" ||
      Extension == ".tpp") {
    return SPT_CppHeader;
  }
  return SPT_CppSource;
}

const std::string SpacesForStatement = "        "; // Eight spaces
const std::string SpacesForArg = "        ";       // Eight spaces

const std::string &getFmtEndStatement(void) {
  const static std::string EndStatement = ";\n";
  return EndStatement;
}

const std::string &getFmtStatementIndent(std::string &BaseIndent) {
  const static std::string FmtStatementIndent = BaseIndent + SpacesForStatement;
  return FmtStatementIndent;
}

const std::string &getFmtEndArg(void) {
  const static std::string EndArg = ",\n";
  return EndArg;
}

const std::string &getFmtArgIndent(std::string &BaseIndent) {
  const static std::string FmtArgIndent = BaseIndent + SpacesForArg;
  return FmtArgIndent;
}

std::vector<std::string> split(const std::string &Str, char Delim) {
  std::vector<std::string> V;
  std::stringstream S(Str);
  std::string Token;
  while (std::getline(S, Token, Delim))
    V.push_back(Token);

  return V;
}

/// Find the innermost (closest) block (CompoundStmt) where S is located
/// TODO: Go across macros
const clang::CompoundStmt *findImmediateBlock(const clang::Stmt *S) {
  if (!S)
    return nullptr;

  auto &Context = dpct::DpctGlobalInfo::getContext();
  auto Parents = Context.getParents(*S);
  while (Parents.size() == 1) {
    auto *Parent = Parents[0].get<Stmt>();
    if (Parent) {
      if (Parent->getStmtClass() == Stmt::StmtClass::CompoundStmtClass &&
          !Parent->getBeginLoc().isMacroID())
        return dyn_cast<CompoundStmt>(Parent);
      Parents = Context.getParents(*Parent);
    } else {
      Parents = Context.getParents(Parents[0]);
    }
  }

  return nullptr;
}

// A worklist-based BFS algorithm to find the innermost (closest) block
// where D is located
const clang::CompoundStmt *findImmediateBlock(const ValueDecl *D) {
  if (!D)
    return nullptr;

  // CS points to the CompoundStmt that is the body of the belonging function
  const CompoundStmt *CS = nullptr;
  if (D->getDeclContext()->getDeclKind() == Decl::Kind::Block) {
    auto BD = static_cast<const BlockDecl *>(D->getDeclContext());
    CS = BD->getCompoundBody();
  } else if (D->getDeclContext()->getDeclKind() == Decl::Kind::CXXMethod) {
    auto BD = static_cast<const CXXMethodDecl *>(D->getDeclContext());
    CS = dyn_cast<CompoundStmt>(BD->getBody());
  } else if (D->getDeclContext()->getDeclKind() == Decl::Kind::CXXConstructor) {
    auto BD = static_cast<const CXXConstructorDecl *>(D->getDeclContext());
    CS = dyn_cast<CompoundStmt>(BD->getBody());
  } else if (D->getDeclContext()->getDeclKind() == Decl::Kind::CXXDestructor) {
    auto BD = static_cast<const CXXDestructorDecl *>(D->getDeclContext());
    CS = dyn_cast<CompoundStmt>(BD->getBody());
  } else if (D->getLexicalDeclContext()->getDeclKind() ==
             Decl::Kind::Function) {
    auto BD = static_cast<const FunctionDecl *>(D->getDeclContext());
    CS = dyn_cast<CompoundStmt>(BD->getBody());
  }
  if (!CS)
    return nullptr;
  // Worklist
  std::deque<const CompoundStmt *> WL;
  WL.push_back(CS);

  while (!WL.empty()) {
    const CompoundStmt *CS = WL.front();
    WL.pop_front();
    for (auto Iter = CS->body_begin(); Iter != CS->body_end(); ++Iter) {
      // For a DeclStmt, check if TypeName and ArgName match
      if ((*Iter)->getStmtClass() == Stmt::StmtClass::DeclStmtClass) {
        DeclStmt *DS = dyn_cast<DeclStmt>(*Iter);
        for (auto It = DS->decl_begin(); It != DS->decl_end(); ++It) {
          VarDecl *VD = dyn_cast<VarDecl>(*It);
          if (VD == D)
            return CS;
        }
      }
      // Add nested CompoundStmt to the worklist for later search, BFS
      else if ((*Iter)->getStmtClass() == Stmt::StmtClass::CompoundStmtClass) {
        const CompoundStmt *CS = dyn_cast<CompoundStmt>(*Iter);
        WL.push_back(CS);
      }
    }
  }

  return nullptr;
}

const clang::FunctionDecl *getImmediateOuterFuncDecl(const clang::Stmt *S) {
  if (!S)
    return nullptr;

  auto &Context = dpct::DpctGlobalInfo::getContext();
  auto Parents = Context.getParents(*S);
  while (Parents.size() == 1) {
    if (auto *Parent = Parents[0].get<Decl>())
      if (auto FD = dyn_cast<clang::FunctionDecl>(Parent))
        return FD;

    Parents = Context.getParents(Parents[0]);
  }

  return nullptr;
}

const clang::CUDAKernelCallExpr *getParentKernelCall(const clang::Expr *E) {
  if (!E)
    return nullptr;

  auto &Context = dpct::DpctGlobalInfo::getContext();
  auto Parents = Context.getParents(*E);
  while (Parents.size() == 1) {
    if (auto KC = Parents[0].get<clang::CUDAKernelCallExpr>())
      return KC;

    Parents = Context.getParents(Parents[0]);
  }

  return nullptr;
}

bool callingFuncHasDeviceAttr(const CallExpr *CE) {
  auto FD = getImmediateOuterFuncDecl(CE);
  return FD && FD->hasAttr<CUDADeviceAttr>();
}

// Determine if a Stmt and a ValueDecl are in the same scope
bool isInSameScope(const Stmt *S, const ValueDecl *D) {
  if (!S || !D)
    return false;

  // Find the innermost block of D and S
  const auto *CS1 = findImmediateBlock(D);
  const auto *CS2 = findImmediateBlock(S);

  if (!CS1 || !CS2)
    return false;

  return CS1 == CS2;
}

// Iteratively get the inner ValueDecl of a potentially nested expression
// with implicit casts
const DeclRefExpr *getInnerValueDecl(const Expr *Arg) {
  auto DRE = dyn_cast<DeclRefExpr>(Arg->IgnoreImpCasts());
  while (!DRE) {
    if (auto UO = dyn_cast<UnaryOperator>(Arg->IgnoreImpCasts()))
      Arg = UO->getSubExpr();
    else
      return nullptr;
    DRE = dyn_cast<DeclRefExpr>(Arg->IgnoreImpCasts());
  }
  return DRE;
}

// Check if a string starts with the prefix
bool startsWith(const std::string &Str, const std::string &Prefix) {
  return Prefix.size() <= Str.size() &&
         std::equal(Prefix.begin(), Prefix.end(), Str.begin());
}

bool startsWith(const std::string &Str, char C) {
  return Str.size() && Str[0] == C;
}

// Check if a string ends with the suffix
bool endsWith(const std::string &Str, const std::string &Suffix) {
  return Suffix.size() <= Str.size() &&
         std::equal(Suffix.rbegin(), Suffix.rend(), Str.rbegin());
}

bool endsWith(const std::string &Str, char C) {
  return Str.size() && Str[Str.size() - 1] == C;
}

const clang::Stmt *getParentStmt(DynTypedNode Node) {
  if (auto S = Node.get<Stmt>()) {
    return getParentStmt(S);
  } else if (auto D = Node.get<Decl>()) {
    return getParentStmt(D);
  }
  return nullptr;
}

const clang::Stmt *getNonImplicitCastParentStmt(const clang::Stmt *S) {
  if (!S)
    return nullptr;
  const clang::Stmt *P = getParentStmt(S);
  while (P) {
    if (!dyn_cast<ImplicitCastExpr>(P)) {
      return P;
    } else {
      P = getParentStmt(P);
    }
  }
  return nullptr;
}

const clang::Stmt *
getNonImplicitCastNonParenExprParentStmt(const clang::Stmt *S) {
  if (!S)
    return nullptr;
  const clang::Stmt *P = getParentStmt(S);
  while (P) {
    if (!dyn_cast<ImplicitCastExpr>(P) && !dyn_cast<ParenExpr>(P)) {
      return P;
    } else {
      P = getParentStmt(P);
    }
  }
  return nullptr;
}

const clang::Stmt *getParentStmt(const clang::Stmt *S, bool SkipNonWritten) {
  if (!S)
    return nullptr;

  auto &Context = dpct::DpctGlobalInfo::getContext();
  auto Parents = Context.getParents(*S);
  assert(Parents.size() >= 1);
  if (Parents.size() >= 1) {
    const auto *P = Parents[0].get<Stmt>();
    if (SkipNonWritten && P) {
      if (const auto *CleanUp = dyn_cast<ExprWithCleanups>(P))
        return getParentStmt(CleanUp, SkipNonWritten);
    }
    return P;
  }

  return nullptr;
}

const clang::Stmt *getParentStmt(const clang::Decl *D) {
  if (!D)
    return nullptr;

  auto &Context = dpct::DpctGlobalInfo::getContext();
  auto Parents = Context.getParents(*D);
  assert(Parents.size() >= 1);
  if (Parents.size() >= 1)
    return Parents[0].get<Stmt>();

  return nullptr;
}

const clang::Decl *getParentDecl(const clang::Decl *D) {
  if (!D)
    return nullptr;

  auto &Context = dpct::DpctGlobalInfo::getContext();
  auto Parents = Context.getParents(*D);
  assert(Parents.size() >= 1);
  if (Parents.size() >= 1)
    return Parents[0].get<Decl>();

  return nullptr;
}

// Find the ancestor DeclStmt node
// Assumes: E != nullptr
const DynTypedNode getAncestorDeclStmtNode(const clang::Expr *E) {
  auto &Context = dpct::DpctGlobalInfo::getContext();
  auto ParentNodes = Context.getParents(*E);
  DynTypedNode ParentNode;
  while (!ParentNodes.empty()) {
    ParentNode = ParentNodes[0];
    if (ParentNode.get<DeclStmt>())
      break;
    ParentNodes = Context.getParents(ParentNode);
  }
  return ParentNode;
}

// Find the ancestor DeclStmt of E
// Assumes: E != nullptr
const clang::DeclStmt *getAncestorDeclStmt(const clang::Expr *E) {
  return getAncestorDeclStmtNode(E).get<DeclStmt>();
}

void getTheOuterMostExprOrValueDecl(const clang::Expr *E,
                                    const clang::Expr *&ResultExpr,
                                    const clang::ValueDecl *&ResultDecl) {
  auto &Context = dpct::DpctGlobalInfo::getContext();
  auto ParentNodes = Context.getParents(*E);
  DynTypedNode ParentNode;
  ResultExpr = E;
  ResultDecl = nullptr;
  while (!ParentNodes.empty()) {
    ParentNode = ParentNodes[0];
    if (!ParentNode.get<Expr>() && !ParentNode.get<ValueDecl>()) {
      break;
    }

    if (ParentNode.get<Expr>()) {
      ResultDecl = nullptr;
      ResultExpr = ParentNode.get<Expr>();
    } else if (ParentNode.get<ValueDecl>()) {
      ResultExpr = nullptr;
      ResultDecl = ParentNode.get<ValueDecl>();
    }

    ParentNodes = Context.getParents(ParentNode);
  }
}

const std::shared_ptr<clang::DynTypedNode>
getParentNode(const std::shared_ptr<clang::DynTypedNode> N) {
  if (!N)
    return nullptr;

  auto &Context = dpct::DpctGlobalInfo::getContext();
  auto Parents = Context.getParents(*N);
  if (Parents.size() >= 1)
    return std::make_shared<clang::DynTypedNode>(Parents[0]);

  return nullptr;
}

const std::shared_ptr<clang::DynTypedNode>
getNonImplicitCastParentNode(const std::shared_ptr<clang::DynTypedNode> N) {
  if (!N)
    return nullptr;

  auto P = getParentNode(N);
  while (P) {
    if (!P->get<ImplicitCastExpr>()) {
      return P;
    } else {
      P = getParentNode(P);
    }
  }
  return nullptr;
}

// Determine if S is a single line statement inside
// a if/while/do while/for statement
bool IsSingleLineStatement(const clang::Stmt *S) {
  auto ParentStmt = getParentStmt(S);
  if (!ParentStmt)
    return false;

  auto ParentStmtClass = ParentStmt->getStmtClass();
  return ParentStmtClass == Stmt::StmtClass::IfStmtClass ||
         ParentStmtClass == Stmt::StmtClass::WhileStmtClass ||
         ParentStmtClass == Stmt::StmtClass::DoStmtClass ||
         ParentStmtClass == Stmt::StmtClass::ForStmtClass;
}

// Find the nearest non-Expr non-Decl ancestor node of Expr E
// Assumes: E != nullptr
const DynTypedNode findNearestNonExprNonDeclAncestorNode(const clang::Expr *E) {
  auto &Context = dpct::DpctGlobalInfo::getContext();
  auto ParentNodes = Context.getParents(*E);
  DynTypedNode LastNode = DynTypedNode::create(*E), ParentNode;
  while (!ParentNodes.empty()) {
    ParentNode = ParentNodes[0];
    bool IsSingleStmt = ParentNode.get<CompoundStmt>() ||
                        ParentNode.get<IfStmt>() || ParentNode.get<ForStmt>() ||
                        ParentNode.get<WhileStmt>() || ParentNode.get<DoStmt>();
    if (!ParentNode.getSourceRange().getBegin().isMacroID() && IsSingleStmt) {
      break;
    }
    LastNode = ParentNode;
    ParentNodes = Context.getParents(LastNode);
  }
  return LastNode;
}

// Find the nearest non-Expr non-Decl ancestor statement of Expr E
// Assumes: E != nullptr
const clang::Stmt *findNearestNonExprNonDeclAncestorStmt(const clang::Expr *E) {
  return findNearestNonExprNonDeclAncestorNode(E).get<Stmt>();
}

SourceRange getScopeInsertRange(const MemberExpr *ME) {
  return getScopeInsertRange(ME, ME->getBeginLoc(), ME->getEndLoc());
}

SourceRange getScopeInsertRange(const Expr *E,
                                const SourceLocation &FuncNameBegin,
                                const SourceLocation &FuncCallEnd) {
  SourceLocation StmtBegin, StmtEndAfterSemi;
  auto &Context = dpct::DpctGlobalInfo::getContext();
  auto &SM = dpct::DpctGlobalInfo::getSourceManager();
  auto ParentNode = Context.getParents(*E);
  DynTypedNode AncestorStmt;
  SourceLocation StmtEnd;
  if (ParentNode.empty()) {
    StmtBegin = FuncNameBegin;
    StmtEnd = FuncCallEnd;
  } else if (!ParentNode[0].get<Expr>() && !ParentNode[0].get<Decl>()) {
    StmtBegin = FuncNameBegin;
    StmtEnd = FuncCallEnd;
  } else {
    AncestorStmt = findNearestNonExprNonDeclAncestorNode(E);
    StmtBegin = AncestorStmt.getSourceRange().getBegin();
    StmtEnd = AncestorStmt.getSourceRange().getEnd();
    if (StmtBegin.isMacroID())
      StmtBegin = SM.getExpansionLoc(StmtBegin);
    if (StmtEnd.isMacroID())
      StmtEnd = SM.getExpansionLoc(StmtEnd);
  }

  if (AncestorStmt.get<Expr>()) {
    StmtEnd = StmtEnd.getLocWithOffset(Lexer::MeasureTokenLength(
        SM.getExpansionLoc(StmtEnd), SM,
        dpct::DpctGlobalInfo::getContext().getLangOpts()));
  }

  StmtEndAfterSemi = StmtEnd.getLocWithOffset(Lexer::MeasureTokenLength(
      SM.getExpansionLoc(StmtEnd), SM,
      dpct::DpctGlobalInfo::getContext().getLangOpts()));
  return {StmtBegin, StmtEndAfterSemi};
}

std::string getCanonicalPath(SourceLocation Loc) {
  auto &SM = dpct::DpctGlobalInfo::getSourceManager();
  std::string Path = SM.getFilename(SM.getExpansionLoc(Loc)).str();
  makeCanonical(Path);
  return Path;
}

bool containOnlyDigits(const std::string &str) {
  return std::all_of(str.begin(), str.end(), ::isdigit);
}

void replaceSubStr(std::string &Str, const std::string &SubStr,
                   const std::string &Repl) {
  auto P = Str.find(SubStr);
  if (P != std::string::npos)
    Str.replace(P, SubStr.size(), Repl);
}
void replaceSubStrAll(std::string &Str, const std::string &SubStr,
                      const std::string &Repl) {
  auto P = Str.find(SubStr);
  while (P != std::string::npos) {
    Str.replace(P, SubStr.size(), Repl);
    P = Str.find(SubStr);
  }
}

/// Get the immediate ancestor with type \tparam T of \param S
template <typename T> const T *getImmediateAncestor(const Stmt *S) {
  if (!S)
    return nullptr;

  auto &Context = dpct::DpctGlobalInfo::getContext();
  auto Parents = Context.getParents(*S);
  while (Parents.size() == 1) {
    if (auto *Parent = Parents[0].get<T>()) {
      return Parent;
    } else {
      Parents = Context.getParents(Parents[0]);
    }
  }

  return nullptr;
}

/// Find the FunctionDecl where \param S is located
const FunctionDecl *getFunctionDecl(const Stmt *S) {
  return getImmediateAncestor<FunctionDecl>(S);
}

/// Get the CallExpr where \param S is referenced
const CallExpr *getCallExpr(const Stmt *S) {
  return getImmediateAncestor<CallExpr>(S);
}

/// Check if \param E is an expr that loads the address of \param DRE,
/// ignoring any casts and parens.
bool isAddressOfExpr(const Expr *E, const DeclRefExpr *DRE) {
  E = E->IgnoreCasts()->IgnoreParens();
  if (auto UO = dyn_cast<UnaryOperator>(E))
    if (UO->getOpcode() == UO_AddrOf)
      if (auto DRE2 = dyn_cast<DeclRefExpr>(UO->getSubExpr()))
        if (DRE->getDecl() == DRE2->getDecl())
          return true;
  return false;
}

/// Check if \param CE allocates memory pointed to by \param Arg
bool isCudaMemoryAllocation(const DeclRefExpr *Arg, const CallExpr *CE) {
  auto FD = CE->getDirectCallee();
  if (!FD)
    return false;
  auto FuncName = FD->getNameAsString();
  if (FuncName == "cudaMalloc" || FuncName == "cudaMallocPitch") {
    if (!CE->getNumArgs())
      return false;
    if (isAddressOfExpr(CE->getArg(0), Arg))
      return true;
  }
  return false;
}

/// This function traverses all the nodes in the AST represented by \param Root
/// in a depth-first manner, until the node \param Sentinal is reached, to check
/// if the pointer \param Arg to a piece of memory is used as lvalue after the
/// most recent memory allocation until \param Sentinal.
///
/// \param Arg: the expr that represents a reference to a declared variable
/// \param Root: the root of an AST
/// \param Sentinal: the sentinal node indicating termination of traversal
/// \param CurrentScope: the current scope of searching
/// \param UsedInScope: the map recording used-as-lavlue status for all scopes
/// \param Done: if current searching should stop or not
///
/// devPtr (T *) can be initialized in the following ways:
///   1. cudaMalloc(&devPtr, size);
///   2. cudaMallocPitch(&devPtr, pitch, width, height);
/// where "&devPtr" can be surrounded by arbitrary number of cast or paren
/// expressions.
/// If a new allocation happens on the memory pointed to by devPtr, \Used is
/// reset to false.
///
/// devPtr (T *) can be used as lvalue in the various ways:
///   1. devPtr = devPtr + 1;
///   2. devPtr = devPtr - 1;
///   3. devPtr += 1;
///   4. devPtr -= 1;
///   5. mod(&devPtr); // void mod(int **);
///   6. mod(devPtr);  // void mod(int *&);
///   ...
/// In a Clang AST, \param Arg is judged of used-as-lvalue when it is not under
/// a LValueToRValue cast node in the AST, which covers all the above cases.
/// Each used-as-lvalue scenario sets \param Used to true.
///
/// If the memory is never seen to be allocated in the traversing process,
/// \param Used is conservatively treated as true.
void findUsedAsLvalue(const DeclRefExpr *Arg, const Stmt *Root,
                      const Stmt *Sentinal,
                      std::vector<const Stmt *> &CurrentScope,
                      std::map<std::vector<const Stmt *>, bool> &UsedInScope,
                      bool &Done) {
  // Done with searching when Sentinal is reached.
  if (!Arg || !Root || !Sentinal)
    return;
  if (Root == Sentinal) {
    Done = true;
    return;
  }

  if (auto DRE = dyn_cast<DeclRefExpr>(Root)) {
    if (DRE->getType()->isPointerType()) {
      if (DRE->getDecl() != Arg->getDecl())
        return;
      if (auto *Parent = getParentStmt(DRE))
        if (auto *ICE = dyn_cast<ImplicitCastExpr>(Parent))
          if (ICE->getCastKind() == CK_LValueToRValue)
            return;
      // Arg is used as lvalue
      UsedInScope[CurrentScope] = true;
    }
  } else if (auto CE = dyn_cast<CallExpr>(Root)) {
    if (isCudaMemoryAllocation(Arg, CE))
      UsedInScope[CurrentScope] = false;
    else
      for (auto It = CE->arg_begin(); !Done && It != CE->arg_end(); ++It)
        findUsedAsLvalue(Arg, *It, Sentinal, CurrentScope, UsedInScope, Done);
  } else if (auto IS = dyn_cast<IfStmt>(Root)) {
    // Condition
    findUsedAsLvalue(Arg, IS->getCond(), Sentinal, CurrentScope, UsedInScope,
                     Done);
    if (Done)
      return;
    bool Used = UsedInScope[CurrentScope];

    // Then branch
    CurrentScope.push_back(IS->getThen());
    UsedInScope[CurrentScope] = Used;
    findUsedAsLvalue(Arg, IS->getThen(), Sentinal, CurrentScope, UsedInScope,
                     Done);
    if (Done)
      return;
    CurrentScope.pop_back();

    // Else branch
    if (auto ElseBranch = IS->getElse()) {
      CurrentScope.push_back(ElseBranch);
      UsedInScope[CurrentScope] = Used;
      findUsedAsLvalue(Arg, ElseBranch, Sentinal, CurrentScope, UsedInScope,
                       Done);
      if (Done)
        return;
      CurrentScope.pop_back();
    }
  } else if (auto WS = dyn_cast<WhileStmt>(Root)) {
    // Condition
    findUsedAsLvalue(Arg, WS->getCond(), Sentinal, CurrentScope, UsedInScope,
                     Done);
    if (Done)
      return;

    // Body
    bool Used = UsedInScope[CurrentScope];
    CurrentScope.push_back(WS->getBody());
    UsedInScope[CurrentScope] = Used;
    findUsedAsLvalue(Arg, WS->getBody(), Sentinal, CurrentScope, UsedInScope,
                     Done);
    if (Done)
      return;
    CurrentScope.pop_back();
  } else if (auto FS = dyn_cast<ForStmt>(Root)) {
    // Initialization
    findUsedAsLvalue(Arg, FS->getInit(), Sentinal, CurrentScope, UsedInScope,
                     Done);
    if (Done)
      return;
    // Condition
    findUsedAsLvalue(Arg, FS->getCond(), Sentinal, CurrentScope, UsedInScope,
                     Done);
    if (Done)
      return;
    // Increment
    findUsedAsLvalue(Arg, FS->getInc(), Sentinal, CurrentScope, UsedInScope,
                     Done);
    if (Done)
      return;

    // Body
    bool Used = UsedInScope[CurrentScope];
    CurrentScope.push_back(FS->getBody());
    UsedInScope[CurrentScope] = Used;
    findUsedAsLvalue(Arg, FS->getBody(), Sentinal, CurrentScope, UsedInScope,
                     Done);
    if (Done)
      return;
    CurrentScope.pop_back();
  } else {
    // Finishes when Sentinal is reached or we're done searching current
    // children nodes
    for (auto It = Root->child_begin(); !Done && It != Root->child_end(); ++It)
      findUsedAsLvalue(Arg, *It, Sentinal, CurrentScope, UsedInScope, Done);
  }
}

/// This function checks if the pointer \param Arg to a piece of memory is used
/// as lvalue after the memory is allocated, until \param S in the same
/// function. If the memory not allocated before \param S in the same function,
/// \param Arg is considered used-as-lvalue before \param S.
bool isArgUsedAsLvalueUntil(const DeclRefExpr *Arg, const Stmt *S) {
  // Global variables are always treated as used-as-lvalue
  if (Arg->getDecl()->isDefinedOutsideFunctionOrMethod())
    return true;

  auto *FD = getFunctionDecl(S);
  if (!FD)
    return true;

  auto *CS = FD->getBody();

  std::vector<const Stmt *> CurrentScope{CS};
  // If \param Arg is used as lvalue before \param S in the scope
  std::map<std::vector<const Stmt *>, bool> UsedInScope;
  UsedInScope[CurrentScope] = true;

  // If we are done with searching (\param S has been reached)
  bool Done = false;

  // Traverse from the function body
  findUsedAsLvalue(Arg, CS, S, CurrentScope, UsedInScope, Done);

  return UsedInScope[CurrentScope];
}

/// This function gets the length from \p eRange begin to the trailing
/// spaces end of \p Range end.
/// \param SourceRange Range
/// \param SM SourceManager.
/// \return The result length.
unsigned int getLenIncludingTrailingSpaces(SourceRange Range,
                                           const SourceManager &SM) {
  const char *C = SM.getCharacterData(
      Lexer::getLocForEndOfToken(Range.getEnd(), 0, SM, LangOptions()));
  while (C && *C) {
    if (!isspace(*C)) {
      break;
    }
#if defined(__linux__)
    if (*C == '\n') {
      break;
    }
#elif defined(_WIN32)
    if (*C == '\r') {
      break;
    }
#else
#error Only support Windows and Linux.
#endif
    ++C;
  }
  return C - SM.getCharacterData(Range.getBegin());
}

/// This function gets the statement nodes of the initialization, condition or
/// increment parts of the \p Node.
/// \param Node The statement node which is if, for, do, while or switch.
/// \return The result statement nodes vector.
std::vector<const Stmt *> getConditionNode(DynTypedNode Node) {
  std::vector<const Stmt *> Res;
  if (const IfStmt *CondtionNode = Node.get<IfStmt>()) {
    Res.push_back(CondtionNode->getCond());
    Res.push_back(CondtionNode->getConditionVariableDeclStmt());
  } else if (const ForStmt *CondtionNode = Node.get<ForStmt>()) {
    Res.push_back(CondtionNode->getCond());
    Res.push_back(CondtionNode->getInc());
    Res.push_back(CondtionNode->getInit());
    Res.push_back(CondtionNode->getConditionVariableDeclStmt());
  } else if (const WhileStmt *CondtionNode = Node.get<WhileStmt>()) {
    Res.push_back(CondtionNode->getCond());
    Res.push_back(CondtionNode->getConditionVariableDeclStmt());
  } else if (const DoStmt *CondtionNode = Node.get<DoStmt>()) {
    Res.push_back(CondtionNode->getCond());
  } else if (const SwitchStmt *CondtionNode = Node.get<SwitchStmt>()) {
    Res.push_back(CondtionNode->getCond());
    Res.push_back(CondtionNode->getConditionVariableDeclStmt());
  }
  return Res;
}

/// This function gets the expression nodes of the condition part of the \p Node
/// \param Node The statement node which is if, for, do, while or switch.
/// \return The result statement nodes vector.
std::vector<const Stmt *> getConditionExpr(DynTypedNode Node) {
  std::vector<const Stmt *> Res;
  if (const IfStmt *CondtionNode = Node.get<IfStmt>()) {
    Res.push_back(CondtionNode->getCond());
  } else if (const ForStmt *CondtionNode = Node.get<ForStmt>()) {
    Res.push_back(CondtionNode->getCond());
  } else if (const WhileStmt *CondtionNode = Node.get<WhileStmt>()) {
    Res.push_back(CondtionNode->getCond());
  } else if (const DoStmt *CondtionNode = Node.get<DoStmt>()) {
    Res.push_back(CondtionNode->getCond());
  } else if (const SwitchStmt *CondtionNode = Node.get<SwitchStmt>()) {
    Res.push_back(CondtionNode->getCond());
  }
  return Res;
}

/// This function checks whether call expression \p CE is a child node of the
/// initialization, condition or increment part of for, while, do, if or switch.
///
/// If the original CallExpr is in if, switch, return or the init part of for
/// statement, and meet one of the following cases, then \p CanAvoidUsingLambda
/// will be true and \p SL will be set as the begin location of the Flow
/// Control statement:
/// 1. The CallExpr is the statement;
/// 2. The statement is a var decl using C init style and the CallExpr is the
///    init expression;
/// 3. The statement is an assign operator and the CallExpr is the RHS;
/// 4. The statement is a binary operator and the CallExpr is the LHS (avoiding
///    the short - circuit - evaluation);
/// \param E The expression to be checked.
bool isConditionOfFlowControl(const clang::CallExpr *CE,
                              std::string &OriginStmtType,
                              bool &CanAvoidUsingLambda, SourceLocation &SL) {
  CanAvoidUsingLambda = false;
  auto &Context = dpct::DpctGlobalInfo::getContext();
  auto ParentNodes = Context.getParents(*CE);
  DynTypedNode ParentNode;
  std::vector<DynTypedNode> AncestorNodes;
  bool FoundStmtHasCondition = false;
  while (!ParentNodes.empty()) {
    ParentNode = ParentNodes[0];
    AncestorNodes.push_back(ParentNode);
    if (ParentNode.get<IfStmt>() || ParentNode.get<ForStmt>() ||
        ParentNode.get<WhileStmt>() || ParentNode.get<DoStmt>() ||
        ParentNode.get<SwitchStmt>()) {
      if (ParentNode.get<IfStmt>())
        OriginStmtType = "if";
      else if (ParentNode.get<ForStmt>())
        OriginStmtType = "for";
      else if (ParentNode.get<WhileStmt>())
        OriginStmtType = "while";
      else if (ParentNode.get<DoStmt>())
        OriginStmtType = "do";
      else
        OriginStmtType = "switch";

      FoundStmtHasCondition = true;
      break;
    }
    ParentNodes = Context.getParents(ParentNode);
  }
  if (!FoundStmtHasCondition)
    return false;

  std::vector<const Stmt *> CondtionNodes;
  CondtionNodes = getConditionNode(AncestorNodes[AncestorNodes.size() - 1]);

  for (auto CondtionNode : CondtionNodes) {
    if (CondtionNode == nullptr)
      continue;
    for (const auto &Node : AncestorNodes) {
      if (Node.get<Stmt>() && Node.get<Stmt>() == CondtionNode) {
        // if the expression in else if, we can only use lambda
        auto P = getParentStmt(AncestorNodes[AncestorNodes.size() - 1]);
        if (!P)
          return true;
        auto CS = dyn_cast<CompoundStmt>(P);
        if (!CS)
          return true;
        // if the expression in condition of do-while or while,
        // we can only use lambda
        if (OriginStmtType == "do" || OriginStmtType == "while")
          return true;
        // if the expression in condition of for and not the Init part,
        // we can only use lambda
        if (OriginStmtType == "for") {
          auto FS = AncestorNodes[AncestorNodes.size() - 1].get<ForStmt>();
          if (!FS)
            return true;
          if (CondtionNode != FS->getInit())
            return true;
        }

        auto DS = dyn_cast<DeclStmt>(CondtionNode);
        auto E = dyn_cast<Expr>(CondtionNode);
        const BinaryOperator *BO = nullptr;
        if (DS && DS->isSingleDecl()) {
          // E.g., if(int a = functionCall()){}
          const VarDecl *VD = dyn_cast<VarDecl>(DS->getSingleDecl());
          if (VD && VD->hasInit()) {
            if (VD->getInitStyle() == VarDecl::InitializationStyle::CInit &&
                VD->getInit()->IgnoreImplicit() == CE) {
              CanAvoidUsingLambda = true;
              SL = dpct::DpctGlobalInfo::getSourceManager().getExpansionLoc(
                  AncestorNodes[AncestorNodes.size() - 1]
                      .get<Stmt>()
                      ->getBeginLoc());
            }
          }
        } else if (E && (BO = dyn_cast<BinaryOperator>(E->IgnoreImplicit()))) {
          // E.g., if(functionCall() && a){}
          //    or if(a = functionCall()){}
          if ((BO->getLHS()->IgnoreImplicit() == CE) ||
              (BO->getOpcode() == BO_Assign &&
               BO->getRHS()->IgnoreImplicit() == CE)) {
            CanAvoidUsingLambda = true;
            SL = dpct::DpctGlobalInfo::getSourceManager().getExpansionLoc(
                AncestorNodes[AncestorNodes.size() - 1]
                    .get<Stmt>()
                    ->getBeginLoc());
          }
        } else if (E && E->IgnoreImplicit() == CE) {
          // E.g., if(functionCall()){}
          CanAvoidUsingLambda = true;
          SL = dpct::DpctGlobalInfo::getSourceManager().getExpansionLoc(
              AncestorNodes[AncestorNodes.size() - 1]
                  .get<Stmt>()
                  ->getBeginLoc());
        }

        return true;
      }
    }
    if (CE == CondtionNode) {
      CanAvoidUsingLambda = true;
      SL = dpct::DpctGlobalInfo::getSourceManager().getExpansionLoc(
          AncestorNodes[AncestorNodes.size() - 1].get<Stmt>()->getBeginLoc());
      return true;
    }
  }
  return false;
}

/// If \p OnlyCheckConditionExpr is false, this function checks whether
/// expression \p E is a child node of the initialization, condition or
/// increment part of for, while, do, if or switch, if \p OnlyCheckConditionExpr
/// is true, only checks whether \p E is a child node of the condition part of
/// for, while, do, if or switch.
/// \param E The expression to be checked.
/// \return The result.
bool isConditionOfFlowControl(const clang::Expr *E,
                              bool OnlyCheckConditionExpr) {
  auto &Context = dpct::DpctGlobalInfo::getContext();
  auto ParentNodes = Context.getParents(*E);
  DynTypedNode ParentNode;
  std::vector<DynTypedNode> AncestorNodes;
  bool FoundStmtHasCondition = false;
  while (!ParentNodes.empty()) {
    ParentNode = ParentNodes[0];
    AncestorNodes.push_back(ParentNode);
    if (ParentNode.get<IfStmt>() || ParentNode.get<ForStmt>() ||
        ParentNode.get<WhileStmt>() || ParentNode.get<DoStmt>() ||
        ParentNode.get<SwitchStmt>()) {
      FoundStmtHasCondition = true;
      break;
    }
    ParentNodes = Context.getParents(ParentNode);
  }
  if (!FoundStmtHasCondition)
    return false;
  std::vector<const Stmt *> CondtionNodes;
  if (OnlyCheckConditionExpr) {
    CondtionNodes = getConditionExpr(AncestorNodes[AncestorNodes.size() - 1]);
  } else {
    CondtionNodes = getConditionNode(AncestorNodes[AncestorNodes.size() - 1]);
  }

  for (auto CondtionNode : CondtionNodes) {
    if (CondtionNode == nullptr)
      continue;
    for (const auto &Node : AncestorNodes) {
      if (Node.get<Stmt>() && Node.get<Stmt>() == CondtionNode)
        return true;
    }
    if (E == CondtionNode)
      return true;
  }
  return false;
}

/// This function used in BLAS and Random migration. It generates the buffer
/// declaration and return the buffer name.
/// \param PointerName The origin pointer name string.
/// \param TypeAsStr The type of the origin pointer.
/// \param IndentStr The indentation.
/// \param [out] BufferDecl The buffer declaration string.
/// \return The buffer name.
std::string getBufferNameAndDeclStr(const std::string &PointerName,
                                    const std::string &TypeAsStr,
                                    const std::string &IndentStr,
                                    std::string &BufferDecl) {
  std::string BufferTempName =
      PointerName + "_buf_ct" +
      std::to_string(dpct::DpctGlobalInfo::getSuffixIndexInRuleThenInc());
  BufferDecl = "auto " + BufferTempName + " = " + MapNames::getDpctNamespace() +
               "get_buffer<" + TypeAsStr + ">(" + PointerName + ");" + getNL() +
               IndentStr;
  return BufferTempName;
}
/// This function used in BLAS and Random migration. It generates the buffer
/// declaration and return the buffer name.
/// \param Arg The origin pointer argument expression.
/// \param TypeAsStr The type of the origin pointer.
/// \param IndentStr The indentation.
/// \param [out] BufferDecl The buffer declaration string.
/// \return The buffer name.
std::string getBufferNameAndDeclStr(const Expr *Arg,
                                    const std::string &TypeAsStr,
                                    const std::string &IndentStr,
                                    std::string &BufferDecl) {
  std::string PointerName = dpct::ExprAnalysis::ref(Arg);
  std::string BufferTempName =
      getTempNameForExpr(Arg, true, true) + "buf_ct" +
      std::to_string(dpct::DpctGlobalInfo::getSuffixIndexInRuleThenInc());
  BufferDecl = "auto " + BufferTempName + " = " + MapNames::getDpctNamespace() +
               "get_buffer<" + TypeAsStr + ">(" + PointerName + ");" + getNL() +
               IndentStr;
  return BufferTempName;
}

// Recursively travers the subtree under \p S for all the reference of \p VD,
// store and return matched nodes in \p Result
void VarReferencedInFD(const Stmt *S, const ValueDecl *VD,
                       std::vector<const clang::DeclRefExpr *> &Result) {
  if (!S)
    return;

  if (auto DRF = dyn_cast<DeclRefExpr>(S)) {
    if (DRF->getDecl() == VD) {
      Result.push_back(DRF);
    }
  }
  for (auto It = S->child_begin(); It != S->child_end(); ++It) {
    VarReferencedInFD(*It, VD, Result);
  }
}

// Get the length of spaces until the next new line char, including the length
// of new line chars ('\r' and '\n').
// Return 0 if there is non-space char before the next new line char.
int getLengthOfSpacesToEndl(const char *CharData) {
  if (!CharData)
    return 0;
  int Len = 0;
  while (CharData && *CharData) {
    if (*CharData == '\n')
      return Len + 1;
    if (*CharData == '\r')
      return Len + 2;
    if (isspace(*CharData)) {
      ++CharData;
      ++Len;
    } else {
      return 0;
    }
  }
  return 0;
}

/// Determine whether \p A and \p B is in the same line.
/// \param A A source location.
/// \param B Another source location.
/// \param SM SourceManager.
/// \param [out] Invalid Is both source locations are valid.
/// \return The result.
bool isInSameLine(clang::SourceLocation A, clang::SourceLocation B,
                  const clang::SourceManager &SM, bool &Invalid) {
  auto ALocInfo = SM.getDecomposedLoc(A);
  auto BLocInfo = SM.getDecomposedLoc(B);
  bool InValidFlag = false;
  auto ALineNumber =
      SM.getLineNumber(ALocInfo.first, ALocInfo.second, &InValidFlag);
  if (InValidFlag) {
    Invalid = true;
    return false;
  }
  auto BLineNumber =
      SM.getLineNumber(BLocInfo.first, BLocInfo.second, &InValidFlag);
  if (InValidFlag) {
    Invalid = true;
    return false;
  }
  Invalid = false;
  if (ALineNumber == BLineNumber)
    return true;
  else
    return false;
}

/// Get the source range of a function call when it is in a function-like macro:
/// #define FUNCTION_LIKE_MACRO(X) fun(X)
/// FUNCTION_LIKE_MACRO(anAPICall());
///                     |         |
///              range begin   range end
/// \param CE A fucntion call in a function-like macro.
/// \return The source range of \p CE.
SourceRange getFunctionRange(const CallExpr *CE) {
  SourceManager &SM = dpct::DpctGlobalInfo::getSourceManager();
  auto Begin = CE->getBeginLoc();
  Begin = SM.getImmediateSpellingLoc(Begin);
  Begin = SM.getExpansionLoc(Begin);

  auto End = CE->getEndLoc();
  End = SM.getImmediateSpellingLoc(End);
  End = SM.getExpansionLoc(End);

  End = End.getLocWithOffset(Lexer::MeasureTokenLength(
      SM.getExpansionLoc(End), SM,
      dpct::DpctGlobalInfo::getContext().getLangOpts()));
  return SourceRange(Begin, End);
}

// If Flags[i] is true, then the range of Repls[i] will be calculated.
static std::vector<clang::tooling::Range>
calculateRangesWithFlag(const clang::tooling::Replacements &Repls,
                        std::vector<bool> Flags) {
  std::vector<clang::tooling::Range> Ranges;

  int Diff = 0;
  for (const auto &R : Repls) {
    Ranges.emplace_back(/*offset*/ R.getOffset() + Diff,
                        /*length*/ R.getReplacementText().size());
    Diff = Diff + R.getReplacementText().size() - R.getLength();
  }

  std::vector<clang::tooling::Range> RangesAfterFilter;
  int Size = Ranges.size();
  for (int i = 0; i < Size; ++i) {
    if (Flags[i])
      RangesAfterFilter.push_back(Ranges[i]);
  }
  return RangesAfterFilter;
}

/// Calculate the ranges of the input \p Repls which has NOT set NotFormatFlags.
/// \param Repls Replacements with format flags.
/// \return The result ranges.
std::vector<clang::tooling::Range>
calculateRangesWithFormatFlag(const clang::tooling::Replacements &Repls) {
  std::vector<bool> FormatFlags;
  for (const auto &R : Repls) {
    if (R.getNotFormatFlag())
      FormatFlags.push_back(false);
    else
      FormatFlags.push_back(true);
  }
  return calculateRangesWithFlag(Repls, FormatFlags);
}

/// Calculate the ranges of the input \p Repls which has set
/// BlockLevelFormatFlags. \param Repls Replacements with lambda flags. \return
/// The result ranges.
std::vector<clang::tooling::Range> calculateRangesWithBlockLevelFormatFlag(
    const clang::tooling::Replacements &Repls) {
  std::vector<bool> BlockLevelFormatFlags;

  for (const auto &R : Repls) {
    if (R.getBlockLevelFormatFlag())
      BlockLevelFormatFlags.push_back(true);
    else
      BlockLevelFormatFlags.push_back(false);
  }

  return calculateRangesWithFlag(Repls, BlockLevelFormatFlags);
}

/// Calculate the ranges of the input \p Ranges after \p Repls is applied to
/// the files.
/// \param Repls Replacements to apply.
/// \param Ranges Ranges before applying the replacements.
/// \return The result ranges.
std::vector<clang::tooling::Range>
calculateUpdatedRanges(const clang::tooling::Replacements &Repls,
                       const std::vector<clang::tooling::Range> &Ranges) {
  std::vector<clang::tooling::Range> Result;
  for (const auto &R : Ranges) {
    unsigned int BOffset = Repls.getShiftedCodePosition(R.getOffset());
    unsigned int EOffset =
        Repls.getShiftedCodePosition(R.getOffset() + R.getLength());
    if (BOffset > EOffset)
      continue;
    Result.emplace_back(BOffset, EOffset - BOffset);
  }
  return Result;
}

/// Determine if \param S is assigned or not
/// \param S A Stmt node
/// \return True if S is assigned and false if S is not assigned
bool isAssigned(const Stmt *S) {
  const auto *P = getParentStmt(S, true);
  return !P || (!dyn_cast<CompoundStmt>(P) && !dyn_cast<ForStmt>(P) &&
                !dyn_cast<WhileStmt>(P) && !dyn_cast<DoStmt>(P) &&
                !dyn_cast<IfStmt>(P));
}

/// Determine if \param S is in return statement or not
/// \param S A Stmt node
/// \return True if S is in return statement and false if S is not
bool isInRetStmt(const clang::Stmt *S) {
  if (auto ParentStmt = getParentStmt(S)) {
    if (ParentStmt->getStmtClass() == Stmt::StmtClass::ReturnStmtClass) {
      return true;
    }
  }
  return false;
}

/// Compute a temporary variable name for \param E
/// \param E The Expr based on which the temp name is computed
/// \param KeepLastUnderline A boolean value indicating if the last underline
/// is to be perserved or not
/// \return A temporary variable name

std::string getTempNameForExpr(const Expr *E, bool HandleLiteral,
                               bool KeepLastUnderline, bool IsInMacroDefine,
                               SourceLocation CallBegin, SourceLocation CallEnd) {
  SourceManager &SM = dpct::DpctGlobalInfo::getSourceManager();
  E = E->IgnoreCasts();
  bool RangeInCall = false;
  SourceLocation TokenBegin;
  SourceLocation ExprEndLoc;
  if (CallBegin.isValid() && CallEnd.isValid()) {
    auto Range = getRangeInRange(E, CallBegin, CallEnd);
    auto DLBegin = SM.getDecomposedLoc(Range.first);
    auto DLEnd = SM.getDecomposedLoc(Range.second);
    if (DLBegin.first == DLEnd.first &&
        DLBegin.second <= DLEnd.second) {
      TokenBegin = Range.first;
      ExprEndLoc = Range.second;
      RangeInCall = true;
    }
  }
  // Fallback to Range while CallBegin/End is not valid or getRangeInRange dose
  // not return a valid range
  if (!RangeInCall) {
    auto Range =
        getTheLastCompleteImmediateRange(E->getBeginLoc(), E->getEndLoc());
    TokenBegin = Range.first;
    ExprEndLoc = Range.second;
  }

  std::string IdString;
  llvm::raw_string_ostream OS(IdString);
  Token Tok;
  while (SM.getCharacterData(TokenBegin) <= SM.getCharacterData(ExprEndLoc)) {
    if (Lexer::getRawToken(TokenBegin, Tok, SM,
                           dpct::DpctGlobalInfo::getContext().getLangOpts(),
                           true)) {
      break;
    }
    if (Tok.isAnyIdentifier()) {
      OS << Tok.getRawIdentifier() << "_";
    } else if (HandleLiteral && Tok.isLiteral()) {
      OS << std::string(Tok.getLiteralData(), 0, Tok.getLength()) << "_";
    }
    TokenBegin = Tok.getEndLoc();
  }
  OS.flush();
  if (!KeepLastUnderline)
    IdString.pop_back();

  // If the fisrt char is digit, add "ct_" as prefix
  if (!IdString.empty() && isdigit(IdString[0]))
    IdString = "ct_" + IdString;

  return IdString;
}
// Check if an Expr is the outer most function-like macro
// E.g. MACRO_A(MACRO_B(x,y),z)
// Where MACRO_A is outer most and MACRO_B, x, y, z are not.
bool isOuterMostMacro(const Stmt *E) {
  auto &CT = dpct::DpctGlobalInfo::getContext();
  std::string ExpandedExpr, ExpandedParent;
  // Save the preprocessing result of E in ExpandedExpr
  llvm::raw_string_ostream StreamE(ExpandedExpr);
  E->printPretty(StreamE, nullptr, CT.getPrintingPolicy());
  StreamE.flush();
  std::shared_ptr<DynTypedNode> P =
      std::make_shared<DynTypedNode>(DynTypedNode::create(*E));
  // Find a parent stmt whose preprocessing result is different from
  // ExpandedExpr Since some parent is not writable. (is not shown in the
  // preprocessing result), a while loop is required to find the first writable
  // ancestor.
  do {
    ExpandedParent = "";
    P = getParentNode(P);
    if (!P)
      return true;
    llvm::raw_string_ostream StreamP(ExpandedParent);
    P->print(StreamP, CT.getPrintingPolicy());
    StreamP.flush();
  } while (!ExpandedParent.compare(ExpandedExpr));

  return !isInsideFunctionLikeMacro(E->getBeginLoc(), E->getEndLoc(), P);
}

bool isSameLocation(const SourceLocation L1, const SourceLocation L2) {
  auto LocInfo1 = dpct::DpctGlobalInfo::getInstance().getLocInfo(L1);
  auto LocInfo2 = dpct::DpctGlobalInfo::getInstance().getLocInfo(L2);
  if (LocInfo1.first.compare(LocInfo2.first) ||
      LocInfo1.second != LocInfo2.second) {
    return false;
  }
  return true;
}

bool isInsideFunctionLikeMacro(const SourceLocation BeginLoc,
                               const SourceLocation EndLoc,
                               const std::shared_ptr<DynTypedNode> Parent) {

  if (!BeginLoc.isMacroID() || !EndLoc.isMacroID()) {
    return false;
  }

  auto &SM = dpct::DpctGlobalInfo::getSourceManager();
  // If the begin/end location are different macro expansions,
  // the expression is a combination of different macros
  // which makes it outer-most.
  auto ExpansionBegin = SM.getExpansionRange(BeginLoc).getBegin();
  auto ExpansionEnd = SM.getExpansionRange(EndLoc).getBegin();
  if (!isSameLocation(ExpansionBegin, ExpansionEnd)) {
    return false;
  }

  // Since SM.getExpansionLoc() will always return the range of the outer-most
  // macro. If the expanded location of the parent stmt and E are the same, E is
  // inside a function-like macro.
  // E.g. MACRO_A(MACRO_B(x,y),z) where E is the PP
  // result of MACRO_B and Parent is the PP result of MACRO_A,
  // SM.getExpansionLoc(E) is at the beginning of MACRO_A, same as
  // SM.getExpansionLoc(Parent), in the source code. E is not outer-most.
  if (Parent->getSourceRange().getBegin().isValid() &&
      Parent->getSourceRange().getBegin().isMacroID()) {
    if (isSameLocation(SM.getExpansionLoc(Parent->getSourceRange().getBegin()),
                       SM.getExpansionLoc(BeginLoc))) {
      if (Parent->getSourceRange().getEnd().isValid() &&
          Parent->getSourceRange().getEnd().isMacroID()) {
        if (isSameLocation(
                SM.getExpansionLoc(Parent->getSourceRange().getEnd()),
                SM.getExpansionLoc(EndLoc))) {
          return true;
        }
      }
    }
  }

  // Another case which should return true is
  // #define MacroA(x) = x
  // When MacroA is used for default arguments in function definition
  // like foo(int x MacroA(0)) and the ASTMatcher matches the "0" in the
  // expansion, since the parent of x in the AST is "int x MacroA(0)" not "= x",
  // previous check cannot detect the "0" is inside a function like macro.
  // Should check if the expansion is the whole macro definition.

  // Get the location of "x" in "#define MacroA(x) = x"
  SourceLocation ImmediateSpellingBegin =
      SM.getExpansionLoc(SM.getImmediateSpellingLoc(BeginLoc));
  SourceLocation ImmediateSpellingEnd =
      SM.getExpansionLoc(SM.getImmediateSpellingLoc(EndLoc));
  SourceLocation ImmediateExpansionBegin =
      SM.getSpellingLoc(SM.getImmediateExpansionRange(BeginLoc).getBegin());
  SourceLocation ImmediateExpansionEnd =
      SM.getSpellingLoc(SM.getImmediateExpansionRange(EndLoc).getEnd());

  // Check if one of the 4 combinations of begin&end matches a macro def
  // ExpansionBegin & ExpansionEnd
  auto It = dpct::DpctGlobalInfo::getExpansionRangeToMacroRecord().find(
      getCombinedStrFromLoc(ImmediateExpansionBegin));
  if (It != dpct::DpctGlobalInfo::getExpansionRangeToMacroRecord().end() &&
      It->second->TokenIndex == 0 &&
      (!It->second->FilePath.compare(
           dpct::DpctGlobalInfo::getLocInfo(ImmediateExpansionEnd).first) &&
       It->second->ReplaceTokenEndOffset ==
           dpct::DpctGlobalInfo::getLocInfo(ImmediateExpansionEnd).second)) {
    return false;
  }

  // ExpansionBegin & SpellingEnd
  It = dpct::DpctGlobalInfo::getExpansionRangeToMacroRecord().find(
      getCombinedStrFromLoc(ImmediateExpansionBegin));
  if (It != dpct::DpctGlobalInfo::getExpansionRangeToMacroRecord().end() &&
      It->second->TokenIndex == 0 &&
      (!It->second->FilePath.compare(
           dpct::DpctGlobalInfo::getLocInfo(ImmediateSpellingEnd).first) &&
       It->second->ReplaceTokenEndOffset ==
           dpct::DpctGlobalInfo::getLocInfo(ImmediateSpellingEnd).second)) {
    return false;
  }

  // SpellingBegin & ExpansionEnd
  It = dpct::DpctGlobalInfo::getExpansionRangeToMacroRecord().find(
      getCombinedStrFromLoc(ImmediateSpellingBegin));
  if (It != dpct::DpctGlobalInfo::getExpansionRangeToMacroRecord().end() &&
      It->second->TokenIndex == 0 &&
      (!It->second->FilePath.compare(
           dpct::DpctGlobalInfo::getLocInfo(ImmediateExpansionEnd).first) &&
       It->second->ReplaceTokenEndOffset ==
           dpct::DpctGlobalInfo::getLocInfo(ImmediateExpansionEnd).second)) {
    return false;
  }

  // SpellingBegin & SpellingEnd
  It = dpct::DpctGlobalInfo::getExpansionRangeToMacroRecord().find(
      getCombinedStrFromLoc(ImmediateSpellingBegin));
  if (It != dpct::DpctGlobalInfo::getExpansionRangeToMacroRecord().end() &&
      It->second->TokenIndex == 0 &&
      (!It->second->FilePath.compare(
           dpct::DpctGlobalInfo::getLocInfo(ImmediateSpellingEnd).first) &&
       It->second->ReplaceTokenEndOffset ==
           dpct::DpctGlobalInfo::getLocInfo(ImmediateSpellingEnd).second)) {
    return false;
  }

  return true;
}

bool isLocationStraddle(SourceLocation BeginLoc, SourceLocation EndLoc) {
  auto &SM = dpct::DpctGlobalInfo::getSourceManager();
  auto SpellingBegin = SM.getSpellingLoc(BeginLoc);
  auto SpellingEnd = SM.getSpellingLoc(EndLoc);
  auto ItSpellingBegin =
      dpct::DpctGlobalInfo::getExpansionRangeToMacroRecord().find(
          getCombinedStrFromLoc(SpellingBegin));
  auto ItSpellingEnd =
      dpct::DpctGlobalInfo::getExpansionRangeToMacroRecord().find(
          getCombinedStrFromLoc(SpellingEnd));

  if ((BeginLoc.isMacroID() && EndLoc.isFileID()) ||
      (BeginLoc.isFileID() && EndLoc.isMacroID())) {
    return true;
  }

  // Different expansion but same define, e.g. AAA * AAA
  if (BeginLoc.isMacroID() && EndLoc.isMacroID()) {
    auto ExpansionBegin = SM.getExpansionRange(BeginLoc).getBegin();
    auto ExpansionEnd = SM.getExpansionRange(EndLoc).getBegin();
    auto DLExpanBegin = SM.getDecomposedLoc(ExpansionBegin);
    auto DLExpanEnd = SM.getDecomposedLoc(ExpansionEnd);
    if (DLExpanBegin.first != DLExpanEnd.first ||
        DLExpanBegin.second != DLExpanEnd.second) {
      return true;
    }
  }

  // If begin and end are both not in macro define, not straddle
  if (ItSpellingBegin ==
          dpct::DpctGlobalInfo::getExpansionRangeToMacroRecord().end() &&
      ItSpellingEnd ==
          dpct::DpctGlobalInfo::getExpansionRangeToMacroRecord().end()) {
    return false;
  }

  // If only one of begin and end is in macro define, straddle
  if (ItSpellingBegin ==
          dpct::DpctGlobalInfo::getExpansionRangeToMacroRecord().end() ||
      ItSpellingEnd ==
          dpct::DpctGlobalInfo::getExpansionRangeToMacroRecord().end()) {
    return true;
  }

  // If DL.first(the FileId) or DL.second(the location) is different which means
  // begin and end are in different macro define, straddle
  if (ItSpellingBegin->second->FilePath.compare(
          ItSpellingEnd->second->FilePath) ||
      ItSpellingBegin->second->ReplaceTokenBeginOffset !=
          ItSpellingEnd->second->ReplaceTokenBeginOffset) {
    return true;
  }

  return false;
}

// Check if an Expr is partially in function-like macro
bool isExprStraddle(const Stmt *S) {
  // Remove the outer func-like macro before checking straddle
  auto Range = getRangeInsideFuncLikeMacro(S);
  return isLocationStraddle(Range.getBegin(), Range.getEnd());
}

// Check if an Expr contains macro
bool isContainMacro(const Expr *E) {
  auto &SM = dpct::DpctGlobalInfo::getSourceManager();
  auto &Map = dpct::DpctGlobalInfo::getExpansionRangeBeginMap();
  SourceLocation ExprBegin = getStmtExpansionSourceRange(E).getBegin();
  SourceLocation ExprEnd = getStmtExpansionSourceRange(E).getEnd();

  SourceLocation Loc = ExprBegin;
  while (Loc <= ExprEnd) {
    if (Map.find(getCombinedStrFromLoc(Loc)) != Map.end()) {
      return true;
    }
    auto Tok = Lexer::findNextToken(
        Loc, SM, dpct::DpctGlobalInfo::getContext().getLangOpts());
    if (Tok.has_value())
      Loc = Tok.value().getLocation();
    else
      return false;
  }
  return false;
}

/// Get the dereference name of the expression \p E.
std::string getDrefName(const Expr *E) {
  std::ostringstream OS;
  printDerefOp(OS, E, nullptr);
  return OS.str();
}

const CXXRecordDecl *getParentRecordDecl(const ValueDecl *DD) {
  if (!DD)
    return nullptr;

  auto &Context = dpct::DpctGlobalInfo::getContext();
  auto Parents = Context.getParents(*DD);
  assert(Parents.size() >= 1);
  if (Parents.size() >= 1)
    return Parents[0].get<CXXRecordDecl>();

  return nullptr;
}

/// Get All Decls for a VarDecl or a FieldDecl
/// E.g for a VarDecl:
/// |-DeclStmt
///   |-VarDecl
///   `-VarDecl
/// or for a FieldDecl:
/// |-CXXRecordDecl
///   |-FieldDecl
///   |-FieldDecl
std::vector<const DeclaratorDecl *> getAllDecls(const DeclaratorDecl *DD) {
  std::vector<const DeclaratorDecl *> Decls;
  // For VarDecl, All VarDecls share the same parent DeclStmt with it
  if (auto P = getParentStmt(DD)) {
    if (auto DS = dyn_cast<DeclStmt>(P)) {
      for (auto It = DS->decl_begin(); It != DS->decl_end(); ++It) {
        if (auto DD2 = dyn_cast<DeclaratorDecl>(*It))
            Decls.push_back(DD2);
      }
    }
  }
  // For FieldDecl, All FieldDecls share the same BeginLoc with it
  else if (auto P = getParentRecordDecl(DD)) {
    for (auto It = P->decls_begin(); It != P->decls_end(); ++It) {
      if (auto DD2 = dyn_cast<DeclaratorDecl>(*It))
        if (DD2->getBeginLoc() == DD->getBeginLoc())
            Decls.push_back(DD2);
    }
  }
  return Decls;
}

/// Deduce the type for a QualType
/// \param QT QualType of the type
/// \param TypeName The name of the type
/// \HasConst If QT's parent has const qualifier
/// \return The pointer type string
std::string deducePointerType(QualType QT, std::string TypeName,
                              bool HasConst) {
  if (auto ET = dyn_cast<ElaboratedType>(QT))
    if (auto RT = dyn_cast<RecordType>(ET->desugar()))
      if (RT->getDecl()->getNameAsString() == TypeName)
        return HasConst ? "* const " : "*";
  return "";
}

/// Deduce the type for a DeclaratorDecl
/// \param DD The DeclaratorDecl of the type
/// \param TypeName The name of the type
/// \return The pointer type string
std::string deducePointerType(const DeclaratorDecl *DD, std::string TypeName) {
  std::string Result;
  auto DDT = DD->getType();
  if (DDT->isPointerType()) {
    auto PT = DDT->getPointeeType();
    // cudaStream_t
    Result = deducePointerType(PT, TypeName, DDT.getQualifiers().hasConst());
    // cudaStream_t *
    if (auto TDT = dyn_cast<TypedefType>(PT)) {
      if (TDT->desugar()->isPointerType()) {
        auto PT2 = TDT->desugar()->getPointeeType();
        Result =
            deducePointerType(PT2, TypeName, PT.getQualifiers().hasConst());
      }
    }
  }
  // cudaStream_t &
  else if (auto LVRT = dyn_cast<LValueReferenceType>(DDT)) {
    auto PT = LVRT->Type::getPointeeType();
    if (auto TDT = dyn_cast<TypedefType>(PT)) {
      if (TDT->desugar()->isPointerType()) {
        auto PT2 = TDT->desugar()->getPointeeType();
        Result =
            deducePointerType(PT2, TypeName, PT.getQualifiers().hasConst());
      }
    }
  }
  // cudaStream_t &&
  else if (auto RVRT = dyn_cast<RValueReferenceType>(DDT)) {
    auto PT = RVRT->Type::getPointeeType();
    if (auto TDT = dyn_cast<TypedefType>(PT)) {
      if (TDT->desugar()->isPointerType()) {
        auto PT2 = TDT->desugar()->getPointeeType();
        Result =
            deducePointerType(PT2, TypeName, PT.getQualifiers().hasConst());
      }
    }
  }
  // cudaStream_t [] and cudaStream_t *[]
  else if (auto CAT = dyn_cast<clang::ArrayType>(DDT)) {
    QualType FinalET = CAT->getElementType();
    while(dyn_cast<clang::ArrayType>(FinalET)) {
      FinalET = dyn_cast<clang::ArrayType>(FinalET)->getElementType();
    }
    auto PT = FinalET->getPointeeType();
    // cudaStream_t []
    Result = deducePointerType(PT, TypeName, DDT.getQualifiers().hasConst());
    // cudaStream_t *[]
    if (auto TDT = dyn_cast<TypedefType>(PT)) {
      if (TDT->desugar()->isPointerType()) {
        auto PT2 = TDT->desugar()->getPointeeType();
        Result =
            deducePointerType(PT2, TypeName, PT.getQualifiers().hasConst());
      }
    }
  }
  return Result;
}

/// Check whether the input expression \p E is a single token which is an
/// identifier or literal.
bool isAnIdentifierOrLiteral(const Expr *E) {
  SourceManager &SM = dpct::DpctGlobalInfo::getSourceManager();
  E = E->IgnoreCasts();
  dpct::ExprAnalysis EA(E);
  auto BeginLoc = EA.getExprBeginSrcLoc();
  auto EndLoc = EA.getExprEndSrcLoc();
  Token BeginTok, EndTok;
  if (!Lexer::getRawToken(BeginLoc, BeginTok, SM,
                          dpct::DpctGlobalInfo::getContext().getLangOpts(),
                          true) &&
      !Lexer::getRawToken(EndLoc, EndTok, SM,
                          dpct::DpctGlobalInfo::getContext().getLangOpts(),
                          true)) {
    if ((BeginTok.getLocation() != EndTok.getLocation()) ||
        (BeginTok.getLength() != EndTok.getLength())) {
      return false;
    }
    if (BeginTok.isAnyIdentifier() || BeginTok.isLiteral()) {
      return true;
    } else {
      return false;
    }
  } else {
    return false;
  }
}

/// Check whether the \p E meets the conditions 1, 2 and 3
/// 1. a sizeof operator
/// 2. the operand is type
/// 3. The type is same as \p TypeStr
bool isSameSizeofTypeWithTypeStr(const Expr *E, const std::string &TypeStr) {
  auto UETTE = dyn_cast<UnaryExprOrTypeTraitExpr>(E->IgnoreImpCasts());
  if (!UETTE) {
    return false;
  }

  if (UETTE->getKind() != UETT_SizeOf) {
    return false;
  }

  if (!UETTE->isArgumentType()) {
    return false;
  }

  auto ArguQT = UETTE->getArgumentType();
  if (dpct::DpctGlobalInfo::getReplacedTypeName(ArguQT) == TypeStr) {
    return true;
  } else {
    return false;
  }
}

bool isInReturnStmt(const Expr *E, SourceLocation &OuterInsertLoc) {
  auto &Context = dpct::DpctGlobalInfo::getContext();
  auto ParentNodes = Context.getParents(*E);
  DynTypedNode ParentNode;
  const ReturnStmt *RS = nullptr;
  while (!ParentNodes.empty()) {
    ParentNode = ParentNodes[0];
    RS = ParentNode.get<ReturnStmt>();
    if (RS) {
      OuterInsertLoc = dpct::DpctGlobalInfo::getSourceManager().getExpansionLoc(
          RS->getBeginLoc());
      return true;
    }
    ParentNodes = Context.getParents(ParentNode);
  }
  return false;
}

std::string getHashStrFromLoc(SourceLocation Loc) {
  auto R = dpct::DpctGlobalInfo::getLocInfo(Loc);
  std::string Ret = std::to_string(
      std::hash<std::string>()(dpct::buildString(R.first, R.second)));
  return Ret;
}

bool IsTypeChangedToPointer(const DeclRefExpr *DRE) {
  auto D = DRE->getDecl();
  auto T = D->getType();
  if (auto VD = dyn_cast<VarDecl>(D))
    return T->isScalarType() &&
           (VD->hasAttr<CUDASharedAttr>() || VD->hasAttr<CUDAGlobalAttr>() ||
            VD->hasAttr<CUDADeviceAttr>());
  return false;
}

SourceLocation getBeginLocOfPreviousEmptyMacro(SourceLocation Loc) {
  auto &Map = dpct::DpctGlobalInfo::getEndOfEmptyMacros();
  auto It = Map.find(getHashStrFromLoc(Loc));
  if (It != Map.end()) {
    return It->second;
  }
  return Loc;
}

unsigned int getEndLocOfFollowingEmptyMacro(SourceLocation Loc) {
  auto &SM = dpct::DpctGlobalInfo::getSourceManager();
  auto &Map = dpct::DpctGlobalInfo::getBeginOfEmptyMacros();
  Token Tok;
  bool Ret = Lexer::getRawToken(
      Loc.getLocWithOffset(Lexer::MeasureTokenLength(
          Loc, SM, dpct::DpctGlobalInfo::getContext().getLangOpts())),
      Tok, SM, dpct::DpctGlobalInfo::getContext().getLangOpts(), true);
  if (Ret)
    return 0;

  SourceLocation EndOfToken = SM.getExpansionLoc(Tok.getLocation());
  while (Tok.isNot(tok::eof) && Tok.is(tok::comment)) {
    Lexer::getRawToken(
        EndOfToken.getLocWithOffset(Lexer::MeasureTokenLength(
            EndOfToken, SM, dpct::DpctGlobalInfo::getContext().getLangOpts())),
        Tok, SM, dpct::DpctGlobalInfo::getContext().getLangOpts(), true);
    EndOfToken = SM.getExpansionLoc(Tok.getEndLoc());
    ;
  }

  auto EndOfTokenLocInfo = dpct::DpctGlobalInfo::getLocInfo(EndOfToken);
  std::string EndOfTokenKey = std::to_string(std::hash<std::string>()(
      dpct::buildString(EndOfTokenLocInfo.first, EndOfTokenLocInfo.second)));
  auto OriginalLocInfo = dpct::DpctGlobalInfo::getLocInfo(Loc);

  auto It = Map.find(EndOfTokenKey);
  if (It != Map.end()) {
    return It->second + EndOfTokenLocInfo.second - OriginalLocInfo.second;
  }
  return 0;
}

std::string
getNestedNameSpecifierString(const clang::NestedNameSpecifier *NNS) {
  if (!NNS)
    return std::string();
  std::string Result;
  llvm::raw_string_ostream OS(Result);
  NNS->print(OS, dpct::DpctGlobalInfo::getContext().getPrintingPolicy());
  OS.flush();
  if (StringRef(Result).startswith("::"))
    Result = Result.substr(2);
  return Result;
}
std::string
getNestedNameSpecifierString(const clang::NestedNameSpecifierLoc &NNSL) {
  if (NNSL)
    return getNestedNameSpecifierString(NNSL.getNestedNameSpecifier());
  return std::string();
}

bool needExtraParens(const Expr *E) {
  switch (E->IgnoreImplicitAsWritten()->getStmtClass()) {
  case Stmt::DeclRefExprClass:
  case Stmt::MemberExprClass:
  case Stmt::ParenExprClass:
  case Stmt::CallExprClass:
  case Stmt::IntegerLiteralClass:
  case Stmt::FloatingLiteralClass:
  case Stmt::StringLiteralClass:
  case Stmt::ArraySubscriptExprClass:
  case Stmt::CStyleCastExprClass:
  case Stmt::CXXStaticCastExprClass:
    return false;
  case Stmt::CXXConstructExprClass: {
    auto Ctor = static_cast<const CXXConstructExpr *>(E);
    if (Ctor->getParenOrBraceRange().isInvalid() && Ctor->getNumArgs() == 1)
      return needExtraParens(Ctor->getArg(0));
    else
      return true;
  }
  case Stmt::CXXOperatorCallExprClass: {
    if (auto COCE = dyn_cast<CXXOperatorCallExpr>(E)) {
      if (COCE->getOperator() == clang::OO_Subscript)
        return false;
    }
    return true;
  }
  default:
    return true;
  }
}

// Get the range of an Expr in the largest (the outermost) macro definition
// e.g.
// line 1: #define MACRO_A 3
// line 2: #define MACRO_B 2 + MACRO_A
// line 3: MACRO_B
// The result of PP: 2 + 3
// The search path of Beginloc "2": line 2 --> line 3
// Since line 3 is not MacroID, stops at line 2.
// The search path of EndLoc "3": line 1 --> line 2 --> line 3
// Since line 3 is not MacroID, stops at line 2.
// Return std::pair(line 2 "2", line 2 "3")
std::pair<clang::SourceLocation, clang::SourceLocation>
getTheOneBeforeLastImmediateExapansion(const clang::SourceLocation Begin,
                                       const clang::SourceLocation End) {
  auto &SM = dpct::DpctGlobalInfo::getSourceManager();
  auto ResultBegin = Begin;
  auto ResultEnd = End;

  // Keep calling getImmediateExpansionRange until the next location is not
  // macro
  if (ResultBegin.isMacroID()) {
    while (SM.getImmediateExpansionRange(ResultBegin).getBegin().isMacroID()) {
      ResultBegin = SM.getImmediateExpansionRange(ResultBegin).getBegin();
    }
  }
  if (ResultEnd.isMacroID()) {
    while (SM.getImmediateExpansionRange(ResultEnd).getEnd().isMacroID()) {
      ResultEnd = SM.getImmediateExpansionRange(ResultEnd).getEnd();
    }
  }
  return std::pair<clang::SourceLocation, clang::SourceLocation>(ResultBegin,
                                                                 ResultEnd);
}

// To remove the correct kernel range, DPCT need to find the Begin/End pair in
// which the Begin/End are at the same macro define. e.g.
// Line 1: #define CCC <<<1,1>>>()
// Line 2: #define KERNEL(A, B) templatefoo<A,B>CCC
// Line 3: #define CALL_KERNEL(C, D) KERNEL(C, D); int a = 0;
// Line 4: void templatefoo2() { CALL_KERNEL(8, 9) }
// There are 3 candidates of the kernel range,
// 1. Line 4 "CALL_KERNEL2(8, AAA)"
// 2. Line 3 "KERNEL(C, D)"
// 3. Line 2 "templatefoo<A,B>CCC"
// The 3rd candidate is the best choice.
// However, the original begin/end location of the kernel call is at Line 2 "t"
// and Line1 ")". This function will calculate the macro expansion level of the
// begin and end location.
// The macro expansion level of Line 2 "t" is 3. (Line4 -> Line3 -> Line2)
// The macro expansion level of Line1 ")" is 4. (Line4 -> Line3 -> Line2 ->
// Line1) And this function will perform a while loop to find the lowest common
// ancestor of the begin and end.
std::pair<clang::SourceLocation, clang::SourceLocation>
getTheLastCompleteImmediateRange(clang::SourceLocation BeginLoc,
                                 clang::SourceLocation EndLoc) {
  auto &SM = dpct::DpctGlobalInfo::getSourceManager();
  auto BeginLevel = calculateExpansionLevel(BeginLoc, true);
  auto EndLevel = calculateExpansionLevel(EndLoc, false);
  while ((BeginLevel > 0 || EndLevel > 0) &&
         (isLocationStraddle(BeginLoc, EndLoc) ||
          ((BeginLoc.isMacroID() && !dpct::DpctGlobalInfo::isInAnalysisScope(
                                        SM.getSpellingLoc(BeginLoc))) ||
           (EndLoc.isMacroID() && !dpct::DpctGlobalInfo::isInAnalysisScope(
                                      SM.getSpellingLoc(EndLoc)))))) {
    if (BeginLevel > EndLevel) {
      BeginLoc = SM.getImmediateExpansionRange(BeginLoc).getBegin();
      BeginLevel--;
    } else {
      EndLoc = SM.getImmediateExpansionRange(EndLoc).getEnd();
      EndLevel--;
    }
  }
  BeginLoc = SM.getSpellingLoc(BeginLoc);
  EndLoc = SM.getSpellingLoc(EndLoc);
  return std::pair<clang::SourceLocation, clang::SourceLocation>(BeginLoc,
                                                                 EndLoc);
}

bool isInRange(SourceLocation PB, SourceLocation PE, SourceLocation Loc) {
  auto &SM = dpct::DpctGlobalInfo::getSourceManager();
  auto PBDC = SM.getDecomposedLoc(PB);
  auto DC = SM.getDecomposedLoc(Loc);
  if (PBDC.first != DC.first || PBDC.second > DC.second) {
    return false;
  }
  auto PEDC = SM.getDecomposedLoc(PE);
  if (PEDC.first != DC.first || DC.second > PEDC.second) {
    return false;
  }
  return true;
}

bool isInRange(SourceLocation PB, SourceLocation PE, StringRef FilePath,
               size_t Offset) {
  auto PBLC = dpct::DpctGlobalInfo::getInstance().getLocInfo(PB);
  if (PBLC.first != FilePath || PBLC.second > Offset) {
    return false;
  }
  auto PELC = dpct::DpctGlobalInfo::getInstance().getLocInfo(PE);
  if (PELC.first != FilePath || Offset > PELC.second) {
    return false;
  }
  return true;
}

SourceLocation getLocInRange(SourceLocation Loc, SourceRange Range) {
  auto &SM = dpct::DpctGlobalInfo::getSourceManager();
  auto BeginCandidate = Loc;
  if (Loc.isMacroID() &&
      !isInRange(Range.getBegin(), Range.getEnd(), BeginCandidate)) {
    // Try getImmediateSpellingLoc
    // e.g. M1(call(M2))
    BeginCandidate = SM.getImmediateSpellingLoc(Loc);
    if (BeginCandidate.isMacroID()) {
      BeginCandidate = SM.getExpansionLoc(BeginCandidate);
    }
    if (!isInRange(Range.getBegin(), Range.getEnd(), BeginCandidate)) {
      // Try getImmediateExpansionRange
      // e.g. #define M1(x) call(x)
      BeginCandidate =
          SM.getSpellingLoc(SM.getImmediateExpansionRange(Loc).getBegin());
      if (!isInRange(Range.getBegin(), Range.getEnd(), BeginCandidate)) {
        BeginCandidate = SM.getSpellingLoc(Loc);
      }
    }
  }
  return BeginCandidate;
}

std::pair<SourceLocation, SourceLocation>
getRangeInRange(const Stmt *E, SourceLocation RangeBegin,
                SourceLocation RangeEnd, bool IncludeLastToken) {
  return getRangeInRange(E->getSourceRange(), RangeBegin, RangeEnd,
                         IncludeLastToken);
}

void traversePossibleLocations(const SourceLocation &SL,
                               const SourceLocation &RangeBegin,
                               const SourceLocation &RangeEnd,
                               SourceLocation &Result, bool IsBegin) {
  auto &SM = dpct::DpctGlobalInfo::getSourceManager();
  if (!SL.isValid())
    return;

  if (!SL.isMacroID()) {
    if (isInRange(RangeBegin, RangeEnd, SL)) {
      if (Result.isValid()) {
        if (IsBegin && SM.getDecomposedLoc(Result).second <
                           SM.getDecomposedLoc(SL).second) {
          Result = SL;
        } else if (!IsBegin && SM.getDecomposedLoc(Result).second >
                                   SM.getDecomposedLoc(SL).second) {
          Result = SL;
        } else {
          return;
        }
      }
      Result = SL;
    }
    return;
  }

  if (IsBegin) {
    traversePossibleLocations(SM.getImmediateExpansionRange(SL).getBegin(),
                              RangeBegin, RangeEnd, Result, IsBegin);
  } else {
    traversePossibleLocations(SM.getImmediateExpansionRange(SL).getEnd(),
                              RangeBegin, RangeEnd, Result, IsBegin);
  }
  traversePossibleLocations(SM.getImmediateSpellingLoc(SL), RangeBegin,
                            RangeEnd, Result, IsBegin);
}

std::pair<SourceLocation, SourceLocation>
getRangeInRange(SourceRange Range, SourceLocation SearchRangeBegin,
                SourceLocation SearchRangeEnd, bool IncludeLastToken) {
  auto &SM = dpct::DpctGlobalInfo::getSourceManager();
  auto &Context = dpct::DpctGlobalInfo::getContext();
  SourceLocation ResultBegin = SourceLocation();
  SourceLocation ResultEnd = SourceLocation();
  traversePossibleLocations(Range.getBegin(), SearchRangeBegin, SearchRangeEnd,
                            ResultBegin, true);
  traversePossibleLocations(Range.getEnd(), SearchRangeBegin, SearchRangeEnd,
                            ResultEnd, false);
  if (ResultBegin.isValid() && ResultEnd.isValid()) {
    if (isSameLocation(ResultBegin, ResultEnd)) {
      auto It = dpct::DpctGlobalInfo::getExpansionRangeBeginMap().find(
          getCombinedStrFromLoc(ResultBegin));
      if(It != dpct::DpctGlobalInfo::getExpansionRangeBeginMap().end()){
        // If the begin/end loc are at the same location
        // and the loc is another macro expand,
        // recursively search for a more precise range.
        auto MacroDefBegin = It->second.getBegin();
        auto MacroDefEnd = It->second.getEnd();
        auto MacroDefEndTokenLength =
            Lexer::MeasureTokenLength(MacroDefEnd, SM, Context.getLangOpts());
        MacroDefEnd = MacroDefEnd.getLocWithOffset(MacroDefEndTokenLength);
        auto InnerResult = getRangeInRange(Range, MacroDefBegin, MacroDefEnd, false);
        // If the new range covers the entire macro, use the original range,
        // otherwise, use the inner range.
        if (isInRange(It->second.getBegin(), It->second.getEnd(),
                      InnerResult.first) &&
            isInRange(It->second.getBegin(), It->second.getEnd(),
                      InnerResult.second) &&
            (!isSameLocation(It->second.getBegin(), InnerResult.first) ||
             !isSameLocation(It->second.getEnd(), InnerResult.second))) {
          ResultBegin = InnerResult.first;
          ResultEnd = InnerResult.second;
        }
      }
    }
    ResultBegin = SM.getExpansionLoc(ResultBegin);
    ResultEnd = SM.getExpansionLoc(ResultEnd);
    if (IncludeLastToken) {
      auto LastTokenLength =
          Lexer::MeasureTokenLength(ResultEnd, SM, Context.getLangOpts());
      ResultEnd = ResultEnd.getLocWithOffset(LastTokenLength);
    }
    return std::pair<SourceLocation, SourceLocation>(ResultBegin, ResultEnd);
  }
  return std::pair<SourceLocation, SourceLocation>(Range.getBegin(),
                                                   Range.getEnd());
}

unsigned int calculateIndentWidth(const CUDAKernelCallExpr *Node,
                                  clang::SourceLocation SL, bool &Flag) {
  Flag = true;
  if (!Node)
    return dpct::DpctGlobalInfo::getCodeFormatStyle().IndentWidth;

  auto &Context = dpct::DpctGlobalInfo::getContext();
  auto &SM = dpct::DpctGlobalInfo::getSourceManager();
  std::string IndentStr = getIndent(SL, SM).str();
  unsigned int Len = 0;
  for (const auto &C : IndentStr) {
    if (C == '\t')
      Len = Len + dpct::DpctGlobalInfo::getCodeFormatStyle().TabWidth;
    else
      Len++;
  }

  unsigned int CompoundStmtCounter = 0;
  clang::DynTypedNodeList Parents = Context.getParents(*Node);
  while (!Parents.empty()) {
    auto Cur = Parents[0];

    if (Cur.get<clang::CompoundStmt>()) {
      if (!Context.getParents(Cur).empty() &&
          !Context.getParents(Cur)[0].get<clang::IfStmt>() &&
          !Context.getParents(Cur)[0].get<clang::ForStmt>() &&
          !Context.getParents(Cur)[0].get<clang::WhileStmt>()) {
        CompoundStmtCounter++;
      }
    } else if (Cur.get<clang::IfStmt>() || Cur.get<clang::ForStmt>() ||
               Cur.get<clang::WhileStmt>()) {
      CompoundStmtCounter++;
    }

    Parents = Context.getParents(Cur);
  }

  unsigned int Result = 0;
  if (CompoundStmtCounter) {
    Result = Len / CompoundStmtCounter;
    if (Len % CompoundStmtCounter != 0)
      Flag = false;
  } else {
    Result = Len;
  }

  return Result == 0 ? dpct::DpctGlobalInfo::getCodeFormatStyle().IndentWidth
                     : Result;
}

bool isIncludedFile(const std::string &CurrentFile,
                    const std::string &CheckingFile) {
  auto CurrentFileInfo =
      dpct::DpctGlobalInfo::getInstance().insertFile(CurrentFile);
  auto CheckingFileInfo =
      dpct::DpctGlobalInfo::getInstance().insertFile(CheckingFile);

  std::deque<std::shared_ptr<dpct::DpctFileInfo>> Q(
      CurrentFileInfo->getIncludedFilesInfoSet().begin(),
      CurrentFileInfo->getIncludedFilesInfoSet().end());

  std::unordered_set<std::shared_ptr<dpct::DpctFileInfo>> InsertedFile;
  InsertedFile = CurrentFileInfo->getIncludedFilesInfoSet();

  while (!Q.empty()) {
    if (Q.front() == nullptr) {
      Q.pop_front();
      continue;
    } else if (Q.front() == CheckingFileInfo) {
      return true;
    } else {
      for (const auto &IncludeFile : Q.front()->getIncludedFilesInfoSet()) {
        if (InsertedFile.find(IncludeFile) == InsertedFile.end()) {
          Q.insert(Q.end(), IncludeFile);
          InsertedFile.insert(IncludeFile);
        }
      }
      Q.pop_front();
    }
  }
  return false;
}

std::string getCombinedStrFromLoc(const clang::SourceLocation Loc) {
  auto &SM = dpct::DpctGlobalInfo::getSourceManager();
  if (SM.isWrittenInScratchSpace(Loc)) {
    return Loc.printToString(SM);
  }
  auto LocInfo = dpct::DpctGlobalInfo::getLocInfo(Loc);
  return LocInfo.first + ":" + std::to_string(LocInfo.second);
}

std::string getFinalCastTypeNameStr(std::string CastTypeName) {
  // Since curandState and other state types have same prefix (e.g.,
  // curandStateXORWOW_t), we need choose a result which matches longest.
  std::map<size_t /*replaced length*/,
           std::pair<std::string::size_type /*BeginLoc*/,
                     std::string /*replacing text*/>,
           std::greater<size_t>>
      ReplaceLengthStringMap;

  for (auto &Pair : MapNames::DeviceRandomGeneratorTypeMap) {
    std::string::size_type BeginLoc = CastTypeName.find(Pair.first);
    if (BeginLoc != std::string::npos) {
      ReplaceLengthStringMap.insert(std::make_pair(
          Pair.first.size(), std::make_pair(BeginLoc, Pair.second)));
    }
  }

  if (!ReplaceLengthStringMap.empty()) {
    const auto BeginIter = ReplaceLengthStringMap.begin();
    CastTypeName.replace(BeginIter->second.first, BeginIter->first,
                         BeginIter->second.second);
  }
  return CastTypeName;
}

bool isLexicallyInLocalScope(const clang::Decl *D) {
  const DeclContext *LDC = D->getLexicalDeclContext();
  while (true) {
    if (LDC->isFunctionOrMethod())
      return true;
    if (!isa<TagDecl>(LDC))
      return false;
    if (const auto *CRD = dyn_cast<CXXRecordDecl>(LDC))
      if (CRD->isLambda())
        return true;
    LDC = LDC->getLexicalParent();
  }
  return false;
}

const DeclaratorDecl *getHandleVar(const Expr *Arg) {
  const Expr *NoImpCastE = nullptr;
  if (auto UO = dyn_cast<UnaryOperator>(Arg->IgnoreImpCasts())) {
    if (UO->getOpcode() == UO_AddrOf) {
      NoImpCastE = UO->getSubExpr()->IgnoreImpCasts();
    }
  } else {
    NoImpCastE = Arg->IgnoreImpCasts();
  }
  if (!NoImpCastE)
    return nullptr;

  if (auto ASE = dyn_cast<ArraySubscriptExpr>(NoImpCastE))
    NoImpCastE = ASE->getBase()->IgnoreImpCasts();

  if (auto DeclRef = dyn_cast<DeclRefExpr>(NoImpCastE)) {
    if (dyn_cast<VarDecl>(DeclRef->getDecl()))
      return dyn_cast<DeclaratorDecl>(DeclRef->getDecl());
  } else if (auto Member = dyn_cast<MemberExpr>(NoImpCastE)) {
    if (dyn_cast<FieldDecl>(Member->getMemberDecl()))
      return dyn_cast<DeclaratorDecl>(Member->getMemberDecl());
  }
  return nullptr;
}

clang::RecordDecl *getRecordDecl(clang::QualType QT) {
  clang::QualType PointeeType;

  if (const auto *PT = QT->getAs<clang::PointerType>())
    PointeeType = PT->getPointeeType();
  else if (const auto *RT = QT->getAs<clang::ReferenceType>())
    PointeeType = RT->getPointeeType();
  else
    return QT->getAsRecordDecl();

  if (const auto *RT = PointeeType->getAs<clang::RecordType>())
    return dyn_cast<clang::RecordDecl>(RT->getDecl());

  return nullptr;
}

bool checkPointerInStructRecursively(const RecordDecl *R) {
  std::deque<const clang::RecordDecl *> Q;
  Q.push_back(R);
  while (!Q.empty()) {
    if (Q.front() == nullptr) {
      Q.pop_front();
      continue;
    } else {
      for (const auto &I : Q.front()->fields()) {
        if (I->getType()->isPointerType()) {
          return true;
        }
      }
      for (const auto &I : Q.front()->decls()) {
        if (const clang::RecordDecl *RD = dyn_cast<clang::RecordDecl>(I)) {
          Q.insert(Q.end(), RD);
        }
      }
      Q.pop_front();
    }
  }
  return false;
}

bool checkPointerInStructRecursively(const clang::DeclRefExpr *DRE) {
  return checkPointerInStructRecursively(getRecordDecl(DRE->getType()));
}

SourceLocation getImmSpellingLocRecursive(const SourceLocation Loc) {
  auto &SM = dpct::DpctGlobalInfo::getSourceManager();
  if (SM.isMacroArgExpansion(Loc) &&
      SM.isMacroArgExpansion(SM.getImmediateSpellingLoc(Loc))) {
    return getImmSpellingLocRecursive(SM.getImmediateSpellingLoc(Loc));
  }
  return Loc;
}

bool getTypeRange(const clang::VarDecl *PVD, clang::SourceRange &SR) {
  auto &SM = dpct::DpctGlobalInfo::getSourceManager();
  auto BeginLoc = SM.getExpansionLoc(PVD->getBeginLoc());
  auto EndLoc = SM.getExpansionLoc(PVD->getEndLoc());
  std::string IdentifyName = PVD->getName().str();

  auto TokenBegin = BeginLoc;
  Token Tok;
  while (SM.getFileOffset(TokenBegin) <= SM.getFileOffset(EndLoc)) {
    if (Lexer::getRawToken(TokenBegin, Tok, SM,
                           dpct::DpctGlobalInfo::getContext().getLangOpts(),
                           true)) {
      break;
    }
    if (Tok.isAnyIdentifier()) {
      if (Tok.getRawIdentifier().str() == IdentifyName) {
        SR = clang::SourceRange(BeginLoc, TokenBegin);
        return true;
      }
    }
    TokenBegin = Tok.getEndLoc();
  }

  return false;
}

llvm::StringRef getCalleeName(const CallExpr *CE) {
  auto &SM = dpct::DpctGlobalInfo::getSourceManager();
  const char *Start = SM.getCharacterData(
      getStmtExpansionSourceRange(CE->getCallee()).getBegin());
  const char *End = Start;
  int StrSize = 0;
  while (((int)(*End) >= (int)'A' && (int)(*End) <= (int)'Z') ||
         ((int)(*End) >= (int)'a' && (int)(*End) <= (int)'z') ||
         ((int)(*End) >= (int)'0' && (int)(*End) <= (int)':') ||
         (int)(*End) == (int)'_') {
    StrSize++;
    End++;
  }
  return llvm::StringRef(Start, StrSize);
}

// Usually used to find a correct removing range.
// Return the source range which is not straddle and mostly
// close to the spelling location.
// Let Begin/End are the begin/end location of pow(1, 2)
// #define CALL(x) x
// #define FUNC_NAME pow
// #define ARGS (1, 2)
// #define ALL FUNC_NAME ARGS
// ex 1. CALL(CALL(CALL(FUNC_NAME ARGS)))
// Result: Range of "FUNC_NAME ARGS"
// ex 2. CALL(pow(1, 2))
// Result: Range of "pow(1, 2)"
// ex 3. CALL(ALL)
// Result: Range of "FUNC_NAME ARGS" (in the definition of "ALL")
SourceRange getDefinitionRange(SourceLocation Begin, SourceLocation End) {
  auto &SM = dpct::DpctGlobalInfo::getSourceManager();

  // if one of begin/end location is FileID, the only valid range is the
  // expansion location.
  // e.g.
  // #define REAL_NAME(x) x
  // #define FUNC_NAME REAL_NAME(foo) FUNC_NAME(args);
  if (Begin.isFileID() || End.isFileID()) {
    return SourceRange(SM.getExpansionLoc(Begin), SM.getExpansionLoc(End));
  }

  // Remove the outer func-like macro
  // ex. CALL(CALL(CALL(FUNC_NAME ARGS)));
  // the following while will skip the 3 CALL()
  SourceLocation PreBegin = Begin;
  SourceLocation PreEnd = End;

  while (SM.isMacroArgExpansion(Begin) && SM.isMacroArgExpansion(End)) {
    PreBegin = Begin;
    PreEnd = End;
    Begin = SM.getImmediateSpellingLoc(Begin);
    End = SM.getImmediateSpellingLoc(End);
  }

  // if there is still either one of begin/end is macro arg expansion
  if (SM.isMacroArgExpansion(Begin) || SM.isMacroArgExpansion(End)) {
    // In cases like CALL(FUNC_NAME CALL(ARGS))
    // If the Begin location is always the 1st token of macro defines,
    // it's safe to use the expansion location as the begin location.
    // ex.
    // #define FOO foo
    // #define FUNC_NAME FOO
    bool BeginIsAlwaysTheFirstToken = true;
    if (SM.isMacroArgExpansion(Begin)) {
      auto NextBegin = Begin;
      while (NextBegin.isMacroID()) {
        NextBegin = SM.getImmediateExpansionRange(NextBegin).getBegin();
        auto It = dpct::DpctGlobalInfo::getExpansionRangeToMacroRecord().find(
            getCombinedStrFromLoc(SM.getSpellingLoc(NextBegin)));
        if (It !=
            dpct::DpctGlobalInfo::getExpansionRangeToMacroRecord().end()) {
          auto TokenIndex = It->second->TokenIndex;
          if (TokenIndex != 0)
            BeginIsAlwaysTheFirstToken = false;
        }
      }
    }
    // No precise range available which can be removed without delete extra
    // syntax.
    // ex. CALL(FUNC_NAME CALL(ARGS))
    // After the above while loop, begin/end will be the Range of "FUNC_NAME
    // CALL(ARGS)". However, FUNC_NAME and CALL() may contain syntax which are
    // not belong to the CallExpr.
    // ex.
    // #define FUNC_NAME a = pow
    // #define CALL(x) x;
    // The "a =" and ";" will be removed if we use the range "FUNC_NAME
    // CALL(ARGS)" therefore, return a range with length 0 for the caller of
    // getDefinitionRange() to handle the exception.
    if (!BeginIsAlwaysTheFirstToken)
      return SourceRange(SM.getSpellingLoc(Begin), SM.getSpellingLoc(Begin));
  }

  // If the begin/end are not in the same macro arg, no precise range available.
  // Using PreBegin/PreEnd because they contain the info of the last func-like
  // macro.
  if (!isLocInSameMacroArg(PreBegin, PreEnd)) {
    return SourceRange(SM.getSpellingLoc(Begin), SM.getSpellingLoc(Begin));
  }

  std::tie(Begin, End) = getTheLastCompleteImmediateRange(Begin, End);

  return SourceRange(Begin, End);
}

bool isLocInSameMacroArg(SourceLocation Begin, SourceLocation End) {
  auto &SM = dpct::DpctGlobalInfo::getSourceManager();
  // Both Begin/End are not macro arg, treat as they are in same macro arg
  if (!SM.isMacroArgExpansion(Begin) && !SM.isMacroArgExpansion(End)) {
    return true;
  }
  if (SM.isMacroArgExpansion(Begin) && SM.isMacroArgExpansion(End)) {
    // Check if begin/end are the same macro arg, so use getBegin() for
    // both loc
    Begin = SM.getSpellingLoc(SM.getImmediateExpansionRange(Begin).getBegin());
    End = SM.getSpellingLoc(SM.getImmediateExpansionRange(End).getBegin());
    auto ItMatchBegin =
        dpct::DpctGlobalInfo::getExpansionRangeToMacroRecord().find(
            getCombinedStrFromLoc(Begin));
    auto ItMatchEnd =
        dpct::DpctGlobalInfo::getExpansionRangeToMacroRecord().find(
            getCombinedStrFromLoc(End));
    if (ItMatchBegin !=
            dpct::DpctGlobalInfo::getExpansionRangeToMacroRecord().end() &&
        ItMatchEnd !=
            dpct::DpctGlobalInfo::getExpansionRangeToMacroRecord().end() &&
        ItMatchBegin == ItMatchEnd) {
      // The whole kernel call is in a single macro arg
      return true;
    }
  }
  return false;
}

const CompoundStmt *
findTheOuterMostCompoundStmtUntilMeetControlFlowNodes(const CallExpr *CE) {
  const CompoundStmt *LatestCS = nullptr;
  if (!CE)
    return LatestCS;

  auto &Context = dpct::DpctGlobalInfo::getContext();
  auto Parents = Context.getParents(*CE);
  const Stmt *LastStmt = dyn_cast<Stmt>(CE);
  while (Parents.size() > 0) {
    auto *Parent = Parents[0].get<Stmt>();
    if (Parent) {
      if (Parent->getStmtClass() == Stmt::StmtClass::CompoundStmtClass) {
        LatestCS = dyn_cast<CompoundStmt>(Parent);
      } else if (Parent->getStmtClass() == Stmt::StmtClass::DoStmtClass) {
        const DoStmt *DS = dyn_cast<DoStmt>(Parent);
        if (!DS)
          break;
        const Expr *Cond = DS->getCond();
        Expr::EvalResult ER;
        if (!Cond->isTypeDependent() && !Cond->isValueDependent() &&
            Cond->EvaluateAsInt(ER, dpct::DpctGlobalInfo::getContext())) {
          int64_t Value = ER.Val.getInt().getExtValue();
          // If the Cond is 0, it means this Do-stmt just execute once
          if (Value != 0) {
            break;
          }
        }
      } else if (Parent->getStmtClass() == Stmt::StmtClass::IfStmtClass) {
        const IfStmt *IS = dyn_cast<IfStmt>(Parent);
        // If the node is cond, it means node just execute once
        if (IS->getCond() != LastStmt) {
          break;
        }
      }

      LastStmt = Parent;
      Parents = Context.getParents(*Parent);
    } else {
      // It means Parent[0] is Decl
      Parents = Context.getParents(Parents[0]);
    }
  }

  return LatestCS;
}

bool isInMacroDefinition(SourceLocation BeginLoc, SourceLocation EndLoc) {
  auto Range = getDefinitionRange(BeginLoc, EndLoc);
  auto ItBegin = dpct::DpctGlobalInfo::getExpansionRangeToMacroRecord().find(
      getCombinedStrFromLoc(Range.getBegin()));
  if (ItBegin == dpct::DpctGlobalInfo::getExpansionRangeToMacroRecord().end()) {
    return false;
  }
  return true;
}

bool isPartOfMacroDef(SourceLocation BeginLoc, SourceLocation EndLoc) {
  if (dpct::DpctGlobalInfo::getSourceManager().isWrittenInScratchSpace(BeginLoc))
    return false;
  auto ItBegin = dpct::DpctGlobalInfo::getExpansionRangeToMacroRecord().find(
      getCombinedStrFromLoc(BeginLoc));
  auto ItEnd = dpct::DpctGlobalInfo::getExpansionRangeToMacroRecord().find(
      getCombinedStrFromLoc(EndLoc));
  // If any of begin/end is not in the macro or the begin is not the 1st token
  // or the end is not the last macro
  if (ItBegin == dpct::DpctGlobalInfo::getExpansionRangeToMacroRecord().end() ||
      ItEnd == dpct::DpctGlobalInfo::getExpansionRangeToMacroRecord().end() ||
      ItBegin->second->TokenIndex != 0 ||
      ItEnd->second->TokenIndex != ItEnd->second->NumTokens - 1) {
    return true;
  }
  return false;
}

// This function will construct some union-find sets when traverse the
// call-graph of device/global functions.
// E.g.,
// g1->d1->d2
// g2->d3->d2
//
// This function will be called for each global function.
// 1st, visit g1, check its child d1, d1 is not visited, so set d1's parent ptr
// value to g1's head (current is g1 itself).
// Now the set becomes to:
// g1<-d1
// 2nd, visit d1, check its child d2, d2 is not visited, so set d2's parent ptr
// value to d1's head (current is g1).
// Now the set becomes to:
// g1<-d1
//  |<-d2
// 3nd, visit d2, it does not have child, so return.
// 4th, visit g2, check its child d3, d3 is not visited, so set d3's parent ptr
// value to g2's head (current is g2 itself).
// Now the set becomes to:
// g1<-d1  g2<-d3
//  |<-d2
// 5th, visit d3, check its child d2, d2 is visited, so set d3's head node's
// parent ptr value to d2's head.
// Now the set becomes to:
// g1<-d1
//  |<-d2
//  |<-g2<-d3
//
// Finally, all nodes can be represented by their head node. If we want to
// change or get the field value of a node, we need to change or get the field
// value of the node's head. In this function, we will change or get the field
// "Dim" (the value is used in the dimension of nd_item/nd_range).
// In this example, there is only one head node g1.
void constructUnionFindSetRecursively(
    std::shared_ptr<dpct::DeviceFunctionInfo> DFIPtr) {
  if (!DFIPtr)
    return;

  if (DFIPtr->ConstructGraphVisited)
    return;

  auto CallExprMap = DFIPtr->getCallExprMap();
  DFIPtr->ConstructGraphVisited = true;

  dpct::MemVarMap *CurHead = dpct::MemVarMap::getHead(&(DFIPtr->getVarMap()));
  if (!CurHead)
    return;

  std::vector<std::shared_ptr<dpct::DeviceFunctionInfo>> RelatedDFI;
  for (auto &Item : CallExprMap) {
    auto FuncInfoPtr = Item.second->getFuncInfo();
    if (!FuncInfoPtr)
      continue;
    RelatedDFI.push_back(FuncInfoPtr);
  }
  auto RelatedDFIFromSpellingLoc =
      dpct::DpctGlobalInfo::getDFIVecRelatedFromSpellingLoc(DFIPtr);
  RelatedDFI.insert(RelatedDFI.end(), RelatedDFIFromSpellingLoc.begin(),
                    RelatedDFIFromSpellingLoc.end());

  for (auto &FuncInfoPtr : RelatedDFI) {
    if (!FuncInfoPtr)
      continue;
    if (FuncInfoPtr == DFIPtr)
      continue;
    if (FuncInfoPtr->getVarMap().hasItem() && DFIPtr->getVarMap().hasItem()) {
      dpct::MemVarMap *HeadOfTheChild =
          dpct::MemVarMap::getHead(&(FuncInfoPtr->getVarMap()));
      if (!HeadOfTheChild)
        continue;
      if (FuncInfoPtr->ConstructGraphVisited) {
        // Update dim
        HeadOfTheChild->Dim = std::max(HeadOfTheChild->Dim, CurHead->Dim);
        // Update head node
        CurHead->Parent = HeadOfTheChild;
        CurHead = CurHead->Parent;
      } else {
        // Update dim
        CurHead->Dim = std::max(CurHead->Dim, HeadOfTheChild->Dim);
        // Update head node
        HeadOfTheChild->Parent = CurHead;
      }
    }

    constructUnionFindSetRecursively(FuncInfoPtr);
  }
}

// To find if device variable \pExpr has __share__ attribute,
// if it has, HasSharedAttr is set true.
// if \pExpr is in an if/while/do while/for statement
// \pNeedReport is set true.
// To handle six kinds of cases:
// case1: extern __shared__ uint32_t share_array[];
//        atomicAdd(&share_array[0], 1);
// case2: extern __shared__ uint32_t share_array[];
//        uint32_t *p = &share_array[0];
//        atomicAdd(p, 1);
// case3: __shared__ uint32_t share_v;
//        atomicAdd(&share_v, 1);
// case4: __shared__ uint32_t share_v;
//        uint32_t *p = &share_v;
//        atomicAdd(p, 1);
// case5: extern __shared__ uint32_t share_array[];
//        atomicAdd(share_array, 1);
// case6: __shared__ uint32_t share_v;
//        uint32_t *p;
//        p = &share_v;
//        atomicAdd(p, 1);
void getShareAttrRecursive(const Expr *Expr, bool &HasSharedAttr,
                           bool &NeedReport) {
  if (!Expr)
    return;

  if (dyn_cast_or_null<CallExpr>(Expr)) {
    NeedReport = true;
    return;
  }

  if (auto UO = dyn_cast_or_null<UnaryOperator>(Expr)) {
    if (UO->getOpcode() == UnaryOperatorKind::UO_AddrOf) {
      Expr = UO->getSubExpr();
    }
  }

  if (auto BO = dyn_cast_or_null<BinaryOperator>(Expr)) {
    getShareAttrRecursive(BO->getLHS(), HasSharedAttr, NeedReport);
    getShareAttrRecursive(BO->getRHS(), HasSharedAttr, NeedReport);
  }

  if (auto ASE = dyn_cast_or_null<ArraySubscriptExpr>(Expr)) {
    Expr = ASE->getBase();
  }

  if (auto CSCE = dyn_cast_or_null<CStyleCastExpr>(Expr)) {
    Expr = CSCE->getSubExpr();
  }

  const clang::Expr *AssignedExpr = NULL;
  const FunctionDecl *FuncDecl = NULL;
  if (auto DRE = dyn_cast_or_null<DeclRefExpr>(
          Expr->IgnoreImplicitAsWritten()->IgnoreParens())) {
    if (isa<ParmVarDecl>(DRE->getDecl())) {
      NeedReport = true;
      return;
    }

    if (auto VD = dyn_cast_or_null<VarDecl>(DRE->getDecl())) {
      if (VD->hasAttr<CUDASharedAttr>()) {
        HasSharedAttr = true;
        return;
      }

      AssignedExpr = VD->getInit();
      if (FuncDecl = dyn_cast_or_null<FunctionDecl>(VD->getDeclContext())) {
        std::vector<const DeclRefExpr *> Refs;
        VarReferencedInFD(FuncDecl->getBody(), VD, Refs);
        for (auto const &Ref : Refs) {
          if (Ref == DRE)
            break;

          if (auto BO = dyn_cast_or_null<BinaryOperator>(getParentStmt(Ref))) {
            if (BO->getLHS() == Ref && BO->getOpcode() == BO_Assign &&
                !clang::dpct::DpctGlobalInfo::checkSpecificBO(DRE, BO))
              AssignedExpr = BO->getRHS();
          }
        }
      }
    }
  } else if (auto BO = dyn_cast_or_null<BinaryOperator>(
                 Expr->IgnoreImplicitAsWritten()->IgnoreParens())) {

    getShareAttrRecursive(BO->getLHS(), HasSharedAttr, NeedReport);
    getShareAttrRecursive(BO->getRHS(), HasSharedAttr, NeedReport);
  }

  if (AssignedExpr) {
    // if AssignedExpr in a if/while/do while/for statement,
    // it is necessary to report a warning message.
    if (isInCtrlFlowStmt(AssignedExpr, FuncDecl,
                         dpct::DpctGlobalInfo::getContext())) {
      NeedReport = true;
    }
    getShareAttrRecursive(AssignedExpr, HasSharedAttr, NeedReport);
  }
}

llvm::SmallVector<clang::ast_matchers::BoundNodes, 1U>
findDREInScope(const clang::Stmt *Scope,
               const std::vector<std::string> &IgnoreTypes) {
  using namespace clang::ast_matchers;
  if (IgnoreTypes.empty()) {
    auto VarReferenceMatcher = findAll(declRefExpr().bind("VarReference"));
    return match(VarReferenceMatcher, *Scope,
                 clang::dpct::DpctGlobalInfo::getContext());
  }
  auto VarReferenceWithIgnoreTypesMatcher =
      findAll(declRefExpr(unless(to(varDecl(internal::Matcher<NamedDecl>(
                              new internal::HasNameMatcher(IgnoreTypes))))))
                  .bind("VarReference"));
  return match(VarReferenceWithIgnoreTypesMatcher, *Scope,
               clang::dpct::DpctGlobalInfo::getContext());
}

/// Find all the DRE sub-expression of \p E
/// \param [in] E The input expression
/// \param [out] DRESet The DREs which are found by this function
/// \param [out] HasCallExpr The flag means if there is CallExpr in \p E
/// \param [in] IgnoreTypes Ignore DREs with these type name
void findDREs(const Expr *E, std::set<const clang::DeclRefExpr *> &DRESet,
              bool &HasCallExpr, const std::vector<std::string> &IgnoreTypes) {
  if (!E)
    return;

  auto DREResults = findDREInScope(E);
  for (auto &Result : DREResults) {
    const DeclRefExpr *MatchedDRE =
        Result.getNodeAs<DeclRefExpr>("VarReference");
    if (!MatchedDRE)
      continue;
    DRESet.insert(MatchedDRE);
  }

  auto CallExprMatcher = clang::ast_matchers::findAll(
      clang::ast_matchers::callExpr().bind("CallExpr"));
  auto CEResults = clang::ast_matchers::match(
      CallExprMatcher, *E, clang::dpct::DpctGlobalInfo::getContext());
  for (auto &Result : CEResults) {
    const CallExpr *MatchedCE = Result.getNodeAs<CallExpr>("CallExpr");
    if (MatchedCE) {
      HasCallExpr = true;
      return;
    }
  }
}

void checkDREIsPrivate(const DeclRefExpr *DRE, LocalVarAddrSpaceEnum &Result) {
  Result = LocalVarAddrSpaceEnum::AS_CannotDeduce;
  if (const ParmVarDecl *PVD = dyn_cast_or_null<ParmVarDecl>(DRE->getDecl())) {
    if (const clang::FunctionDecl *FD =
            dyn_cast_or_null<FunctionDecl>(PVD->getParentFunctionOrMethod())) {
      if (FD->hasAttr<CUDAGlobalAttr>()) {
        Result = LocalVarAddrSpaceEnum::AS_IsGlobal;
        return;
      } else if (FD->hasAttr<CUDADeviceAttr>()) {
        return;
      }
    }
    return;
  }

  const clang::VarDecl *VD = dyn_cast_or_null<VarDecl>(DRE->getDecl());
  if (!VD)
    return;

  if (!VD->getType()->isReferenceType() && !VD->getType()->isPointerType()) {
    // If the type is neither a reference nor a pointer, treat it as a local
    // variable, its address space is private.
    Result = LocalVarAddrSpaceEnum::AS_IsPrivate;
    return;
  }

  const clang::FunctionDecl *FuncDecl =
      dyn_cast_or_null<FunctionDecl>(VD->getDeclContext());
  if (!FuncDecl)
    return;

  const clang::CompoundStmt *CS =
      dyn_cast_or_null<CompoundStmt>(FuncDecl->getBody());
  if (!CS)
    return;

  // Using ASTMatcher to find all DRE in current CompoundStmt, if that DRE's
  // declaration is same as VD, push that DRE into Refs. The DRE in Refs will be
  // checked in the next step.
  auto MatchedResults = findDREInScope(CS);
  std::vector<const DeclRefExpr *> Refs;
  for (auto &Result : MatchedResults) {
    const DeclRefExpr *DRE = Result.getNodeAs<DeclRefExpr>("VarReference");
    if (!DRE)
      continue;
    if (DRE->getDecl() == VD) {
      Refs.push_back(DRE);
    }
  }

  LocalVarAddrSpaceEnum LastAssignmentResult =
      LocalVarAddrSpaceEnum::AS_CannotDeduce;
  if (VD->hasInit()) {
    LocalVarAddrSpaceEnum InitExprResult =
        LocalVarAddrSpaceEnum::AS_CannotDeduce;
    checkIsPrivateVar(VD->getInit(), InitExprResult);
    LastAssignmentResult = InitExprResult;
  }
  bool CanLastReferenceBeDeduced = false;
  // Check each DRE before current statement, try to deduce their address space
  // Currently only check the assignment like "var1 = ExprContainsDRE;"
  // If the DRE is in control flow statement, then its address cannot be
  // deduced. Else, call checkIsPrivateVar() to the right hand side Expr's
  // address space. Finally, using the address space of the last DRE as result.
  for (auto const &Ref : Refs) {
    // Break means tool will not check the assignments after the DRE
    if (Ref == DRE)
      break;

    if (auto BO = dyn_cast_or_null<BinaryOperator>(getParentStmt(Ref))) {
      if (BO->getLHS() == Ref && BO->getOpcode() == BO_Assign &&
          !clang::dpct::DpctGlobalInfo::checkSpecificBO(DRE, BO)) {
        if (isInCtrlFlowStmt(BO->getRHS(), FuncDecl,
                             dpct::DpctGlobalInfo::getContext())) {
          LastAssignmentResult = LocalVarAddrSpaceEnum::AS_CannotDeduce;
          CanLastReferenceBeDeduced = false;
        } else {
          LocalVarAddrSpaceEnum SubResult =
              LocalVarAddrSpaceEnum::AS_CannotDeduce;
          checkIsPrivateVar(BO->getRHS(), SubResult);
          LastAssignmentResult = SubResult;
          CanLastReferenceBeDeduced = true;
        }
      }
    } else {
      CanLastReferenceBeDeduced = false;
    }
  }

  if (!CanLastReferenceBeDeduced)
    LastAssignmentResult = LocalVarAddrSpaceEnum::AS_CannotDeduce;
  if (LastAssignmentResult == LocalVarAddrSpaceEnum::AS_CannotDeduce)
    return;

  Result = LastAssignmentResult;
  return;
}

// This function will check the address space of the input argument "Expr"
// Step1: find all DeclRefExpr in the "Expr"
// Step2: get each DeclRefExpr's address space
// Step3: merge the results
void checkIsPrivateVar(const Expr *Expr, LocalVarAddrSpaceEnum &Result) {
  Result = LocalVarAddrSpaceEnum::AS_CannotDeduce;
  bool HasCallExpr = false;
  std::set<const clang::DeclRefExpr *> DRESet;
  findDREs(Expr, DRESet, HasCallExpr);
  if (HasCallExpr)
    return;
  std::set<LocalVarAddrSpaceEnum> ResultSet;
  for (const auto &DRE : DRESet) {
    LocalVarAddrSpaceEnum ThisDREResult =
        LocalVarAddrSpaceEnum::AS_CannotDeduce;
    checkDREIsPrivate(DRE, ThisDREResult);
    ResultSet.insert(ThisDREResult);
  }

  if (ResultSet.count(LocalVarAddrSpaceEnum::AS_CannotDeduce))
    return;

  if (ResultSet.size() == 1) {
    Result = *ResultSet.begin();
    return;
  }
}

/// Determine whether a variable represented by DeclRefExpr is unmodified
/// 1. func(..., T Val(pass by value), ...)
/// 2. ... = Val
/// 3. { ...
///      Val;
///      ...}
/// The variable is unmodified in above cases
/// \param [in] DRE Input DeclRefExpr
/// \returns If variable not modified, return false
bool isModifiedRef(const clang::DeclRefExpr *DRE) {
  auto &CT = dpct::DpctGlobalInfo::getContext();
  const clang::Stmt *P = CT.getParents(*DRE)[0].get<ImplicitCastExpr>();
  if (!P) {
    P = DRE;
  }
  if (auto CE = CT.getParents(*P)[0].get<CallExpr>()) {
    int index, ArgNum = CE->getNumArgs();
    for (index = 0; index < ArgNum; index++) {
      if (CE->getArg(index)->IgnoreImplicit() == DRE)
        break;
    }
    if (index == ArgNum)
      return true;
    if (auto CalleeDecl = CE->getDirectCallee()) {
      auto ParaDecl = CalleeDecl->getParamDecl(index);
      if (ParaDecl && !ParaDecl->getType()->isReferenceType()) {
        return false;
      }
    }
  } else if (auto BO = CT.getParents(*P)[0].get<BinaryOperator>()) {
    if (BO->getRHS()->IgnoreImplicit() == DRE)
      return false;
  } else if (CT.getParents(*P)[0].get<CompoundStmt>()) {
    return false;
  }
  return true;
}

bool isDefaultStream(const Expr *StreamArg) {
  StreamArg = StreamArg->IgnoreCasts();
  if (isa<CXXNullPtrLiteralExpr>(StreamArg) || isa<GNUNullExpr>(StreamArg)) {
    return true;
  } else if (auto DAE = dyn_cast<CXXDefaultArgExpr>(StreamArg)) {
    return isDefaultStream(DAE->getExpr());
  } else if (auto Paren = dyn_cast<ParenExpr>(StreamArg)) {
    return isDefaultStream(Paren->getSubExpr());
  }
  Expr::EvalResult Result{};
  if (!StreamArg->isValueDependent() &&
      StreamArg->EvaluateAsInt(Result, dpct::DpctGlobalInfo::getContext())) {
    // 0 or 1 (cudaStreamLegacy) or 2 (cudaStreamPerThread)
    // all migrated to default queue;
    return Result.Val.getInt().getZExtValue() < 3;
  }
  return false;
}

const NamedDecl *getNamedDecl(const clang::Type *TypePtr) {
  const NamedDecl *ND = nullptr;
  if (!TypePtr)
    return ND;
  if (TypePtr->isRecordType()) {
    ND = TypePtr->castAs<clang::RecordType>()->getDecl();
  } else if (TypePtr->isEnumeralType()) {
    ND = TypePtr->castAs<clang::EnumType>()->getDecl();
  } else if (TypePtr->getTypeClass() == clang::Type::Typedef) {
    ND = TypePtr->castAs<clang::TypedefType>()->getDecl();
  } else if (TypePtr->getTypeClass() == clang::Type::ConstantArray) {
    if (auto Array = TypePtr->getAsArrayTypeUnsafe()) {
      ND =
          getNamedDecl(Array->getElementType().getCanonicalType().getTypePtr());
    }
  }
  return ND;
}
bool isTypeInAnalysisScope(const clang::Type *TypePtr) {
  bool IsInAnalysisScope = false;
  if (const auto *ND = getNamedDecl(TypePtr)) {
    auto Loc = ND->getBeginLoc();
    if (dpct::DpctGlobalInfo::isInAnalysisScope(Loc))
      IsInAnalysisScope = true;
  }
  return IsInAnalysisScope;
}

/// This function will find all assignments to the DRE of \p HandleDecl in
/// the range of \p CS.
/// The result is returned by \p Refs.
void findAssignments(const clang::DeclaratorDecl *HandleDecl,
                     const clang::CompoundStmt *CS,
                     std::vector<const clang::DeclRefExpr *> &Refs) {
  if (!HandleDecl)
    return;
  auto MatchedResults = findDREInScope(CS);

  for (auto &Result : MatchedResults) {
    const DeclRefExpr *DRE = Result.getNodeAs<DeclRefExpr>("VarReference");
    if (!DRE)
      continue;
    if (DRE->getDecl() == HandleDecl) {
      if (auto BO = dyn_cast_or_null<BinaryOperator>(getParentStmt(DRE))) {
        if (BO->getLHS() == DRE && BO->getOpcode() == BO_Assign) {
          // case1: handle = another_handle;
          Refs.push_back(DRE);
          continue;
        }
      }

      auto FunctionCall =
          clang::dpct::DpctGlobalInfo::findAncestor<CallExpr>(DRE);
      if (!FunctionCall)
        continue;
      auto Expr =
          clang::dpct::DpctGlobalInfo::getChildExprOfTargetAncestor<CallExpr>(
              DRE);
      if (!Expr)
        continue;

      const auto FunctionDecl = FunctionCall->getDirectCallee();
      if (!FunctionDecl)
        continue;

      for (size_t Idx = 0, ArgNum = FunctionCall->getNumArgs(); Idx < ArgNum;
           ++Idx) {
        if (FunctionCall->getArg(Idx) == Expr) {
          // case2: void foo(handle_t &h);
          //        foo(handle);
          if (FunctionDecl->getParamDecl(Idx)->getType()->isReferenceType() ||
              FunctionDecl->getParamDecl(Idx)->getType()->isPointerType()) {
            Refs.push_back(DRE);
            continue;
          }
        }
      }
    }
  }
}

/// Find the flow control ancestor node for \p S
/// If \p RangeLimit is not nullptr, find flow control node in the sub-tree of
/// \p RangeLimit
/// If \p RangeLimit is nullptr, find flow control node in all ancestors of
/// \p S Return nullptr if no flow control found in the range.
/// Otherwise it will return the flow control node.
const clang::Stmt *
getAncestorFlowControl(const clang::Stmt *S,
                       const clang::CompoundStmt *RangeLimit = nullptr) {
  if (!S)
    return nullptr;

  auto &Context = dpct::DpctGlobalInfo::getContext();
  auto Parents = Context.getParents(*S);
  while (Parents.size() >= 1) {
    auto *Parent = Parents[0].get<Stmt>();
    if (Parent) {
      if (Parent == RangeLimit)
        return nullptr;
      auto StmtClass = Parent->getStmtClass();
      if (StmtClass == Stmt::StmtClass::WhileStmtClass ||
          StmtClass == Stmt::StmtClass::ForStmtClass ||
          StmtClass == Stmt::StmtClass::IfStmtClass ||
          StmtClass == Stmt::StmtClass::SwitchStmtClass ||
          StmtClass == Stmt::StmtClass::CallExprClass) {
        return Parent;
      } else if (StmtClass == Stmt::StmtClass::DoStmtClass) {
        const Expr *CondExpr = dyn_cast<DoStmt>(Parent)->getCond();
        Expr::EvalResult ER;
        if (!CondExpr->isValueDependent() &&
            CondExpr->EvaluateAsInt(ER, Context)) {
          int64_t Value = ER.Val.getInt().getExtValue();
          if (Value != 0) {
            return Parent;
          }
        }
      }
    }
    Parents = Context.getParents(Parents[0]);
  }

  return nullptr;
}

bool isAncestorOf(const Stmt *Descendant, const Stmt *Ancestor) {
  if (!Descendant)
    return false;

  auto &Context = dpct::DpctGlobalInfo::getContext();
  auto Parents = Context.getParents(*Descendant);
  while (Parents.size() >= 1) {
    auto *Parent = Parents[0].get<Stmt>();
    if (Parent) {
      if (Parent == Ancestor)
        return true;
    }
    Parents = Context.getParents(Parents[0]);
  }

  return false;
}

/// This function first finds the nearest flow control stmt in the
/// ancestors of \p S, then returns the body (CompoundStmt) of that
/// flow control stmt
const clang::Stmt *getBodyofAncestorFCStmt(const clang::Stmt *S) {
  auto FlowControlStmt = getAncestorFlowControl(S, nullptr);
  if (!FlowControlStmt)
    return nullptr;
  auto StmtClass = FlowControlStmt->getStmtClass();
  const clang::Stmt *CS = nullptr;
  switch (StmtClass) {
  case Stmt::StmtClass::WhileStmtClass: {
    CS = dyn_cast<WhileStmt>(FlowControlStmt)->getBody();
    break;
  }
  case Stmt::StmtClass::ForStmtClass: {
    CS = dyn_cast<ForStmt>(FlowControlStmt)->getBody();
    break;
  }
  case Stmt::StmtClass::DoStmtClass: {
    CS = dyn_cast<DoStmt>(FlowControlStmt)->getBody();
    break;
  }
  case Stmt::StmtClass::IfStmtClass: {
    if (isAncestorOf(S, dyn_cast<IfStmt>(FlowControlStmt)->getThen())) {
      CS = dyn_cast<IfStmt>(FlowControlStmt)->getThen();
    } else {
      CS = dyn_cast<IfStmt>(FlowControlStmt)->getElse();
    }
    break;
  }
  case Stmt::StmtClass::SwitchStmtClass:
    return nullptr;
  case Stmt::StmtClass::CallExprClass:
    return nullptr;
  default:
    break;
  }
  return CS;
}

bool analyzeMemcpyOrder(
    const clang::CompoundStmt *CS,
    std::vector<std::pair<const Stmt *, MemcpyOrderAnalysisNodeKind>>
        &MemcpyOrderVec,
    std::vector<unsigned int> &DREOffsetVec) {
  using namespace clang::ast_matchers;
  const static std::unordered_set<std::string> SpecialFuncNameSet = {
      "printf", "cudaGetErrorString", "exit", "cudaDeviceSynchronize"};

  std::set<const clang::Expr *> SrcExprs;
  std::map<const clang::Expr *, bool> DstExprsMap;
  std::set<const clang::Expr *> ExcludeExprs;
  std::set<const clang::Expr *> MemcpyCallExprs;

  // 1. Find all CallExprs in this scope. If there is any CallExpr
  //    between two memcpy() API calls(except the kernel call on default stream).
  //    The wait() must be added after the previous memcpy() call.
  auto CallExprMatcher = findAll(callExpr().bind("CallExpr"));
  auto MatchedResults =
      match(CallExprMatcher, *CS, clang::dpct::DpctGlobalInfo::getContext());
  for (auto &Result : MatchedResults) {
    const CallExpr *CE = Result.getNodeAs<CallExpr>("CallExpr");
    if (!CE)
      return false;
    std::string FuncName = "";
    if (CE->getDirectCallee()) {
      FuncName = CE->getDirectCallee()->getNameInfo().getName().getAsString();
    } else {
      if (auto ULE = dyn_cast_or_null<UnresolvedLookupExpr>(CE->getCallee())) {
        FuncName = ULE->getNameInfo().getAsString();
      }
    }
    if (FuncName.empty())
      return false;

    ExcludeExprs.insert(CE->getCallee());

    if (FuncName == "cudaMemcpy" || FuncName == "cudaMemcpyFromSymbol" ||
        FuncName == "cudaMemcpyToSymbol") {
      MemcpyCallExprs.insert(CE);
      if (getAncestorFlowControl(CE, CS)) {
        MemcpyOrderVec.emplace_back(
            CE, MemcpyOrderAnalysisNodeKind::MOANK_MemcpyInFlowControl);
      } else {
        // Record the first and second argument of memcpy
        int DirectionArgIndex = 4;
        if (FuncName == "cudaMemcpy") {
          DirectionArgIndex = 3;
        }
        if (auto Direction =
                dyn_cast<DeclRefExpr>(CE->getArg(DirectionArgIndex))) {
          auto CpyKind = Direction->getDecl()->getName();
          if (CpyKind == "cudaMemcpyDeviceToHost" ||
              CpyKind == "cudaMemcpyHostToHost") {
            DstExprsMap.insert({CE->getArg(0), true});
          } else {
            DstExprsMap.insert({CE->getArg(0), false});
          }
        } else {
          DstExprsMap.insert({CE->getArg(0), true});
        }
        SrcExprs.insert(CE->getArg(1));
        ExcludeExprs.insert(CE->getArg(0));
        ExcludeExprs.insert(CE->getArg(1));
        MemcpyOrderVec.emplace_back(CE,
                                    MemcpyOrderAnalysisNodeKind::MOANK_Memcpy);
      }
    } else if (SpecialFuncNameSet.count(FuncName)) {
      MemcpyOrderVec.emplace_back(
          CE, MemcpyOrderAnalysisNodeKind::MOANK_SpecialCallExpr);
    } else {
      const CUDAKernelCallExpr* KCall = dyn_cast<CUDAKernelCallExpr>(CE);
      if(!KCall) {
        KCall = dyn_cast_or_null<CUDAKernelCallExpr>(getParentStmt(CE));
      }
      if(KCall) {
        const CallExpr *Config = KCall->getConfig();
        if (Config) {
          // Record the pointer DRE used as argument of kernel call on default
          // stream in ExcludeExprs. Because those pointers are accessed on device
          // and on default stream, the q.memcpy before this kernel call don't
          // need wait for those DREs.   
          if ((Config->getNumArgs() == 4) &&
               isDefaultStream(Config->getArg(3))) {
            for(auto Arg: KCall->arguments()) {
              if (auto DRE = dyn_cast_or_null<DeclRefExpr>(
                      Arg->IgnoreImplicitAsWritten())) {
                if (DRE->getType()->isPointerType()) {
                  ExcludeExprs.insert(DRE);
                }
              }
            }
            MemcpyOrderVec.emplace_back(
                CE, MemcpyOrderAnalysisNodeKind::MOANK_KernelCallExpr);
            continue;
          }
        }
      }
      MemcpyOrderVec.emplace_back(
          CE, MemcpyOrderAnalysisNodeKind::MOANK_OtherCallExpr);
    }
  }

  std::set<const clang::DeclRefExpr *> AllDREsInCS;
  auto Results = findDREInScope(CS, {"cudaError_t"});
  for (auto &Result : Results) {
    const DeclRefExpr *MatchedDRE =
        Result.getNodeAs<DeclRefExpr>("VarReference");
    if (!MatchedDRE)
      continue;
    AllDREsInCS.insert(MatchedDRE);
  }

  // Find all DREs related with the first and the second arguments
  // of the memcpy APIs. If there is any related DRE between two
  // memcpy calls(except the pointer DRE used as arguments of kernel
  // call on default stream), wait() must be added after the previous memcpy.
  // The method to find all related DREs:
  // Find all memcpy APIs in this scope, insert the DREs in the first
  // and second arguments of those APIs into the DRE set.
  // Find DREs in the scope, if the DRE's declaration is not a local variable,
  // insert the declaration into VD set.
  // do {
  //   1. Add the declarations of DREs in the DRE set into VD set.
  //   2. Find all DREs in this scope related to the declarations in VD set.
  //   3. Add the DRE which is related to the DRE in step2 into DRE set.
  // } while (DRE set is changed or VD set is changed);
  std::set<const clang::DeclRefExpr *> SrcDRESet;
  std::set<const clang::DeclRefExpr *> DstDRESet;
  std::set<const clang::DeclRefExpr *> ExcludeDRESet;
  std::set<const clang::ValueDecl *> DstVDSet;
  std::set<const clang::ValueDecl *> VDSet;
  std::set<const clang::ValueDecl *> NewVDSet;
  std::set<const clang::DeclRefExpr *> NewDRESet;
  std::set<const clang::ValueDecl *> ProcessedVDSet;
  std::set<const void *> ProcessedExprOrDecl;
  bool HasCallExpr = true;
  for (const auto &E : SrcExprs)
    findDREs(E, SrcDRESet, HasCallExpr, {"cudaError_t"});
  for (const auto &E : DstExprsMap) {
    DstDRESet.clear();
    findDREs(E.first, DstDRESet, HasCallExpr, {"cudaError_t"});
    for (const auto &DRE : DstDRESet) {
      if (auto D = dyn_cast_or_null<clang::VarDecl>(DRE->getDecl())) {
        NewVDSet.insert(D);
        if (E.second) {
          DstVDSet.insert(D);
        }
      }
    }
  }
  for (const auto &E : ExcludeExprs)
    findDREs(E, ExcludeDRESet, HasCallExpr, {"cudaError_t"});
  for (const auto &DRE : AllDREsInCS) {
    if (DRE->getDecl()->getKind() == clang::Decl::Kind::EnumConstant) {
      ExcludeDRESet.insert(DRE);
    }
  }
  for (const auto &DRE : SrcDRESet) {
    if (auto D = dyn_cast_or_null<clang::VarDecl>(DRE->getDecl())) {
      NewVDSet.insert(D);
    }
  }
  for (const auto &DRE : AllDREsInCS) {
    if (ExcludeDRESet.count(DRE))
      continue;
    if (const clang::VarDecl *VD =
            dyn_cast_or_null<clang::VarDecl>(DRE->getDecl())) {
      if (!VD->isLocalVarDecl()) {
        NewVDSet.insert(VD);
      }
    }
  }
  do {
    VDSet = NewVDSet;
    NewVDSet.clear();
    ProcessedVDSet.insert(VDSet.begin(), VDSet.end());
    for (const auto &DRE : AllDREsInCS) {
      if (!DRE->getDecl() || !VDSet.count(DRE->getDecl()))
        continue;
      const clang::Expr *ExprScope = nullptr;
      const clang::ValueDecl *DeclScope = nullptr;
      getTheOuterMostExprOrValueDecl(DRE, ExprScope, DeclScope);
      llvm::SmallVector<BoundNodes, 1U> Results;
      if (ExprScope) {
        if (ProcessedExprOrDecl.count(ExprScope)) {
          continue;
        }
        ProcessedExprOrDecl.insert(ExprScope);
        Results = findDREInScope(ExprScope);
      } else if (DeclScope) {
        if (ProcessedExprOrDecl.count(DeclScope)) {
          continue;
        }
        ProcessedExprOrDecl.insert(DeclScope);
        if (!ProcessedVDSet.count(DeclScope)) {
          if (const clang::VarDecl *VarD = dyn_cast<VarDecl>(DeclScope)) {
            if (!VarD->hasInit() ||
                !MemcpyCallExprs.count(VarD->getInit()->IgnoreImplicit())) {
              NewVDSet.insert(DeclScope);
            }
          }
        }
        auto VarReferenceMatcher = valueDecl(forEachDescendant(
            declRefExpr(unless(to(varDecl(hasAnyName("cudaError_t")))))
                .bind("VarReference")));
        Results = match(VarReferenceMatcher, *DeclScope,
                        clang::dpct::DpctGlobalInfo::getContext());
      }
      for (auto &Result : Results) {
        const DeclRefExpr *MatchedDRE =
            Result.getNodeAs<DeclRefExpr>("VarReference");
        if (!MatchedDRE) {
          continue;
        }
        auto D = dyn_cast_or_null<clang::VarDecl>(MatchedDRE->getDecl());
        if (!D) {
          continue;
        }
        bool IsGlobalVar = !D->isLocalVarDecl();
        if (auto ParentCE = clang::dpct::DpctGlobalInfo::findAncestor<CallExpr>(
                MatchedDRE)) {
          if (auto Callee = ParentCE->getDirectCallee()) {
            if (!IsGlobalVar && !DstVDSet.count(D) &&
                dpct::DpctGlobalInfo::isInCudaPath(Callee->getBeginLoc())) {
              continue;
            }
          }
        }
        if (auto ImpCastExpr = dpct::DpctGlobalInfo::getContext()
                                   .getParents(*MatchedDRE)[0]
                                   .get<ImplicitCastExpr>()) {
          if ((ImpCastExpr->getCastKind() == CastKind::CK_LValueToRValue) &&
              !IsGlobalVar && !DstVDSet.count(D)) {
            bool IsDerefOrArraySubExpr = false;
            if (dpct::DpctGlobalInfo::getContext()
                    .getParents(*ImpCastExpr)[0]
                    .get<ArraySubscriptExpr>()) {
              IsDerefOrArraySubExpr = true;
            } else if (auto UO = dpct::DpctGlobalInfo::getContext()
                                     .getParents(*ImpCastExpr)[0]
                                     .get<UnaryOperator>()) {
              if (UO->getOpcode() == UnaryOperatorKind::UO_Deref) {
                IsDerefOrArraySubExpr = true;
              }
            }
            if (!IsDerefOrArraySubExpr) {
              continue;
            }
          }
        } else if (auto BO = clang::dpct::DpctGlobalInfo::findAncestor<
                       BinaryOperator>(MatchedDRE)) {
          if ((BO->getOpcode() == BO_Assign) &&
              (MatchedDRE == dyn_cast_or_null<DeclRefExpr>(
                                 BO->getLHS()->IgnoreImplicitAsWritten()))) {
            if (auto RHSCE = dyn_cast_or_null<CallExpr>(
                    BO->getRHS()->IgnoreImplicitAsWritten())) {
              if (auto Callee = RHSCE->getDirectCallee()) {
                dpct::DpctGlobalInfo::isInCudaPath(Callee->getBeginLoc());
                continue;
              }
            }
          }
        }
        NewDRESet.insert(MatchedDRE);
        if (!ProcessedVDSet.count(D)) {
          NewVDSet.insert(D);
        }
      }
    }
    for (const auto &NewDRE : NewDRESet) {
      if (ExcludeDRESet.count(NewDRE))
        continue;
      auto Offset =
          clang::dpct::DpctGlobalInfo::getLocInfo(NewDRE->getBeginLoc()).second;
      DREOffsetVec.push_back(Offset);
    }
    NewDRESet.clear();
  } while (!NewVDSet.empty());
  std::sort(DREOffsetVec.begin(), DREOffsetVec.end());
  return true;
}

/// This function is used to check if the ".wait()" can be omitted in the
/// migrated code of cudaMemcpy/cudaMemcpyFromSymbol/cudaMemcpyToSymbol API.
/// Rule:
/// This function analyze the code in five scopes respectively (The CompoundStmt
/// of WhileStmt/ForStmt/DoStmt/IfStmt/FuncDecl). The analysis result of the
/// outer scope will not affect the inner scope.
///
/// Traverse all CallExpr nodes between current cudaMemcpy CallExpr and
/// the next cudaMemcpy CallExpr with pre-order.
/// If (1) all these CallExpr nodes are in the SpecialFuncNameSet(e.g.,
/// printf()), and (2) current and next cudaMemcpy are not in flow control
/// Stmt, then the ".wait()" for current cudaMemcpy can be omitted.
///
/// E.g.,
/// \code
/// void foo() {
///   cudaMemcpy();
///   cudaMemcpy();
///   while (true) {
///     cudaMemcpy();
///     cudaMemcpy();
///     if (true) {
///       cudaMemcpy();
///       cudaMemcpy();
///     }
///   }
/// }
/// \endcode
/// will be migrated to
/// \code
/// void foo() {
///   q_ct1.emecpy();
///   q_ct1.emecpy().wait();
///   while (true) {
///     q_ct1.emecpy();
///     q_ct1.emecpy().wait();
///     if (true) {
///       q_ct1.emecpy();
///       q_ct1.emecpy().wait();
///     }
///   }
/// }
/// \endcode
bool canOmitMemcpyWait(const clang::CallExpr *CE) {
  if (!CE)
    return false;

  const clang::CompoundStmt *CS =
      dyn_cast_or_null<clang::CompoundStmt>(getBodyofAncestorFCStmt(CE));
  if (!CS) {
    auto FD = clang::dpct::DpctGlobalInfo::findAncestor<FunctionDecl>(CE);
    if (!FD)
      return false;
    CS = dyn_cast_or_null<CompoundStmt>(FD->getBody());
  }

  if (!CS)
    return false;

  // The cudaMemcpy function typically operates synchronously. However, when
  // copying from host to device using pageable host memory, its behavior
  // becomes asynchronous. If the --optimize-migration option is used during
  // migration, the migration tool assumes host memory is pageable and migrates
  // cudaMemcpy into an asynchronous memcpy from host to device, which can
  // improve performance by permitting concurrent memory transfer with other
  // task. However, the cudaMemcpy copies content of host memory to the staging
  // memory for DMA transfer to device memory before return.
  // In the following 3 cases, the content of host memory may change. Therefore,
  // the migration tool still migrates cudaMemcpy into a synchronous memcpy to
  // ensure the copy behavior is correct:
  // 1. The cudaMemcpy called within a control flow statement, which is usually
  // accompanied by an overwrite of host memory.
  // E.g.,
  // for(int i = 0; i < 10; i++) {
  //   *host_src = i;
  //   cudaMemcpy(device_dst, host_src, sizeof(int), cudaMemcpyHostToDevice);
  // }
  //
  // 2. The host pointer freed after cudaMemcpy with no cudaDeviceSynchronize()
  // between them.
  // E.g.,
  // cudaMemcpy(device_dst, host_src, sizeof(int), cudaMemcpyHostToDevice);
  // free(host_src);
  //
  // 3. The address-of operation is used in src expression, an error may occur 
  // if temporary variables are used. The temporary variable may become unavailable
  // before the copy is completed.
  // E.g.,
  // void test() {
  //   int data;
  //   ...
  //   cudaMemcpy(device_dst, &data, sizeof(int), cudaMemcpyHostToDevice);
  // }
  if (auto Direction = dyn_cast<DeclRefExpr>(CE->getArg(3))) {
    auto CpyKind = Direction->getDecl()->getName();
    if (CpyKind == "cudaMemcpyDeviceToDevice") {
      return true;
    }
    if (CpyKind == "cudaMemcpyHostToDevice" &&
        dpct::DpctGlobalInfo::isOptimizeMigration()) {
      if (auto Body = getBodyofAncestorFCStmt(CE)) {
        if (dpct::DpctGlobalInfo::isAncestor(Body, CE)) {
          return false;
        }
      }
      auto SrcExpr = CE->getArg(1);
      auto AddrOfMatcher =
          clang::ast_matchers::findAll(clang::ast_matchers::unaryOperator(
              clang::ast_matchers::hasOperatorName("&")));
      auto AddrOfMatchedResults = clang::ast_matchers::match(
          AddrOfMatcher, *SrcExpr, clang::dpct::DpctGlobalInfo::getContext());
      if (AddrOfMatchedResults.size() == 0) {
        auto SyncPointMatcher = clang::ast_matchers::findAll(
            clang::ast_matchers::callExpr(
                clang::ast_matchers::callee(clang::ast_matchers::functionDecl(
                    clang::ast_matchers::hasAnyName("cudaDeviceSynchronize"))))
                .bind("SyncPoint"));
        auto CEBegLocInfo = dpct::DpctGlobalInfo::getLocInfo(CE->getBeginLoc());
        auto CEEndLocInfo = dpct::DpctGlobalInfo::getLocInfo(CE->getEndLoc());
        std::set<const clang::DeclRefExpr *> DRESet;
        bool HasCallExpr = false;
        bool isSrcPointerFreedAfterCE = false;
        std::vector<const clang::DeclRefExpr *> DREMatchResult;
        std::set<const clang::DeclRefExpr *> SrcDRESet;
        std::set<unsigned int> SyncPointOffset;
        auto SyncPointMatchedResults = clang::ast_matchers::match(
            SyncPointMatcher, *CS, clang::dpct::DpctGlobalInfo::getContext());
        for (auto &SP : SyncPointMatchedResults) {
          if (const CallExpr *SPCE = SP.getNodeAs<CallExpr>("SyncPoint")) {
            if (auto Body = getBodyofAncestorFCStmt(SPCE)) {
              if (dpct::DpctGlobalInfo::isAncestor(Body, SPCE)) {
                continue;
              }
            }
            SyncPointOffset.insert(
                dpct::DpctGlobalInfo::getLocInfo(SPCE->getBeginLoc()).second);
          }
        }
        auto checkIfSrcPointerFreedAfterCE = [&]() {
          for (auto &D : DREMatchResult) {
            if (auto ParentCE =
                    clang::dpct::DpctGlobalInfo::findAncestor<CallExpr>(D)) {
              auto DC = ParentCE->getDirectCallee();
              if (!DC) {
                continue;
              }
              std::string FuncName = getFunctionName(DC);
              auto DRELocInfo =
                  dpct::DpctGlobalInfo::getLocInfo(D->getEndLoc());
              if ((FuncName == "free" || FuncName == "cudaFreeHost") &&
                  (DRELocInfo.second > CEEndLocInfo.second)) {
                bool FreeAfterSyncPoint = false;
                for (auto &Offset : SyncPointOffset) {
                  if ((Offset > CEEndLocInfo.second) &&
                      (Offset < DRELocInfo.second)) {
                    FreeAfterSyncPoint = true;
                    break;
                  }
                }
                if (!FreeAfterSyncPoint) {
                  return true;
                }
              }
            }
          }
          return false;
        };
        findDREs(SrcExpr, SrcDRESet, HasCallExpr);
        for (auto &SrcDRE : SrcDRESet) {
          findAllVarRef(SrcDRE, DREMatchResult);
          if (isSrcPointerFreedAfterCE = checkIfSrcPointerFreedAfterCE()) {
            break;
          }
          DREMatchResult.clear();
        }
        if (!isSrcPointerFreedAfterCE) {
          clang::dpct::DiagnosticsUtils::report(
              CEBegLocInfo.first, CEBegLocInfo.second,
              clang::dpct::Diagnostics::WAIT_REMOVE, true, false);
          return true;
        }
      }
    }
  }

  auto &SM = clang::dpct::DpctGlobalInfo::getSourceManager();
  auto CSLocInfo = clang::dpct::DpctGlobalInfo::getLocInfo(
      SM.getExpansionLoc(CS->getBeginLoc()));
  auto FileInfo =
      clang::dpct::DpctGlobalInfo::getInstance().insertFile(CSLocInfo.first);
  auto &Map = FileInfo->getMemcpyOrderAnalysisResultMap();
  auto Iter = Map.find(CS);

  std::vector<std::pair<const Stmt *, MemcpyOrderAnalysisNodeKind>>
      MemcpyOrderVec;
  std::vector<unsigned int> DREOffsetVec;
  if (Iter == Map.end()) {
    if (!analyzeMemcpyOrder(CS, MemcpyOrderVec, DREOffsetVec))
      return false;
    Map.insert(std::make_pair(
        CS, dpct::MemcpyOrderAnalysisInfo(MemcpyOrderVec, DREOffsetVec)));
  } else {
    MemcpyOrderVec = Iter->second.MemcpyOrderVec;
    DREOffsetVec = Iter->second.DREOffsetVec;
  }

  bool StartSearch = false;

  for (const auto &S : MemcpyOrderVec) {
    if (S.first == CE) {
      if (S.second == MemcpyOrderAnalysisNodeKind::MOANK_MemcpyInFlowControl)
        return false;
      StartSearch = true;
      continue;
    }
    if (StartSearch) {
      if (S.second == MemcpyOrderAnalysisNodeKind::MOANK_OtherCallExpr) {
        if (!dpct::DpctGlobalInfo::isAncestor(CE, S.first)) {
          return false;
        }
      }
      if (S.second == MemcpyOrderAnalysisNodeKind::MOANK_MemcpyInFlowControl) {
        return false;
      }
      if (S.second == MemcpyOrderAnalysisNodeKind::MOANK_Memcpy) {
        unsigned int CurrentCallExprEndOffset =
            clang::dpct::DpctGlobalInfo::getLocInfo(CE->getEndLoc()).second;
        unsigned int NextCallExprEndOffset =
            clang::dpct::DpctGlobalInfo::getLocInfo(S.first->getEndLoc())
                .second;
        auto FirstDREAfterCurrentCallExprEndLoc = std::lower_bound(
            DREOffsetVec.begin(), DREOffsetVec.end(), CurrentCallExprEndOffset);
        if (FirstDREAfterCurrentCallExprEndLoc == DREOffsetVec.end())
          return true;
        if (*FirstDREAfterCurrentCallExprEndLoc <= NextCallExprEndOffset)
          return false;

        return true;
      }
    }
  }

  return false;
}
/// Check if \p E contains a sizeof(Type) sub-expression
/// \param [in] E The input expression
/// \returns The check result
bool containSizeOfType(const Expr *E) {
  auto SizeOfMatcher = clang::ast_matchers::findAll(
      clang::ast_matchers::unaryExprOrTypeTraitExpr(
          clang::ast_matchers::ofKind(UETT_SizeOf))
          .bind("sizeof"));
  auto MatchedResults = clang::ast_matchers::match(
      SizeOfMatcher, *E, clang::dpct::DpctGlobalInfo::getContext());
  for (const auto &Res : MatchedResults) {
    const UnaryExprOrTypeTraitExpr *UETTE =
        Res.getNodeAs<UnaryExprOrTypeTraitExpr>("sizeof");
    if (!UETTE)
      continue;
    if (UETTE->isArgumentType())
      return true;
  }
  return false;
}

void findRelatedAssignmentRHS(const clang::DeclRefExpr *DRE,
                              std::set<const clang::Expr *> &RHSSet) {
  auto VD = dyn_cast_or_null<VarDecl>(DRE->getDecl());
  if (!VD)
    return;
  auto FD = clang::dpct::DpctGlobalInfo::findAncestor<FunctionDecl>(DRE);
  if (!FD)
    return;
  auto CS = dyn_cast_or_null<CompoundStmt>(FD->getBody());
  if (!CS)
    return;

  auto VarReferenceMatcher = clang::ast_matchers::findAll(
      clang::ast_matchers::declRefExpr().bind("VarReference"));
  auto MatchedResults = clang::ast_matchers::match(
      VarReferenceMatcher, *CS, clang::dpct::DpctGlobalInfo::getContext());
  std::vector<const DeclRefExpr *> Refs;
  for (auto &Result : MatchedResults) {
    const DeclRefExpr *MatchedDRE =
        Result.getNodeAs<DeclRefExpr>("VarReference");
    if (!MatchedDRE)
      continue;
    if (MatchedDRE->getDecl() == VD) {
      // Break means tool will not check the assignments after the DRE
      if (MatchedDRE == DRE)
        break;
      Refs.push_back(MatchedDRE);
    }
  }

  if (VD->hasInit()) {
    RHSSet.insert(VD->getInit());
  }
  for (auto const &Ref : Refs) {
    if (auto BO = dyn_cast_or_null<BinaryOperator>(getParentStmt(Ref))) {
      if (BO->getLHS() == Ref && BO->getOpcode() == BO_Assign &&
          !clang::dpct::DpctGlobalInfo::checkSpecificBO(DRE, BO)) {
        RHSSet.insert(BO->getRHS());
      }
    }
  }
}

/// Check if \p E and all expressions relative to \p E contain a sizeof(Type)
/// sub-expression
/// E.g., this function will check expression: a1, a2, b1, b2
/// int a1 = b1;
/// int a2 = b2;
/// __shared__ float mem[a1 + a2];
/// \param [in] E The input expression
/// \param [out] ExprContainSizeofType The expression contains a sizeof(Type)
/// \returns The check result
bool checkIfContainSizeofTypeRecursively(
    const clang::Expr *E, const clang::Expr *&ExprContainSizeofType) {
  if (containSizeOfType(E)) {
    ExprContainSizeofType = E;
    return true;
  }

  bool HasCallExpr = false;
  std::set<const clang::DeclRefExpr *> DRESet;
  findDREs(E, DRESet, HasCallExpr);
  for (const auto &DRE : DRESet) {
    std::set<const clang::Expr *> RHSSet;
    findRelatedAssignmentRHS(DRE, RHSSet);
    for (const auto &RHS : RHSSet) {
      if (checkIfContainSizeofTypeRecursively(RHS, ExprContainSizeofType))
        return true;
    }
  }
  return false;
}

bool maybeDependentCubType(const clang::TypeSourceInfo *TInfo) {
  QualType CanType = TInfo->getType().getCanonicalType();
  auto *CanTyPtr = CanType.split().Ty;

  // template <int THREADS_PER_BLOCK>
  //  __global__ void Kernel() {
  //  1.  typedef cub::BlockScan<int, THREADS_PER_BLOCK> BlockScan;
  //  2.  __shared__ typename BlockScan::TempStorage temp1;
  //  }

  // Handle 1
  auto isCubRecordType = [&](const clang::Type *T) -> bool {
    if (auto *SpecType = dyn_cast<TemplateSpecializationType>(T)) {
      auto *TemplateDecl = SpecType->getTemplateName().getAsTemplateDecl();
      auto *Ctx = TemplateDecl->getDeclContext();
      auto *CubNS = dyn_cast<NamespaceDecl>(Ctx);
      while (CubNS) {
        if (CubNS->isInlineNamespace()) {
          CubNS = dyn_cast<NamespaceDecl>(CubNS->getDeclContext());
          continue;
        }
        break;
      }
      return CubNS && CubNS->getCanonicalDecl()->getName() == "cub";
    }
    return false;
  };

  // Handle 2
  if (const auto *DNT = dyn_cast<DependentNameType>(CanTyPtr)) {
    auto *QNNS = DNT->getQualifier();
    // *::TempStorage must be a type.
    if (QNNS->getKind() == NestedNameSpecifier::TypeSpec) {
      auto *QNNSType = QNNS->getAsType();
      return isCubRecordType(QNNSType);
    }
  }

  return isCubRecordType(CanTyPtr);
}

bool isCubVar(const VarDecl *VD) {
  QualType CanType = VD->getType().getCanonicalType();
  std::string CanonicalTypeStr = CanType.getAsString();
  // 1.process non-template case
  if (!isTypeInAnalysisScope(VD->getType().getCanonicalType().getTypePtr()) &&
      (CanonicalTypeStr.find("struct cub::") == 0 ||
       CanonicalTypeStr.find("class cub::") == 0)) {
    return true;
  }
  // 2.process template cases
  if (CanonicalTypeStr.find("::TempStorage") != std::string::npos) {
    if (maybeDependentCubType(VD->getTypeSourceInfo()))
      return true;

    std::string TypeParameterName;
    auto findTypeParameterName = [&](DependentNameTypeLoc DNT) {
      std::string Name;
      if (auto NNS = DNT.getQualifierLoc().getNestedNameSpecifier()) {
        if (auto Type = NNS->getAsType()) {
          Name = QualType(Type, 0).getAsString();
        }
      }
      return Name;
    };
    if (auto CAT = VD->getTypeSourceInfo()
                       ->getTypeLoc()
                       .getAs<ConstantArrayTypeLoc>()) {
      auto DNT = CAT.getElementLoc().getAs<DependentNameTypeLoc>();
      TypeParameterName = findTypeParameterName(DNT);
    } else if (auto DNT = VD->getTypeSourceInfo()
                              ->getTypeLoc()
                              .getAs<DependentNameTypeLoc>()) {
      TypeParameterName = findTypeParameterName(DNT);
    }
    // if type not template parameter and not include cub::
    // then we return false
    if (TypeParameterName.empty()) {
      return false;
    }
    auto DeviceFuncDecl =
        clang::dpct::DpctGlobalInfo::findAncestor<FunctionDecl>(VD);
    if (!DeviceFuncDecl)
      return false;
    std::string DeviceFuncName = DeviceFuncDecl->getNameAsString();
    auto FuncTemp = DeviceFuncDecl->getDescribedFunctionTemplate();
    if (!FuncTemp) {
      return false;
    }
    auto TempPara = FuncTemp->getTemplateParameters();
    int index = -1;
    for (auto Itr = TempPara->begin(); Itr != TempPara->end(); Itr++) {
      if ((*Itr)->getNameAsString() == TypeParameterName) {
        index = Itr - TempPara->begin();
        break;
      }
    }
    if (index == -1) {
      return false;
    }
    auto isCub = [&](const Expr *Callee) {
      auto DRE = dyn_cast<DeclRefExpr>(Callee);
      if (!DRE) {
        return false;
      }
      auto TemplateArgsList = DRE->template_arguments();
      if (TemplateArgsList.size() < (unsigned)index) {
        return false;
      }
      if (TemplateArgsList[index].getArgument().getKind() ==
          TemplateArgument::ArgKind::Type) {
        auto CanonicalType = TemplateArgsList[index]
                                 .getArgument()
                                 .getAsType()
                                 .getCanonicalType();
        std::string ArgTy = CanonicalType.getAsString();
        if (!isTypeInAnalysisScope(CanonicalType.getTypePtr()) &&
            ArgTy.find("class cub::") == 0) {
          return true;
        }
      }
      return false;
    };
    if (DeviceFuncDecl->hasAttr<CUDADeviceAttr>()) {
      auto DevCallMatcher = ast_matchers::callExpr(
                                ast_matchers::callee(ast_matchers::functionDecl(
                                    ast_matchers::hasName(DeviceFuncName))))
                                .bind("devcall");
      auto DevCallMatchResult = ast_matchers::match(
          DevCallMatcher, clang::dpct::DpctGlobalInfo::getContext());
      // if no MatchResult, then we return false.
      bool Result = DevCallMatchResult.size();
      for (auto &Element : DevCallMatchResult) {
        if (auto CE = Element.getNodeAs<CallExpr>("devcall")) {
          auto Callee = CE->getCallee()->IgnoreImplicitAsWritten();
          Result = Result && isCub(Callee);
        }
      }
      return Result;
    } else if (DeviceFuncDecl->hasAttr<CUDAGlobalAttr>()) {
      auto KernelCallMatcher =
          ast_matchers::cudaKernelCallExpr(
              ast_matchers::callee(ast_matchers::functionDecl(
                  ast_matchers::hasName(DeviceFuncName))))
              .bind("kernelcall");
      auto KernelMatchResult = ast_matchers::match(
          KernelCallMatcher, clang::dpct::DpctGlobalInfo::getContext());
      bool Result = KernelMatchResult.size();
      for (auto &Element : KernelMatchResult) {
        if (auto CE = Element.getNodeAs<CUDAKernelCallExpr>("kernelcall")) {
          auto Callee = CE->getCallee()->IgnoreImplicitAsWritten();
          Result = Result && isCub(Callee);
        }
      }
      return Result;
    }
    return false;
  }
  return false;
}
const std::string &getItemName() {
  const static std::string ItemName = "item" + getCTFixedSuffix();
  return ItemName;
}

/// This function finds all variable reference in declaration scope.
/// \param [in] DRE The input variable reference
/// \param [out] MatchResult The vector contains all variable reference
/// \param [in] IsGlobalVariableAllowed Whether find reference for global
/// variable
void findAllVarRef(const clang::DeclRefExpr *DRE,
                   std::vector<const clang::DeclRefExpr *> &MatchResult,
                   bool IsGlobalVariableAllowed) {
  if (!DRE) {
    return;
  }
  using namespace ast_matchers;
  llvm::SmallVector<clang::ast_matchers::BoundNodes, 1U> RefMatchResult;
  if (auto VD = dyn_cast<VarDecl>(DRE->getDecl())) {
    if (auto Scope = dpct::DpctGlobalInfo::findAncestor<CompoundStmt>(VD)) {
      auto RefMatcher = compoundStmt(forEachDescendant(
          declRefExpr(to(varDecl(hasName(VD->getNameAsString()))))
              .bind("Reference")));
      RefMatchResult = ast_matchers::match(RefMatcher, *Scope,
                                           dpct::DpctGlobalInfo::getContext());
    } else {
      if (!IsGlobalVariableAllowed) {
        return;
      } else if (auto Scope =
                     dpct::DpctGlobalInfo::findAncestor<TranslationUnitDecl>(
                         VD)) {
        auto RefMatcher = translationUnitDecl(forEachDescendant(
            declRefExpr(to(varDecl(hasName(VD->getNameAsString()))))
                .bind("Reference")));
        RefMatchResult = ast_matchers::match(
            RefMatcher, *Scope, dpct::DpctGlobalInfo::getContext());
      }
    }

    for (auto &Element : RefMatchResult) {
      if (auto MatchedDRE = Element.getNodeAs<DeclRefExpr>("Reference")) {
        MatchResult.push_back(MatchedDRE);
      }
    }
  }
  return;
}

/// Check if this Exprssion's value used or not
/// \param [in] E The input Expression
/// \param [out] Result The check result
/// \return return true if the check result is valid
bool isExprUsed(const clang::Expr *E, bool &Result) {
  if (!E) {
    return false;
  }
  auto &Context = dpct::DpctGlobalInfo::getContext();
  auto Parents = Context.getParents(*E);
  if (Parents.size() != 1) {
    return false;
  }
  auto ParentNode = Context.getParents(*E)[0];
  if (ParentNode.get<CompoundStmt>()) {
    Result = false;
  } else if (auto P = ParentNode.get<ForStmt>()) {
    if (P->getBody() == E ||
        dpct::DpctGlobalInfo::isAncestor(P->getBody(), E)) {
      Result = false;
    }
  } else if (auto P = ParentNode.get<WhileStmt>()) {
    if (P->getBody() == E ||
        dpct::DpctGlobalInfo::isAncestor(P->getBody(), E)) {
      Result = false;
    }
  } else if (auto P = ParentNode.get<SwitchStmt>()) {
    if (P->getBody() == E ||
        dpct::DpctGlobalInfo::isAncestor(P->getBody(), E)) {
      Result = false;
    }
  } else if (auto P = ParentNode.get<DoStmt>()) {
    if (P->getBody() == E ||
        dpct::DpctGlobalInfo::isAncestor(P->getBody(), E)) {
      Result = false;
    }
  } else if (auto P = ParentNode.get<IfStmt>()) {
    if (P->getCond() == E ||
        dpct::DpctGlobalInfo::isAncestor(P->getCond(), E)) {
      Result = true;
    } else {
      Result = false;
    }
  } else if (auto P = ParentNode.get<CaseStmt>()) {
    if (P->getRHS() == E || dpct::DpctGlobalInfo::isAncestor(P->getRHS(), E)) {
      Result = false;
    }
  } else {
    Result = true;
  }
  return true;
}

std::string getRemovedAPIWarningMessage(std::string FuncName) {
    auto Msg = MapNames::RemovedAPIWarningMessage.find(FuncName);
    if (Msg != MapNames::RemovedAPIWarningMessage.end()) {
      return Msg->second;
    }
    return "";
}

bool isUserDefinedDecl(const clang::Decl *D) {
  std::string InFile = dpct::DpctGlobalInfo::getLocInfo(D).first;
  bool InInstallPath = isChildOrSamePath(DpctInstallPath, InFile);
  bool InCudaPath = dpct::DpctGlobalInfo::isInCudaPath(D->getLocation());
  if (InInstallPath || InCudaPath)
    return false;
  return true;
}

void insertHeaderForTypeRule(std::string Name, clang::SourceLocation Loc) {
  auto It = MapNames::TypeNamesMap.find(Name);
  if (It == MapNames::TypeNamesMap.end())
    return;
  for (auto ItHeader = It->second->Includes.begin();
       ItHeader != It->second->Includes.end(); ItHeader++) {
    dpct::DpctGlobalInfo::getInstance().insertHeader(Loc, *ItHeader);
  }
}

std::string getBaseTypeStr(const CallExpr *CE) {
  auto ME = dyn_cast<MemberExpr>(CE->getCallee()->IgnoreImpCasts());
  if (!ME)
    return "";
  auto Base = ME->getBase()->IgnoreImpCasts();
  if (!Base)
    return "";
  return dpct::DpctGlobalInfo::getTypeName(Base->getType().getCanonicalType());
}

std::string getParamTypeStr(const CallExpr *CE, unsigned int Idx) {
  if (CE->getNumArgs() <= Idx)
    return "";
  if (!CE->getDirectCallee())
    return "";
  if (!CE->getDirectCallee()->getParamDecl(Idx))
    return "";
  return CE->getDirectCallee()
      ->getParamDecl(Idx)
      ->getType()
      .getCanonicalType()
      .getUnqualifiedType()
      .getAsString();
}

std::string getArgTypeStr(const clang::CallExpr *CE, unsigned int Idx) {
  if (CE->getNumArgs() <= Idx)
    return "";
  const Expr *E = CE->getArg(Idx);
  if (!E) {
    return "";
  }
  return E->IgnoreImplicitAsWritten()
      ->getType()
      .getCanonicalType()
      .getUnqualifiedType()
      .getAsString();
}

std::string getFunctionName(const clang::FunctionDecl *Node) {
  std::string FunctionName;
  llvm::raw_string_ostream OS(FunctionName);
  auto PP = dpct::DpctGlobalInfo::getContext().getPrintingPolicy();
  if (const auto *CMD = dyn_cast<clang::CXXMethodDecl>(Node)) {
    const CXXRecordDecl *CRD = CMD->getParent();
    CRD->printName(OS, PP);
    OS << "::";
  }
  Node->getNameInfo().printName(OS, PP);
  OS.flush();
  return FunctionName;
}
std::string getFunctionName(const clang::UnresolvedLookupExpr *Node) {
  return Node->getNameInfo().getName().getAsString();
}
std::string getFunctionName(const clang::FunctionTemplateDecl *Node) {
  return getFunctionName(Node->getTemplatedDecl());
}
bool isLambda(const clang::FunctionDecl *FD) {
  if (const auto *CMD = dyn_cast<clang::CXXMethodDecl>(FD)) {
    const CXXRecordDecl *CRD = CMD->getParent();
    return CRD->isLambda();
  }
  return false;
}

const clang::LambdaExpr *
getImmediateOuterLambdaExpr(const clang::FunctionDecl *FuncDecl) {
  if (FuncDecl && FuncDecl->hasAttr<clang::CUDADeviceAttr>() &&
      FuncDecl->getAttr<clang::CUDADeviceAttr>()->isImplicit() &&
      FuncDecl->hasAttr<clang::CUDAHostAttr>() &&
      FuncDecl->getAttr<clang::CUDAHostAttr>()->isImplicit()) {
    auto *LE = dpct::DpctGlobalInfo::findAncestor<clang::LambdaExpr>(FuncDecl);
    if (LE && LE->getLambdaClass() && LE->getLambdaClass()->isLambda() &&
        isLexicallyInLocalScope(LE->getLambdaClass())) {
      return LE;
    }
  }
  return nullptr;
}

// Implementation copied from clang/lib/AST/Decl.cpp
// Helper function: returns true if QT is or contains a type
// having a postfix component.
bool typeIsPostfix(clang::QualType QT) {
  using namespace clang;
  while (true) {
    const auto *const T = QT.getTypePtr();
    switch (T->getTypeClass()) {
    default:
      return false;
    case clang::Type::Pointer:
      QT = cast<clang::PointerType>(T)->getPointeeType();
      break;
    case clang::Type::BlockPointer:
      QT = cast<BlockPointerType>(T)->getPointeeType();
      break;
    case clang::Type::MemberPointer:
      QT = cast<MemberPointerType>(T)->getPointeeType();
      break;
    case clang::Type::LValueReference:
    case clang::Type::RValueReference:
      QT = cast<ReferenceType>(T)->getPointeeType();
      break;
    case clang::Type::PackExpansion:
      QT = cast<PackExpansionType>(T)->getPattern();
      break;
    case clang::Type::Paren:
    case clang::Type::ConstantArray:
    case clang::Type::DependentSizedArray:
    case clang::Type::IncompleteArray:
    case clang::Type::VariableArray:
    case clang::Type::FunctionProto:
    case clang::Type::FunctionNoProto:
      return true;
    }
  }
}

// Check if the given pointer is only accessed on host side.
bool isPointerHostAccessOnly(const clang::ValueDecl *VD) {
  llvm::SmallVector<clang::ast_matchers::BoundNodes, 1U> MatchResult;
  using namespace clang::ast_matchers;
  auto &SM = clang::dpct::DpctGlobalInfo::getSourceManager();
  auto &CTX = clang::dpct::DpctGlobalInfo::getContext();
  auto LocInfo =
      dpct::DpctGlobalInfo::getLocInfo(SM.getExpansionLoc(VD->getBeginLoc()));
  auto &Map = dpct::DpctGlobalInfo::getMallocHostInfoMap();
  std::string Key = LocInfo.first + "*" + std::to_string(LocInfo.second);
  if(Map.count(Key)){
    return Map[Key];
  }
  auto &Val = Map[Key];
  Val = false;
  auto PtrName = VD->getName();
  auto PtrMatcher =
      findAll(declRefExpr(to(varDecl(hasName(PtrName)))).bind("PtrVar"));
  if(auto FD = dyn_cast<FunctionDecl>(VD->getDeclContext())) {
    auto Def = FD->getDefinition();
    if(!Def) {
      return false;
    }
    if(Def->hasAttr<CUDADeviceAttr>() || Def->hasAttr<CUDAGlobalAttr>()) {
      return false;
    }
    auto Body = Def->getBody();
    MatchResult = ast_matchers::match(PtrMatcher, *Body, CTX);
  }
  if (!MatchResult.size()) {
    return false;
  }
  // Match all DeclRefExpr for given pointer, check if each DeclRefExpr is used in
  // host access. If all DeclRefExpr is used in host access, then the pointer is
  // only accessed on host side.
  // If DeclRefExpr usage meets one or more of the following 3 conditions, it's used in
  // host access.
  // Condition 1: Used in pointer dereference expr and the expr value category is rvalue.
  // Condition 2: Used in array subscript expr and the expr value category is rvalue.
  // Condition 3: Used in some C and CUDA runtime functions, e.g., printf, cudaMemcpy.
  for (auto &SubResult : MatchResult) {
    bool HostAccess = false;
    const DeclRefExpr *PtrDRE = SubResult.getNodeAs<DeclRefExpr>("PtrVar");
    if (!PtrDRE) {
      return false;
    } else if(PtrDRE->getDecl() != VD) {
      continue;
    }
    const Stmt* S = PtrDRE;
    const CallExpr *CE = nullptr;
    bool needFindParent = true;
    while(needFindParent) {
      S = getParentStmt(S);
      switch(S->getStmtClass()) {
        // Condition 1
        case clang::Stmt::StmtClass::UnaryOperatorClass : {
          auto UO = dyn_cast<UnaryOperator>(S);
          if (!UO) {
            return false;
          }
          auto OpCode = UO->getOpcode();
          if(OpCode == UnaryOperatorKind::UO_AddrOf) {
            break;
          } else if(OpCode != UnaryOperatorKind::UO_Deref) {
            return false;
          }
          LLVM_FALLTHROUGH;
        }
        // Condition 2
        case clang::Stmt::StmtClass::ArraySubscriptExprClass: {
          auto RValueExpr =
              dyn_cast_or_null<ImplicitCastExpr>(getParentStmt(S));
          if (RValueExpr &&
              (RValueExpr->getCastKind() == CastKind::CK_LValueToRValue)) {
            HostAccess = true;
          }
          needFindParent =false;
          break;
        }
        case clang::Stmt::StmtClass::CallExprClass:{
          CE = dyn_cast<CallExpr>(S);
          needFindParent =false;
          break;
        }
        case clang::Stmt::StmtClass::ImplicitCastExprClass :
        case clang::Stmt::StmtClass::ParenExprClass:
        case clang::Stmt::StmtClass::CStyleCastExprClass:
        case clang::Stmt::StmtClass::CXXConstCastExprClass:
        case clang::Stmt::StmtClass::CXXStaticCastExprClass:
        case clang::Stmt::StmtClass::CXXDynamicCastExprClass:
        case clang::Stmt::StmtClass::CXXReinterpretCastExprClass:{
          break;
        }
        default : {
          return false;
        }
      }
    }
    // Condition 3
    if (CE) {
      auto CEDecl = CE->getDirectCallee();
      if (!CEDecl) {
        return false;
      }
      auto FuncName = CEDecl->getName();
      SourceLocation CEDeclLoc = SM.getExpansionLoc(CEDecl->getLocation());
      if (dpct::DpctGlobalInfo::isInCudaPath(CEDeclLoc)) {
        if (FuncName == "cudaFreeHost" || FuncName == "cudaMallocHost") {
          HostAccess = true;
        } else if (FuncName == "cudaMemcpy" || FuncName == "cudaMemcpyAsync") {
          if (auto Enum = dyn_cast<DeclRefExpr>(CE->getArg(3))) {
            auto CpyKind = Enum->getDecl()->getName();
            if (CpyKind == "cudaMemcpyHostToHost" ||
                (CpyKind == "cudaMemcpyHostToDevice" &&
                 clang::dpct::DpctGlobalInfo::isAncestor(CE->getArg(1),
                                                         PtrDRE)) ||
                (CpyKind == "cudaMemcpyDeviceToHost" &&
                 clang::dpct::DpctGlobalInfo::isAncestor(CE->getArg(0),
                                                         PtrDRE))) {
              HostAccess = true;
            }
          }
        }
      } else {
        if (SM.isInSystemHeader(CEDeclLoc) &&
            (FuncName == "memset" || FuncName == "memcpy" ||
             FuncName == "printf")) {
          HostAccess = true;
        } else {
          int ArgIndex = -1, ArgNums = CE->getNumArgs();
          for (int index = 0; index < ArgNums; index++) {
            if (clang::dpct::DpctGlobalInfo::isAncestor(CE->getArg(index),
                                                        PtrDRE)) {
              ArgIndex = index;
              break;
            }
          }
          if ((ArgIndex >= 0) &&
              isPointerHostAccessOnly(CEDecl->getParamDecl(ArgIndex))) {
            HostAccess = true;
          }
        }
      }
    }
    if (!HostAccess) {
      return false;
    }
  }
  return (Val = true);
}

std::string getBaseTypeRemoveTemplateArguments(const clang::MemberExpr *ME) {
  auto QT = ME->getBase()->IgnoreImpCasts()->getType();
  if (const auto ice = dyn_cast<ImplicitCastExpr>(ME->getBase())) {
    if (ice->getCastKind() == CastKind::CK_UncheckedDerivedToBase)
      QT = ME->getBase()->getType();
  }
  if (ME->isArrow())
    QT = QT->getPointeeType();
  const auto CT = QT.getCanonicalType();
  if (const auto RT = dyn_cast<clang::RecordType>(CT.getTypePtr())) {
    if(const clang::CXXRecordDecl* RD = RT->getAsCXXRecordDecl()) {
      if(RD->getIdentifier() == nullptr) {
        return dpct::DpctGlobalInfo::getUnqualifiedTypeName(QT);
      }
    }
    return RT->getDecl()->getQualifiedNameAsString();
  } else {
    return dpct::DpctGlobalInfo::getUnqualifiedTypeName(QT);
  }
}

bool containIterationSpaceBuiltinVar(const clang::Stmt *Node) {
  if (!Node)
    return false;
  using namespace clang::ast_matchers;
  auto BuiltinMatcher = findAll(
      memberExpr(hasObjectExpression(opaqueValueExpr(
                     hasSourceExpression(declRefExpr(to(varDecl(hasAnyName(
                         "threadIdx", "blockDim", "blockIdx", "gridDim"))))))),
                 hasParent(implicitCastExpr(
                     hasParent(callExpr(hasParent(pseudoObjectExpr()))))))
          .bind("memberExpr"));
  auto MatchedResults =
      match(BuiltinMatcher, *Node, clang::dpct::DpctGlobalInfo::getContext());
  return MatchedResults.size();
}

bool containBuiltinWarpSize(const clang::Stmt *Node) {
  if (!Node)
    return false;
  using namespace clang::ast_matchers;
  auto BuiltinMatcher =
      findAll(declRefExpr(to(varDecl(hasName("warpSize")).bind("VD"))));
  auto MatchedResults =
      match(BuiltinMatcher, *Node, clang::dpct::DpctGlobalInfo::getContext());
  for (const auto &Res : MatchedResults) {
    const clang::VarDecl *VD = Res.getNodeAs<clang::VarDecl>("VD");
    if (!VD)
      continue;
    if (!clang::dpct::DpctGlobalInfo::isInAnalysisScope(VD->getLocation()))
      return true;
  }
  return false;
}

bool isCapturedByLambda(const clang::TypeLoc *TL) {
  using namespace dpct;
  const FieldDecl *FD = DpctGlobalInfo::findAncestor<clang::FieldDecl>(TL);
  if (!FD)
    return false;
  const LambdaExpr *LE = DpctGlobalInfo::findAncestor<clang::LambdaExpr>(TL);
  if (!LE)
    return false;
  for (const auto &D : LE->getLambdaClass()->decls()) {
    const FieldDecl *FieldDeclItem = dyn_cast<clang::FieldDecl>(D);
    if (FieldDeclItem && (FieldDeclItem == FD))
      return true;
  }
  return false;
}

std::string getAddressSpace(const CallExpr *C, int ArgIdx) {
  bool HasAttr = false;
  bool NeedReport = false;
  const Expr *Arg = C->getArg(ArgIdx);
  if (!Arg) {
    return "";
  }
  getShareAttrRecursive(Arg, HasAttr, NeedReport);
  if (HasAttr && !NeedReport)
    return "local_space";
  LocalVarAddrSpaceEnum LocalVarCheckResult =
      LocalVarAddrSpaceEnum::AS_CannotDeduce;
  checkIsPrivateVar(Arg, LocalVarCheckResult);
  if (LocalVarCheckResult == LocalVarAddrSpaceEnum::AS_IsPrivate) {
    return "private_space";
  } else if (LocalVarCheckResult == LocalVarAddrSpaceEnum::AS_IsGlobal) {
    return "global_space";
  } else {
    clang::dpct::ExprAnalysis EA(Arg);
    auto LocInfo =
        dpct::DpctGlobalInfo::getInstance().getLocInfo(C->getBeginLoc());
    clang::dpct::DiagnosticsUtils::report(
        LocInfo.first, LocInfo.second,
        clang::dpct::Diagnostics::UNDEDUCED_ADDRESS_SPACE, true, false,
        EA.getReplacedString());
    return "global_space";
  }
}

std::string getNameSpace(const NamespaceDecl *NSD) {
  if (!NSD)
    return "";
  std::string NameSpace =
      getNameSpace(dyn_cast<NamespaceDecl>(NSD->getDeclContext()));
  if (!NameSpace.empty() && !NSD->isInlineNamespace())
    return NameSpace + "::" + NSD->getName().str();
  else if (NameSpace.empty() && !NSD->isInlineNamespace())
    return NSD->getName().str();
  return NameSpace;
}

bool isFromCUDA(const Decl *D) {
  SourceLocation DeclLoc =
      dpct::DpctGlobalInfo::getSourceManager().getExpansionLoc(
          D->getLocation());
  std::string DeclLocFilePath = dpct::DpctGlobalInfo::getLocInfo(DeclLoc).first;
  makeCanonical(DeclLocFilePath);

  // clang hacked the declarations of std::min/std::max
  // In original code, the declaration should be in standard lib,
  // but clang need to add device version overload, so it hacked the
  // resolution by adding a special attribute.
  // So we need treat function which is declared in this file as it
  // is from standard lib.
  SmallString<512> AlgorithmFileInCudaWrapper = StringRef(DpctInstallPath);
  path::append(AlgorithmFileInCudaWrapper, Twine("lib"), Twine("clang"),
               Twine(CLANG_VERSION_MAJOR_STRING), Twine("include"));
  path::append(AlgorithmFileInCudaWrapper, Twine("cuda_wrappers"),
               Twine("algorithm"));

  if (AlgorithmFileInCudaWrapper.str().str() == DeclLocFilePath) {
    return false;
  }

  return (isChildPath(dpct::DpctGlobalInfo::getCudaPath(), DeclLocFilePath) ||
          isChildPath(DpctInstallPath, DeclLocFilePath));
}

namespace clang {
namespace dpct {
void requestFeature(HelperFeatureEnum Feature) {
  if (Feature == HelperFeatureEnum::none) {
    return;
  }
  DpctGlobalInfo::setNeedDpctDeviceExt();
}

std::string getDpctVersionStr() {
  std::string Str;
  llvm::raw_string_ostream OS(Str);
  OS << DPCT_VERSION_MAJOR << "." << DPCT_VERSION_MINOR << "."
     << DPCT_VERSION_PATCH;
  return OS.str();
}

void requestHelperFeatureForEnumNames(const std::string Name) {
  auto HelperFeatureIter =
      clang::dpct::EnumConstantRule::EnumNamesMap.find(Name);
  if (HelperFeatureIter != clang::dpct::EnumConstantRule::EnumNamesMap.end()) {
    requestFeature(HelperFeatureIter->second->RequestFeature);
    return;
  }
  auto CuDNNHelperFeatureIter =
      clang::dpct::CuDNNTypeRule::CuDNNEnumNamesHelperFeaturesMap.find(Name);
  if (CuDNNHelperFeatureIter !=
      clang::dpct::CuDNNTypeRule::CuDNNEnumNamesHelperFeaturesMap.end()) {
    requestFeature(CuDNNHelperFeatureIter->second);
  }
}
void requestHelperFeatureForTypeNames(const std::string Name) {
  auto HelperFeatureIter = MapNames::TypeNamesMap.find(Name);
  if (HelperFeatureIter != MapNames::TypeNamesMap.end()) {
    requestFeature(HelperFeatureIter->second->RequestFeature);
    return;
  }
  auto CuDNNHelperFeatureIter = MapNames::CuDNNTypeNamesMap.find(Name);
  if (CuDNNHelperFeatureIter != MapNames::CuDNNTypeNamesMap.end()) {
    requestFeature(CuDNNHelperFeatureIter->second->RequestFeature);
  }
}
} // namespace dpct
} // namespace clang
