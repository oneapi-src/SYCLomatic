//===--- Utility.cpp -----------------------------------*- C++ -*---===//
//
// Copyright (C) 2018 - 2020 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===---------------------------------------------------------------===//

#include "Utility.h"
#include "AnalysisInfo.h"
#include "Debug.h"
#include "ExprAnalysis.h"
#include "MapNames.h"
#include "SaveNewFiles.h"
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

using namespace llvm;
using namespace clang;
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

int getCurrnetColumn(SourceLocation Loc, const SourceManager &SM) {
  auto LocInfo = SM.getDecomposedLoc(Loc);
  auto Buffer = SM.getBufferData(LocInfo.first);
  // Find last line end.
  auto begin = Buffer.find_last_of('\n', LocInfo.second);
  if (begin == StringRef::npos) {
    // We're at the beginning of the file.
    begin = 0;
  }
  return LocInfo.second - begin;
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

// Get textual representation of the Stmt.
std::string getStmtSpelling(const Stmt *S) {
  std::string Str;
  if (!S)
    return Str;
  auto &SM = dpct::DpctGlobalInfo::getSourceManager();
  SourceLocation BeginLoc, EndLoc;
  if (S->getBeginLoc().isMacroID() && !isOuterMostMacro(S)) {
    BeginLoc = SM.getImmediateSpellingLoc(S->getBeginLoc());
    EndLoc = SM.getImmediateSpellingLoc(S->getEndLoc());
    if (EndLoc.isMacroID()) {
      // if the immediate spelling location of
      // a macro arg is another macro, get the expansion loc
      EndLoc = SM.getExpansionLoc(EndLoc);
    }
    if (BeginLoc.isMacroID()) {
      // if the immediate spelling location of
      // a macro arg is another macro, get the expansion loc
      BeginLoc = SM.getExpansionLoc(BeginLoc);
    }
  } else {
    BeginLoc = SM.getExpansionLoc(S->getBeginLoc());
    EndLoc = SM.getExpansionLoc(S->getEndLoc());
  }

  int Length =
      SM.getFileOffset(EndLoc) - SM.getFileOffset(BeginLoc) +
      Lexer::MeasureTokenLength(
          EndLoc, SM, dpct::DpctGlobalInfo::getContext().getLangOpts());
  Str = std::string(SM.getCharacterData(BeginLoc), Length);
  return Str;
}

std::string getStmtExpansion(const Stmt *S, const ASTContext &Context) {
  const SourceManager &SM = Context.getSourceManager();
  SourceLocation Begin(S->getBeginLoc()), _End(S->getEndLoc());
  SourceLocation End(Lexer::getLocForEndOfToken(_End, 0, SM, LangOptions()));
  if (Begin.isMacroID())
    Begin = SM.getExpansionLoc(Begin);
  if (End.isMacroID())
    End = SM.getExpansionLoc(End);
  return std::string(SM.getCharacterData(Begin),
                     SM.getCharacterData(End) - SM.getCharacterData(Begin));
}

SourceProcessType GetSourceFileType(llvm::StringRef SourcePath) {
  SmallString<256> FilePath = SourcePath;
  auto Extension = path::extension(FilePath);

  if (Extension == ".cu") {
    return TypeCudaSource;
  } else if (Extension == ".cuh") {
    return TypeCudaHeader;
  } else if (Extension == ".cpp" || Extension == ".cxx" || Extension == ".cc" ||
             Extension == ".c" || Extension == ".C") {
    return TypeCppSource;
  } else if (Extension == ".hpp" || Extension == ".hxx" || Extension == ".h" ||
             Extension == ".hh" || Extension == ".inl" || Extension == ".inc" ||
             Extension == ".INL" || Extension == ".INC" ||
             Extension == ".TPP" || Extension == ".tpp") {
    return TypeCppHeader;
  } else {
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
      return TypeCppSource;
    }
    auto &IncludingFileSet = dpct::DpctGlobalInfo::getIncludingFileSet();
    if (IncludingFileSet.find(SourcePath.str()) != end(IncludingFileSet)) {
      return TypeCppHeader;
    }
    return TypeCppSource;
  }
}

std::vector<std::string>
ruleTopoSort(std::vector<std::vector<std::string>> &TableRules) {
  std::vector<std::string> Vec;

  std::vector<std::list<int>> AdjacencyList;
  std::vector<int> InDegree;
  std::stack<int> Stack;
  std::vector<std::string> RuleNames;

  int n = TableRules.size();
  AdjacencyList.assign(n, std::list<int>());
  InDegree.assign(n, 0);

  for (int i = 0; i < n; i++) {
    RuleNames.push_back(TableRules[i].at(0));
  }

  for (int i = 0; i < n; i++) {
    for (std::vector<std::string>::iterator it = TableRules[i].begin() + 1;
         it != TableRules[i].end(); ++it) {

      // if detect rule depend on itself,  then just ignore
      if (*it == *TableRules[i].begin()) {
        continue;
      }

      std::vector<std::string>::iterator index =
          find(RuleNames.begin(), RuleNames.end(), *it);
      if (index != RuleNames.end()) {
        AdjacencyList[i].push_back(index - RuleNames.begin());
        InDegree[index - RuleNames.begin()]++;
      }
    }
  }

  for (int i = 0; i < n; i++)
    if (InDegree[i] == 0)
      Stack.push(i);

  while (!Stack.empty()) {
    int v = Stack.top();
    Stack.pop();
    InDegree[v] = -1;

    for (std::list<int>::iterator it = AdjacencyList[v].begin();
         it != AdjacencyList[v].end(); it++) {
      InDegree[*it]--;
      if (InDegree[*it] == 0)
        Stack.push(*it);
    }
    AdjacencyList[v].clear();
    Vec.push_back(RuleNames[v]);
  }
  if (Vec.size() != InDegree.size()) {
    std::cout << "Error: Two rules have dependency on each other！\n";
    dpct::DebugInfo::ShowStatus(MigrationError);
    exit(MigrationError);
  }

  return Vec;
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
  } else if (D->getLexicalDeclContext()->getDeclKind() ==
             Decl::Kind::Function) {
    auto BD = static_cast<const FunctionDecl *>(D->getDeclContext());
    CS = dyn_cast<CompoundStmt>(BD->getBody());
  }

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

// Iteratively get the inner ValueDecl of a potetionally nested expression
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

const clang::Stmt *getParentStmt(ast_type_traits::DynTypedNode Node) {
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

const clang::Stmt *getParentStmt(const clang::Stmt *S) {
  if (!S)
    return nullptr;

  auto &Context = dpct::DpctGlobalInfo::getContext();
  auto Parents = Context.getParents(*S);
  assert(Parents.size() >= 1);
  if (Parents.size() >= 1)
    return Parents[0].get<Stmt>();

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
const ast_type_traits::DynTypedNode
getAncestorDeclStmtNode(const clang::Expr *E) {
  auto &Context = dpct::DpctGlobalInfo::getContext();
  auto ParentNodes = Context.getParents(*E);
  ast_type_traits::DynTypedNode ParentNode;
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

const std::shared_ptr<clang::ast_type_traits::DynTypedNode>
getParentNode(const std::shared_ptr<clang::ast_type_traits::DynTypedNode> N) {
  if (!N)
    return nullptr;

  auto &Context = dpct::DpctGlobalInfo::getContext();
  auto Parents = Context.getParents(*N);
  // if (Parents.size() == 1)
  return std::make_shared<clang::ast_type_traits::DynTypedNode>(Parents[0]);

  // return nullptr;
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
const ast_type_traits::DynTypedNode
findNearestNonExprNonDeclAncestorNode(const clang::Expr *E) {
  auto &Context = dpct::DpctGlobalInfo::getContext();
  auto ParentNodes = Context.getParents(*E);
  ast_type_traits::DynTypedNode LastNode, ParentNode;
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
  ast_type_traits::DynTypedNode AncestorStmt;
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
    // Initilization
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
std::vector<const Stmt *> getConditionNode(ast_type_traits::DynTypedNode Node) {
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
std::vector<const Stmt *> getConditionExpr(ast_type_traits::DynTypedNode Node) {
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
/// will be true and \p SL will be set as the begin location of the FlowControl
/// statement:
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
  ast_type_traits::DynTypedNode ParentNode;
  std::vector<ast_type_traits::DynTypedNode> AncestorNodes;
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
    for (auto Node : AncestorNodes) {
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
  ast_type_traits::DynTypedNode ParentNode;
  std::vector<ast_type_traits::DynTypedNode> AncestorNodes;
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
    for (auto Node : AncestorNodes) {
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
  // TODO: reinterpret will copy more data
  BufferDecl = "auto " + BufferTempName + " = dpct::get_buffer<" + TypeAsStr +
               ">(" + PointerName + ");" + getNL() + IndentStr;
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
  // TODO: reinterpret will copy more data
  BufferDecl = "auto " + BufferTempName + " = dpct::get_buffer<" + TypeAsStr +
               ">(" + PointerName + ");" + getNL() + IndentStr;
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

/// Calculate the ranges of the input \p Repls which has NOT set NotFormatFlags.
/// \param Repls Replacements with format flags.
/// \return The result ranges.
std::vector<clang::tooling::Range>
calculateRangesWithFormatFlag(const clang::tooling::Replacements &Repls) {
  std::vector<bool> NotFormatFlags;
  std::vector<clang::tooling::Range> Ranges;

  int Diff = 0;
  for (auto R : Repls) {
    if (R.getNotFormatFlag())
      NotFormatFlags.push_back(true);
    else
      NotFormatFlags.push_back(false);
    Ranges.emplace_back(/*offset*/ R.getOffset() + Diff,
                        /*length*/ R.getReplacementText().size());

    Diff = Diff + R.getReplacementText().size() - R.getLength();
  }

  std::vector<clang::tooling::Range> RangesAfterFilter;
  int Size = Ranges.size();
  for (int i = 0; i < Size; ++i) {
    if (!NotFormatFlags[i])
      RangesAfterFilter.push_back(Ranges[i]);
  }
  return RangesAfterFilter;
}

/// Determine if \param S is assigned or not
/// \param S A Stmt node
/// \return True if S is assigned and false if S is not assigned
bool isAssigned(const Stmt *S) {
  auto P = getParentStmt(S);
  return !P || (!dyn_cast<CompoundStmt>(P) && !dyn_cast<ForStmt>(P) &&
                !dyn_cast<WhileStmt>(P) && !dyn_cast<DoStmt>(P) &&
                !dyn_cast<IfStmt>(P));
}

/// Compute a temporary variable name for \param E
/// \param E The Expr based on which the temp name is computed
/// \param KeepLastUnderline A boolean value indicating if the last underline
/// is to be perserved or not
/// \return A temporary variable name

std::string getTempNameForExpr(const Expr *E, bool HandleLiteral,
                               bool KeepLastUnderline, bool IsInMacroDefine) {
  SourceManager &SM = dpct::DpctGlobalInfo::getSourceManager();
  E = E->IgnoreCasts();
  dpct::ArgumentAnalysis EA(E, IsInMacroDefine);
  auto TokenBegin = EA.getExprBeginSrcLoc();
  auto ExprEndLoc = EA.getExprEndSrcLoc();
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
  std::shared_ptr<ast_type_traits::DynTypedNode> P =
      std::make_shared<ast_type_traits::DynTypedNode>(
          ast_type_traits::DynTypedNode::create(*E));
  // Find a parent stmt whose preprocessing result is different from
  // ExpandedExpr Since some parent is not writable.(is not shown in the
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

bool isInsideFunctionLikeMacro(
    const SourceLocation BeginLoc, const SourceLocation EndLoc,
    const std::shared_ptr<ast_type_traits::DynTypedNode> Parent) {

  if (!BeginLoc.isMacroID() || !EndLoc.isMacroID()) {
    return false;
  }

  auto &SM = dpct::DpctGlobalInfo::getSourceManager();
  // If the begin/end location are different macro expansions,
  // the expression is a combination of different macros
  // which makes it outer-most.
  if (SM.getCharacterData(SM.getExpansionRange(BeginLoc).getEnd()) <
      SM.getCharacterData(SM.getExpansionLoc(EndLoc))) {
    return false;
  }

  // Since SM.getExpansionLoc() will always return the range of the outer-most
  // macro. If the expanded location of the parent stmt and E are the same, E is
  // inside a function-like macro.
  // E.g. MACRO_A(MACRO_B(x,y),z) where E is the PP
  // result of MACRO_B and Parent is the PP result of MACRO_A,
  // SM.getExpansionLoc(E) is at the begining of MACRO_A, same as
  // SM.getExpansionLoc(Parent), in the source code. E is not outer-most.
  if (Parent->getSourceRange().getBegin().isValid() &&
    Parent->getSourceRange().getBegin().isMacroID()) {
    if (SM.getCharacterData(
      SM.getExpansionLoc(Parent->getSourceRange().getBegin())) ==
      SM.getCharacterData(SM.getExpansionLoc(BeginLoc))) {
      if (Parent->getSourceRange().getEnd().isValid() &&
        Parent->getSourceRange().getEnd().isMacroID()) {
        if (SM.getCharacterData(
          SM.getExpansionLoc(Parent->getSourceRange().getEnd())) ==
          SM.getCharacterData(SM.getExpansionLoc(EndLoc))) {
          return true;
        }
      }
    }
  }

  // Another case which should to return true is
  // #define MacroA(x) = x
  // When MacroA is used for default arguments in function definition
  // like foo(int x MacroA(0)) and the ASTMatcher matches the "0" in the
  // expansion, since the parent of x in the AST is "int x MacroA(0)" not "= x",
  // previous check cannot detect the "0" is inside a function like macro.
  // Should check if the expansion is the whole macro definition.

  // Get the location of "x" in "#define MacroA(x) = x"
  SourceLocation ImmediateSpellingBegin = SM.getImmediateSpellingLoc(BeginLoc);
  SourceLocation ImmediateSpellingEnd = SM.getImmediateSpellingLoc(EndLoc);
  SourceLocation ImmediateExpansionBegin =
      SM.getImmediateExpansionRange(BeginLoc).getBegin();
  SourceLocation ImmediateExpansionEnd =
      SM.getImmediateExpansionRange(EndLoc).getEnd();

  // Check if one of the 4 combinations of begin&end matches a macro def
  // ExpansionBegin & ExpansionEnd
  auto It = dpct::DpctGlobalInfo::getExpansionRangeToMacroRecord().find(
      SM.getCharacterData(ImmediateExpansionBegin));
  if (It != dpct::DpctGlobalInfo::getExpansionRangeToMacroRecord().end() &&
      It->second->TokenIndex == 0 &&
      SM.getCharacterData(It->second->ReplaceTokenEnd) ==
          SM.getCharacterData(ImmediateExpansionEnd)) {
    return false;
  }
  // ExpansionBegin & SpellingEnd
  It = dpct::DpctGlobalInfo::getExpansionRangeToMacroRecord().find(
      SM.getCharacterData(ImmediateExpansionBegin));
  if (It != dpct::DpctGlobalInfo::getExpansionRangeToMacroRecord().end() &&
      It->second->TokenIndex == 0 &&
      SM.getCharacterData(It->second->ReplaceTokenEnd) ==
          SM.getCharacterData(ImmediateSpellingEnd)) {
    return false;
  }
  // SpellingBegin & ExpansionEnd
  It = dpct::DpctGlobalInfo::getExpansionRangeToMacroRecord().find(
      SM.getCharacterData(ImmediateSpellingBegin));
  if (It != dpct::DpctGlobalInfo::getExpansionRangeToMacroRecord().end() &&
      It->second->TokenIndex == 0 &&
      SM.getCharacterData(It->second->ReplaceTokenEnd) ==
          SM.getCharacterData(ImmediateExpansionEnd)) {
    return false;
  }
  // SpellingBegin & SpellingEnd
  It = dpct::DpctGlobalInfo::getExpansionRangeToMacroRecord().find(
      SM.getCharacterData(ImmediateSpellingBegin));
  if (It != dpct::DpctGlobalInfo::getExpansionRangeToMacroRecord().end() &&
      It->second->TokenIndex == 0 &&
      SM.getCharacterData(It->second->ReplaceTokenEnd) ==
          SM.getCharacterData(ImmediateSpellingEnd)) {
    return false;
  }

  return true;
}

// Check if an Expr is partially in function-like macro
bool isExprStraddle(const Stmt *S, ExprSpellingStatus *SpellingStatus) {
  auto &SM = dpct::DpctGlobalInfo::getSourceManager();
  bool HasMacroDefine = false;
  bool HasMacroExpansion = false;
  *SpellingStatus = NoType;

  // For all tokens of S, check whether it's in the define or the expansion part
  for (auto It = S->child_begin(); It != S->child_end(); It++) {
    bool ArgIsDefine = false;
    SourceLocation BeginLoc;
    // Instead of calling isOuterMostMacro to decide which Loc is correct,
    // calling getImmediateSpellingLoc anyway
    // because we only care about the consistency between tokens.
    BeginLoc = SM.getImmediateSpellingLoc(It->getBeginLoc());

    // Check if the token in the define part of a function-like macro.
    auto ItMatch = dpct::DpctGlobalInfo::getExpansionRangeToMacroRecord().find(
        SM.getCharacterData(BeginLoc));
    if (ItMatch !=
            dpct::DpctGlobalInfo::getExpansionRangeToMacroRecord().end() &&
        ItMatch->second->IsFunctionLike) {
      ArgIsDefine = true;
      HasMacroDefine = true;
    } else {
      HasMacroExpansion = true;
    }
    ExprSpellingStatus ChildSpellingStatus;
    // Recursively check the child node
    if (isExprStraddle(*It, &ChildSpellingStatus)) {
      return true;
    }
    // If the child's descendent has different SpellingStatus with the child
    // itself, it is straddle. In the following example,
    // "(double)x" is the parent and is in macro define
    // while "0" is the child and is in macro expansion
    // #define macro(x) (double)x
    // macro(0);
    if ((ChildSpellingStatus == IsExpansion && ArgIsDefine) ||
        (ChildSpellingStatus == IsDefine && !ArgIsDefine)) {
      return true;
    }
  }
  // If some children are in the define part and others are in the expansion
  // part, the Expr is a straddle node and no consist SpellingStatus to set.
  if (HasMacroDefine && HasMacroExpansion) {
    return true;
  }
  // When all children have consist SpellingStatus, record and return the
  // status.
  if (HasMacroDefine) {
    *SpellingStatus = IsDefine;
  } else {
    *SpellingStatus = IsExpansion;
  }
  return false;
}

/// Check the expression \p E is an address-of expression like "&aaa".
bool isSimpleAddrOf(const Expr *E) {
  if (auto UO = dyn_cast<UnaryOperator>(E)) {
    if (UO->getOpcode() == UO_AddrOf) {
      return true;
    }
  }
  return false;
}
/// Check the expression \p E is an address-of expression like "&aaa".
/// On Windows, the AST node is OverloadedOperatorKind instead of UnaryOperator.
bool isCOCESimpleAddrOf(const Expr *E) {
  if (auto COCE = dyn_cast<CXXOperatorCallExpr>(E)) {
    if (COCE->getOperator() == clang::OverloadedOperatorKind::OO_Amp &&
        COCE->getNumArgs() == 1) {
      return true;
    }
  }
  return false;
}
/// Remove the address-of operator in the expression \p E.
/// \param E An expression has checked by isSimpleAddrOf() or
/// isCOCESimpleAddrOf().
/// \param isCOCE If the expression is checked by isCOCESimpleAddrOf(), it need
/// be true.
/// \return The expression string without the address-of operator.
std::string getNameStrRemovedAddrOf(const Expr *E, bool isCOCE) {
  if (isCOCE) {
    auto COCE = dyn_cast<CXXOperatorCallExpr>(E);
    if (!COCE) {
      return "";
    }
    dpct::ExprAnalysis SEA;
    SEA.analyze(COCE->getArg(0));
    return SEA.getReplacedString();
  } else {
    auto UO = dyn_cast<UnaryOperator>(E);
    if (!UO) {
      return "";
    }
    dpct::ExprAnalysis SEA;
    SEA.analyze(UO->getSubExpr());
    return SEA.getReplacedString();
  }
}

/// Get the dereference name of the expression \p E.
std::string getDrefName(const Expr *E) {
  if (isSimpleAddrOf(E)) {
    return getNameStrRemovedAddrOf(E, false);
  } else if (isCOCESimpleAddrOf(E)) {
    return getNameStrRemovedAddrOf(E, true);
  } else {
    dpct::ExprAnalysis EA(E);
    if (isAnIdentifierOrLiteral(E)) {
      return "*" + EA.getReplacedString();
    } else {
      return "*(" + EA.getReplacedString() + ")";
    }
  }
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

/// Get sibling Decls for a VarDecl or a FieldDecl
/// E.g for a VarDecl:
/// |-DeclStmt
///   |-VarDecl
///   `-VarDecl
/// or for a FieldDecl:
/// |-CXXRecordDecl
///   |-FieldDecl
///   |-FieldDecl
std::vector<const DeclaratorDecl *> getSiblingDecls(const DeclaratorDecl *DD) {
  std::vector<const DeclaratorDecl *> Decls;
  // For VarDecl, sibling VarDecls share the same parent DeclStmt with it
  if (auto P = getParentStmt(DD)) {
    if (auto DS = dyn_cast<DeclStmt>(P)) {
      for (auto It = DS->decl_begin(); It != DS->decl_end(); ++It) {
        if (auto DD2 = dyn_cast<DeclaratorDecl>(*It))
          if (DD2 != DD)
            Decls.push_back(DD2);
      }
    }
  }
  // For FieldDecl, sibling FieldDecls share the same BeginLoc with it
  else if (auto P = getParentRecordDecl(DD)) {
    for (auto It = P->decls_begin(); It != P->decls_end(); ++It) {
      if (auto DD2 = dyn_cast<DeclaratorDecl>(*It))
        if (DD2->getBeginLoc() == DD->getBeginLoc())
          if (DD2 != DD)
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
  else if (auto CAT = dyn_cast<ConstantArrayType>(DDT)) {
    auto PT = CAT->getElementType()->getPointeeType();
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
/// 1. an sizeof operator
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

std::string addIndirectionIfNecessary(const Expr *E) {
  if (isSimpleAddrOf(E->IgnoreImplicit())) {
    return getNameStrRemovedAddrOf(E->IgnoreImplicit());
  } else if (isCOCESimpleAddrOf(E)) {
    return getNameStrRemovedAddrOf(E->IgnoreImplicit(), true);
  } else {
    dpct::ExprAnalysis EA;
    EA.analyze(E);
    return "*(" + EA.getReplacedString() + ")";
  }
}

bool isInReturnStmt(const Expr *E, SourceLocation &OuterInsertLoc) {
  auto &Context = dpct::DpctGlobalInfo::getContext();
  auto ParentNodes = Context.getParents(*E);
  ast_type_traits::DynTypedNode ParentNode;
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
  std::stringstream CombinedStr;
  CombinedStr << std::hex
              << std::hash<std::string>()(dpct::buildString(R.first, R.second));
  return CombinedStr.str();
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

SourceLocation getEndLocOfFollowingEmptyMacro(SourceLocation Loc) {
  auto &SM = dpct::DpctGlobalInfo::getSourceManager();
  auto &Map = dpct::DpctGlobalInfo::getBeginOfEmptyMacros();
  Token Tok;
  Lexer::getRawToken(
      Loc.getLocWithOffset(Lexer::MeasureTokenLength(
          Loc, SM, dpct::DpctGlobalInfo::getContext().getLangOpts())),
      Tok, SM, dpct::DpctGlobalInfo::getContext().getLangOpts(), true);

  SourceLocation EndOfToken = SM.getExpansionLoc(Tok.getLocation());
  while (Tok.isNot(tok::eof) && Tok.is(tok::comment)) {
    Lexer::getRawToken(
        EndOfToken.getLocWithOffset(Lexer::MeasureTokenLength(
            EndOfToken, SM, dpct::DpctGlobalInfo::getContext().getLangOpts())),
        Tok, SM, dpct::DpctGlobalInfo::getContext().getLangOpts(), true);
    EndOfToken = SM.getExpansionLoc(Tok.getEndLoc());
    ;
  }

  auto It = Map.find(getHashStrFromLoc(EndOfToken));
  if (It != Map.end()) {
    return It->second;
  }
  return Loc;
}

std::string
getNestedNameSpecifierString(const clang::NestedNameSpecifier *NNS) {
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
    return false;
  case Stmt::CXXConstructExprClass: {
    auto Ctor = static_cast<const CXXConstructExpr *>(E);
    if (Ctor->getParenOrBraceRange().isInvalid() && Ctor->getNumArgs() == 1)
      return needExtraParens(Ctor->getArg(0));
    else
      return true;
  }
  default:
    return true;
  }
}

bool isPredefinedStreamHandle(const Expr *E) {
  Expr::EvalResult ER;
  if (auto PE = dyn_cast<ParenExpr>(E->IgnoreImplicit())) {
    if (auto CSCE = dyn_cast<CStyleCastExpr>(PE->getSubExpr())) {
      if (!CSCE->getSubExpr()->isValueDependent() &&
          CSCE->getSubExpr()->EvaluateAsInt(
              ER, dpct::DpctGlobalInfo::getContext())) {
        int64_t Value = ER.Val.getInt().getExtValue();
        if (Value == 1 || Value == 2) {
          // cudaStreamLegacy is ((cudaStream_t)0x1)
          // cudaStreamPerThread is ((cudaStream_t)0x2)
          return true;
        }
      }
    }
  } else if (!E->IgnoreImplicit()->isValueDependent() &&
             E->IgnoreImplicit()->EvaluateAsInt(
                 ER, dpct::DpctGlobalInfo::getContext())) {
    int64_t Value = ER.Val.getInt().getExtValue();
    if (Value == 0) {
      // cudaStreamDefault is 0x00
      return true;
    }
  } else if (dyn_cast<GNUNullExpr>(E->IgnoreImplicit()) ||
             dyn_cast<CXXNullPtrLiteralExpr>(E->IgnoreImplicit())) {
      // default stream can be used as NULL, __null, nullptr
      return true;
  }
  return false;
}