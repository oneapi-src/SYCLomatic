//===--- Utility.cpp -----------------------------------*- C++ -*---===//
//
// Copyright (C) 2018 - 2019 Intel Corporation. All rights reserved.
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
#include "SaveNewFiles.h"
#include "clang/AST/ASTContext.h"
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
std::string getStmtSpelling(const Stmt *S, const ASTContext &Context) {
  std::string Str;
  if(!S)
    return Str;
  auto &SM = Context.getSourceManager();
  SourceLocation BeginLoc, EndLoc;
  if (SM.isMacroArgExpansion(S->getBeginLoc())) {
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

  int Length = SM.getFileOffset(EndLoc) - SM.getFileOffset(BeginLoc) +
               Lexer::MeasureTokenLength(EndLoc, SM, Context.getLangOpts());
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
             Extension == ".INL" || Extension == ".INC") {
    return TypeCppHeader;
  } else {
    std::string ErrMsg =
        "[ERROR] Not support \"" + Extension.str() + "\" file type!\n";
    dpct::PrintMsg(ErrMsg);
    std::exit(MigrationErrorNotSupportFileType);
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

// Find the innermost (closest) block where S is located
const clang::CompoundStmt *findImmediateBlock(const clang::Stmt *S) {
  if (!S)
    return nullptr;

  auto &Context = dpct::DpctGlobalInfo::getContext();
  auto Parents = Context.getParents(*S);
  while (Parents.size() == 1) {
    auto *Parent = Parents[0].get<Stmt>();
    if (Parent) {
      if (Parent->getStmtClass() == Stmt::StmtClass::CompoundStmtClass)
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

const clang::Stmt *getParentStmt(const clang::Stmt *S) {
  if (!S)
    return nullptr;

  auto &Context = dpct::DpctGlobalInfo::getContext();
  auto Parents = Context.getParents(*S);
  assert(Parents.size() == 1);
  if (Parents.size() == 1)
    return Parents[0].get<Stmt>();

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
const ast_type_traits::DynTypedNode
findNearestNonExprNonDeclAncestorNode(const clang::Expr *E) {
  auto &Context = dpct::DpctGlobalInfo::getContext();
  auto ParentNodes = Context.getParents(*E);
  ast_type_traits::DynTypedNode LastNode, ParentNode;
  while (!ParentNodes.empty()) {
    ParentNode = ParentNodes[0];
    if (!ParentNode.get<Expr>() && !ParentNode.get<Decl>()) {
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
  ast_type_traits::DynTypedNode LastNode;
  SourceLocation StmtEnd;
  if (ParentNode.empty()) {
    StmtBegin = FuncNameBegin;
    StmtEnd = FuncCallEnd;
  } else if (!ParentNode[0].get<Expr>() && !ParentNode[0].get<Decl>()) {
    StmtBegin = FuncNameBegin;
    StmtEnd = FuncCallEnd;
  } else {
    auto AncestorStmt = findNearestNonExprNonDeclAncestorNode(E);
    StmtBegin = AncestorStmt.getSourceRange().getBegin();
    StmtEnd = AncestorStmt.getSourceRange().getEnd();
    if (StmtBegin.isMacroID())
      StmtBegin = SM.getExpansionLoc(StmtBegin);
    if (StmtEnd.isMacroID())
      StmtEnd = SM.getExpansionLoc(StmtEnd);
  }

  Optional<Token> TokSharedPtr;
  TokSharedPtr = Lexer::findNextToken(StmtEnd, SM, LangOptions());
  Token TokSemi = TokSharedPtr.getValue();
  StmtEndAfterSemi = TokSemi.getEndLoc();
  return {StmtBegin, StmtEndAfterSemi};
}

std::string getCanonicalPath(SourceLocation Loc) {
  auto &SM = dpct::DpctGlobalInfo::getSourceManager();
  std::string Path = SM.getFilename(SM.getExpansionLoc(Loc));
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
