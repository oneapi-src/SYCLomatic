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

#include "AnalysisInfo.h"
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

bool isChildPath(const std::string &Root, const std::string &Child) {
#if defined(_WIN64)
  std::string LocalRoot = StringRef(Root).lower();
  std::string LocalChild = StringRef(Child).lower();
#elif defined(__linux__)
  std::string LocalRoot = Root;
  std::string LocalChild = Child;
#else
#error Only support windows and Linux.
#endif

  auto Diff = mismatch(path::begin(LocalRoot), path::end(LocalRoot),
                       path::begin(LocalChild));
  // LocalRoot is not considered prefix of LocalChild if they are equal.
  return Diff.first == path::end(LocalRoot) &&
         Diff.second != path::end(LocalChild);
}

bool isSamePath(const std::string &Root, const std::string &Child) {
#if defined(_WIN64)
  std::string LocalRoot = StringRef(Root).lower();
  std::string LocalChild = StringRef(Child).lower();
#elif defined(__linux__)
  std::string LocalRoot = Root;
  std::string LocalChild = Child;
#else
#error Only support windows and Linux.
#endif
  auto Diff = mismatch(path::begin(LocalRoot), path::end(LocalRoot),
                       path::begin(LocalChild));
  return Diff.first == path::end(LocalRoot) &&
         Diff.second == path::end(LocalChild);
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
  auto &SM = Context.getSourceManager();
  SourceLocation BeginLoc, EndLoc;
  if (SM.isMacroArgExpansion(S->getBeginLoc())) {
    BeginLoc = SM.getSpellingLoc(S->getBeginLoc());
    EndLoc = SM.getSpellingLoc(S->getEndLoc());
  } else {
    BeginLoc = SM.getExpansionLoc(S->getBeginLoc());
    EndLoc = SM.getExpansionLoc(S->getEndLoc());
  }
  auto Length = SM.getDecomposedLoc(EndLoc).second -
                SM.getDecomposedLoc(BeginLoc).second +
                Lexer::MeasureTokenLength(EndLoc, SM, Context.getLangOpts());
  return std::string(SM.getCharacterData(BeginLoc), Length);
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
             Extension == ".c") {
    return TypeCppSource;
  } else if (Extension == ".hpp" || Extension == ".hxx" || Extension == ".h" ||
             Extension == ".hh") {
    return TypeCppHeader;
  } else {
    llvm::errs() << "[ERROR] Not support\"" << Extension << "\" file type!\n";
    std::exit(1);
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

std::vector<std::string> split(const std::string &str, char delim) {
  std::vector<std::string> vs;
  std::stringstream ss(str);
  std::string token;
  while (std::getline(ss, token, delim))
    vs.push_back(token);

  return vs;
}

// Find the innermost (closest) block where S is located
const clang::CompoundStmt *findImmediateBlock(const clang::Stmt *S) {
  if (!S)
    return nullptr;

  auto &Context = syclct::SyclctGlobalInfo::getContext();
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

bool startsWith(std::string str, std::string s) { return str.rfind(s, 0) == 0; }
