//===--------------- LibraryAPIMigration.h --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DPCT_LIBRARY_API_MIGRATION_H
#define DPCT_LIBRARY_API_MIGRATION_H

#include "ExprAnalysis.h"
#include "MapNames.h"

#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ParentMapContext.h"
#include "clang/Format/Format.h"
#include "clang/Frontend/CompilerInstance.h"

#include <string>
#include <vector>

namespace clang {
namespace dpct {

struct LibraryAPIStmts {
  LibraryAPIStmts() {}

  LibraryAPIStmts &operator<<(const LibraryAPIStmts &InputStmts) {
    S.insert(S.end(), InputStmts.S.begin(), InputStmts.S.end());
    return *this;
  }
  LibraryAPIStmts &operator<<(const std::vector<std::string> &InputStmts) {
    S.insert(S.end(), InputStmts.begin(), InputStmts.end());
    return *this;
  }
  LibraryAPIStmts &operator<<(const std::string &InputStmt) {
    if (InputStmt.empty())
      return *this;

    S.push_back(InputStmt);
    return *this;
  }

  std::string getAsString(std::string IndentStr, bool IsNLAtBegin) {
    std::ostringstream OS;
    for (const auto &Stmt : S) {
      if (IsNLAtBegin)
        OS << getNL() << IndentStr << Stmt; // For suffix string
      else
        OS << Stmt << getNL() << IndentStr; // For prefix string
    }
    return OS.str();
  }
  std::vector<std::string> S;
};

struct LibraryMigrationFlags {
  bool NeedUseLambda = false;
  bool CanAvoidUsingLambda = false;
  bool IsMacroArg = false;
  bool CanAvoidBrace = false;
  bool IsAssigned = false;
  bool MoveOutOfMacro = false;
  std::string OriginStmtType;
  bool IsPrefixEmpty = false;
  bool IsSuffixEmpty = false;
  bool IsPrePrefixEmpty = false;
};
struct LibraryMigrationLocations {
  SourceLocation PrefixInsertLoc;
  SourceLocation SuffixInsertLoc;
  SourceLocation OuterInsertLoc;
  SourceLocation FuncNameBegin;
  SourceLocation FuncCallEnd;
  SourceLocation OutOfMacroInsertLoc;
  unsigned int Len = 0;
  SourceLocation FuncPtrDeclBegin;
  SourceLocation FuncPtrDeclHandleTypeBegin;
  unsigned int FuncPtrDeclLen = 0;
};
struct LibraryMigrationStrings {
  std::string PrePrefixInsertStr;
  std::string PrefixInsertStr;
  std::string Repl;
  std::string SuffixInsertStr;
  std::string IndentStr;
};

void initVars(const CallExpr *CE, const VarDecl *VD, const BinaryOperator *BO,
              LibraryMigrationFlags &Flags,
              LibraryMigrationStrings &ReplaceStrs,
              LibraryMigrationLocations &Locations);

} // namespace dpct
} // namespace clang

#endif // !DPCT_LIBRARY_API_MIGRATION_H
