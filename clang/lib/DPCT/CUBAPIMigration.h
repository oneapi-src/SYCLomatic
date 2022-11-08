//===--------------- CUBAPIMigration.h --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_DPCT_CUBAPIMIGRATION_H
#define CLANG_DPCT_CUBAPIMIGRATION_H

#include "ASTTraversal.h"
#include "llvm/ADT/StringRef.h"

namespace clang {
namespace dpct {

class CubTypeRule : public NamedMigrationRule<CubTypeRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);

  static bool CanMappingToSyclNativeBinaryOp(StringRef OpTypeName);
  static bool CanMappingToSyclBinaryOp(StringRef OpTypeName);
};

class CubDeviceLevelRule : public NamedMigrationRule<CubDeviceLevelRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

class CubRule : public NamedMigrationRule<CubRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);

private:
  struct ParamAssembler {
    std::string &ParamListRef;
    ParamAssembler(std::string &List) : ParamListRef(List){};
    ParamAssembler &operator<<(std::string Param) {
      if (Param.empty()) {
        return *this;
      }
      if (ParamListRef.empty()) {
        ParamListRef = Param;
      } else {
        ParamListRef += ", " + Param;
      }
      return *this;
    };
  };
  static int PlaceholderIndex;
  std::string getOpRepl(const Expr *Operator);
  void processCubDeclStmt(const DeclStmt *DS);
  void processCubTypeDef(const TypedefDecl *TD);
  void processCubFuncCall(const CallExpr *CE, bool FuncCallUsed = false);
  void processCubMemberCall(const CXXMemberCallExpr *MC);
  void processTypeLoc(const TypeLoc *TL);

  void processDeviceLevelFuncCall(const CallExpr *CE, bool FuncCallUsed);
  void processThreadLevelFuncCall(const CallExpr *CE, bool FuncCallUsed);
  void processWarpLevelFuncCall(const CallExpr *CE, bool FuncCallUsed);
  void processBlockLevelMemberCall(const CXXMemberCallExpr *MC);
  void processWarpLevelMemberCall(const CXXMemberCallExpr *MC);

public:
  /// Pseudo code:
  /// loop_1 {
  ///   ...
  ///   tempstorage = nullptr;
  ///   ...
  ///   loop_j {
  ///     ...
  ///     loop_N {
  ///       func(tempstorage, ...);
  ///       tempstorage = ...
  ///     }
  ///   }
  /// }
  /// The callexpr is redundant if following two conditions are meet:
  /// (1) No modified reference between tempstorage initialization and callexpr.
  /// (2) No modified reference in loop_j or deeper loop.
  /// The redundant callexpr can be remove safely.
  static bool isRedundantCallExpr(const CallExpr *C);

  /// Analyze temp_storage and temp_storage_size argument to determine
  /// whether these two argument and related decl or cudaMalloc can be
  /// removed.
  /// If the d_temp_storage and temp_storage_bytes only used in
  /// Reduce/Min/Max/Sum and cudaMalloc, then we can remove related decl
  /// and cudaMalloc*.
  /// Pseudo code:
  /// loop_1 {
  ///   ...
  ///   tempstorage = nullptr;
  ///   ...
  ///   loop_j {
  ///     ...
  ///     loop_N {
  ///       func(tempstorage, ...);
  ///       tempstorage = ...
  ///     }
  ///   }
  /// }
  /// The callexpr is redundant if following two conditions are meet:
  /// (1) No modified reference between tempstorage initialization and callexpr.
  /// (2) No modified reference in loop_j or deeper loop.
  /// The redundant callexpr can be remove safely.
  static void removeRedundantTempVar(const CallExpr *CE);
};

} // namespace dpct
} // namespace clang

#endif // CLANG_DPCT_CUBAPIMIGRATION_H
