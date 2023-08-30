//===---- GroupFunctionAnalyzer.h -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_DPCT_GROUP_FUNCTION_ANALYZER_H
#define CLANG_DPCT_GROUP_FUNCTION_ANALYZER_H

#include "AnalysisInfo.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "llvm/Support/SaveAndRestore.h"

namespace clang::dpct {

/// GroupFunctionCallInControlFlowAnalyzer - Check wether the input code
/// call group function in control flow.
class GroupFunctionCallInControlFlowAnalyzer
    : public RecursiveASTVisitor<GroupFunctionCallInControlFlowAnalyzer> {
  friend class RecursiveASTVisitor<GroupFunctionCallInControlFlowAnalyzer>;
  const ASTContext &Context;
  FunctionDecl *FD = nullptr;
  llvm::SmallVector<Stmt *, 32> Stmts;
  llvm::SmallSet<CallExpr *, 32> SideEffectCallExprs;
  bool SideEffects = false;

#define VISIT_NODE(CLASS)                                                      \
  bool Visit(CLASS *S);                                                        \
  void PostVisit(CLASS *S);                                                    \
  bool Traverse##CLASS(CLASS *Node) {                                          \
    if (!Visit(Node))                                                          \
      return false;                                                            \
    if (!RecursiveASTVisitor<                                                  \
            GroupFunctionCallInControlFlowAnalyzer>::Traverse##CLASS(Node))    \
      return false;                                                            \
    PostVisit(Node);                                                           \
    return true;                                                               \
  }

  VISIT_NODE(ForStmt)
  VISIT_NODE(DoStmt)
  VISIT_NODE(WhileStmt)
  VISIT_NODE(SwitchStmt)
  VISIT_NODE(IfStmt)
  VISIT_NODE(ReturnStmt)
  VISIT_NODE(CallExpr)
#undef VISIT_NODE

  bool isSyncThreadsCallExpr(const CallExpr *CE) const;
  bool isDeviceFunctionCallExprWithSideEffects(const CallExpr *CE) const;
  void checkEnterCondPathAndPush(Stmt *S);
  void checkExitCondPathAndPop(Stmt *S);
  bool checkConstEval(const Stmt *S) const;

public:
  GroupFunctionCallInControlFlowAnalyzer(const ASTContext &Ctx)
      : Context(Ctx) {}
  void checkCallGroupFunctionInControlFlow(FunctionDecl *FD);
  void noteCallGroupFunctionInControlFlow(CallExpr *CE);

  using const_ref_range =
      llvm::iterator_range<decltype(SideEffectCallExprs)::const_iterator>;
  const_ref_range calls() const {
    return const_ref_range(SideEffectCallExprs.begin(),
                           SideEffectCallExprs.end());
  }
};

} // namespace clang::dpct

#endif // CLANG_DPCT_GROUP_FUNCTION_ANALYZER_H
