//===---- GroupFunctionAnalyzer.cpp -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GroupFunctionAnalyzer.h"

using namespace clang;
using namespace dpct;

bool GroupFunctionCallInControlFlowAnalyzer::Visit(IfStmt *S) {
  checkEnterCondPathAndPush(S);
  return true;
}
void GroupFunctionCallInControlFlowAnalyzer::PostVisit(IfStmt *S) {
  checkExitCondPathAndPop(S);
}

bool GroupFunctionCallInControlFlowAnalyzer::Visit(SwitchStmt *S) {
  checkEnterCondPathAndPush(S);
  return true;
}
void GroupFunctionCallInControlFlowAnalyzer::PostVisit(SwitchStmt *S) {
  checkExitCondPathAndPop(S);
}

bool GroupFunctionCallInControlFlowAnalyzer::Visit(ForStmt *S) {
  checkEnterCondPathAndPush(S);
  return true;
}
void GroupFunctionCallInControlFlowAnalyzer::PostVisit(ForStmt *S) {
  checkExitCondPathAndPop(S);
}

bool GroupFunctionCallInControlFlowAnalyzer::Visit(DoStmt *S) {
  checkEnterCondPathAndPush(S);
  return true;
}
void GroupFunctionCallInControlFlowAnalyzer::PostVisit(DoStmt *S) {
  checkExitCondPathAndPop(S);
}

bool GroupFunctionCallInControlFlowAnalyzer::Visit(WhileStmt *S) {
  checkEnterCondPathAndPush(S);
  return true;
}
void GroupFunctionCallInControlFlowAnalyzer::PostVisit(WhileStmt *S) {
  checkExitCondPathAndPop(S);
}

bool GroupFunctionCallInControlFlowAnalyzer::Visit(ReturnStmt *S) {
  Stmts.push_back(S);
  return true;
}
void GroupFunctionCallInControlFlowAnalyzer::PostVisit(ReturnStmt *S) {
  Stmts.pop_back();
}

bool GroupFunctionCallInControlFlowAnalyzer::Visit(CallExpr *CE) {
  // This CallExpr in conditional code.
  if (!Stmts.empty()) {

    // Whether the callee is __syncthreads().
    if (isSyncThreadsCallExpr(CE)) {
      noteCallGroupFunctionInControlFlow(CE);
      return true;
    }

    // Whether the callee is another device function on analysis scope.
    if (isDeviceFunctionCallExprWithSideEffects(CE)) {
      noteCallGroupFunctionInControlFlow(CE);

      // We need to recursively check the device function.
      checkCallGroupFunctionInControlFlow(CE->getDirectCallee());
    }
  }

  return true;
}
void GroupFunctionCallInControlFlowAnalyzer::PostVisit(CallExpr *) {}

bool GroupFunctionCallInControlFlowAnalyzer::isSyncThreadsCallExpr(
    const CallExpr *CE) const {
  const auto *FD = CE->getDirectCallee();
  return FD && FD->getName() == "__syncthreads" &&
         (FD->hasAttr<CUDADeviceAttr>() || FD->hasAttr<CUDAGlobalAttr>());
}

bool GroupFunctionCallInControlFlowAnalyzer::
    isDeviceFunctionCallExprWithSideEffects(const CallExpr *CE) const {
  const auto *FD = CE->getDirectCallee();
  if (!FD || FD->isTemplateInstantiation())
    return false;
  auto FnInfo = DeviceFunctionDecl::LinkRedecls(FD);
  if (!FnInfo)
    return false;
  return FnInfo->hasCallGroupFunctionInControlFlow();
}

void GroupFunctionCallInControlFlowAnalyzer::
    checkCallGroupFunctionInControlFlow(FunctionDecl *FD) {
  llvm::SaveAndRestore<bool> SvaedSideEffects(SideEffects);
  llvm::SaveAndRestore<FunctionDecl *> SavedFD(this->FD);
  auto FnInfo = DeviceFunctionDecl::LinkRedecls(FD);
  if (!FnInfo)
    return;

  if (FnInfo->hasSideEffectsAnalyzed())
    return;

  FnInfo->setHasSideEffectsAnalyzed();
  this->FD = FD;
  SideEffects = false;
  this->TraverseDecl(FD);
  FnInfo->setCallGroupFunctionInControlFlow(SideEffects);
}

void GroupFunctionCallInControlFlowAnalyzer::noteCallGroupFunctionInControlFlow(
    CallExpr *CE) {
  if (!FD || FD->isTemplateInstantiation())
    return;
  SideEffects = true;
  SideEffectCallExprs.insert(CE);
  auto FnInfo = DeviceFunctionDecl::LinkRedecls(FD);
  auto CallInfo = FnInfo->addCallee(CE);
  CallInfo->setHasSideEffects();
}

void GroupFunctionCallInControlFlowAnalyzer::checkEnterCondPathAndPush(
    Stmt *S) {
  if (!checkConstEval(S))
    Stmts.push_back(S);
}
void GroupFunctionCallInControlFlowAnalyzer::checkExitCondPathAndPop(Stmt *S) {
  if (!Stmts.empty() && Stmts.back() == S)
    Stmts.pop_back();
}

bool GroupFunctionCallInControlFlowAnalyzer::checkConstEval(
    const Stmt *S) const {
  if (!S)
    return false;

  if (const auto *E = dyn_cast<Expr>(S)) {
    if (!E->isValueDependent() && E->isEvaluatable(Context))
      return true;
  }

  switch (S->getStmtClass()) {
  case Stmt::ForStmtClass: {
    const auto *F = dyn_cast<ForStmt>(S);
    if (F->getInit() && !checkConstEval(F->getInit()))
      return false;
    if (F->getCond() && !checkConstEval(F->getCond()))
      return false;
    if (F->getInc() && !checkConstEval(F->getInc()))
      return false;
    return true;
  }
  case Stmt::DoStmtClass: {
    const auto *D = dyn_cast<DoStmt>(S);
    return checkConstEval(D->getCond());
  }
  case Stmt::WhileStmtClass: {
    const auto *W = dyn_cast<WhileStmt>(S);
    return checkConstEval(W->getCond());
  }
  case Stmt::SwitchStmtClass: {
    const auto *W = dyn_cast<SwitchStmt>(S);
    return checkConstEval(W->getCond());
  }
  case Stmt::IfStmtClass: {
    const auto *I = dyn_cast<IfStmt>(S);
    if (I->isConstexpr() || I->isConsteval())
      return true;
    return checkConstEval(I->getCond());
  }
  default:
    break;
  }
  return false;
}
