//===--------------- BarrierFenceSpaceAnalyzer.cpp ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "BarrierFenceSpaceAnalyzer.h"
#include "AnalysisInfo.h"

#include "llvm/Support/Casting.h"

#include <algorithm>

using namespace llvm;

bool clang::dpct::BarrierFenceSpaceAnalyzer::Visit(clang::ForStmt *FS) {
  Level Lvl;
  Lvl.CurrentLoc = CurrentLevel.CurrentLoc;
  Lvl.LevelBeginLoc = FS->getBeginLoc();
  LevelStack.push(CurrentLevel);
  CurrentLevel = Lvl;
  return true;
}
void clang::dpct::BarrierFenceSpaceAnalyzer::PostVisit(clang::ForStmt *FS) {
  if (!CurrentLevel.SyncCallsVec.empty()) {
    CurrentLevel.SyncCallsVec.front().second.Predecessors.push_back(
        clang::SourceRange(CurrentLevel.CurrentLoc, FS->getEndLoc()));
    CurrentLevel.SyncCallsVec.back().second.Successors.push_back(
        clang::SourceRange(CurrentLevel.LevelBeginLoc,
                           CurrentLevel.FirstSyncBeginLoc));

    LevelMap.insert(std::make_pair(LevelStack.size(), CurrentLevel));
  }
  CurrentLevel = LevelStack.top();
  LevelStack.pop();
  return;
}
bool clang::dpct::BarrierFenceSpaceAnalyzer::Visit(clang::CallExpr *CE) {
  const clang::FunctionDecl *FuncDecl = CE->getDirectCallee();
  std::string FuncName;
  if (FuncDecl)
    FuncName = FuncDecl->getNameInfo().getName().getAsString();

  if (FuncName == "__syncthreads") {
    if (!clang::dpct::DpctGlobalInfo::findAncestor<IfStmt>(CE) &&
        !clang::dpct::DpctGlobalInfo::findAncestor<DoStmt>(CE) &&
        !clang::dpct::DpctGlobalInfo::findAncestor<WhileStmt>(CE) &&
        !clang::dpct::DpctGlobalInfo::findAncestor<SwitchStmt>(CE)) {
      if (LevelStack.size() > 1) {
        // We will further refine it if meet real request.
        return false;
      }

      if (CurrentLevel.FirstSyncBeginLoc.isInvalid()) {
        CurrentLevel.FirstSyncBeginLoc = CE->getBeginLoc();
      }

      clang::SourceRange Range(CurrentLevel.CurrentLoc, CE->getBeginLoc());
      if (!CurrentLevel.SyncCallsVec.empty()) {
        CurrentLevel.SyncCallsVec.back().second.Successors.push_back(Range);
      }
      CurrentLevel.SyncCallsVec.emplace_back(CE, SyncCallInfo({Range}, {}));

      unsigned int LevelNum = LevelStack.size();
      auto UpperBoundIter = LevelMap.upper_bound(LevelNum);
      for (auto Iter = UpperBoundIter; Iter != LevelMap.end(); Iter++) {
        Iter->second.SyncCallsVec.back().second.Successors.push_back(
            clang::SourceRange(Iter->second.CurrentLoc, CE->getBeginLoc()));
        LevelVec.push_back(Iter->second);
      }
      LevelMap.erase(UpperBoundIter, LevelMap.end());

      CurrentLevel.CurrentLoc = CE->getEndLoc();
    }
  } else {
    if (auto FD = CE->getDirectCallee()) {
      std::string FuncName = FD->getNameInfo().getName().getAsString();
      if (!AllowedDeviceFunctions.count(FuncName) ||
          isUserDefinedFunction(FD)) {
        return false;
      }
    }
  }
  return true;
}
void clang::dpct::BarrierFenceSpaceAnalyzer::PostVisit(clang::CallExpr *) {}

bool clang::dpct::BarrierFenceSpaceAnalyzer::Visit(clang::DeclRefExpr *DRE) {
  // Collect all DREs and its Decl
  DREDeclMap.insert(std::make_pair(DRE, DRE->getDecl()));
  return true;
}
void clang::dpct::BarrierFenceSpaceAnalyzer::PostVisit(clang::DeclRefExpr *) {}

bool clang::dpct::BarrierFenceSpaceAnalyzer::Visit(clang::GotoStmt *) {
  // We will further refine it if meet real request.
  // By default, goto/label stmt is not supported.
  return false;
}
void clang::dpct::BarrierFenceSpaceAnalyzer::PostVisit(clang::GotoStmt *) {}

bool clang::dpct::BarrierFenceSpaceAnalyzer::Visit(clang::LabelStmt *) {
  // We will further refine it if meet real request.
  // By default, goto/label stmt is not supported.
  return false;
}
void clang::dpct::BarrierFenceSpaceAnalyzer::PostVisit(clang::LabelStmt *) {}

bool clang::dpct::BarrierFenceSpaceAnalyzer::Visit(clang::MemberExpr *ME) {
  if (ME->getType()->isPointerType() || ME->getType()->isArrayType()) {
    return false;
  }
  return true;
}
void clang::dpct::BarrierFenceSpaceAnalyzer::PostVisit(clang::MemberExpr *) {}
bool clang::dpct::BarrierFenceSpaceAnalyzer::Visit(
    clang::CXXDependentScopeMemberExpr *) {
  return false;
}
void clang::dpct::BarrierFenceSpaceAnalyzer::PostVisit(
    clang::CXXDependentScopeMemberExpr *) {}
bool clang::dpct::BarrierFenceSpaceAnalyzer::Visit(
    clang::CXXConstructExpr *CCE) {
  auto Ctor = CCE->getConstructor();
  std::string CtorName = Ctor->getParent()->getQualifiedNameAsString();
  if (AllowedDeviceFunctions.count(CtorName) && !isUserDefinedFunction(Ctor))
    return true;
  return false;
}
void clang::dpct::BarrierFenceSpaceAnalyzer::PostVisit(
    clang::CXXConstructExpr *) {}

bool clang::dpct::BarrierFenceSpaceAnalyzer::traverseFunction(
    const clang::FunctionDecl *FD) {
  CurrentLevel.CurrentLoc = FD->getBody()->getBeginLoc();
  CurrentLevel.LevelBeginLoc = FD->getBody()->getBeginLoc();

  if (!this->TraverseDecl(const_cast<clang::FunctionDecl *>(FD))) {
    return false;
  }

  if (!CurrentLevel.SyncCallsVec.empty()) {
    CurrentLevel.SyncCallsVec.back().second.Successors.push_back(
        clang::SourceRange(CurrentLevel.CurrentLoc, FD->getEndLoc()));
  }
  for (auto &Iter : LevelMap) {
    Iter.second.SyncCallsVec.back().second.Successors.push_back(
        clang::SourceRange(Iter.second.CurrentLoc, FD->getEndLoc()));
    LevelVec.push_back(Iter.second);
  }
  LevelMap.clear();
  LevelVec.push_back(CurrentLevel);
  return true;
}

bool isPointerOperationSafe(const clang::Expr *Pointer) {
  auto P = getNonImplicitCastNonParenExprParentStmt(Pointer);
  if (!P)
    return false;

  if (auto ASE = dyn_cast<clang::ArraySubscriptExpr>(P)) {
    auto PP = getNonImplicitCastNonParenExprParentStmt(ASE);
    if (auto UO = dyn_cast_or_null<clang::UnaryOperator>(PP)) {
      if (UO->getOpcode() == clang::UnaryOperatorKind::UO_AddrOf) {
        return isPointerOperationSafe(UO);
      }
    }
    return true;
  }
  if (auto UO = dyn_cast<clang::UnaryOperator>(P)) {
    if (UO->getOpcode() == clang::UnaryOperatorKind::UO_Deref) {
      auto PP = getNonImplicitCastNonParenExprParentStmt(UO);
      if (auto OuterUO = dyn_cast_or_null<clang::UnaryOperator>(PP)) {
        if (OuterUO->getOpcode() == clang::UnaryOperatorKind::UO_AddrOf) {
          return isPointerOperationSafe(OuterUO);
        }
      }
      return true;
    }
    return false;
  }

  // Special case for atomicAdd.
  // If the Pointer is the first arg of atomicAdd function, return true.
  if (auto CE = dyn_cast<clang::CallExpr>(P)) {
    if (auto FD = CE->getDirectCallee()) {
      std::string FuncName = FD->getNameInfo().getName().getAsString();
      if (FuncName == "atomicAdd" && (CE->getArg(0) == Pointer) &&
          !isUserDefinedFunction(FD)) {
        return true;
      }
    }
  }
  return false;
}

bool clang::dpct::BarrierFenceSpaceAnalyzer::canSetLocalFenceSpace(
    const clang::CallExpr *CE) {
  if (CE->getBeginLoc().isMacroID() || CE->getEndLoc().isMacroID())
    return false;
  auto FD = dpct::DpctGlobalInfo::findAncestor<clang::FunctionDecl>(CE);
  if (!FD)
    return false;
  if (!FD->hasAttr<clang::CUDAGlobalAttr>())
    return false;
  if (FD->getTemplateSpecializationKind() !=
          TemplateSpecializationKind::TSK_Undeclared ||
      FD->getDescribedFunctionTemplate()) {
    return false;
  }

  CELoc = getHashStrFromLoc(CE->getBeginLoc());
  FDLoc = getHashStrFromLoc(FD->getBeginLoc());

  auto FDIter = CachedResults.find(FDLoc);
  if (FDIter != CachedResults.end()) {
    auto CEIter = FDIter->second.find(CELoc);
    if (CEIter != FDIter->second.end()) {
      return CEIter->second;
    } else {
      return false;
    }
  }

  // analyze this FD
  // Traverse AST, analysis the context info of kernel calling sycthreads()
  //   1. Find each syncthreads call's predecessor parts and successor parts.
  //   2. When meet __device__ function is called, if the device function is not
  //      in allow list, exit.
  //   3. Check all DREs(Declare Ref Expr), if __device__ variable is used,
  //      exit.
  if (!traverseFunction(FD)) {
    setFalseForThisFunctionDecl();
    return false;
  }

  std::unordered_set<clang::ParmVarDecl *> PointerParams;
  for (auto &Param : FD->parameters()) {
    if (Param->getType()->isPointerType()) {
      PointerParams.insert(Param);
    }
  }

  // Check whether each input pointer of kernel is "safe" (no an alias name
  // used) or not. If it is not "safe", exit.
  std::set<clang::SourceLocation> DRELocs;
  for (auto &Iter : DREDeclMap) {
    if (auto VD = dyn_cast_or_null<VarDecl>(Iter.second)) {
      if (VD->hasAttr<clang::CUDADeviceAttr>() &&
          !(VD->getName().str() == "threadIdx" ||
            VD->getName().str() == "blockIdx" ||
            VD->getName().str() == "blockDim" ||
            VD->getName().str() == "gridDim")) {
        setFalseForThisFunctionDecl();
        return false; // not support to check __device__ variables
      }
    }
    if (dyn_cast<clang::ParmVarDecl>(Iter.second) &&
        PointerParams.count(dyn_cast<clang::ParmVarDecl>(Iter.second))) {
      if (!isPointerOperationSafe(Iter.first)) {
        setFalseForThisFunctionDecl();
        return false; // avoid the alias of input pointers
      }
      DRELocs.insert(Iter.first->getBeginLoc());
    }
  }

  auto isInRanges = [](clang::SourceLocation SL,
                       std::vector<clang::SourceRange> Ranges) -> bool {
    auto &SM = dpct::DpctGlobalInfo::getSourceManager();
    for (auto &Range : Ranges) {
      if (SM.getFileOffset(Range.getBegin()) < SM.getFileOffset(SL) &&
          SM.getFileOffset(SL) < SM.getFileOffset(Range.getEnd())) {
        return true;
      }
    }
    return false;
  };

  auto containsMacro = [](clang::SourceLocation SL,
                          std::vector<clang::SourceRange> Ranges) -> bool {
    if (SL.isMacroID())
      return true;
    for (auto &Range : Ranges) {
      if (Range.getBegin().isMacroID() || Range.getEnd().isMacroID()) {
        return true;
      }
    }
    return false;
  };

  // DRELoc: input pointer parameter's usage location
  // For a syncthreads call, tool will use local_space fence for this barrier,
  // if it meets:
  // For arbitrary input pointer of kernel, all DREs of this pointer are only
  // used in either predecessor parts or successor parts
  for (auto &I : LevelVec) {
    for (auto &SyncCall : I.SyncCallsVec) {
      bool Result = true;
      for (auto &Loc : DRELocs) {
        if (containsMacro(Loc, SyncCall.second.Predecessors) ||
            containsMacro(Loc, SyncCall.second.Successors) ||
            (isInRanges(Loc, SyncCall.second.Predecessors) &&
             isInRanges(Loc, SyncCall.second.Successors))) {
          Result = false;
          break;
        }
      }
      CachedResults[FDLoc][getHashStrFromLoc(SyncCall.first->getBeginLoc())] =
          Result;
    }
  }

  // find the result in the new map
  FDIter = CachedResults.find(FDLoc);
  if (FDIter != CachedResults.end()) {
    auto CEIter = FDIter->second.find(CELoc);
    if (CEIter != FDIter->second.end()) {
      return CEIter->second;
    } else {
      return false;
    }
  }
  return false;
}

std::unordered_map<std::string, std::unordered_map<std::string, bool>>
    clang::dpct::BarrierFenceSpaceAnalyzer::CachedResults;

// Functions in this set should not create alias name for input pointer
const std::unordered_set<std::string>
    clang::dpct::BarrierFenceSpaceAnalyzer::AllowedDeviceFunctions = {
        "__popc",
        "atomicAdd",
        "__fetch_builtin_x",
        "__fetch_builtin_y",
        "__fetch_builtin_z",
        "uint4"};
