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

bool clang::dpct::BarrierFenceSpaceAnalyzer::Visit(clang::IfStmt *IS) {
  // No special process, treat as one block
  return true;
}
void clang::dpct::BarrierFenceSpaceAnalyzer::PostVisit(clang::IfStmt *IS) {
  // No special process, treat as one block
}
bool clang::dpct::BarrierFenceSpaceAnalyzer::Visit(clang::SwitchStmt *SS) {
  // No special process, treat as one block
  return true;
}
void clang::dpct::BarrierFenceSpaceAnalyzer::PostVisit(clang::SwitchStmt *SS) {
  // No special process, treat as one block
}

bool clang::dpct::BarrierFenceSpaceAnalyzer::Visit(clang::ForStmt *FS) {
  LoopRange.push_back(FS->getSourceRange());
  return true;
}
void clang::dpct::BarrierFenceSpaceAnalyzer::PostVisit(clang::ForStmt *FS) {
  LoopRange.pop_back();
}
bool clang::dpct::BarrierFenceSpaceAnalyzer::Visit(clang::DoStmt *DS) {
  LoopRange.push_back(DS->getSourceRange());
  return true;
}
void clang::dpct::BarrierFenceSpaceAnalyzer::PostVisit(clang::DoStmt *DS) {
  LoopRange.pop_back();
}
bool clang::dpct::BarrierFenceSpaceAnalyzer::Visit(clang::WhileStmt *WS) {
  LoopRange.push_back(WS->getSourceRange());
  return true;
}
void clang::dpct::BarrierFenceSpaceAnalyzer::PostVisit(clang::WhileStmt *WS) {
  LoopRange.pop_back();
}
bool clang::dpct::BarrierFenceSpaceAnalyzer::Visit(clang::CallExpr *CE) {
  const clang::FunctionDecl *FuncDecl = CE->getDirectCallee();
  std::string FuncName;
  if (FuncDecl)
    FuncName = FuncDecl->getNameInfo().getName().getAsString();

  if (FuncName == "__syncthreads") {
    SyncCallInfo SCI;
    SCI.Predecessors.push_back(
        clang::SourceRange(FD->getBody()->getBeginLoc(), CE->getBeginLoc()));
    SCI.Successors.push_back(
        clang::SourceRange(CE->getEndLoc(), FD->getBody()->getEndLoc()));
    if (!LoopRange.empty()) {
      SCI.Predecessors.push_back(LoopRange.front());
      SCI.Successors.push_back(LoopRange.front());
    }
    SyncCallsVec.push_back(std::make_pair(CE, SCI));
  } else {
    if (auto FD = CE->getDirectCallee()) {
      std::string FuncName = FD->getNameInfo().getName().getAsString();
      if (!AllowedDeviceFunctions.count(FuncName) || isUserDefinedDecl(FD)) {
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
  if (AllowedDeviceFunctions.count(CtorName) && !isUserDefinedDecl(Ctor))
    return true;
  return false;
}
void clang::dpct::BarrierFenceSpaceAnalyzer::PostVisit(
    clang::CXXConstructExpr *) {}

bool clang::dpct::BarrierFenceSpaceAnalyzer::traverseFunction(
    const clang::FunctionDecl *FD) {
  if (!this->TraverseDecl(const_cast<clang::FunctionDecl *>(FD))) {
    return false;
  }
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
          !isUserDefinedDecl(FD)) {
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

  this->FD = FD;
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
  std::map<clang::ValueDecl *, std::set<clang::SourceLocation>> DRELocs;
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
      DRELocs[Iter.second].insert(Iter.first->getBeginLoc());
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
  for (auto &SyncCall : SyncCallsVec) {
    bool Result = true;
    for (auto &Pair : DRELocs) {
      if (Pair.second.size() == 1) {
        continue;
      }
      std::optional<bool> DREInPredecessors;
      for (auto &Loc : Pair.second) {
        if (containsMacro(Loc, SyncCall.second.Predecessors) ||
            containsMacro(Loc, SyncCall.second.Successors)) {
          Result = false;
          break;
        }
        if (isInRanges(Loc, SyncCall.second.Predecessors)) {
          if (DREInPredecessors.has_value()) {
            if (DREInPredecessors.value() != true) {
              Result = false;
              break;
            }
          } else {
            DREInPredecessors = true;
          }
        }
        if (isInRanges(Loc, SyncCall.second.Successors)) {
          if (DREInPredecessors.has_value()) {
            if (DREInPredecessors.value() != false) {
              Result = false;
              break;
            }
          } else {
            DREInPredecessors = false;
          }
        }
      }
    }
    CachedResults[FDLoc][getHashStrFromLoc(SyncCall.first->getBeginLoc())] =
        Result;
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
