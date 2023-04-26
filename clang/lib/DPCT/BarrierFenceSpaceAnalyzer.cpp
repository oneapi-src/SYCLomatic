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

// Functions in this set should not create alias name for input pointer
const std::unordered_set<std::string> AllowedDeviceFunctions = {
        "__popc",
        "atomicAdd",
        "__fetch_builtin_x",
        "__fetch_builtin_y",
        "__fetch_builtin_z",
        "uint4",
        "sqrtf",
        "__expf"};

bool clang::dpct::ReadWriteOrderAnalyzer::visitIterationNode(clang::Stmt *S) {
  Level Lvl;
  Lvl.CurrentLoc = CurrentLevel.CurrentLoc;
  Lvl.LevelBeginLoc = S->getBeginLoc();
  LevelStack.push(CurrentLevel);
  CurrentLevel = Lvl;
  return true;
}
void clang::dpct::ReadWriteOrderAnalyzer::PostVisitIterationNode(
    clang::Stmt *S) {
  if (!CurrentLevel.SyncCallsVec.empty()) {
    CurrentLevel.SyncCallsVec.front().second.Predecessors.push_back(
        clang::SourceRange(CurrentLevel.CurrentLoc, S->getEndLoc()));
    CurrentLevel.SyncCallsVec.back().second.Successors.push_back(
        clang::SourceRange(CurrentLevel.LevelBeginLoc,
                           CurrentLevel.FirstSyncBeginLoc));

    LevelMap.insert(std::make_pair(LevelStack.size(), CurrentLevel));
  }
  CurrentLevel = LevelStack.top();
  LevelStack.pop();
  return;
}

bool clang::dpct::ReadWriteOrderAnalyzer::Visit(clang::IfStmt *IS) {
  // No special process, treat `then` block and `else` block as one block
  return true;
}
void clang::dpct::ReadWriteOrderAnalyzer::PostVisit(clang::IfStmt *IS) {
  // No special process, treat `then` block and `else` block as one block
}

bool clang::dpct::ReadWriteOrderAnalyzer::Visit(clang::ForStmt *FS) {
  return visitIterationNode(FS);
}
void clang::dpct::ReadWriteOrderAnalyzer::PostVisit(clang::ForStmt *FS) {
  PostVisitIterationNode(FS);
}
bool clang::dpct::ReadWriteOrderAnalyzer::Visit(clang::DoStmt *DS) {
  return visitIterationNode(DS);
}
void clang::dpct::ReadWriteOrderAnalyzer::PostVisit(clang::DoStmt *DS) {
  PostVisitIterationNode(DS);
}
bool clang::dpct::ReadWriteOrderAnalyzer::Visit(clang::WhileStmt *WS) {
  return visitIterationNode(WS);
}
void clang::dpct::ReadWriteOrderAnalyzer::PostVisit(clang::WhileStmt *WS) {
  PostVisitIterationNode(WS);
}
bool clang::dpct::ReadWriteOrderAnalyzer::Visit(clang::CallExpr *CE) {
  const clang::FunctionDecl *FuncDecl = CE->getDirectCallee();
  std::string FuncName;
  if (FuncDecl)
    FuncName = FuncDecl->getNameInfo().getName().getAsString();

  if (FuncName == "__syncthreads") {
    if (!clang::dpct::DpctGlobalInfo::findAncestor<SwitchStmt>(CE)) {
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
          isUserDefinedDecl(FD)) {
        return false;
      }
    }
  }
  return true;
}
void clang::dpct::ReadWriteOrderAnalyzer::PostVisit(clang::CallExpr *) {}

bool clang::dpct::ReadWriteOrderAnalyzer::Visit(clang::DeclRefExpr *DRE) {
  // Collect all DREs and its Decl
  DREDeclMap.insert(std::make_pair(DRE, DRE->getDecl()));
  return true;
}
void clang::dpct::ReadWriteOrderAnalyzer::PostVisit(clang::DeclRefExpr *) {}

bool clang::dpct::ReadWriteOrderAnalyzer::Visit(clang::GotoStmt *) {
  // We will further refine it if meet real request.
  // By default, goto/label stmt is not supported.
  return false;
}
void clang::dpct::ReadWriteOrderAnalyzer::PostVisit(clang::GotoStmt *) {}

bool clang::dpct::ReadWriteOrderAnalyzer::Visit(clang::LabelStmt *) {
  // We will further refine it if meet real request.
  // By default, goto/label stmt is not supported.
  return false;
}
void clang::dpct::ReadWriteOrderAnalyzer::PostVisit(clang::LabelStmt *) {}

bool clang::dpct::ReadWriteOrderAnalyzer::Visit(clang::MemberExpr *ME) {
  if (ME->getType()->isPointerType() || ME->getType()->isArrayType()) {
    return false;
  }
  return true;
}
void clang::dpct::ReadWriteOrderAnalyzer::PostVisit(clang::MemberExpr *) {}
bool clang::dpct::ReadWriteOrderAnalyzer::Visit(
    clang::CXXDependentScopeMemberExpr *) {
  return false;
}
void clang::dpct::ReadWriteOrderAnalyzer::PostVisit(
    clang::CXXDependentScopeMemberExpr *) {}
bool clang::dpct::ReadWriteOrderAnalyzer::Visit(
    clang::CXXConstructExpr *CCE) {
  auto Ctor = CCE->getConstructor();
  std::string CtorName = Ctor->getParent()->getQualifiedNameAsString();
  if (AllowedDeviceFunctions.count(CtorName) && !isUserDefinedDecl(Ctor))
    return true;
  return false;
}
void clang::dpct::ReadWriteOrderAnalyzer::PostVisit(
    clang::CXXConstructExpr *) {}

bool clang::dpct::ReadWriteOrderAnalyzer::traverseFunction(
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
          !isUserDefinedDecl(FD)) {
        return true;
      }
    }
  }
  return false;
}

bool clang::dpct::ReadWriteOrderAnalyzer::analyze(
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
    clang::dpct::ReadWriteOrderAnalyzer::CachedResults;

bool clang::dpct::GlobalPointerReferenceCountAnalyzer::Visit(
    clang::DeclRefExpr *DRE) {
  clang::VarDecl *VD = dyn_cast_or_null<VarDecl>(DRE->getDecl());
  if (!VD) {
    return true;
  }
  if ((VD->hasAttr<clang::CUDADeviceAttr>() ||
       VD->hasAttr<clang::HIPManagedAttr>()) &&
      isUserDefinedDecl(VD)) {
    HasGlobalDeviceVariable = true;
  }

  auto I = NonconstPointerDecls.find(VD);
  if (I != NonconstPointerDecls.end()) {
    // Check if pointers in NonconstPointerDecls is passed to other functions
    if (auto FuncCall =
            clang::dpct::DpctGlobalInfo::findAncestor<clang::CallExpr>(DRE)) {
      if (auto FD = FuncCall->getDirectCallee()) {
        if (isUserDefinedDecl(FD) ||
            AllowedDeviceFunctions.count(
                FD->getNameInfo().getName().getAsString())) {
          return false;
        }
      }
    }
    if (!UnderVarDeclOrBinaryOP.empty() && UnderVarDeclOrBinaryOP.top()) {
      if (UnderVarDeclOrBinaryOP.top()->getType()->isPointerType()) {
        if (!UnderVarDeclOrBinaryOP.top()
                 ->getType()
                 ->getPointeeType()
                 .isConstQualified()) {
          VarInfo VI(I->second.ID);
          NonconstPointerDecls.insert(
              std::make_pair(UnderVarDeclOrBinaryOP.top(), VI));
          return true;
        }
      }
    }
    I->second.ReferencedDRENumber = I->second.ReferencedDRENumber + 1;
  }
  return true;
}
void clang::dpct::GlobalPointerReferenceCountAnalyzer::PostVisit(
    clang::DeclRefExpr *) {}

bool clang::dpct::GlobalPointerReferenceCountAnalyzer::Visit(
    clang::BinaryOperator *BO) {
  if (BO->getOpcode() == clang::BinaryOperatorKind::BO_AddAssign ||
      BO->getOpcode() == clang::BinaryOperatorKind::BO_SubAssign ||
      BO->getOpcode() == clang::BinaryOperatorKind::BO_Assign) {
    if (clang::DeclRefExpr *NewDRE =
            dyn_cast_or_null<clang::DeclRefExpr>(BO->getLHS())) {
      if (clang::VarDecl *NewVD =
              dyn_cast_or_null<clang::VarDecl>(NewDRE->getDecl())) {
        UnderVarDeclOrBinaryOP.push(NewVD);
        return true;
      }
    }
  }
  if (UnderVarDeclOrBinaryOP.empty()) {
    UnderVarDeclOrBinaryOP.push(nullptr);
  } else {
    UnderVarDeclOrBinaryOP.push(UnderVarDeclOrBinaryOP.top());
  }
  return true;
}
void clang::dpct::GlobalPointerReferenceCountAnalyzer::PostVisit(
    clang::BinaryOperator *) {
  UnderVarDeclOrBinaryOP.pop();
}

bool clang::dpct::GlobalPointerReferenceCountAnalyzer::Visit(
    clang::VarDecl *VD) {
  if (VD->hasInit()) {
    UnderVarDeclOrBinaryOP.push(VD);
    return true;
  }
  if (UnderVarDeclOrBinaryOP.empty()) {
    UnderVarDeclOrBinaryOP.push(nullptr);
  } else {
    UnderVarDeclOrBinaryOP.push(UnderVarDeclOrBinaryOP.top());
  }
  return true;
}
void clang::dpct::GlobalPointerReferenceCountAnalyzer::PostVisit(
    clang::VarDecl *) {
  UnderVarDeclOrBinaryOP.pop();
}

bool clang::dpct::GlobalPointerReferenceCountAnalyzer::countReference(
    const clang::FunctionDecl *FD) {
  // Collect all non-const pointers which point to fundamental type
  for (auto &Param : FD->parameters()) {
    if (Param->getType()->isPointerType()) {
      if (!Param->getType()->getPointeeType()->isFundamentalType()) {
        return false;
      } else {
        if (!Param->getType()->getPointeeType().isConstQualified()) {
          NonconstPointerDecls.insert(
              std::make_pair(Param, VarInfo(NonconstPointerDecls.size())));
        }
      }
    } else {
      if (!Param->getType()->isFundamentalType()) {
        return false;
      }
    }
  }

  if (!(this->TraverseDecl(const_cast<clang::FunctionDecl *>(FD)))) {
    return false;
  }
  if (HasGlobalDeviceVariable)
    return false;

  // If pointers in NonconstPointerDecls only used once, then we can use
  // local_space
  for (const auto OuterI : NonconstPointerDecls) {
    size_t Count = 0;
    size_t ID = OuterI.second.ID;
    for (const auto InnerI : NonconstPointerDecls) {
      if (InnerI.second.ID == ID) {
        Count = Count + InnerI.second.ReferencedDRENumber;
      }
    }
    if (Count > 1)
      return false;
  }

  return true;
}

bool clang::dpct::GlobalPointerReferenceCountAnalyzer::analyze(
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

  std::string FDLocStr = getHashStrFromLoc(FD->getBeginLoc());
  auto Iter = CachedResults.find(FDLocStr);
  if (Iter != CachedResults.end()) {
    return Iter->second;
  }

  bool Result = countReference(FD);
  CachedResults[FDLocStr] = Result;
  return Result;
}

std::unordered_map<std::string, bool>
    clang::dpct::GlobalPointerReferenceCountAnalyzer::CachedResults;
