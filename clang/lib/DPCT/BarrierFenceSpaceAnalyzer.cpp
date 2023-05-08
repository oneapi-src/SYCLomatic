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

bool clang::dpct::ReadWriteOrderAnalyzer::Visit(clang::IfStmt *IS) {
  // No special process, treat as one block
  return true;
}
void clang::dpct::ReadWriteOrderAnalyzer::PostVisit(clang::IfStmt *IS) {
  // No special process, treat as one block
}
bool clang::dpct::ReadWriteOrderAnalyzer::Visit(clang::SwitchStmt *SS) {
  // No special process, treat as one block
  return true;
}
void clang::dpct::ReadWriteOrderAnalyzer::PostVisit(clang::SwitchStmt *SS) {
  // No special process, treat as one block
}

bool clang::dpct::ReadWriteOrderAnalyzer::Visit(clang::ForStmt *FS) {
  LoopRange.push_back(FS->getSourceRange());
  return true;
}
void clang::dpct::ReadWriteOrderAnalyzer::PostVisit(clang::ForStmt *FS) {
  LoopRange.pop_back();
}
bool clang::dpct::ReadWriteOrderAnalyzer::Visit(clang::DoStmt *DS) {
  LoopRange.push_back(DS->getSourceRange());
  return true;
}
void clang::dpct::ReadWriteOrderAnalyzer::PostVisit(clang::DoStmt *DS) {
  LoopRange.pop_back();
}
bool clang::dpct::ReadWriteOrderAnalyzer::Visit(clang::WhileStmt *WS) {
  LoopRange.push_back(WS->getSourceRange());
  return true;
}
void clang::dpct::ReadWriteOrderAnalyzer::PostVisit(clang::WhileStmt *WS) {
  LoopRange.pop_back();
}
bool clang::dpct::ReadWriteOrderAnalyzer::Visit(clang::CallExpr *CE) {
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
  std::map<clang::ValueDecl*, std::set<clang::SourceLocation>> DRELocs;
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
    clang::dpct::ReadWriteOrderAnalyzer::CachedResults;

bool canBeTreatedAsPrivateMemoryAccess(int KernelDim,
                                       const clang::DeclRefExpr *DRE,
                                       const clang::FunctionDecl *FD) {
  if (KernelDim != 1) {
    return false;
  }
  // Check if this DRE(Ptr) matches pattern: Ptr[Idx]
  // clang-format off
  //    ArraySubscriptExpr <col:7, col:16> 'float' lvalue
  //    |-ImplicitCastExpr <col:7> 'float *' <LValueToRValue>
  //    | `-DeclRefExpr <col:7> 'float *' lvalue ParmVar 0x555a6c216d68 'Ptr' 'float *'
  //    `-ImplicitCastExpr <col:12> 'int' <LValueToRValue>
  //      `-DeclRefExpr <col:12> 'int' lvalue Var 0x555a6c217078 'Idx' 'int'
  // clang-format on
  auto IsParentArraySubscriptExpr =
      [](const clang::DeclRefExpr *Node) -> const clang::ArraySubscriptExpr * {
    auto ICE =
        clang::dpct::DpctGlobalInfo::findParent<clang::ImplicitCastExpr>(Node);
    if (!ICE || (ICE->getCastKind() != clang::CastKind::CK_LValueToRValue))
      return nullptr;
    auto ASE =
        clang::dpct::DpctGlobalInfo::findParent<clang::ArraySubscriptExpr>(ICE);
    if (!ASE)
      return nullptr;
    return ASE;
  };
  const clang::ArraySubscriptExpr *ASE = IsParentArraySubscriptExpr(DRE);
  if (!ASE)
    return false;

  // IdxVD must be local variable and must be defined in this function
  const clang::DeclRefExpr *IdxDRE =
      dyn_cast_or_null<clang::DeclRefExpr>(ASE->getIdx()->IgnoreImpCasts());
  if (!IdxDRE)
    return false;
  const clang::VarDecl *IdxVD =
      dyn_cast_or_null<clang::VarDecl>(IdxDRE->getDecl());
  
  if (!IdxVD->isLocalVarDecl())
    return false;
  const clang::FunctionDecl *IdxVDContext =
      clang::dpct::DpctGlobalInfo::findAncestor<clang::FunctionDecl>(IdxVD);
  if (!IdxVDContext || (IdxVDContext != FD))
    return false;

  // VD's DRE should only be used as rvalue
  auto DREMatcher = clang::ast_matchers::findAll(
      clang::ast_matchers::declRefExpr().bind("DRE"));
  auto MatchedResults = clang::ast_matchers::match(
      DREMatcher, *(FD->getBody()), clang::dpct::DpctGlobalInfo::getContext());
  for (const auto &Res : MatchedResults) {
    const clang::DeclRefExpr *RefDRE = Res.getNodeAs<clang::DeclRefExpr>("DRE");
    if (RefDRE->getDecl() != IdxVD) {
      continue;
    }
    auto ICE = clang::dpct::DpctGlobalInfo::findParent<clang::ImplicitCastExpr>(
        RefDRE);
    if (!ICE || (ICE->getCastKind() != clang::CastKind::CK_LValueToRValue))
      return false;
  }

  // Check if Index variable match pattern: blockIdx.x * blockDim.x + threadIdx.x
  if (!IdxVD->hasInit())
    return false;
  auto IsIterationSpaceBuiltinVar =
      [](const clang::PseudoObjectExpr *Node, const std::string &BuiltinNameRef,
         const std::string &FieldNameRef) -> bool {
    if (!Node)
      return false;
    using namespace clang::ast_matchers;
    auto BuiltinMatcher = findAll(
        memberExpr(
            hasObjectExpression(opaqueValueExpr(hasSourceExpression(
                declRefExpr(to(varDecl(hasAnyName("threadIdx", "blockDim",
                                                  "blockIdx"))))
                    .bind("declRefExpr")))),
            hasParent(implicitCastExpr(hasParent(callExpr().bind("callExpr")))))
            .bind("memberExpr"));
    auto MatchedResults =
        match(BuiltinMatcher, *Node, clang::dpct::DpctGlobalInfo::getContext());
    if (MatchedResults.size() != 1)
      return false;
    const auto Res = MatchedResults[0];
    auto ME = Res.getNodeAs<clang::MemberExpr>("memberExpr");
    auto CE = Res.getNodeAs<clang::CallExpr>("callExpr");
    auto DRE = Res.getNodeAs<clang::DeclRefExpr>("declRefExpr");
    if (!ME || !CE || !DRE)
      return false;
    auto POE =
        clang::dpct::DpctGlobalInfo::findParent<clang::PseudoObjectExpr>(CE);
    if (!POE || (POE != Node))
      return false;
    StringRef BuiltinName = DRE->getDecl()->getName();
    StringRef FieldName = ME->getMemberDecl()->getName();
    if (BuiltinName == BuiltinNameRef && FieldName == FieldNameRef)
      return true;
    return false;
  };
  const clang::Expr* InitExpr = IdxVD->getInit()->IgnoreImpCasts();
  // Case 1: blockIdx.x * blockDim.x + threadIdx.x
  // Case 2: blockDim.x * blockIdx.x + threadIdx.x
  // Case 3: threadIdx.x + blockIdx.x * blockDim.x
  // Case 4: threadIdx.x + blockDim.x * blockIdx.x
  const clang::BinaryOperator* BOAdd = dyn_cast<clang::BinaryOperator>(InitExpr);
  if (!BOAdd || BOAdd->getOpcode() != clang::BinaryOperatorKind::BO_Add)
    return false;
  const clang::BinaryOperator *BOMul =
      dyn_cast<clang::BinaryOperator>(BOAdd->getLHS());
  if (BOMul && IsIterationSpaceBuiltinVar(
                   dyn_cast<clang::PseudoObjectExpr>(BOAdd->getRHS()),
                   "threadIdx", "__fetch_builtin_x")) {
    if (IsIterationSpaceBuiltinVar(
            dyn_cast<clang::PseudoObjectExpr>(BOMul->getLHS()), "blockIdx",
            "__fetch_builtin_x") &&
        IsIterationSpaceBuiltinVar(
            dyn_cast<clang::PseudoObjectExpr>(BOMul->getRHS()), "blockDim",
            "__fetch_builtin_x")) {
      // Case 1
      return true;
    }
    if (IsIterationSpaceBuiltinVar(
            dyn_cast<clang::PseudoObjectExpr>(BOMul->getRHS()), "blockIdx",
            "__fetch_builtin_x") &&
        IsIterationSpaceBuiltinVar(
            dyn_cast<clang::PseudoObjectExpr>(BOMul->getLHS()), "blockDim",
            "__fetch_builtin_x")) {
      // Case 2
      return true;
    }
    return false;
  }
  BOMul = dyn_cast<clang::BinaryOperator>(BOAdd->getRHS());
  if (BOMul && IsIterationSpaceBuiltinVar(
                   dyn_cast<clang::PseudoObjectExpr>(BOAdd->getLHS()),
                   "threadIdx", "__fetch_builtin_x")) {
    if (IsIterationSpaceBuiltinVar(
            dyn_cast<clang::PseudoObjectExpr>(BOMul->getLHS()), "blockIdx",
            "__fetch_builtin_x") &&
        IsIterationSpaceBuiltinVar(
            dyn_cast<clang::PseudoObjectExpr>(BOMul->getRHS()), "blockDim",
            "__fetch_builtin_x")) {
      // Case 3
      return true;
    }
    if (IsIterationSpaceBuiltinVar(
            dyn_cast<clang::PseudoObjectExpr>(BOMul->getRHS()), "blockIdx",
            "__fetch_builtin_x") &&
        IsIterationSpaceBuiltinVar(
            dyn_cast<clang::PseudoObjectExpr>(BOMul->getLHS()), "blockDim",
            "__fetch_builtin_x")) {
      // Case 4
      return true;
    }
    return false;
  }
  return false;
}

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
            !AllowedDeviceFunctions.count(
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

    if (!canBeTreatedAsPrivateMemoryAccess(KernelDim, DRE, FD)) {
      I->second.ReferencedDRENumber = I->second.ReferencedDRENumber + 1;
    }
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

  this->FD = FD;
  auto QueryKernelDim = [](const clang::FunctionDecl *FD) -> int {
    const auto DFD = DpctGlobalInfo::getInstance().findDeviceFunctionDecl(FD);
    if (!DFD)
      return 3;
    const auto FuncInfo = DFD->getFuncInfo();
    if (!FuncInfo)
      return 3;
    const auto MVM =
        MemVarMap::getHeadWithoutPathCompression(&(FuncInfo->getVarMap()));
    if (!MVM)
      return 3;
    return MVM->Dim;
  };
  KernelDim = QueryKernelDim(FD);

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
