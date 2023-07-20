//===--------------- BarrierFenceSpaceAnalyzer.cpp ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "BarrierFenceSpaceAnalyzer.h"
#include "llvm/Support/Casting.h"
#include <algorithm>

using namespace llvm;

bool clang::dpct::BarrierFenceSpaceAnalyzer::Visit(const IfStmt *IS) {
  // No special process, treat as one block
  return true;
}
void clang::dpct::BarrierFenceSpaceAnalyzer::PostVisit(const IfStmt *IS) {
  // No special process, treat as one block
}
bool clang::dpct::BarrierFenceSpaceAnalyzer::Visit(const SwitchStmt *SS) {
  // No special process, treat as one block
  return true;
}
void clang::dpct::BarrierFenceSpaceAnalyzer::PostVisit(const SwitchStmt *SS) {
  // No special process, treat as one block
}

bool clang::dpct::BarrierFenceSpaceAnalyzer::Visit(const ForStmt *FS) {
  LoopRange.push_back(FS->getSourceRange());
  return true;
}
void clang::dpct::BarrierFenceSpaceAnalyzer::PostVisit(
    const clang::ForStmt *FS) {
  LoopRange.pop_back();
}
bool clang::dpct::BarrierFenceSpaceAnalyzer::Visit(const DoStmt *DS) {
  LoopRange.push_back(DS->getSourceRange());
  return true;
}
void clang::dpct::BarrierFenceSpaceAnalyzer::PostVisit(const DoStmt *DS) {
  LoopRange.pop_back();
}
bool clang::dpct::BarrierFenceSpaceAnalyzer::Visit(const WhileStmt *WS) {
  LoopRange.push_back(WS->getSourceRange());
  return true;
}
void clang::dpct::BarrierFenceSpaceAnalyzer::PostVisit(const WhileStmt *WS) {
  LoopRange.pop_back();
}
bool clang::dpct::BarrierFenceSpaceAnalyzer::Visit(const CallExpr *CE) {
  const FunctionDecl *FuncDecl = CE->getDirectCallee();
  if (!FuncDecl)
    return true;
  std::string FuncName = FuncDecl->getNameInfo().getName().getAsString();

  for (const auto &Arg : CE->arguments())
    DeviceFunctionCallArgs.insert(Arg);

  if (FuncName == "__syncthreads") {
    SyncCallInfo SCI;
    SCI.Predecessors.push_back(
        SourceRange(FD->getBody()->getBeginLoc(), CE->getBeginLoc()));
    SCI.Successors.push_back(
        SourceRange(CE->getEndLoc(), FD->getBody()->getEndLoc()));
    if (!LoopRange.empty()) {
      SCI.Predecessors.push_back(LoopRange.front());
      SCI.Successors.push_back(LoopRange.front());
    }
    SyncCallsVec.push_back(std::make_pair(CE, SCI));
  }
  return true;
}
void clang::dpct::BarrierFenceSpaceAnalyzer::PostVisit(const CallExpr *) {}

bool clang::dpct::BarrierFenceSpaceAnalyzer::Visit(const DeclRefExpr *DRE) {
  // Collect all DREs and its Decl
  const auto PVD = dyn_cast<ParmVarDecl>(DRE->getDecl());
  if (!PVD)
    return true;

  TypeAnalyzer TA;
  int ParamterTypeKind = TA.getInputParamterTypeKind(PVD->getType());
  if (ParamterTypeKind == 1) {
    return true;
  } else if (ParamterTypeKind == -1) {
#ifdef __DEBUG_BARRIER_FENCE_SPACE_ANALYZER
    std::cout << "Return False case A: "
              << PVD->getBeginLoc().printToString(
                     DpctGlobalInfo::getSourceManager())
              << ", Type:" << DpctGlobalInfo::getTypeName(PVD->getType())
              << std::endl;
#endif
    return false;
  }

  const auto &Iter = DefUseMap.find(PVD);
  if (Iter != DefUseMap.end()) {
    Iter->second.insert(DRE);
  } else {
    std::set<const DeclRefExpr *> Set;
    Set.insert(DRE);
    DefUseMap.insert(std::make_pair(PVD, Set));
  }
  return true;
}
void clang::dpct::BarrierFenceSpaceAnalyzer::PostVisit(const DeclRefExpr *) {}

bool clang::dpct::BarrierFenceSpaceAnalyzer::Visit(const GotoStmt *GS) {
  // We will further refine it if meet real request.
  // By default, goto/label stmt is not supported.
#ifdef __DEBUG_BARRIER_FENCE_SPACE_ANALYZER
  std::cout << "Return False case B: "
            << GS->getBeginLoc().printToString(
                   DpctGlobalInfo::getSourceManager())
            << std::endl;
#endif
  return false;
}
void clang::dpct::BarrierFenceSpaceAnalyzer::PostVisit(const GotoStmt *) {}

bool clang::dpct::BarrierFenceSpaceAnalyzer::Visit(const LabelStmt *LS) {
  // We will further refine it if meet real request.
  // By default, goto/label stmt is not supported.
#ifdef __DEBUG_BARRIER_FENCE_SPACE_ANALYZER
  std::cout << "Return False case C: "
            << LS->getBeginLoc().printToString(
                   DpctGlobalInfo::getSourceManager())
            << std::endl;
#endif
  return false;
}
void clang::dpct::BarrierFenceSpaceAnalyzer::PostVisit(const LabelStmt *) {}

bool clang::dpct::BarrierFenceSpaceAnalyzer::Visit(const MemberExpr *ME) {
  if (ME->getType()->isPointerType() || ME->getType()->isArrayType()) {
#ifdef __DEBUG_BARRIER_FENCE_SPACE_ANALYZER
    std::cout << "Return False case D: "
              << ME->getBeginLoc().printToString(
                     DpctGlobalInfo::getSourceManager())
              << std::endl;
#endif
    return false;
  }
  return true;
}
void clang::dpct::BarrierFenceSpaceAnalyzer::PostVisit(const MemberExpr *) {}
bool clang::dpct::BarrierFenceSpaceAnalyzer::Visit(
    const CXXDependentScopeMemberExpr *CDSME) {
#ifdef __DEBUG_BARRIER_FENCE_SPACE_ANALYZER
  std::cout << "Return False case E: "
            << CDSME->getBeginLoc().printToString(
                   DpctGlobalInfo::getSourceManager())
            << std::endl;
#endif
  return false;
}
void clang::dpct::BarrierFenceSpaceAnalyzer::PostVisit(
    const CXXDependentScopeMemberExpr *) {}

bool clang::dpct::BarrierFenceSpaceAnalyzer::Visit(
    const CXXConstructExpr *CCE) {
  for (const auto &Arg : CCE->arguments())
    DeviceFunctionCallArgs.insert(Arg);
  return true;
}
void clang::dpct::BarrierFenceSpaceAnalyzer::PostVisit(
    const CXXConstructExpr *) {}

std::set<const clang::DeclRefExpr *>
clang::dpct::BarrierFenceSpaceAnalyzer::matchAllDRE(const VarDecl *TargetDecl,
                                                    const Stmt *Range) {
  std::set<const DeclRefExpr *> Set;
  if (!TargetDecl || !Range) {
    return Set;
  }
  auto DREMatcher = ast_matchers::findAll(
      ast_matchers::declRefExpr(ast_matchers::isDeclSameAs(TargetDecl))
          .bind("DRE"));
  auto MatchedResults =
      ast_matchers::match(DREMatcher, *Range, DpctGlobalInfo::getContext());
  for (auto Node : MatchedResults) {
    if (auto DRE = Node.getNodeAs<DeclRefExpr>("DRE"))
      Set.insert(DRE);
  }
  return Set;
}

std::pair<std::set<const clang::DeclRefExpr *>,
          std::set<const clang::VarDecl *>>
clang::dpct::BarrierFenceSpaceAnalyzer::isAssignedToAnotherDREOrVD(
    const DeclRefExpr *CurrentDRE) {
  std::set<const DeclRefExpr *> ResultDRESet;
  std::set<const VarDecl *> ResultVDSet;
  findAncestorInFunctionScope<Stmt>(
      CurrentDRE, FD,
      [&](const DynTypedNode &Parent,
          const DynTypedNode &Current) -> const void * {
        const auto BO = Parent.get<BinaryOperator>();
        const auto VD = Parent.get<VarDecl>();
        if (BO && BO->isAssignmentOp() &&
            (BO->getRHS() == Current.get<Expr>()) &&
            BO->getLHS()->getType()->isPointerType()) {
          auto DREMatcher =
              ast_matchers::findAll(ast_matchers::declRefExpr().bind("DRE"));
          auto MatchedResults = ast_matchers::match(
              DREMatcher, *(BO->getRHS()), DpctGlobalInfo::getContext());
          for (auto Node : MatchedResults) {
            if (auto DRE = Node.getNodeAs<DeclRefExpr>("DRE"))
              ResultDRESet.insert(DRE);
          }
        } else if (VD && VD->getType()->isPointerType()) {
          if (!VD->getType()->getPointeeType().isConstQualified())
            ResultVDSet.insert(VD);
        }
        return nullptr;
      });
#ifdef __DEBUG_BARRIER_FENCE_SPACE_ANALYZER
  const auto &SM = DpctGlobalInfo::getSourceManager();
  std::cout << "CurrentDRE:" << CurrentDRE->getBeginLoc().printToString(SM)
            << std::endl;
  for (const auto Item : ResultDRESet) {
    std::cout << "    AnotherDRE:" << Item->getBeginLoc().printToString(SM)
              << std::endl;
  }
  for (const auto Item : ResultVDSet) {
    std::cout << "    AnotherVD:" << Item->getBeginLoc().printToString(SM)
              << std::endl;
  }
#endif
  return std::make_pair(ResultDRESet, ResultVDSet);
}

clang::dpct::BarrierFenceSpaceAnalyzer::AccessMode
clang::dpct::BarrierFenceSpaceAnalyzer::getAccessKind(
    const DeclRefExpr *CurrentDRE) {
  bool FoundDeref = false;
  bool FoundBO = false;
  const BinaryOperator *BO = nullptr;
  const UnaryOperator *UOInBO = nullptr;
  const ArraySubscriptExpr *ASEInBO = nullptr;
  const ParmVarDecl *PVD = nullptr;

  auto &Context = DpctGlobalInfo::getContext();
  DynTypedNode Current = DynTypedNode::create(*CurrentDRE);
  DynTypedNodeList Parents = Context.getParents(Current);
  while (!Parents.empty()) {
    if (Parents[0].get<FunctionDecl>() && Parents[0].get<FunctionDecl>() == FD)
      break;

    const auto CE = Parents[0].get<CallExpr>();
    UOInBO = Parents[0].get<UnaryOperator>();
    ASEInBO = Parents[0].get<ArraySubscriptExpr>();
    BO = Parents[0].get<BinaryOperator>();

    if (CE && CE->getDirectCallee()) {
      int Idx = 0;
      for (const auto Arg : CE->arguments()) {
        if (Arg == Current.get<Expr>())
          break;
        Idx++;
      }
      if (dyn_cast<CXXOperatorCallExpr>(CE) || dyn_cast<CXXMemberCallExpr>(CE))
        Idx--;
      if (Idx >= 0)
        PVD = CE->getDirectCallee()->getParamDecl(Idx);
      else {
        if (const auto CMD = dyn_cast<CXXMethodDecl>(CE->getDirectCallee())) {
          if (CMD->isImplicit())
            return AccessMode::Write;
        }
        return AccessMode::ReadWrite;
      }
    }
    if (!FoundBO && BO && BO->isAssignmentOp() &&
        (BO->getLHS() == Current.get<Expr>()) && FoundDeref) {
      FoundBO = true;
    } else if (!FoundDeref && UOInBO &&
               (UOInBO->getOpcode() == UnaryOperatorKind::UO_Deref)) {
      FoundDeref = true;
    } else if (!FoundDeref && ASEInBO &&
               (ASEInBO->getBase() == Current.get<Expr>())) {
      FoundDeref = true;
    }

    Current = Parents[0];
    Parents = Context.getParents(Current);
  }

  if (PVD) {
    if (PVD->getType()->isPointerType()) {
      if (PVD->getType()->getPointeeType().isConstQualified())
        return AccessMode::Read;
      return AccessMode::ReadWrite;
    }
    if (PVD->getType()->isLValueReferenceType()) {
      if (PVD->getType().getNonReferenceType().isConstQualified())
        return AccessMode::Read;
      return AccessMode::ReadWrite;
    }
    return AccessMode::Read;
  }

  if (BO) {
    if (BO->getOpcode() == BinaryOperatorKind::BO_Assign) {
      return AccessMode::Write;
    }
    return AccessMode::ReadWrite;
  }

  // No BO or Call in ancestor
  return AccessMode::Read;
}

bool clang::dpct::BarrierFenceSpaceAnalyzer::isPotentialGlobalMemoryAccess(
    std::shared_ptr<DeviceFunctionInfo> DFI, bool IsInGlobalFunction) {
  if (!DFI)
    return false;

  const MemVarInfoMap &MVIM =
      DFI->getVarMap().getMap(MemVarInfo::VarScope::Global);
  for (const auto &VarInfo : MVIM) {
    auto Attr = VarInfo.second->getAttr();
    if (Attr == MemVarInfo::VarAttrKind::Device ||
        Attr == MemVarInfo::VarAttrKind::Managed)
      return true;
  }
  for (const auto &Call : DFI->getCallExprMap()) {
    if (Call.second->getName() == "__syncthreads" && !IsInGlobalFunction) {
      // Currently we do not support the analysis for __syncthreads in
      // __device__ function
      return false;
    }
    if (isPotentialGlobalMemoryAccess(Call.second->getFuncInfo(), false))
      return true;
  }
  return false;
}

clang::dpct::BarrierFenceSpaceAnalyzerResult
clang::dpct::BarrierFenceSpaceAnalyzer::canSetLocalFenceSpace(
    const CallExpr *CE, bool IsDryRun) {
  this->IsDryRun = IsDryRun;
#ifdef __DEBUG_BARRIER_FENCE_SPACE_ANALYZER
  std::cout << "BarrierFenceSpaceAnalyzer Analyzing ..." << std::endl;
#endif
  if (CE->getBeginLoc().isMacroID() || CE->getEndLoc().isMacroID()) {
#ifdef __DEBUG_BARRIER_FENCE_SPACE_ANALYZER
    std::cout << "Return False case F: CE->getBeginLoc().isMacroID() || "
                 "CE->getEndLoc().isMacroID()"
              << std::endl;
#endif
    return BarrierFenceSpaceAnalyzerResult(false, false, GlobalFunctionName);
  }
  auto FD = DpctGlobalInfo::findAncestor<FunctionDecl>(CE);
  if (!FD) {
#ifdef __DEBUG_BARRIER_FENCE_SPACE_ANALYZER
    std::cout << "Return False case G: !FD" << std::endl;
#endif
    return BarrierFenceSpaceAnalyzerResult(false, false, GlobalFunctionName);
  }
  if (!FD->hasAttr<CUDAGlobalAttr>()) {
#ifdef __DEBUG_BARRIER_FENCE_SPACE_ANALYZER
    std::cout << "Return False case H: !FD->hasAttr<CUDAGlobalAttr>()"
              << std::endl;
#endif
    return BarrierFenceSpaceAnalyzerResult(false, false, GlobalFunctionName);
  }

  CELoc = getHashStrFromLoc(CE->getBeginLoc());
  FDLoc = getHashStrFromLoc(FD->getBeginLoc());

  auto FDIter = CachedResults.find(FDLoc);
  if (!IsDryRun) {
    if (FDIter != CachedResults.end()) {
      auto CEIter = FDIter->second.find(CELoc);
      if (CEIter != FDIter->second.end()) {
        return CEIter->second;
      } else {
        return BarrierFenceSpaceAnalyzerResult(false, false,
                                               GlobalFunctionName);
      }
    }
  }

  if (isPotentialGlobalMemoryAccess(DeviceFunctionDecl::LinkRedecls(FD),
                                    true)) {
    setFalseForThisFunctionDecl();
#ifdef __DEBUG_BARRIER_FENCE_SPACE_ANALYZER
    std::cout << "Return False case I: Found device/managed variable usage"
              << std::endl;
#endif
    return BarrierFenceSpaceAnalyzerResult(false, false, GlobalFunctionName);
  }

  this->FD = FD;
  GlobalFunctionName = FD->getDeclName().getAsString();

  auto QueryKernelDim = [](const FunctionDecl *FD) -> int {
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

#ifdef __DEBUG_BARRIER_FENCE_SPACE_ANALYZER
  std::cout << "Before traversing current __global__ function" << std::endl;
#endif

  // analyze this FD
  // Traverse AST, analysis the context info of kernel calling sycthreads()
  //   1. Find each syncthreads call's predecessor parts and successor parts.
  //   2. When meet __device__ function is called, if the device function is not
  //      in allow list, exit.
  //   3. Check all DREs(Declare Ref Expr), if __device__ variable is used,
  //      exit.
  if (!this->TraverseDecl(const_cast<FunctionDecl *>(FD))) {
    setFalseForThisFunctionDecl();
    return BarrierFenceSpaceAnalyzerResult(false, false, GlobalFunctionName);
  }

  auto getSize =
      [](const std::unordered_map<const ParmVarDecl *,
                                  std::set<const DeclRefExpr *>> &DefUseMap)
      -> std::size_t {
    std::size_t Size = 0;
    for (const auto &Pair : DefUseMap) {
      Size = Size + Pair.second.size();
    }
    return Size;
  };

#ifdef __DEBUG_BARRIER_FENCE_SPACE_ANALYZER
  std::cout << "DefUseMap init value:" << std::endl;
  for (const auto &Pair : DefUseMap) {
    const auto &SM = DpctGlobalInfo::getSourceManager();
    std::cout << "Decl:" << Pair.first->getBeginLoc().printToString(SM)
              << std::endl;
    for (const auto &Item : Pair.second) {
      std::cout << "    DRE:" << Item->getBeginLoc().printToString(SM)
                << std::endl;
    }
  }
#endif

  // Collect all used positions
  std::size_t MapSize = getSize(DefUseMap);
  do {
    MapSize = getSize(DefUseMap);
    std::set<const DeclRefExpr *> NewDRESet;
    for (auto &Pair : DefUseMap) {
      const ParmVarDecl *CurDecl = Pair.first;
      std::set<const DeclRefExpr *> CurDRESet = Pair.second;
      std::set<const DeclRefExpr *> MatchedResult =
          matchAllDRE(CurDecl, FD->getBody());
      CurDRESet.insert(MatchedResult.begin(), MatchedResult.end());
      NewDRESet.clear();
      for (const auto &DRE : CurDRESet) {
        const auto &SetPair = isAssignedToAnotherDREOrVD(DRE);
        for (const auto AnotherDRE : SetPair.first) {
          std::set<const DeclRefExpr *> AnotherDREMatchedResult = matchAllDRE(
              dyn_cast_or_null<VarDecl>(AnotherDRE->getDecl()), FD->getBody());
          NewDRESet.insert(AnotherDREMatchedResult.begin(),
                           AnotherDREMatchedResult.end());
        }
        for (const auto AnotherVD : SetPair.second) {
          std::set<const DeclRefExpr *> AnotherDREMatchedResult =
              matchAllDRE(AnotherVD, FD->getBody());
          NewDRESet.insert(AnotherDREMatchedResult.begin(),
                           AnotherDREMatchedResult.end());
        }
      }
      if (!NewDRESet.empty())
        Pair.second.insert(NewDRESet.begin(), NewDRESet.end());
    }
  } while (getSize(DefUseMap) != MapSize);

#ifdef __DEBUG_BARRIER_FENCE_SPACE_ANALYZER
  std::cout << "DefUseMap after collection:" << std::endl;
  for (const auto &Pair : DefUseMap) {
    const auto &SM = DpctGlobalInfo::getSourceManager();
    std::cout << "Decl:" << Pair.first->getBeginLoc().printToString(SM)
              << std::endl;
    for (const auto &Item : Pair.second) {
      std::cout << "    DRE:" << Item->getBeginLoc().printToString(SM)
                << std::endl;
    }
  }
#endif

  std::map<const ParmVarDecl *,
           std::set<std::pair<const DeclRefExpr *, AccessMode>>>
      DeclUsedMap;
  for (const auto &Pair : DefUseMap) {
    for (const auto &Item : Pair.second) {
      if (isAccessingMemory(Item)) {
        if (isNoOverlappingAccessAmongWorkItems(KernelDim, Item)) {
          MayDependOn1DKernel = true;
        } else {
          DeclUsedMap[Pair.first].insert(
              std::make_pair(Item, getAccessKind(Item)));
        }
      }
    }
  }

  // Convert DRE to Location for comparing
  std::map<const ParmVarDecl *, std::set<std::pair<SourceLocation, AccessMode>>>
      DeclUsedLocsMap;
  for (const auto &Pair : DeclUsedMap) {
    for (const auto &Item : Pair.second) {
      DeclUsedLocsMap[Pair.first].insert(
          std::make_pair(Item.first->getBeginLoc(), Item.second));
    }
  }

#ifdef __DEBUG_BARRIER_FENCE_SPACE_ANALYZER
  std::cout << "===== DeclUsedLocsMap contnet: =====" << std::endl;
  for (const auto &DeclSetPair : DeclUsedLocsMap) {
    const auto &SM = DpctGlobalInfo::getSourceManager();
    std::cout << "Decl:" << DeclSetPair.first->getBeginLoc().printToString(SM)
              << std::endl;
    for (const auto &Pair : DeclSetPair.second) {
      std::cout << "    DRE:" << Pair.first.printToString(SM)
                << ", AccessMode:" << (int)(Pair.second) << std::endl;
    }
  }
  std::cout << "===== DeclUsedLocsMap contnet end =====" << std::endl;
#endif
#ifdef __DEBUG_BARRIER_FENCE_SPACE_ANALYZER
  std::cout << "===== SyncCall info contnet: =====" << std::endl;
  for (const auto &SyncCall : SyncCallsVec) {
    const auto &SM = DpctGlobalInfo::getSourceManager();
    std::cout << "SyncCall:" << SyncCall.first->getBeginLoc().printToString(SM)
              << std::endl;
    std::cout << "    Predecessors:" << std::endl;
    for (const auto &Range : SyncCall.second.Predecessors) {
      std::cout << "        [" << Range.getBegin().printToString(SM) << ", "
                << Range.getEnd().printToString(SM) << "]" << std::endl;
    }
    std::cout << "    Successors:" << std::endl;
    for (const auto &Range : SyncCall.second.Successors) {
      std::cout << "        [" << Range.getBegin().printToString(SM) << ", "
                << Range.getEnd().printToString(SM) << "]" << std::endl;
    }
  }
  std::cout << "===== SyncCall info contnet end =====" << std::endl;
#endif

  if (IsDryRun) {
    for (auto &SyncCall : SyncCallsVec) {
      if (CE == SyncCall.first) {
        return BarrierFenceSpaceAnalyzerResult(
            isValidAccessPattern(DeclUsedLocsMap, SyncCall.second),
            MayDependOn1DKernel, GlobalFunctionName);
      }
    }
    return BarrierFenceSpaceAnalyzerResult(false, false, GlobalFunctionName);
  }
  for (auto &SyncCall : SyncCallsVec) {
    CachedResults[FDLoc][getHashStrFromLoc(SyncCall.first->getBeginLoc())] =
        BarrierFenceSpaceAnalyzerResult(
            isValidAccessPattern(DeclUsedLocsMap, SyncCall.second),
            MayDependOn1DKernel, GlobalFunctionName);
  }

  // find the result in the new map
  FDIter = CachedResults.find(FDLoc);
  if (FDIter != CachedResults.end()) {
    auto CEIter = FDIter->second.find(CELoc);
    if (CEIter != FDIter->second.end()) {
      return CEIter->second;
    } else {
      return BarrierFenceSpaceAnalyzerResult(false, false, GlobalFunctionName);
    }
  }
  return BarrierFenceSpaceAnalyzerResult(false, false, GlobalFunctionName);
}

bool clang::dpct::BarrierFenceSpaceAnalyzer::
    isNoOverlappingAccessAmongWorkItems(int KernelDim, const DeclRefExpr *DRE) {
  using namespace ast_matchers;
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
      [](const DeclRefExpr *Node) -> const ArraySubscriptExpr * {
    auto ICE = DpctGlobalInfo::findParent<ImplicitCastExpr>(Node);
    if (!ICE || (ICE->getCastKind() != CastKind::CK_LValueToRValue))
      return nullptr;
    auto ASE = DpctGlobalInfo::findParent<ArraySubscriptExpr>(ICE);
    if (!ASE)
      return nullptr;
    return ASE;
  };
  const ArraySubscriptExpr *ASE = IsParentArraySubscriptExpr(DRE);
  if (!ASE)
    return false;

  // IdxVD must be local variable and must be defined in this function
  const DeclRefExpr *IdxDRE =
      dyn_cast_or_null<DeclRefExpr>(ASE->getIdx()->IgnoreImpCasts());
  if (!IdxDRE)
    return false;
  const VarDecl *IdxVD = dyn_cast_or_null<VarDecl>(IdxDRE->getDecl());
  if (IdxVD->getKind() != Decl::Var)
    return false;
  const auto *IdxFD = dyn_cast_or_null<FunctionDecl>(IdxVD->getDeclContext());
  if (!IdxFD)
    return false;
  const Stmt *IdxVDContext = IdxFD->getBody();

  // VD's DRE should only be used as rvalue
  auto DREMatcher = findAll(declRefExpr(isDeclSameAs(IdxVD)).bind("DRE"));
  auto MatchedResults =
      match(DREMatcher, *IdxVDContext, DpctGlobalInfo::getContext());
  for (const auto &Res : MatchedResults) {
    const DeclRefExpr *RefDRE = Res.getNodeAs<DeclRefExpr>("DRE");
    auto ICE = DpctGlobalInfo::findParent<ImplicitCastExpr>(RefDRE);
    if (!ICE || (ICE->getCastKind() != CastKind::CK_LValueToRValue))
      return false;
  }

  // Check if Index variable match pattern: blockIdx.x * blockDim.x +
  // threadIdx.x
  if (!IdxVD->hasInit())
    return false;
  auto IsIterationSpaceBuiltinVar =
      [](const PseudoObjectExpr *Node, const std::string &BuiltinNameRef,
         const std::string &FieldNameRef) -> bool {
    if (!Node)
      return false;
    auto BuiltinMatcher = findAll(
        memberExpr(hasObjectExpression(opaqueValueExpr(hasSourceExpression(
                       declRefExpr(to(varDecl(hasAnyName(
                                       "threadIdx", "blockDim", "blockIdx"))))
                           .bind("declRefExpr")))),
                   hasParent(implicitCastExpr(
                       hasParent(callExpr(hasParent(pseudoObjectExpr()))))))
            .bind("memberExpr"));
    auto MatchedResults =
        match(BuiltinMatcher, *Node, DpctGlobalInfo::getContext());
    if (MatchedResults.size() != 1)
      return false;
    const auto Res = MatchedResults[0];
    auto ME = Res.getNodeAs<MemberExpr>("memberExpr");
    auto DRE = Res.getNodeAs<DeclRefExpr>("declRefExpr");
    if (!ME || !DRE)
      return false;
    StringRef BuiltinName = DRE->getDecl()->getName();
    StringRef FieldName = ME->getMemberDecl()->getName();
    if (BuiltinName == BuiltinNameRef && FieldName == FieldNameRef)
      return true;
    return false;
  };
  const Expr *InitExpr = IdxVD->getInit()->IgnoreImpCasts();
  // Case 1: blockIdx.x * blockDim.x + threadIdx.x
  // Case 2: blockDim.x * blockIdx.x + threadIdx.x
  // Case 3: threadIdx.x + blockIdx.x * blockDim.x
  // Case 4: threadIdx.x + blockDim.x * blockIdx.x
  const BinaryOperator *BOAdd = dyn_cast<BinaryOperator>(InitExpr);
  if (!BOAdd || BOAdd->getOpcode() != BinaryOperatorKind::BO_Add)
    return false;
  const BinaryOperator *BOMul = dyn_cast<BinaryOperator>(BOAdd->getLHS());
  if (BOMul &&
      IsIterationSpaceBuiltinVar(dyn_cast<PseudoObjectExpr>(BOAdd->getRHS()),
                                 "threadIdx", "__fetch_builtin_x")) {
    if (IsIterationSpaceBuiltinVar(dyn_cast<PseudoObjectExpr>(BOMul->getLHS()),
                                   "blockIdx", "__fetch_builtin_x") &&
        IsIterationSpaceBuiltinVar(dyn_cast<PseudoObjectExpr>(BOMul->getRHS()),
                                   "blockDim", "__fetch_builtin_x")) {
      // Case 1
      return true;
    }
    if (IsIterationSpaceBuiltinVar(dyn_cast<PseudoObjectExpr>(BOMul->getRHS()),
                                   "blockIdx", "__fetch_builtin_x") &&
        IsIterationSpaceBuiltinVar(dyn_cast<PseudoObjectExpr>(BOMul->getLHS()),
                                   "blockDim", "__fetch_builtin_x")) {
      // Case 2
      return true;
    }
    return false;
  }
  BOMul = dyn_cast<BinaryOperator>(BOAdd->getRHS());
  if (BOMul &&
      IsIterationSpaceBuiltinVar(dyn_cast<PseudoObjectExpr>(BOAdd->getLHS()),
                                 "threadIdx", "__fetch_builtin_x")) {
    if (IsIterationSpaceBuiltinVar(dyn_cast<PseudoObjectExpr>(BOMul->getLHS()),
                                   "blockIdx", "__fetch_builtin_x") &&
        IsIterationSpaceBuiltinVar(dyn_cast<PseudoObjectExpr>(BOMul->getRHS()),
                                   "blockDim", "__fetch_builtin_x")) {
      // Case 3
      return true;
    }
    if (IsIterationSpaceBuiltinVar(dyn_cast<PseudoObjectExpr>(BOMul->getRHS()),
                                   "blockIdx", "__fetch_builtin_x") &&
        IsIterationSpaceBuiltinVar(dyn_cast<PseudoObjectExpr>(BOMul->getLHS()),
                                   "blockDim", "__fetch_builtin_x")) {
      // Case 4
      return true;
    }
    return false;
  }
  return false;
}

bool clang::dpct::BarrierFenceSpaceAnalyzer::containsMacro(
    const SourceLocation &SL, const SyncCallInfo &SCI) {
  if (SL.isMacroID())
    return true;
  for (auto &Range : SCI.Predecessors) {
    if (Range.getBegin().isMacroID() || Range.getEnd().isMacroID()) {
      return true;
    }
  }
  for (auto &Range : SCI.Successors) {
    if (Range.getBegin().isMacroID() || Range.getEnd().isMacroID()) {
      return true;
    }
  }
  return false;
}

bool clang::dpct::BarrierFenceSpaceAnalyzer::isAccessingMemory(
    const DeclRefExpr *DRE) {
  auto &Context = DpctGlobalInfo::getContext();
  DynTypedNode Current = DynTypedNode::create(*DRE);
  DynTypedNodeList Parents = Context.getParents(Current);
  while (!Parents.empty()) {
    if (Parents[0].get<FunctionDecl>() && Parents[0].get<FunctionDecl>() == FD)
      break;
    const auto UO = Parents[0].get<UnaryOperator>();
    const auto ASE = Parents[0].get<ArraySubscriptExpr>();
    if (UO && (UO->getOpcode() == UnaryOperatorKind::UO_Deref)) {
      return true;
    } else if (ASE && (ASE->getBase() == Current.get<Expr>())) {
      return true;
    }
    Current = Parents[0];
    Parents = Context.getParents(Current);
  }
  return false;
}

bool clang::dpct::BarrierFenceSpaceAnalyzer::isValidAccessPattern(
    const std::map<const ParmVarDecl *,
                   std::set<std::pair<SourceLocation, AccessMode>>>
        &DeclUsedLocsMap,
    const SyncCallInfo &SCI) {
#ifdef __DEBUG_BARRIER_FENCE_SPACE_ANALYZER
  std::cout << "===== isValidAccessPattern =====" << std::endl;
#endif
  for (auto &DeclUsedLocPair : DeclUsedLocsMap) {
    bool FoundRead = false;
    bool FoundWrite = false;
    bool FoundWriteInPredecessors = false;
    bool FoundWriteInSuccessors = false;
    for (auto &LocModePair : DeclUsedLocPair.second) {
      if (LocModePair.first.isMacroID() ||
          (LocModePair.second == AccessMode::ReadWrite)) {
#ifdef __DEBUG_BARRIER_FENCE_SPACE_ANALYZER
        std::cout << "isValidAccessPattern False case 1" << std::endl;
#endif
        return false;
      }
      if (LocModePair.second == AccessMode::Read) {
        FoundRead = true;
      } else if (LocModePair.second == AccessMode::Write) {
        FoundWrite = true;
        if (isInRanges(LocModePair.first, SCI.Predecessors)) {
          FoundWriteInPredecessors = true;
        }
        if (isInRanges(LocModePair.first, SCI.Successors)) {
          FoundWriteInSuccessors = true;
        }
      }
      if (FoundRead && FoundWrite) {
#ifdef __DEBUG_BARRIER_FENCE_SPACE_ANALYZER
        std::cout << "isValidAccessPattern False case 2" << std::endl;
#endif
        return false;
      }
      if (FoundWriteInPredecessors && FoundWriteInSuccessors) {
#ifdef __DEBUG_BARRIER_FENCE_SPACE_ANALYZER
        std::cout << "isValidAccessPattern False case 3" << std::endl;
#endif
        return false;
      }
    }
  }
#ifdef __DEBUG_BARRIER_FENCE_SPACE_ANALYZER
  std::cout << "isValidAccessPattern True" << std::endl;
#endif
  return true;
}

std::unordered_map<
    std::string, std::unordered_map<
                     std::string, clang::dpct::BarrierFenceSpaceAnalyzerResult>>
    clang::dpct::BarrierFenceSpaceAnalyzer::CachedResults;

bool clang::dpct::BarrierFenceSpaceAnalyzer::isInRanges(
    SourceLocation SL, std::vector<SourceRange> Ranges) {
  auto &SM = DpctGlobalInfo::getSourceManager();
  for (auto &Range : Ranges) {
    if (SM.getFileOffset(Range.getBegin()) < SM.getFileOffset(SL) &&
        SM.getFileOffset(SL) < SM.getFileOffset(Range.getEnd())) {
      return true;
    }
  }
  return false;
}
