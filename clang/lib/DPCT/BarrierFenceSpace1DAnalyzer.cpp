//===--------------- BarrierFenceSpace1DAnalyzer.cpp ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "BarrierFenceSpace1DAnalyzer.h"
#include "llvm/Support/Casting.h"
#include <algorithm>

using namespace llvm;

bool clang::dpct::BarrierFenceSpace1DAnalyzer::Visit(const IfStmt *IS) {
  // No special process, treat as one block
  return true;
}
void clang::dpct::BarrierFenceSpace1DAnalyzer::PostVisit(const IfStmt *IS) {
  // No special process, treat as one block
}
bool clang::dpct::BarrierFenceSpace1DAnalyzer::Visit(const SwitchStmt *SS) {
  // No special process, treat as one block
  return true;
}
void clang::dpct::BarrierFenceSpace1DAnalyzer::PostVisit(const SwitchStmt *SS) {
  // No special process, treat as one block
}
bool clang::dpct::BarrierFenceSpace1DAnalyzer::Visit(const CallExpr *CE) {
  const FunctionDecl *FuncDecl = CE->getDirectCallee();
  if (!FuncDecl)
    return true;
  std::string FuncName = FuncDecl->getNameInfo().getName().getAsString();
  if (FuncName == "__syncthreads") {
    SyncCallsVec.push_back(CE);
  }
  return true;
}
void clang::dpct::BarrierFenceSpace1DAnalyzer::PostVisit(const CallExpr *) {}

bool clang::dpct::BarrierFenceSpace1DAnalyzer::Visit(const DeclRefExpr *DRE) {
  // Collect all DREs and its Decl
  const auto *PVD = dyn_cast<ParmVarDecl>(DRE->getDecl());
  if (!PVD)
    return true;

  TypeAnalyzer TA;
  TypeAnalyzer::ParamterTypeKind Kind =
      TA.getInputParamterTypeKind(PVD->getType());
  if (Kind == TypeAnalyzer::ParamterTypeKind::CanSkipAnalysis) {
    return true;
  }
  if (Kind == TypeAnalyzer::ParamterTypeKind::Unsupported) {
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
void clang::dpct::BarrierFenceSpace1DAnalyzer::PostVisit(const DeclRefExpr *) {}

bool clang::dpct::BarrierFenceSpace1DAnalyzer::Visit(const GotoStmt *GS) {
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
void clang::dpct::BarrierFenceSpace1DAnalyzer::PostVisit(const GotoStmt *) {}

bool clang::dpct::BarrierFenceSpace1DAnalyzer::Visit(const LabelStmt *LS) {
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
void clang::dpct::BarrierFenceSpace1DAnalyzer::PostVisit(const LabelStmt *) {}

bool clang::dpct::BarrierFenceSpace1DAnalyzer::Visit(const MemberExpr *ME) {
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
void clang::dpct::BarrierFenceSpace1DAnalyzer::PostVisit(const MemberExpr *) {}
bool clang::dpct::BarrierFenceSpace1DAnalyzer::Visit(
    const CXXDependentScopeMemberExpr *CDSME) {
#ifdef __DEBUG_BARRIER_FENCE_SPACE_ANALYZER
  std::cout << "Return False case E: "
            << CDSME->getBeginLoc().printToString(
                   DpctGlobalInfo::getSourceManager())
            << std::endl;
#endif
  return false;
}
void clang::dpct::BarrierFenceSpace1DAnalyzer::PostVisit(
    const CXXDependentScopeMemberExpr *) {}

/// @brief Check if a DRE is assigned to another DRE.
/// This function checks the ancestors of \p CurrentDRE iteratively.
/// If it finds the parent node is AssignmentOp and \p CurrentDRE is
/// in RHS and the LHS is pointer type, it will insert all DREs in LHS into
/// return value.
/// If it finds the parent node is VarDecl and the VarDecl is pointer type, it
/// will insert the VaeDecl into return value.
/// @param CurrentDRE The DRE to need to be checked.
/// @return Assigned DREs or VDs
std::pair<std::set<const clang::DeclRefExpr *>,
          std::set<const clang::VarDecl *>>
clang::dpct::BarrierFenceSpace1DAnalyzer::isAssignedToAnotherDREOrVD(
    const DeclRefExpr *CurrentDRE) {
  std::set<const DeclRefExpr *> ResultDRESet;
  std::set<const VarDecl *> ResultVDSet;
  findAncestorInFunctionScope<Stmt>(
      CurrentDRE, FD,
      [&](const DynTypedNode &Parent,
          const DynTypedNode &Current) -> const void * {
        const auto *BO = Parent.get<BinaryOperator>();
        const auto *VD = Parent.get<VarDecl>();
        if (BO && BO->isAssignmentOp() &&
            (BO->getRHS() == Current.get<Expr>()) &&
            BO->getLHS()->getType()->isPointerType()) {
          auto DREMatcher =
              ast_matchers::findAll(ast_matchers::declRefExpr().bind("DRE"));
          auto MatchedResults = ast_matchers::match(
              DREMatcher, *(BO->getRHS()), DpctGlobalInfo::getContext());
          for (auto &Node : MatchedResults) {
            if (const auto *DRE = Node.getNodeAs<DeclRefExpr>("DRE"))
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

namespace {
using namespace clang;
using namespace dpct;

template <class NodeTy>
static inline const Stmt *findNearestLoopStmt(const NodeTy *N) {
  if (!N)
    return nullptr;
  auto &Context = DpctGlobalInfo::getContext();
  clang::DynTypedNodeList Parents = Context.getParents(*N);
  while (!Parents.empty()) {
    auto &Cur = Parents[0];
    if (Cur.get<DoStmt>() || Cur.get<ForStmt>() || Cur.get<WhileStmt>())
      return Cur.get<Stmt>();
    Parents = Context.getParents(Cur);
  }
  return nullptr;
}

// Check if this DRE(Ptr) matches pattern: Ptr[Idx]
// clang-format off
//    ArraySubscriptExpr <col:7, col:16> 'float' lvalue
//    |-ImplicitCastExpr <col:7> 'float *' <LValueToRValue>
//    | `-DeclRefExpr <col:7> 'float *' lvalue ParmVar 0x555a6c216d68 'Ptr' 'float *'
//    `-ImplicitCastExpr <col:12> 'int' <LValueToRValue>
//      `-DeclRefExpr <col:12> 'int' lvalue Var 0x555a6c217078 'Idx' 'int'
// clang-format on
const ArraySubscriptExpr *getArraySubscriptExpr(const DeclRefExpr *Node) {
  const auto *ICE = DpctGlobalInfo::findParent<ImplicitCastExpr>(Node);
  if (!ICE || (ICE->getCastKind() != CastKind::CK_LValueToRValue))
    return nullptr;
  const auto *ASE = DpctGlobalInfo::findParent<ArraySubscriptExpr>(ICE);
  if (!ASE)
    return nullptr;
  return ASE;
}

// This function matches the pattern "a += b":
// CompoundAssignOperator '+='
// |-DeclRefExpr
// `-ImplicitCastExpr <LValueToRValue>
//   `-DeclRefExpr
// If it is matched, return true and the 2nd arg is the string of b.
// If it isn't matched. return false.
bool isIncPattern(const DeclRefExpr *DRE, std::string &RHSStr) {
  const CompoundAssignOperator *CO =
      DpctGlobalInfo::findParent<CompoundAssignOperator>(DRE);
  if (!CO)
    return false;
  if (CO->getOpcode() != BinaryOperatorKind::BO_AddAssign)
    return false;
  const DeclRefExpr *RHS =
      dyn_cast<DeclRefExpr>(CO->getRHS()->IgnoreImpCasts());
  if (!RHS)
    return false;
  RHSStr = ExprAnalysis::ref(RHS);
  return true;
}

// This function tracks the definition of Idx Expr in an ArraySubscriptExpr.
// If the Idx Expr is constant, return {idx_init_expr, false, ""}
// E.g.,
//     __global__ void foo() {
//       ...
//       int Idx = threadIdx.x;
//       mem[Idx] = var;
//       ...
//     }
// Return {threadIdx.x, false, ""}
// If the Idx Expr is not constant, but it is in a non-nest loop and the inc
// step is constant, return {idx_init_expr, true, step_str}
// E.g.,
//     __global__ void foo() {
//       ...
//       int Idx = threadIdx.x;
//       loop {
//         mem[Idx] = var;
//         Idx += step;
//       }
//       ...
//     }
// Return {threadIdx.x, true, "step"}
std::tuple<const Expr *, bool, std::string>
getIdxExprOfASE(const ArraySubscriptExpr *ASE) {
  bool IsIdxInc = false;

  // IdxVD must be local variable and must be defined in this function
  const DeclRefExpr *IdxDRE =
      dyn_cast_or_null<DeclRefExpr>(ASE->getIdx()->IgnoreImpCasts());
  if (!IdxDRE)
    return {nullptr, IsIdxInc, ""};
  const VarDecl *IdxVD = dyn_cast_or_null<VarDecl>(IdxDRE->getDecl());
  if (!IdxVD)
    return {nullptr, IsIdxInc, ""};
  if (IdxVD->getKind() != Decl::Var)
    return {nullptr, IsIdxInc, ""};
  const auto *IdxFD = dyn_cast_or_null<FunctionDecl>(IdxVD->getDeclContext());
  if (!IdxFD)
    return {nullptr, IsIdxInc, ""};
  const Stmt *IdxVDContext = IdxFD->getBody();

  std::string IncStr;
  using namespace ast_matchers;
  // VD's DRE should:
  // (1) be used as rvalue; Or
  // (2) meet pattern as "idx += step" and must be in the same loop of ASE
  //     and there is no nested loop
  auto DREMatcher = findAll(declRefExpr(isDeclSameAs(IdxVD)).bind("DRE"));
  auto MatchedResults =
      match(DREMatcher, *IdxVDContext, DpctGlobalInfo::getContext());
  for (const auto &Res : MatchedResults) {
    const DeclRefExpr *RefDRE = Res.getNodeAs<DeclRefExpr>("DRE");
    const auto *ICE = DpctGlobalInfo::findParent<ImplicitCastExpr>(RefDRE);
    if (!ICE || (ICE->getCastKind() != CastKind::CK_LValueToRValue)) {
      const Stmt *NearestLoopStmtOfRefDRE = findNearestLoopStmt(RefDRE);
      const Stmt *NearestLoopStmtOfRefASE = findNearestLoopStmt(ASE);
      if (!NearestLoopStmtOfRefDRE || !NearestLoopStmtOfRefASE ||
          NearestLoopStmtOfRefDRE != NearestLoopStmtOfRefASE ||
          !isIncPattern(RefDRE, IncStr)) {
        return {nullptr, IsIdxInc, IncStr};
      }
      const auto *SecondLoop = findNearestLoopStmt(NearestLoopStmtOfRefDRE);
      if (SecondLoop) {
        return {nullptr, IsIdxInc, IncStr};
      }
      IsIdxInc = true;
    }
  }
  if (!IdxVD->hasInit())
    return {nullptr, IsIdxInc, IncStr};
  return {IdxVD->getInit()->IgnoreImpCasts(), IsIdxInc, IncStr};
}

bool isMeetAnalyisPrerequirements(const CallExpr *CE, const FunctionDecl *&FD) {
#ifdef __DEBUG_BARRIER_FENCE_SPACE_ANALYZER
  std::cout << "BarrierFenceSpaceAnalyzer Analyzing ..." << std::endl;
#endif
  if (CE->getBeginLoc().isMacroID() || CE->getEndLoc().isMacroID()) {
#ifdef __DEBUG_BARRIER_FENCE_SPACE_ANALYZER
    std::cout << "Return False case F: CE->getBeginLoc().isMacroID() || "
                 "CE->getEndLoc().isMacroID()"
              << std::endl;
#endif
    return false;
  }
  FD = DpctGlobalInfo::findAncestor<FunctionDecl>(CE);
  if (!FD) {
#ifdef __DEBUG_BARRIER_FENCE_SPACE_ANALYZER
    std::cout << "Return False case G: !FD" << std::endl;
#endif
    return false;
  }
  if (!FD->hasAttr<CUDAGlobalAttr>()) {
#ifdef __DEBUG_BARRIER_FENCE_SPACE_ANALYZER
    std::cout << "Return False case H: !FD->hasAttr<CUDAGlobalAttr>()"
              << std::endl;
#endif
    return false;
  }
  std::unordered_set<const DeviceFunctionInfo *> Visited{};
  auto DFI = DeviceFunctionDecl::LinkRedecls(FD);
  if (!DFI) {
#ifdef __DEBUG_BARRIER_FENCE_SPACE_ANALYZER
    std::cout << "Return False case J: !DFI" << std::endl;
#endif
    return false;
  }

  if (DFI->getVarMap().hasGlobalMemAcc()) {
#ifdef __DEBUG_BARRIER_FENCE_SPACE_ANALYZER
    std::cout << "Return False case I: Found device/managed variable usage"
              << std::endl;
#endif
    return false;
  }
  return true;
}
} // namespace

void clang::dpct::BarrierFenceSpace1DAnalyzer::constructDefUseMap() {
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

  // Collect all used positions
  std::size_t MapSize = getSize(DefUseMap);
  do {
    MapSize = getSize(DefUseMap);
    std::set<const DeclRefExpr *> NewDRESet;
    for (auto &Pair : DefUseMap) {
      const ParmVarDecl *CurDecl = Pair.first;
      std::set<const DeclRefExpr *> CurDRESet = Pair.second;
      std::set<const DeclRefExpr *> MatchedResult =
          matchTargetDREInScope(CurDecl, FD->getBody());
      CurDRESet.insert(MatchedResult.begin(), MatchedResult.end());
      NewDRESet.clear();
      for (const auto &DRE : CurDRESet) {
        const auto &SetPair = isAssignedToAnotherDREOrVD(DRE);
        for (const auto *AnotherDRE : SetPair.first) {
          std::set<const DeclRefExpr *> AnotherDREMatchedResult =
              matchTargetDREInScope(
                  dyn_cast_or_null<VarDecl>(AnotherDRE->getDecl()),
                  FD->getBody());
          NewDRESet.insert(AnotherDREMatchedResult.begin(),
                           AnotherDREMatchedResult.end());
        }
        for (const auto *AnotherVD : SetPair.second) {
          std::set<const DeclRefExpr *> AnotherDREMatchedResult =
              matchTargetDREInScope(AnotherVD, FD->getBody());
          NewDRESet.insert(AnotherDREMatchedResult.begin(),
                           AnotherDREMatchedResult.end());
        }
      }
      if (!NewDRESet.empty())
        Pair.second.insert(NewDRESet.begin(), NewDRESet.end());
    }
  } while (getSize(DefUseMap) != MapSize);
}

void clang::dpct::BarrierFenceSpace1DAnalyzer::simplifyMap(
    std::map<const ParmVarDecl *, std::set<DREInfo>> &DefDREInfoMap) {
  std::map<const ParmVarDecl *, std::set<const DeclRefExpr *>>
      DefDREInfoMapTemp;
  // simplify DefUseMap
  for (const auto &Pair : DefUseMap) {
    for (const auto &Item : Pair.second) {
      if (isAccessingMemory(Item)) {
        if (hasOverlappingAccessAmongWorkItems(KernelCallBlockDim, Item)) {
          DefDREInfoMapTemp[Pair.first].insert(Item);
        }
      }
    }
  }

  // Convert DRE to Location for comparing
  for (const auto &Pair : DefDREInfoMapTemp) {
    for (const auto &Item : Pair.second) {
      DefDREInfoMap[Pair.first].insert(DREInfo(Item, Item->getBeginLoc()));
    }
  }
}

clang::dpct::BarrierFenceSpace1DAnalyzerResult
clang::dpct::BarrierFenceSpace1DAnalyzer::analyzeFor1DKernel(
    const CallExpr *CE) {
  // Check prerequirements
  const FunctionDecl *FD = nullptr;
  if (!isMeetAnalyisPrerequirements(CE, FD))
    return BarrierFenceSpace1DAnalyzerResult(false, "", false, "");

  // Init values
  this->FD = FD;
  GlobalFunctionName = FD->getDeclName().getAsString();
  auto queryKernelDim = [](const FunctionDecl *FD)
      -> std::pair<int /*kernel dim*/, int /*kernel block dim*/> {
    const auto DFD = DpctGlobalInfo::getInstance().findDeviceFunctionDecl(FD);
    if (!DFD)
      return {3, 3};
    const auto FuncInfo = DFD->getFuncInfo();
    if (!FuncInfo)
      return {3, 3};
    int BlockDim = FuncInfo->KernelCallBlockDim;
    const auto *MVM =
        MemVarMap::getHeadWithoutPathCompression(&(FuncInfo->getVarMap()));
    if (!MVM)
      return {3, BlockDim};
    return {MVM->Dim, BlockDim};
  };
  std::tie(KernelDim, KernelCallBlockDim) = queryKernelDim(FD);

  if (!this->TraverseDecl(const_cast<FunctionDecl *>(FD))) {
    return BarrierFenceSpace1DAnalyzerResult(false, "", false, "");
  }

  constructDefUseMap();
  std::map<const ParmVarDecl *, std::set<DREInfo>> DefLocInfoMap;
  simplifyMap(DefLocInfoMap);

  std::string CELoc = getHashStrFromLoc(CE->getBeginLoc());
  for (auto &SyncCall : SyncCallsVec) {
    if (getHashStrFromLoc(SyncCall->getBeginLoc()) == CELoc) {
      auto Res = isSafeToUseLocalBarrier(DefLocInfoMap);
      return BarrierFenceSpace1DAnalyzerResult(
          std::get<0>(Res), GlobalFunctionName, std::get<1>(Res),
          std::get<2>(Res));
    }
  }
  return BarrierFenceSpace1DAnalyzerResult(false, "", false, "");
}

bool clang::dpct::BarrierFenceSpace1DAnalyzer::
    hasOverlappingAccessAmongWorkItems(int KernelCallBlockDim,
                                       const DeclRefExpr *DRE) {
  if (KernelCallBlockDim != 1) {
    return true;
  }

  const ArraySubscriptExpr *ASE = getArraySubscriptExpr(DRE);
  if (!ASE)
    return true;
  auto Res = getIdxExprOfASE(ASE);
  if (!std::get<0>(Res))
    return true;

  IndexAnalysis IA(std::get<0>(Res));
  if (IA.isDifferenceBetweenThreadIdxXAndIndexConstant()) {
    DREIncStepMap.insert({DRE, std::get<2>(Res)});
  }
  // Check if Index variable has 1:1 mapping to threadIdx.x in a block
  return std::get<1>(Res) || !IA.isStrictlyMonotonic();
}

bool clang::dpct::BarrierFenceSpace1DAnalyzer::containsMacro(
    const SourceLocation &SL) {
  if (SL.isMacroID())
    return true;
  return false;
}

bool clang::dpct::BarrierFenceSpace1DAnalyzer::isAccessingMemory(
    const DeclRefExpr *DRE) {
  auto &Context = DpctGlobalInfo::getContext();
  DynTypedNode Current = DynTypedNode::create(*DRE);
  DynTypedNodeList Parents = Context.getParents(Current);
  while (!Parents.empty()) {
    if (Parents[0].get<FunctionDecl>() && Parents[0].get<FunctionDecl>() == FD)
      break;
    const auto *UO = Parents[0].get<UnaryOperator>();
    const auto *ASE = Parents[0].get<ArraySubscriptExpr>();
    if (UO && (UO->getOpcode() == UnaryOperatorKind::UO_Deref)) {
      return true;
    }
    if (ASE && (ASE->getBase() == Current.get<Expr>())) {
      return true;
    }
    Current = Parents[0];
    Parents = Context.getParents(Current);
  }
  return false;
}

// This function recognizes pattern like below:
// (1) Only 1 access in each iteration for each memory variable
// (2) The step of the `idx` has been identified in previous step.
// for () {
//   ...
//   mem[idx] = var;
//   ...
// }
std::string clang::dpct::BarrierFenceSpace1DAnalyzer::isAnalyzableWriteInLoop(
    const std::set<const DeclRefExpr *> &WriteInLoopDRESet) {
  if (WriteInLoopDRESet.size() > 1) {
#ifdef __DEBUG_BARRIER_FENCE_SPACE_ANALYZER
    std::cout << "isAnalyzableWriteInLoop False case 1" << std::endl;
#endif
    return "";
  }

  const DeclRefExpr *DRE = *WriteInLoopDRESet.begin();
  auto Iter = DREIncStepMap.find(DRE);
  if (Iter == DREIncStepMap.end()) {
#ifdef __DEBUG_BARRIER_FENCE_SPACE_ANALYZER
    std::cout << "isAnalyzableWriteInLoop False case 2" << std::endl;
#endif
    return "";
  }
  return Iter->second;
}

std::tuple<bool /*CanUseLocalBarrier*/,
           bool /*CanUseLocalBarrierWithCondition*/, std::string /*Condition*/>
clang::dpct::BarrierFenceSpace1DAnalyzer::isSafeToUseLocalBarrier(
    const std::map<const ParmVarDecl *, std::set<DREInfo>> &DefDREInfoMap) {
#ifdef __DEBUG_BARRIER_FENCE_SPACE_ANALYZER
  std::cout << "===== isSafeToUseLocalBarrier =====" << std::endl;
#endif
  std::set<std::string> ConditionSet;
  for (auto &DefDREInfo : DefDREInfoMap) {
    std::set<const DeclRefExpr *> RefedDRE;
    for (auto &DREInfo : DefDREInfo.second) {
      if (DREInfo.SL.isMacroID()) {
#ifdef __DEBUG_BARRIER_FENCE_SPACE_ANALYZER
        std::cout << "isSafeToUseLocalBarrier False case 1" << std::endl;
#endif
        return {false, false, ""};
      }
      RefedDRE.insert(DREInfo.DRE);
    }
    if (!RefedDRE.empty()) {
      auto StepStr = isAnalyzableWriteInLoop(RefedDRE);
      if (StepStr.empty()) {
#ifdef __DEBUG_BARRIER_FENCE_SPACE_ANALYZER
        std::cout << "isSafeToUseLocalBarrier False case 3" << std::endl;
#endif
        return {false, false, ""};
      }
      ConditionSet.insert(StepStr);
    }
  }

  if (!ConditionSet.empty()) {
#ifdef __DEBUG_BARRIER_FENCE_SPACE_ANALYZER
    std::cout << "isSafeToUseLocalBarrier True with condition" << std::endl;
#endif
    // local_range(2) loop0     loop1    loop2    ... loopn
    //      0         mem[0]    mem[s]   mem[2s]      mem[ns]
    //      1         mem[1+0]  mem[1+s] mem[1+2s]    mem[1+ns]
    //      2         mem[2+0]  mem[2+s] mem[2+2s]    mem[2+ns]
    //     ...
    //      m         mem[m+0]  mem[m+s] mem[m+2s]    mem[m+ns]
    //
    // We can make sure that there is no overlap in the same iteration since idx
    // should equal to `local_id(2) + C`.
    // Next, we need to make sure there is no overlap among iterations.
    // The memory range in an iteration is `local_range(2)`, then if
    // `s > local_range(2)`, the next iteration start point is larger than
    // previous end, so there is no overlap.

    std::string RHS;
    if (ConditionSet.size() == 1) {
      RHS = *ConditionSet.begin();
    } else {
      RHS = "std::min(";
      for (const auto &C : ConditionSet) {
        RHS = RHS + C + ", ";
      }
      RHS = RHS.substr(0, RHS.size() - 2) + ")";
    }

    // No need to call DpctGlobalInfo::getItem() here.
    // It has been invoked in SyncThreadsRule.
    if (DpctGlobalInfo::getAssumedNDRangeDim() == 1 && KernelDim == 1)
      return {false, true, "item_ct1.get_local_range(0) < " + RHS};
    return {false, true, "item_ct1.get_local_range(2) < " + RHS};
  }
#ifdef __DEBUG_BARRIER_FENCE_SPACE_ANALYZER
  std::cout << "isSafeToUseLocalBarrier True" << std::endl;
#endif
  return {true, false, ""};
}
