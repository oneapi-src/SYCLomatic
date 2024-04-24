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

bool clang::dpct::IntraproceduralAnalyzer::Visit(const ForStmt *FS) {
  LoopRange.push_back(FS->getSourceRange());
  return true;
}
void clang::dpct::IntraproceduralAnalyzer::PostVisit(const clang::ForStmt *FS) {
  LoopRange.pop_back();
}
bool clang::dpct::IntraproceduralAnalyzer::Visit(const DoStmt *DS) {
  LoopRange.push_back(DS->getSourceRange());
  return true;
}
void clang::dpct::IntraproceduralAnalyzer::PostVisit(const DoStmt *DS) {
  LoopRange.pop_back();
}
bool clang::dpct::IntraproceduralAnalyzer::Visit(const WhileStmt *WS) {
  LoopRange.push_back(WS->getSourceRange());
  return true;
}
void clang::dpct::IntraproceduralAnalyzer::PostVisit(const WhileStmt *WS) {
  LoopRange.pop_back();
}
bool clang::dpct::IntraproceduralAnalyzer::Visit(const CallExpr *CE) {
  const FunctionDecl *FuncDecl = CE->getDirectCallee();
  if (!FuncDecl)
    return true;
  std::string FuncName = FuncDecl->getNameInfo().getName().getAsString();

  for (const auto &Arg : CE->arguments())
    DeviceFunctionCallArgs.insert(Arg);

  if (FuncName == "__syncthreads" || isUserDefinedDecl(FuncDecl)) {
    SyncCallInfo SCI;
    SCI.Predecessors.insert(
        SourceRange(FD->getBody()->getBeginLoc(), CE->getBeginLoc()));
    SCI.Successors.insert(
        SourceRange(CE->getEndLoc(), FD->getBody()->getEndLoc()));
    if (!LoopRange.empty()) {
      SCI.Predecessors.insert(LoopRange.front());
      SCI.Successors.insert(LoopRange.front());
    }
    SyncCallsVec.push_back(std::make_pair(CE, SCI));
  }
  return true;
}
void clang::dpct::IntraproceduralAnalyzer::PostVisit(const CallExpr *) {}

bool clang::dpct::IntraproceduralAnalyzer::Visit(const DeclRefExpr *DRE) {
  // Collect all DREs and its Decl
  const auto& PVD = dyn_cast<ParmVarDecl>(DRE->getDecl());
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
void clang::dpct::IntraproceduralAnalyzer::PostVisit(const DeclRefExpr *) {}

bool clang::dpct::IntraproceduralAnalyzer::Visit(const GotoStmt *GS) {
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
void clang::dpct::IntraproceduralAnalyzer::PostVisit(const GotoStmt *) {}

bool clang::dpct::IntraproceduralAnalyzer::Visit(const LabelStmt *LS) {
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
void clang::dpct::IntraproceduralAnalyzer::PostVisit(const LabelStmt *) {}

bool clang::dpct::IntraproceduralAnalyzer::Visit(const MemberExpr *ME) {
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
void clang::dpct::IntraproceduralAnalyzer::PostVisit(const MemberExpr *) {}
bool clang::dpct::IntraproceduralAnalyzer::Visit(
    const CXXDependentScopeMemberExpr *CDSME) {
#ifdef __DEBUG_BARRIER_FENCE_SPACE_ANALYZER
  std::cout << "Return False case E: "
            << CDSME->getBeginLoc().printToString(
                   DpctGlobalInfo::getSourceManager())
            << std::endl;
#endif
  return false;
}
void clang::dpct::IntraproceduralAnalyzer::PostVisit(
    const CXXDependentScopeMemberExpr *) {}

bool clang::dpct::IntraproceduralAnalyzer::Visit(const CXXConstructExpr *CCE) {
  for (const auto &Arg : CCE->arguments())
    DeviceFunctionCallArgs.insert(Arg);
  return true;
}
void clang::dpct::IntraproceduralAnalyzer::PostVisit(const CXXConstructExpr *) {
}

template <class TargetTy, class NodeTy>
static inline const TargetTy *findAncestorInFunctionScope(
    const NodeTy *N, const FunctionDecl *Scope,
    const std::function<const void *(const DynTypedNode &,
                                     const DynTypedNode &)> &Operation) {
  auto &Context = clang::dpct::DpctGlobalInfo::getContext();
  DynTypedNode Current = DynTypedNode::create(*N);
  DynTypedNodeList Parents = Context.getParents(Current);
  while (!Parents.empty()) {
    if (Parents[0].get<FunctionDecl>() &&
        Parents[0].get<FunctionDecl>() == Scope)
      break;
    if (const void *Node = Operation(Parents[0], Current)) {
      return reinterpret_cast<const TargetTy *>(Node);
    }
    Current = Parents[0];
    Parents = Context.getParents(Current);
  }
  return nullptr;
}

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
clang::dpct::IntraproceduralAnalyzer::isAssignedToAnotherDREOrVD(
    const DeclRefExpr *CurrentDRE) {
  std::set<const DeclRefExpr *> ResultDRESet;
  std::set<const VarDecl *> ResultVDSet;
  findAncestorInFunctionScope<Stmt>(
      CurrentDRE, FD,
      [&](const DynTypedNode &Parent,
          const DynTypedNode &Current) -> const void * {
        const auto &BO = Parent.get<BinaryOperator>();
        const auto &VD = Parent.get<VarDecl>();
        if (BO && BO->isAssignmentOp() &&
            (BO->getRHS() == Current.get<Expr>()) &&
            BO->getLHS()->getType()->isPointerType()) {
          auto DREMatcher =
              ast_matchers::findAll(ast_matchers::declRefExpr().bind("DRE"));
          auto MatchedResults = ast_matchers::match(
              DREMatcher, *(BO->getRHS()), DpctGlobalInfo::getContext());
          for (auto &Node : MatchedResults) {
            if (const auto &DRE = Node.getNodeAs<DeclRefExpr>("DRE"))
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

clang::dpct::AccessMode
clang::dpct::IntraproceduralAnalyzer::getAccessKindReadWrite(
    const DeclRefExpr *CurrentDRE) {
  bool FoundDeref = false;
  bool FoundBO = false;
  bool FoundCE = false;
  const BinaryOperator *BO = nullptr;
  const UnaryOperator *UOInBO = nullptr;
  const ArraySubscriptExpr *ASEInBO = nullptr;
  const ParmVarDecl *PVD = nullptr;
  const UnaryOperator *IncDec = nullptr;
  const CallExpr *CE = nullptr;

  auto &Context = DpctGlobalInfo::getContext();
  DynTypedNode Current = DynTypedNode::create(*CurrentDRE);
  DynTypedNodeList Parents = Context.getParents(Current);
  while (!Parents.empty()) {
    if (Parents[0].get<FunctionDecl>() && Parents[0].get<FunctionDecl>() == FD)
      break;

    if (!FoundCE)
      CE = Parents[0].get<CallExpr>();
    if (!FoundDeref)
      UOInBO = Parents[0].get<UnaryOperator>();
    if (!FoundDeref)
      ASEInBO = Parents[0].get<ArraySubscriptExpr>();
    if (!FoundBO)
      BO = Parents[0].get<BinaryOperator>();
    IncDec = Parents[0].get<UnaryOperator>();

    if (!FoundCE && CE && CE->getDirectCallee()) {
      FoundCE = true;
      int Idx = 0;
      for (const auto &Arg : CE->arguments()) {
        if (Arg == Current.get<Expr>())
          break;
        Idx++;
      }
      if (dyn_cast<CXXOperatorCallExpr>(CE) || dyn_cast<CXXMemberCallExpr>(CE))
        Idx--;
      if (Idx >= 0)
        PVD = CE->getDirectCallee()->getParamDecl(Idx);
      else {
        if (const auto &CMD = dyn_cast<CXXMethodDecl>(CE->getDirectCallee())) {
          if (CMD->isImplicit())
            return AccessMode::Write;
        }
        return AccessMode::ReadWrite;
      }
    }
    if (IncDec && (IncDec->isIncrementOp() || IncDec->isDecrementOp()) &&
        FoundDeref) {
      return AccessMode::ReadWrite;
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
  const auto &ICE = DpctGlobalInfo::findParent<ImplicitCastExpr>(Node);
  if (!ICE || (ICE->getCastKind() != CastKind::CK_LValueToRValue))
    return nullptr;
  const auto &ASE = DpctGlobalInfo::findParent<ArraySubscriptExpr>(ICE);
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
    const auto &ICE = DpctGlobalInfo::findParent<ImplicitCastExpr>(RefDRE);
    if (!ICE || (ICE->getCastKind() != CastKind::CK_LValueToRValue)) {
      const Stmt *NearestLoopStmtOfRefDRE = findNearestLoopStmt(RefDRE);
      const Stmt *NearestLoopStmtOfRefASE = findNearestLoopStmt(ASE);
      if (!NearestLoopStmtOfRefDRE || !NearestLoopStmtOfRefASE ||
          NearestLoopStmtOfRefDRE != NearestLoopStmtOfRefASE ||
          !isIncPattern(RefDRE, IncStr)) {
        return {nullptr, IsIdxInc, IncStr};
      }
      const auto &SecondLoop = findNearestLoopStmt(NearestLoopStmtOfRefDRE);
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

bool isMeetAnalyisPrerequirements(const FunctionDecl *FD) {
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

void clang::dpct::IntraproceduralAnalyzer::constructDefUseMap() {
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
          matchTargetDREInScope(CurDecl, FD->getBody());
      CurDRESet.insert(MatchedResult.begin(), MatchedResult.end());
      NewDRESet.clear();
      for (const auto &DRE : CurDRESet) {
        const auto &SetPair = isAssignedToAnotherDREOrVD(DRE);
        for (const auto &AnotherDRE : SetPair.first) {
          std::set<const DeclRefExpr *> AnotherDREMatchedResult =
              matchTargetDREInScope(
                  dyn_cast_or_null<VarDecl>(AnotherDRE->getDecl()),
                  FD->getBody());
          NewDRESet.insert(AnotherDREMatchedResult.begin(),
                           AnotherDREMatchedResult.end());
        }
        for (const auto &AnotherVD : SetPair.second) {
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
}

void clang::dpct::IntraproceduralAnalyzer::simplifyMap(
    std::map<const ParmVarDecl *, std::set<DREInfo>> &DefDREInfoMap) {
  std::map<const ParmVarDecl *,
           std::set<std::pair<const DeclRefExpr *, AccessMode>>>
      DefDREInfoMapTemp;
  // simplify DefUseMap
  for (const auto &Pair : DefUseMap) {
    for (const auto &Item : Pair.second) {
      if (isAccessingMemory(Item)) {
        AccessMode AMRW = getAccessKindReadWrite(Item);
        DefDREInfoMapTemp[Pair.first].insert(std::make_pair(Item, AMRW));
      }
    }
  }

  // Convert DRE to Location for comparing
  for (const auto &Pair : DefDREInfoMapTemp) {
    for (const auto &Item : Pair.second) {
      DefDREInfoMap[Pair.first].insert(
          DREInfo(Item.first, Item.first->getBeginLoc(), Item.second));
    }
  }

#ifdef __DEBUG_BARRIER_FENCE_SPACE_ANALYZER
  std::cout << "===== DefDREInfoMap contnet: =====" << std::endl;
  for (const auto &LocInfo : DefDREInfoMap) {
    const auto &SM = DpctGlobalInfo::getSourceManager();
    std::cout << "Decl:" << LocInfo.first->getBeginLoc().printToString(SM)
              << std::endl;
    for (const auto &Info : LocInfo.second) {
      std::cout << "    DRE:" << Info.SL.printToString(SM)
                << ", AccessMode:" << (int)(Info.AM) << std::endl;
    }
  }
  std::cout << "===== DefDREInfoMap contnet end =====" << std::endl;
#endif
}

bool clang::dpct::InterproceduralAnalyzer::analyze(
    const std::shared_ptr<DeviceFunctionInfo> DFI,
    std::string SyncCallCombinedLoc) {
  // TODO: Need do analysis for all syncthreads call in this DFI's ancestors and
  // this DFI's decendents.
  // The results need be cached and reuse at the next time.

  return false;
}

clang::dpct::IntraproceduralAnalyzerResult
clang::dpct::IntraproceduralAnalyzer::analyze(const FunctionDecl *FD) {
  // Check prerequirements
  if (!isMeetAnalyisPrerequirements(FD))
    return IntraproceduralAnalyzerResult(true);

  // Init values
  this->FD = FD;

  FDLoc = getHashStrFromLoc(FD->getBeginLoc());

#ifdef __DEBUG_BARRIER_FENCE_SPACE_ANALYZER
  std::cout << "Before traversing current __global__ function" << std::endl;
#endif

  if (!this->TraverseDecl(const_cast<FunctionDecl *>(FD))) {
    return IntraproceduralAnalyzerResult(true);
  }

  constructDefUseMap();
  std::map<const ParmVarDecl *, std::set<DREInfo>> DefLocInfoMap;
  simplifyMap(DefLocInfoMap);

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

  std::unordered_map<
      std::string /*call's combined loc string*/,
      std::tuple<
          tooling::UnifiedPath, unsigned int,
          std::unordered_map<unsigned int /*parameter idx*/, AffectedInfo>>>
      Map;
  for (auto &SyncCall : SyncCallsVec) {
    auto LocInfo = DpctGlobalInfo::getLocInfo(SyncCall.first->getBeginLoc());
    Map.insert(
        std::make_pair(getCombinedStrFromLoc(SyncCall.first->getBeginLoc()),
                       std::make_tuple(LocInfo.first, LocInfo.second,
                                       affectedByWhichParameters(
                                           DefLocInfoMap, SyncCall.second))));
  }
  return IntraproceduralAnalyzerResult(Map);
}

bool clang::dpct::IntraproceduralAnalyzer::isAccessingMemory(
    const DeclRefExpr *DRE) {
  auto &Context = DpctGlobalInfo::getContext();
  DynTypedNode Current = DynTypedNode::create(*DRE);
  DynTypedNodeList Parents = Context.getParents(Current);
  while (!Parents.empty()) {
    if (Parents[0].get<FunctionDecl>() && Parents[0].get<FunctionDecl>() == FD)
      break;
    const auto& UO = Parents[0].get<UnaryOperator>();
    const auto& ASE = Parents[0].get<ArraySubscriptExpr>();
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

bool clang::dpct::isInRanges(SourceLocation SL, Ranges Ranges) {
  auto &SM = DpctGlobalInfo::getSourceManager();
  for (auto &Range : Ranges) {
    if (SM.getFileOffset(Range.getBegin()) < SM.getFileOffset(SL) &&
        SM.getFileOffset(SL) < SM.getFileOffset(Range.getEnd())) {
      return true;
    }
  }
  return false;
}

std::tuple<bool /*CanUseLocalBarrier*/,
           bool /*CanUseLocalBarrierWithCondition*/, std::string /*Condition*/>
clang::dpct::InterproceduralAnalyzer::isSafeToUseLocalBarrier(
    const std::map<const ParmVarDecl *, std::set<DREInfo>> &DefDREInfoMap,
    const SyncCallInfo &SCI) {
  return {true, false, ""};
}

std::unordered_map<unsigned int /*parameter idx*/, AffectedInfo>
IntraproceduralAnalyzer::affectedByWhichParameters(
    const std::map<const ParmVarDecl *, std::set<DREInfo>> &DefDREInfoMap,
    const SyncCallInfo &SCI) {
  auto convertPVD2Idx = [](const FunctionDecl *FD, const ParmVarDecl *PVD) {
    unsigned int Idx = 0;
    for (const auto& D : FD->parameters()) {
      if (D == PVD)
        return Idx;
      Idx++;
    }
    assert(0 && "PVD is not in the FD.");
  };

  std::unordered_map<unsigned int /*parameter idx*/, AffectedInfo>
      AffectingParameters;
  for (auto &DefDREInfo : DefDREInfoMap) {
    bool UsedBefore = false;
    bool UsedAfter = false;
    AccessMode AM = Read;
    for (auto &DREInfo : DefDREInfo.second) {
      if (DREInfo.SL.isMacroID()) {
        UsedBefore = true;
        UsedAfter = true;
        AM = ReadWrite;
        break;
      }
      AM = AccessMode(AM | DREInfo.AM);
      if (isInRanges(DREInfo.SL, SCI.Predecessors)) {
        UsedBefore = true;
      }
      if (isInRanges(DREInfo.SL, SCI.Successors)) {
        UsedAfter = true;
      }
    }
    if (AM != Read)
      AffectingParameters.insert(
          std::make_pair(convertPVD2Idx(FD, DefDREInfo.first),
                         AffectedInfo{UsedBefore, UsedAfter, AM}));
  }
  return AffectingParameters;
}
