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
  TypeAnalyzer::ParamterTypeKind Kind =
      TA.getInputParamterTypeKind(PVD->getType());
  if (Kind == TypeAnalyzer::ParamterTypeKind::CanSkipAnalysis) {
    return true;
  } else if (Kind == TypeAnalyzer::ParamterTypeKind::Unsupported) {
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
  for (auto &Node : MatchedResults) {
    if (auto DRE = Node.getNodeAs<DeclRefExpr>("DRE"))
      Set.insert(DRE);
  }
  return Set;
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
          for (auto &Node : MatchedResults) {
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
    if (IncDec && (IncDec->isIncrementOp() || IncDec->isDecrementOp()) &&
        FoundDeref) {
      return AccessMode::ReadWrite;
    } else if (!FoundBO && BO && BO->isAssignmentOp() &&
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
  auto ICE = DpctGlobalInfo::findParent<ImplicitCastExpr>(Node);
  if (!ICE || (ICE->getCastKind() != CastKind::CK_LValueToRValue))
    return nullptr;
  auto ASE = DpctGlobalInfo::findParent<ArraySubscriptExpr>(ICE);
  if (!ASE)
    return nullptr;
  return ASE;
}

// Match pattern:
// CompoundAssignOperator '+='
// |-DeclRefExpr
// `-ImplicitCastExpr <LValueToRValue>
//   `-DeclRefExpr
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

std::tuple<const Expr *, bool, std::string>
getIdxOfASEConstValueExpr(const ArraySubscriptExpr *ASE) {
  bool IsIdxInc = false;

  // IdxVD must be local variable and must be defined in this function
  const DeclRefExpr *IdxDRE =
      dyn_cast_or_null<DeclRefExpr>(ASE->getIdx()->IgnoreImpCasts());
  if (!IdxDRE)
    return {nullptr, IsIdxInc, ""};
  const VarDecl *IdxVD = dyn_cast_or_null<VarDecl>(IdxDRE->getDecl());
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
  auto DREMatcher = findAll(declRefExpr(isDeclSameAs(IdxVD)).bind("DRE"));
  auto MatchedResults =
      match(DREMatcher, *IdxVDContext, DpctGlobalInfo::getContext());
  for (const auto &Res : MatchedResults) {
    const DeclRefExpr *RefDRE = Res.getNodeAs<DeclRefExpr>("DRE");
    auto ICE = DpctGlobalInfo::findParent<ImplicitCastExpr>(RefDRE);
    if (!ICE || (ICE->getCastKind() != CastKind::CK_LValueToRValue)) {
      const Stmt *NearestLoopStmtOfRefDRE = findNearestLoopStmt(RefDRE);
      const Stmt *NearestLoopStmtOfRefASE = findNearestLoopStmt(ASE);
      if (!NearestLoopStmtOfRefDRE || !NearestLoopStmtOfRefASE ||
          NearestLoopStmtOfRefDRE != NearestLoopStmtOfRefASE ||
          !isIncPattern(RefDRE, IncStr)) {
        return {nullptr, IsIdxInc, IncStr};
      }
      auto SecondLoop = findNearestLoopStmt(NearestLoopStmtOfRefDRE);
      if (SecondLoop) {
        return {nullptr, IsIdxInc, IncStr};
      } else {
        IsIdxInc = true;
      }
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

void clang::dpct::BarrierFenceSpaceAnalyzer::constructDefUseMap() {
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
}

void clang::dpct::BarrierFenceSpaceAnalyzer::simplifyMap(
    std::map<const ParmVarDecl *, std::set<DREInfo>> &DefDREInfoMap) {
  std::map<const ParmVarDecl *,
           std::set<std::pair<const DeclRefExpr *, AccessMode>>>
      DefDREInfoMapTemp;
  // simplify DefUseMap
  for (const auto &Pair : DefUseMap) {
    for (const auto &Item : Pair.second) {
      if (isAccessingMemory(Item)) {
        if (!hasOverlappingAccessAmongWorkItems(KernelCallBlockDim, Item)) {
          MayDependOn1DKernel = true;
        } else {
          DefDREInfoMapTemp[Pair.first].insert(
              std::make_pair(Item, getAccessKind(Item)));
        }
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

clang::dpct::BarrierFenceSpaceAnalyzerResult
clang::dpct::BarrierFenceSpaceAnalyzer::analyze(const CallExpr *CE,
                                                bool SkipCacheInAnalyzer) {
  // Check prerequirements
  const FunctionDecl *FD = nullptr;
  if (!isMeetAnalyisPrerequirements(CE, FD))
    return BarrierFenceSpaceAnalyzerResult(false, false, false,
                                           GlobalFunctionName);

  // Init values
  this->SkipCacheInAnalyzer = SkipCacheInAnalyzer;
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
    const auto MVM =
        MemVarMap::getHeadWithoutPathCompression(&(FuncInfo->getVarMap()));
    if (!MVM)
      return {3, BlockDim};
    return {MVM->Dim, BlockDim};
  };
  std::tie(KernelDim, KernelCallBlockDim) = queryKernelDim(FD);

  CELoc = getHashStrFromLoc(CE->getBeginLoc());
  FDLoc = getHashStrFromLoc(FD->getBeginLoc());

  auto FDIter = CachedResults.find(FDLoc);
  if (!SkipCacheInAnalyzer) {
    if (FDIter != CachedResults.end()) {
      auto CEIter = FDIter->second.find(CELoc);
      if (CEIter != FDIter->second.end()) {
        return CEIter->second;
      } else {
        return BarrierFenceSpaceAnalyzerResult(false, false, false,
                                               GlobalFunctionName);
      }
    }
  }

#ifdef __DEBUG_BARRIER_FENCE_SPACE_ANALYZER
  std::cout << "Before traversing current __global__ function" << std::endl;
#endif

  if (!this->TraverseDecl(const_cast<FunctionDecl *>(FD))) {
    if (!SkipCacheInAnalyzer) {
      CachedResults[FDLoc] =
          std::unordered_map<std::string, BarrierFenceSpaceAnalyzerResult>();
    }
    return BarrierFenceSpaceAnalyzerResult(false, false, false,
                                           GlobalFunctionName);
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

  if (SkipCacheInAnalyzer) {
    for (auto &SyncCall : SyncCallsVec) {
      if (CE == SyncCall.first) {
        auto Res = isSafeToUseLocalBarrier(DefLocInfoMap, SyncCall.second);
        return BarrierFenceSpaceAnalyzerResult(
            std::get<0>(Res), std::get<1>(Res), MayDependOn1DKernel,
            GlobalFunctionName, std::get<2>(Res));
      }
    }
    return BarrierFenceSpaceAnalyzerResult(false, false, false,
                                           GlobalFunctionName);
  }
  for (auto &SyncCall : SyncCallsVec) {
    auto Res = isSafeToUseLocalBarrier(DefLocInfoMap, SyncCall.second);
    CachedResults[FDLoc][getHashStrFromLoc(SyncCall.first->getBeginLoc())] =
        BarrierFenceSpaceAnalyzerResult(std::get<0>(Res), std::get<1>(Res),
                                        MayDependOn1DKernel, GlobalFunctionName,
                                        std::get<2>(Res));
  }

  // find the result in the new map
  FDIter = CachedResults.find(FDLoc);
  if (FDIter != CachedResults.end()) {
    auto CEIter = FDIter->second.find(CELoc);
    if (CEIter != FDIter->second.end()) {
      return CEIter->second;
    } else {
      return BarrierFenceSpaceAnalyzerResult(false, false, false,
                                             GlobalFunctionName);
    }
  }
  return BarrierFenceSpaceAnalyzerResult(false, false, false,
                                         GlobalFunctionName);
}

bool clang::dpct::BarrierFenceSpaceAnalyzer::hasOverlappingAccessAmongWorkItems(
    int KernelCallBlockDim, const DeclRefExpr *DRE) {
  using namespace ast_matchers;
  if (KernelCallBlockDim != 1) {
    return true;
  }

  const ArraySubscriptExpr *ASE = getArraySubscriptExpr(DRE);
  if (!ASE)
    return true;
  auto Res = getIdxOfASEConstValueExpr(ASE);
  if (!std::get<0>(Res))
    return true;

  IndexAnalysis IA(std::get<0>(Res));
  if (IA.isDifferenceBetweenThreadIdxXAndIndexConstant()) {
    DREIncStepMap.insert({DRE, std::get<2>(Res)});
  }
  // Check if Index variable has 1:1 mapping to threadIdx.x in a block
  return std::get<1>(Res) || !IA.isStrictlyMonotonic();
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

// This function recognizes pattern like below and check if it is safe to treat
// the write operation as a non-global access.
// for () {
//   ...
//   mem[idx] = var;
//   ... 
// }
std::string clang::dpct::BarrierFenceSpaceAnalyzer::isSafeWriteInLoop(
    const std::set<const DeclRefExpr *> &WILDRESet) {
  if (WILDRESet.size() > 1) {
#ifdef __DEBUG_BARRIER_FENCE_SPACE_ANALYZER
    std::cout << "isSafeWriteInLoop False case 1" << std::endl;
#endif
    return "";
  }

  const DeclRefExpr *DRE = *WILDRESet.begin();
  auto Iter = DREIncStepMap.find(DRE);
  if (Iter == DREIncStepMap.end()) {
#ifdef __DEBUG_BARRIER_FENCE_SPACE_ANALYZER
    std::cout << "isSafeWriteInLoop False case 2" << std::endl;
#endif
    return "";
  }
  return Iter->second;
}

/// @brief Check if it is safe to use local barrier to migrate current
/// __syncthreads call.
/// The requirements to return ture:
///   For each global pointer Decl :
///   (1) all of its reference points must be read accesses
///   or
///   (2) all of its reference points must be wirte accesses and write accesses
///   must all in predecessor or all in successor
/// @param DefUsageInfoMap Saves info of all global memory pointers' reference
/// points
/// @param SCI Saves predecessor ranges and successor ranges of current
/// __syncthreads call
/// @return Is safe or not, and the condition string (if needed)
std::tuple<bool /*CanUseLocalBarrier*/,
           bool /*CanUseLocalBarrierWithCondition*/, std::string /*Condition*/>
clang::dpct::BarrierFenceSpaceAnalyzer::isSafeToUseLocalBarrier(
    const std::map<const ParmVarDecl *, std::set<DREInfo>> &DefDREInfoMap,
    const SyncCallInfo &SCI) {
#ifdef __DEBUG_BARRIER_FENCE_SPACE_ANALYZER
  std::cout << "===== isSafeToUseLocalBarrier =====" << std::endl;
#endif
  std::set<std::string> ConditionSet;
  for (auto &DefDREInfo : DefDREInfoMap) {
    bool FoundRead = false;
    bool FoundWrite = false;
    bool DREInPredecessors = false;
    bool DREInSuccessors = false;
    std::set<const DeclRefExpr *> WriteAfterWriteDRE;
    for (auto &DREInfo : DefDREInfo.second) {
      if (DREInfo.SL.isMacroID() || (DREInfo.AM == AccessMode::ReadWrite)) {
#ifdef __DEBUG_BARRIER_FENCE_SPACE_ANALYZER
        std::cout << "isSafeToUseLocalBarrier False case 1" << std::endl;
#endif
        return {false, false, ""};
      }
      if (DREInfo.AM == AccessMode::Read) {
        FoundRead = true;
      } else if (DREInfo.AM == AccessMode::Write) {
        FoundWrite = true;
      }
      if (isInRanges(DREInfo.SL, SCI.Predecessors)) {
        DREInPredecessors = true;
      }
      if (isInRanges(DREInfo.SL, SCI.Successors)) {
        DREInSuccessors = true;
      }

      if (FoundRead && FoundWrite) {
#ifdef __DEBUG_BARRIER_FENCE_SPACE_ANALYZER
        std::cout << "isSafeToUseLocalBarrier False case 2" << std::endl;
#endif
        return {false, false, ""};
      }
      if (FoundWrite && DREInPredecessors && DREInSuccessors) {
        WriteAfterWriteDRE.insert(DREInfo.DRE);
      }
    }
    if (!WriteAfterWriteDRE.empty()) {
      auto StepStr = isSafeWriteInLoop(WriteAfterWriteDRE);
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
    else
      return {false, true, "item_ct1.get_local_range(2) < " + RHS};
  }
#ifdef __DEBUG_BARRIER_FENCE_SPACE_ANALYZER
  std::cout << "isSafeToUseLocalBarrier True" << std::endl;
#endif
  return {true, false, ""};
}

std::unordered_map<
    std::string, std::unordered_map<
                     std::string, clang::dpct::BarrierFenceSpaceAnalyzerResult>>
    clang::dpct::BarrierFenceSpaceAnalyzer::CachedResults;
