//===--------------- BarrierFenceSpaceAnalyzer.cpp ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "BarrierFenceSpaceAnalyzer.h"
#include "AnalysisInfo.h"
#include "Utility.h"
#include "llvm/Support/Casting.h"
#include <algorithm>
#include <string>
#include <unordered_set>

using namespace llvm;

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

bool clang::dpct::TypeAnalyzer::canBeAnalyzed(const clang::Type *TypePtr) {
  switch (TypePtr->getTypeClass()) {
  case clang::Type::TypeClass::ConstantArray:
    return canBeAnalyzed(dyn_cast<clang::ConstantArrayType>(TypePtr)
                             ->getElementType()
                             .getTypePtr());
  case clang::Type::TypeClass::Pointer:
    PointerLevel++;
    if (PointerLevel >= 2 || IsClass)
      return false;
    IsConstPtr = TypePtr->getPointeeType().isConstQualified();
    return canBeAnalyzed(TypePtr->getPointeeType().getTypePtr());
  case clang::Type::TypeClass::Elaborated:
    return canBeAnalyzed(
        dyn_cast<clang::ElaboratedType>(TypePtr)->desugar().getTypePtr());
  case clang::Type::TypeClass::Typedef:
    return canBeAnalyzed(dyn_cast<clang::TypedefType>(TypePtr)
                             ->getDecl()
                             ->getUnderlyingType()
                             .getTypePtr());
  case clang::Type::TypeClass::Record:
    IsClass = true;
    if (PointerLevel &&
        isUserDefinedDecl(dyn_cast<clang::RecordType>(TypePtr)->getDecl()))
      return false;
    for (const auto &Field :
         dyn_cast<clang::RecordType>(TypePtr)->getDecl()->fields()) {
      if (!canBeAnalyzed(Field->getType().getTypePtr())) {
        return false;
      }
    }
    return true;
  case clang::Type::TypeClass::SubstTemplateTypeParm:
    return canBeAnalyzed(dyn_cast<clang::SubstTemplateTypeParmType>(TypePtr)
                             ->getReplacementType()
                             .getTypePtr());
  case clang::Type::TypeClass::TemplateTypeParm: {
    const clang::TemplateTypeParmType *TTPT =
        dyn_cast<clang::TemplateTypeParmType>(TypePtr);
    const TemplateTypeParmDecl *TTPD = TTPT->getDecl();
    auto Idx = TTPD->getIndex();
    const FunctionTemplateDecl *FTD =
        clang::dpct::DpctGlobalInfo::findParent<FunctionTemplateDecl>(TTPD);
    if (!FTD)
      return false;
    for (const auto &S : FTD->specializations()) {
      const auto TA = S->getTemplateSpecializationArgs()->get(Idx);
      if (TA.getKind() == clang::TemplateArgument::ArgKind::Type) {
        if (!canBeAnalyzed(TA.getAsType().getTypePtr()))
          return false;
      }
    }
    return true;
  }
  default:
    if (TypePtr->isFundamentalType())
      return true;
    else
      return false;
  }
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
clang::dpct::detail::AnalyzerBase::isAssignedToAnotherDREOrVD(
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
  return std::make_pair(ResultDRESet, ResultVDSet);
}

void clang::dpct::detail::AnalyzerBase::constructDefUseMap() {
  auto getSize =
      [](const std::unordered_map<const VarDecl *,
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
      const VarDecl *CurDecl = Pair.first;
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
}

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

  if (FuncName == "__syncthreads" || FuncName == "__barrier_sync" ||
      isUserDefinedDecl(FuncDecl)) {
    SyncCallInfo SCI;
    SCI.IsRealSyncCall =
        (FuncName == "__syncthreads" || FuncName == "__barrier_sync");
    SCI.Predecessors.insert(
        SourceRange(FD->getBody()->getBeginLoc(), CE->getBeginLoc()));
    SCI.Successors.insert(
        SourceRange(CE->getEndLoc(), FD->getBody()->getEndLoc()));
    if (!LoopRange.empty()) {
      SCI.Predecessors.insert(LoopRange.front());
      SCI.Successors.insert(LoopRange.front());
      SCI.IsInLoop = true;
    }
    SyncCallsVec.push_back(std::make_pair(CE, SCI));
  }
  return true;
}
void clang::dpct::IntraproceduralAnalyzer::PostVisit(const CallExpr *) {}

bool clang::dpct::IntraproceduralAnalyzer::Visit(const DeclRefExpr *DRE) {
  // Collect all DREs and its Decl
  const auto &VD = dyn_cast<VarDecl>(DRE->getDecl());
  if (!VD)
    return true;
  if (VD->isLocalVarDecl())
    return true;
  if (isFromCUDA(VD))
    return true;
  if (!isa<ParmVarDecl>(VD))
    return false;

  TypeAnalyzer TA;
  TypeAnalyzer::ParamterTypeKind Kind =
      TA.getInputParamterTypeKind(VD->getType());
  if (Kind == TypeAnalyzer::ParamterTypeKind::CanSkipAnalysis) {
    return true;
  }
  if (Kind == TypeAnalyzer::ParamterTypeKind::Unsupported) {
    return false;
  }

  const auto &Iter = DefUseMap.find(VD);
  if (Iter != DefUseMap.end()) {
    Iter->second.insert(DRE);
  } else {
    std::set<const DeclRefExpr *> Set;
    Set.insert(DRE);
    DefUseMap.insert(std::make_pair(VD, Set));
  }
  return true;
}
void clang::dpct::IntraproceduralAnalyzer::PostVisit(const DeclRefExpr *) {}

bool clang::dpct::IntraproceduralAnalyzer::Visit(const GotoStmt *GS) {
  // We will further refine it if meet real request.
  // By default, goto/label stmt is not supported.
  return false;
}
void clang::dpct::IntraproceduralAnalyzer::PostVisit(const GotoStmt *) {}

bool clang::dpct::IntraproceduralAnalyzer::Visit(const LabelStmt *LS) {
  // We will further refine it if meet real request.
  // By default, goto/label stmt is not supported.
  return false;
}
void clang::dpct::IntraproceduralAnalyzer::PostVisit(const LabelStmt *) {}

bool clang::dpct::IntraproceduralAnalyzer::Visit(const MemberExpr *ME) {
  if (ME->getType()->isPointerType() || ME->getType()->isArrayType()) {
    return false;
  }
  return true;
}
void clang::dpct::IntraproceduralAnalyzer::PostVisit(const MemberExpr *) {}
bool clang::dpct::IntraproceduralAnalyzer::Visit(
    const CXXDependentScopeMemberExpr *CDSME) {
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

    if (!FoundCE && CE && CE->getDirectCallee() &&
        isFromCUDA(CE->getDirectCallee())) {
      FoundCE = true;
      int Idx = 0;
      for (const auto &Arg : CE->arguments()) {
        if (Arg == Current.get<Expr>())
          break;
        Idx++;
      }
      if (Idx > (int)CE->getNumArgs())
        Idx = -1;
      if (Idx > 0 && (dyn_cast<CXXOperatorCallExpr>(CE) ||
                      dyn_cast<CXXMemberCallExpr>(CE)))
        Idx--;
      if (Idx >= 0 && Idx < (int)CE->getDirectCallee()->getNumParams()) {
        PVD = CE->getDirectCallee()->getParamDecl(Idx);
      } else {
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
} // namespace

void clang::dpct::IntraproceduralAnalyzer::simplifyMap(
    std::map<const VarDecl *, std::set<DREInfo>> &DefDREInfoMap) {
  std::map<const VarDecl *,
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
}

AffectedInfo mergeOther(AffectedInfo Me, AffectedInfo Other,
                        bool IsOtherInLoop) {
  if (IsOtherInLoop) {
    Other.UsedBefore = true;
    Other.UsedAfter = true;
  }
  Me.UsedBefore = Me.UsedBefore || Other.UsedBefore;
  Me.UsedAfter = Me.UsedAfter || Other.UsedAfter;
  Me.AM = AccessMode(Me.AM | Other.AM);
  return Me;
}

// Limitation: If func1 is called more than once in func2, we can't
// distinguish the different call sites. So return empty string for now.
std::string findCurCallCombinedLoc(std::string CurCallDeclCombinedLoc,
                                   std::shared_ptr<DeviceFunctionInfo> CurNode,
                                   std::string SyncCallCombinedLoc) {
  if (CurCallDeclCombinedLoc == "__syncthreads")
    return SyncCallCombinedLoc;
  std::vector<std::string> CallLocVec;
  for (const auto &Pair : CurNode->getCallExprMap()) {
    std::shared_ptr<CallFunctionExpr> Call = Pair.second;
    if (CurCallDeclCombinedLoc == Call->getDeclCombinedLoc()) {
      std::string CallLoc = Call->getFilePath().getCanonicalPath().str() + ":" +
                            std::to_string(Call->getCallFuncExprOffset());
      CallLocVec.push_back(CallLoc);
    }
  }
  if (CallLocVec.size() == 1)
    return CallLocVec[0];
  return "";
}

bool clang::dpct::InterproceduralAnalyzer::analyze(
    const std::shared_ptr<DeviceFunctionInfo> InputDFI,
    std::string SyncCallCombinedLoc) {
  if (InputDFI->NonCudaCallNum > 0) {
    return false;
  }
  // Do analysis for all syncthreads call in this DFI's ancestors and
  // this DFI's decendents.
  std::stack<std::tuple<std::weak_ptr<DeviceFunctionInfo> /*node*/,
                        std::string /*CurCallCombinedLoc*/, int /*depth*/>>
      NodeStack;

  std::stack<std::pair<
      std::string /*caller's decl's combined loc str*/,
      std::unordered_map<unsigned int /*parameter idx*/, AffectedInfo>>>
      AffectedByParmsMapInfoStack;
  std::vector<std::pair<
      std::string /*caller's decl's combined loc str*/,
      std::unordered_map<unsigned int /*parameter idx*/, AffectedInfo>>>
      AffectedByParmsMapInfoVec;

  std::set<std::weak_ptr<DeviceFunctionInfo>,
           std::owner_less<std::weak_ptr<DeviceFunctionInfo>>>
      Visited;

  // DFS to find all related DFIs
  NodeStack.push(std::make_tuple(InputDFI, "__syncthreads", 0));
  Visited.insert(InputDFI);
  while (!NodeStack.empty()) {
    auto CurNode = std::get<0>(NodeStack.top()).lock();
    auto CurCallDeclCombinedLoc = std::get<1>(NodeStack.top());
    std::string CurCallCombinedLoc = findCurCallCombinedLoc(
        CurCallDeclCombinedLoc, CurNode, SyncCallCombinedLoc);
    auto CurDepth = std::get<2>(NodeStack.top());
    if (CurCallCombinedLoc.empty()) {
      return false;
    }
    NodeStack.pop();

    int N1 = static_cast<int>(AffectedByParmsMapInfoStack.size()) - CurDepth;
    assert(N1 >= 0 && "N should be greater than or equal to 0");
    if (!AffectedByParmsMapInfoStack.empty())
      AffectedByParmsMapInfoVec.push_back(AffectedByParmsMapInfoStack.top());
    for (int i = 0; i < N1; i++) {
      AffectedByParmsMapInfoStack.pop();
    }

    auto Iter = CurNode->IAR.Map.find(CurCallCombinedLoc);
    if (Iter != CurNode->IAR.Map.end()) {
      const auto &Arg2ParmsMap = std::get<5>(Iter->second);
      bool IsInLoop = std::get<1>(Iter->second);
      // Merge std::get<4>(Iter->second) and
      // AffectedByParmsMapInfoStack.top().second.
      // Then push into AffectedByParmsMapInfoStack.
      std::unordered_map<unsigned int /*parameter idx*/, AffectedInfo>
          CurAffectedbyParmsMap = std::get<4>(Iter->second);
      if (AffectedByParmsMapInfoStack.empty()) {
        AffectedByParmsMapInfoStack.push(std::make_pair(
            CurNode->IAR.CurrentCtxFuncCombinedLoc, CurAffectedbyParmsMap));
      } else {
        const std::unordered_map<unsigned int /*parameter idx*/, AffectedInfo>
            &PrevAffectedbyParmsMap = AffectedByParmsMapInfoStack.top().second;
        for (const auto &P : PrevAffectedbyParmsMap) {
          auto Res = Arg2ParmsMap.find(P.first);
          if (Res != Arg2ParmsMap.end()) {
            for (const auto &ParmIdx : Res->second) {
              CurAffectedbyParmsMap[ParmIdx] = mergeOther(
                  CurAffectedbyParmsMap[ParmIdx], P.second, IsInLoop);
              if (CurAffectedbyParmsMap[ParmIdx].AM == ReadWrite) {
                return false;
              }
              if (CurAffectedbyParmsMap[ParmIdx].AM == Write &&
                  CurAffectedbyParmsMap[ParmIdx].UsedBefore &&
                  CurAffectedbyParmsMap[ParmIdx].UsedAfter) {
                return false;
              }
            }
          }
        }
        AffectedByParmsMapInfoStack.push(std::make_pair(
            CurNode->IAR.CurrentCtxFuncCombinedLoc, CurAffectedbyParmsMap));
      }
    }

    for (const auto &I : CurNode->getParentDFIs()) {
      if (Visited.find(I) != Visited.end()) {
        // Not support analyzing circle in graph
        return false;
      }
      const auto &Iter = I.lock();
      if (Iter->NonCudaCallNum > 1) {
        // The only one non-cuda call should be current node itself
        return false;
      }
      NodeStack.push(std::make_tuple(I, CurNode->IAR.CurrentCtxFuncCombinedLoc,
                                     CurDepth + 1));
      Visited.insert(I);
    }
  }

  if (!AffectedByParmsMapInfoStack.empty())
    AffectedByParmsMapInfoVec.push_back(AffectedByParmsMapInfoStack.top());

  for (const auto &Iter : AffectedByParmsMapInfoVec) {
    const auto &AffectedByParmsMap = Iter.second;
    for (const auto &P : AffectedByParmsMap) {
      if (P.second.AM == ReadWrite) {
        return false;
      }
      if ((P.second.AM == Write) && P.second.UsedBefore && P.second.UsedAfter) {
        return false;
      }
    }
  }

  return true;
}

clang::dpct::IntraproceduralAnalyzerResult
clang::dpct::IntraproceduralAnalyzer::analyze(const FunctionDecl *FD,
                                              DeviceFunctionInfo *DFI) {
  // Check prerequirements
  if (!DFI) {
    return IntraproceduralAnalyzerResult();
  }

  if (DFI->getVarMap().hasGlobalMemAcc()) {
    return IntraproceduralAnalyzerResult();
  }

  // Init values
  this->FD = FD;

  FDLoc = getHashStrFromLoc(FD->getBeginLoc());

  if (!this->TraverseDecl(const_cast<FunctionDecl *>(FD))) {
    return IntraproceduralAnalyzerResult();
  }

  constructDefUseMap();
  std::map<const VarDecl *, std::set<DREInfo>> DefLocInfoMap;
  simplifyMap(DefLocInfoMap);

  generateDRE2VDMap(DefLocInfoMap);

  std::unordered_map<
      std::string /*call's combined loc string*/,
      std::tuple<
          bool /*is real sync call*/, bool /*is in loop*/, tooling::UnifiedPath,
          unsigned int,
          std::unordered_map<unsigned int /*parameter idx*/, AffectedInfo>,
          std::unordered_map<
              unsigned int /*arg idx*/,
              std::set<unsigned int> /*caller parameter(s) idx*/>>>
      Map;
  for (auto &SyncCall : SyncCallsVec) {
    std::unordered_map<unsigned int /*parameter idx*/, AffectedInfo>
        AffectedByParmsMap;
    affectedByWhichParameters(DefLocInfoMap, SyncCall.second,
                              AffectedByParmsMap);
    const auto ArgCallerParmsMap = getArgCallerParmsMap(SyncCall.first);
    auto LocInfo = DpctGlobalInfo::getLocInfo(SyncCall.first->getBeginLoc());
    Map.insert(std::make_pair(
        getCombinedStrFromLoc(SyncCall.first->getBeginLoc()),
        std::make_tuple(SyncCall.second.IsRealSyncCall,
                        SyncCall.second.IsInLoop, LocInfo.first, LocInfo.second,
                        AffectedByParmsMap, ArgCallerParmsMap)));
  }
  return IntraproceduralAnalyzerResult(
      Map, getCombinedStrFromLoc(FD->getBeginLoc()));
}

bool clang::dpct::IntraproceduralAnalyzer::isAccessingMemory(
    const DeclRefExpr *DRE) {
  if (const auto *VD = dyn_cast<VarDecl>(DRE->getDecl())) {
    if (VD->hasGlobalStorage())
      return true;
  }
  auto &Context = DpctGlobalInfo::getContext();
  DynTypedNode Current = DynTypedNode::create(*DRE);
  DynTypedNodeList Parents = Context.getParents(Current);
  while (!Parents.empty()) {
    if (Parents[0].get<FunctionDecl>() && Parents[0].get<FunctionDecl>() == FD)
      break;
    const auto &E = Parents[0].get<Expr>();
    if (E && DeviceFunctionCallArgs.count(E))
      return true;
    const auto &UO = Parents[0].get<UnaryOperator>();
    const auto &ASE = Parents[0].get<ArraySubscriptExpr>();
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

auto convertPVD2Idx = [](const FunctionDecl *FD, const ParmVarDecl *PVD) {
  unsigned int Idx = 0;
  for (const auto &D : FD->parameters()) {
    if (D == PVD)
      return Idx;
    Idx++;
  }
  assert(0 && "PVD is not in the FD.");
};

void IntraproceduralAnalyzer::affectedByWhichParameters(
    const std::map<const VarDecl *, std::set<DREInfo>> &DefDREInfoMap,
    const SyncCallInfo &SCI,
    std::unordered_map<unsigned int /*parameter idx*/, AffectedInfo>
        &AffectingParameters) {
  for (auto &DefDREInfo : DefDREInfoMap) {
    bool UsedBefore = false;
    bool UsedAfter = false;
    AccessMode AM = NotSet;
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
    if (AM != Read) {
      if (const ParmVarDecl *PVD = dyn_cast<ParmVarDecl>(DefDREInfo.first)) {
        AffectingParameters.insert(std::make_pair(
            convertPVD2Idx(FD, PVD), AffectedInfo{UsedBefore, UsedAfter, AM}));
      }
    }
  }
}

void IntraproceduralAnalyzer::generateDRE2VDMap(
    const std::map<const VarDecl *, std::set<DREInfo>> &Map) {
  for (const auto &I : Map) {
    for (const auto &J : I.second) {
      DRE2VDMap.insert(std::make_pair(J.DRE, I.first));
    }
  }
}

std::unordered_map<unsigned int /*arg idx*/,
                   std::set<unsigned int> /*caller parameter(s) idx*/>
IntraproceduralAnalyzer::getArgCallerParmsMap(const CallExpr *CE) {
  std::unordered_map<unsigned int /*arg idx*/,
                     std::set<unsigned int> /*caller parameter(s) idx*/>
      RetMap;
  unsigned int ArgIdx = 0;
  for (const auto &E : CE->arguments()) {
    std::set<const clang::DeclRefExpr *> DRESet;
    bool HasCallExpr;
    findDREs(E, DRESet, HasCallExpr /*un-used output arg*/);
    std::set<unsigned int> CallerParmsIdx;
    for (const auto &DRE : DRESet) {
      const auto &Iter = DRE2VDMap.find(DRE);
      if (Iter != DRE2VDMap.end()) {
        const ParmVarDecl *PVD = dyn_cast<ParmVarDecl>(Iter->second);
        if (!PVD)
          continue;
        CallerParmsIdx.insert(convertPVD2Idx(FD, PVD));
      }
    }
    RetMap.insert(std::make_pair(ArgIdx, CallerParmsIdx));
    ArgIdx++;
  }
  return RetMap;
}

bool clang::dpct::BarrierFenceSpace1DAnalyzer::Visit(const CallExpr *CE) {
  const FunctionDecl *FuncDecl = CE->getDirectCallee();
  if (!FuncDecl)
    return true;
  if (isUserDefinedDecl(FuncDecl))
    return false;
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
  return false;
}
void clang::dpct::BarrierFenceSpace1DAnalyzer::PostVisit(const GotoStmt *) {}

bool clang::dpct::BarrierFenceSpace1DAnalyzer::Visit(const LabelStmt *LS) {
  // We will further refine it if meet real request.
  // By default, goto/label stmt is not supported.
  return false;
}
void clang::dpct::BarrierFenceSpace1DAnalyzer::PostVisit(const LabelStmt *) {}

bool clang::dpct::BarrierFenceSpace1DAnalyzer::Visit(const MemberExpr *ME) {
  if (ME->getType()->isPointerType() || ME->getType()->isArrayType()) {
    return false;
  }
  return true;
}
void clang::dpct::BarrierFenceSpace1DAnalyzer::PostVisit(const MemberExpr *) {}
bool clang::dpct::BarrierFenceSpace1DAnalyzer::Visit(
    const CXXDependentScopeMemberExpr *CDSME) {
  return false;
}
void clang::dpct::BarrierFenceSpace1DAnalyzer::PostVisit(
    const CXXDependentScopeMemberExpr *) {}

namespace {
using namespace clang;
using namespace dpct;

bool isMeetAnalyisPrerequirements(const CallExpr *CE, const FunctionDecl *&FD) {
  if (CE->getBeginLoc().isMacroID() || CE->getEndLoc().isMacroID()) {
    return false;
  }
  FD = DpctGlobalInfo::findAncestor<FunctionDecl>(CE);
  if (!FD) {
    return false;
  }
  if (!FD->hasAttr<CUDAGlobalAttr>()) {
    return false;
  }
  std::unordered_set<const DeviceFunctionInfo *> Visited{};
  auto DFI = DeviceFunctionDecl::LinkRedecls(FD);
  if (!DFI) {
    return false;
  }

  if (DFI->getVarMap().hasGlobalMemAcc()) {
    return false;
  }
  return true;
}
} // namespace

void clang::dpct::BarrierFenceSpace1DAnalyzer::simplifyMap(
    std::map<const VarDecl *, std::set<DREInfo>> &DefDREInfoMap) {
  std::map<const VarDecl *, std::set<const DeclRefExpr *>> DefDREInfoMapTemp;
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
  std::map<const VarDecl *, std::set<DREInfo>> DefLocInfoMap;
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
    return "";
  }

  const DeclRefExpr *DRE = *WriteInLoopDRESet.begin();
  auto Iter = DREIncStepMap.find(DRE);
  if (Iter == DREIncStepMap.end()) {
    return "";
  }
  return Iter->second;
}

std::tuple<bool /*CanUseLocalBarrier*/,
           bool /*CanUseLocalBarrierWithCondition*/, std::string /*Condition*/>
clang::dpct::BarrierFenceSpace1DAnalyzer::isSafeToUseLocalBarrier(
    const std::map<const VarDecl *, std::set<DREInfo>> &DefDREInfoMap) {
  std::set<std::string> ConditionSet;
  for (auto &DefDREInfo : DefDREInfoMap) {
    std::set<const DeclRefExpr *> RefedDRE;
    for (auto &DREInfo : DefDREInfo.second) {
      if (DREInfo.SL.isMacroID()) {
        return {false, false, ""};
      }
      RefedDRE.insert(DREInfo.DRE);
    }
    if (!RefedDRE.empty()) {
      auto StepStr = isAnalyzableWriteInLoop(RefedDRE);
      if (StepStr.empty()) {
        return {false, false, ""};
      }
      ConditionSet.insert(StepStr);
    }
  }

  if (!ConditionSet.empty()) {
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
  return {true, false, ""};
}
