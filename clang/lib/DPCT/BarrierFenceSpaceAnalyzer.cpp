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
  std::string FuncName;
  if (FuncDecl)
    FuncName = FuncDecl->getNameInfo().getName().getAsString();

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
  } else {
    if (auto FD = CE->getDirectCallee()) {
      std::string FuncName = FD->getNameInfo().getName().getAsString();
      if (!AllowedDeviceFunctions.count(FuncName) || isUserDefinedDecl(FD)) {
#ifdef __DEBUG_BARRIER_FENCE_SPACE_ANALYZER
        std::cout << "Return False case A: "
                  << CE->getBeginLoc().printToString(
                         DpctGlobalInfo::getSourceManager())
                  << std::endl;
#endif
        return false;
      }
    }
  }
  return true;
}
void clang::dpct::BarrierFenceSpaceAnalyzer::PostVisit(const CallExpr *) {}

bool clang::dpct::BarrierFenceSpaceAnalyzer::Visit(const DeclRefExpr *DRE) {
  const ValueDecl *VD = DRE->getDecl();
  if (!dyn_cast<FunctionDecl>(VD) &&
      DRE->getDecl()->hasAttr<CUDADeviceAttr>() &&
      !(VD->getName().str() == "threadIdx" ||
        VD->getName().str() == "blockIdx" ||
        VD->getName().str() == "blockDim" ||
        VD->getName().str() == "gridDim")) {
    setFalseForThisFunctionDecl();
#ifdef __DEBUG_BARRIER_FENCE_SPACE_ANALYZER
    std::cout << "Return False case B: "
              << DRE->getBeginLoc().printToString(
                     DpctGlobalInfo::getSourceManager())
              << std::endl;
#endif
    return false; // not support __device__ variables
  }
  // Collect all DREs and its Decl
  const auto PVD = dyn_cast<ParmVarDecl>(DRE->getDecl());
  if (!PVD)
    return true;

  /// \return One of below 3 int values:
  ///  1: can skip analysis
  ///  0: need analysis
  /// -1: unsupport to analyze
  std::function<int(QualType)> getInputParamterTypeKind =
      [](QualType QT) -> int {
    if (QT->isPointerType()) {
      QualType PointeeQT = QT->getPointeeType();
      if (PointeeQT.isConstQualified())
        return 1;
      if (PointeeQT->isFundamentalType())
        return 0;
      return -1;
    }
    if (QT->isFundamentalType()) {
      return 1;
    }
    if (auto ET = dyn_cast<ElaboratedType>(QT.getTypePtr())) {
      if (auto RT = dyn_cast<RecordType>(ET->desugar())) {
        for (const auto &Field : RT->getDecl()->fields()) {
          if (!Field->getType()->isFundamentalType()) {
            return -1;
          }
        }
        return 1;
      }
    }
    return -1;
  };

  int ParamterTypeKind = getInputParamterTypeKind(PVD->getType());
  if (ParamterTypeKind == 1) {
    return true;
  } else if (ParamterTypeKind == -1) {
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
  std::cout << "Return False case F: "
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
  std::cout << "Return False case G: "
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
    std::cout << "Return False case H: "
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
  std::cout << "Return False case I: "
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
  auto Ctor = CCE->getConstructor();
  std::string CtorName = Ctor->getParent()->getQualifiedNameAsString();
  if (AllowedDeviceFunctions.count(CtorName) && !isUserDefinedDecl(Ctor))
    return true;
#ifdef __DEBUG_BARRIER_FENCE_SPACE_ANALYZER
  std::cout << "Return False case J: "
            << CCE->getBeginLoc().printToString(
                   DpctGlobalInfo::getSourceManager())
            << std::endl;
#endif
  return false;
}
void clang::dpct::BarrierFenceSpaceAnalyzer::PostVisit(
    const CXXConstructExpr *) {}

bool clang::dpct::BarrierFenceSpaceAnalyzer::traverseFunction(
    const FunctionDecl *FD) {
  if (!this->TraverseDecl(const_cast<FunctionDecl *>(FD))) {
    return false;
  }
  return true;
}

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

std::set<const clang::DeclRefExpr *>
clang::dpct::BarrierFenceSpaceAnalyzer::isAssignedToAnotherDRE(
    const DeclRefExpr *CurrentDRE) {
  std::set<const DeclRefExpr *> ResultSet;
  auto &Context = DpctGlobalInfo::getContext();
  DynTypedNode Current = DynTypedNode::create(*CurrentDRE);
  DynTypedNodeList Parents = Context.getParents(Current);
  while (!Parents.empty()) {
    if (Parents[0].get<FunctionDecl>() && Parents[0].get<FunctionDecl>() == FD)
      break;
    const auto BO = Parents[0].get<BinaryOperator>();
    if (BO && BO->isAssignmentOp() && (BO->getRHS() == Current.get<Expr>()) &&
        BO->getLHS()->getType()->isPointerType()) {
      auto DREMatcher =
          ast_matchers::findAll(ast_matchers::declRefExpr().bind("DRE"));
      auto MatchedResults = ast_matchers::match(DREMatcher, *(BO->getRHS()),
                                                DpctGlobalInfo::getContext());
      for (auto Node : MatchedResults) {
        if (auto DRE = Node.getNodeAs<DeclRefExpr>("DRE"))
          ResultSet.insert(DRE);
      }
    }
    Current = Parents[0];
    Parents = Context.getParents(Current);
  }

#ifdef __DEBUG_BARRIER_FENCE_SPACE_ANALYZER
  const auto &SM = DpctGlobalInfo::getSourceManager();
  std::cout << "CurrentDRE:" << CurrentDRE->getBeginLoc().printToString(SM)
            << std::endl;
  for (const auto Item : ResultSet) {
    std::cout << "    AnotherDRE:" << Item->getBeginLoc().printToString(SM)
              << std::endl;
  }
#endif
  return ResultSet;
}

const clang::BinaryOperator *
clang::dpct::BarrierFenceSpaceAnalyzer::getAssignmentBinaryOP(
    const DeclRefExpr *CurrentDRE) {
  bool FoundDerefOrArraySubscript = false;
  auto &Context = DpctGlobalInfo::getContext();
  DynTypedNode Current = DynTypedNode::create(*CurrentDRE);
  DynTypedNodeList Parents = Context.getParents(Current);
  while (!Parents.empty()) {
    if (Parents[0].get<FunctionDecl>() && Parents[0].get<FunctionDecl>() == FD)
      break;
    const auto BO = Parents[0].get<BinaryOperator>();
    const auto UO = Parents[0].get<UnaryOperator>();
    const auto ASE = Parents[0].get<ArraySubscriptExpr>();
    if (BO && BO->isAssignmentOp() && (BO->getLHS() == Current.get<Expr>()) &&
        FoundDerefOrArraySubscript) {
      return BO;
    } else if (UO && (UO->getOpcode() == UnaryOperatorKind::UO_Deref)) {
      FoundDerefOrArraySubscript = true;
    } else if (ASE && (ASE->getBase() == Current.get<Expr>())) {
      FoundDerefOrArraySubscript = true;
    }
    Current = Parents[0];
    Parents = Context.getParents(Current);
  }
  return nullptr;
}

clang::dpct::BarrierFenceSpaceAnalyzer::AccessMode
clang::dpct::BarrierFenceSpaceAnalyzer::getAccessKind(
    const DeclRefExpr *CurrentDRE) {
  const BinaryOperator *BO = getAssignmentBinaryOP(CurrentDRE);
  if (!BO) {
    return AccessMode::Read;
  }
  if (BO->getOpcode() == BinaryOperatorKind::BO_Assign) {
    return AccessMode::Write;
  }
  return AccessMode::ReadWrite;
}

bool clang::dpct::BarrierFenceSpaceAnalyzer::canSetLocalFenceSpace(
    const CallExpr *CE) {
#ifdef __DEBUG_BARRIER_FENCE_SPACE_ANALYZER
  std::cout << "BarrierFenceSpaceAnalyzer Analyzing ..." << std::endl;
#endif
  if (CE->getBeginLoc().isMacroID() || CE->getEndLoc().isMacroID())
    return false;
  auto FD = DpctGlobalInfo::findAncestor<FunctionDecl>(CE);
  if (!FD)
    return false;
  if (!FD->hasAttr<CUDAGlobalAttr>())
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

#ifdef __DEBUG_BARRIER_FENCE_SPACE_ANALYZER
  std::cout << "Before traverse" << std::endl;
#endif

  this->FD = FD;

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
        std::set<const DeclRefExpr *> AssignedDREs =
            isAssignedToAnotherDRE(DRE);
        for (const auto AnotherDRE : AssignedDREs) {
          std::set<const DeclRefExpr *> AnotherDREMatchedResult = matchAllDRE(
              dyn_cast_or_null<VarDecl>(AnotherDRE->getDecl()), FD->getBody());
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

  // Convert DRE to <Location, AccessMode> pair for comparing
  std::map<const ParmVarDecl *, std::set<std::pair<SourceLocation, AccessMode>>>
      DeclUsedLocsMap;
  for (const auto &Pair : DefUseMap) {
    for (const auto &Item : Pair.second) {
      if (!isNoOverlappingAccessAmongWorkItems(KernelDim, Item)) {
        DeclUsedLocsMap[Pair.first].insert(
            std::make_pair(Item->getBeginLoc(), getAccessKind(Item)));
      }
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

  for (auto &SyncCall : SyncCallsVec) {
    CachedResults[FDLoc][getHashStrFromLoc(SyncCall.first->getBeginLoc())] =
        isValidAccessPattern(DeclUsedLocsMap, SyncCall.second);
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

bool clang::dpct::BarrierFenceSpaceAnalyzer::isValidAccessPattern(
    const std::map<const ParmVarDecl *,
                   std::set<std::pair<SourceLocation, AccessMode>>>
        &DeclUsedLocsMap,
    const SyncCallInfo &SCI) {
  for (auto &DeclUsedLocPair : DeclUsedLocsMap) {
    bool FoundRead = false;
    bool FoundWrite = false;
    for (auto &LocModePair : DeclUsedLocPair.second) {
      if (containsMacro(LocModePair.first, SCI) ||
          (LocModePair.second == AccessMode::ReadWrite))
        return false;
      if (LocModePair.second == AccessMode::Read) {
        FoundRead = true;
      } else if (LocModePair.second == AccessMode::Write) {
        FoundWrite = true;
      }
      if (FoundRead && FoundWrite)
        return false;
    }
  }
  return true;
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
        "uint4",
        "sqrtf",
        "__expf",
        "fmaf"};

// TODO: Implement more accuracy Predecessors and Successors. Then below code
//       can be used for checking.
#if 0
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

bool clang::dpct::BarrierFenceSpaceAnalyzer::isValidLocationSet(
    const std::set<std::pair<SourceLocation, AccessMode>> &LocationSet,
    const SyncCallInfo &SCI) {
#ifdef __DEBUG_BARRIER_FENCE_SPACE_ANALYZER
  const auto &SM = DpctGlobalInfo::getSourceManager();
  std::cout << "===== isValidLocationSet: =====" << std::endl;
  for (const auto &LocModePair : LocationSet) {
    std::cout << "    DRE:" << LocModePair.first.printToString(SM)
              << ", AccessMode:" << (int)(LocModePair.second) << std::endl;
  }
  std::cout << "Predecessors:" << std::endl;
  for (const auto &Range : SCI.Predecessors) {
    std::cout << "    [" << Range.getBegin().printToString(SM) << ", "
              << Range.getEnd().printToString(SM) << "]" << std::endl;
  }
  std::cout << "Successors:" << std::endl;
  for (const auto &Range : SCI.Successors) {
    std::cout << "    [" << Range.getBegin().printToString(SM) << ", "
              << Range.getEnd().printToString(SM) << "]" << std::endl;
  }
  std::cout << "===== isValidLocationSet end =====" << std::endl;
#endif
  bool DREInPredecessors = false;
  bool DREInSuccessors = false;
  for (auto &LocModePair : LocationSet) {
    if (isInRanges(LocModePair.first, SCI.Predecessors)) {
      DREInPredecessors = true;
    }
    if (isInRanges(LocModePair.first, SCI.Successors)) {
      DREInSuccessors = true;
    }
    if (DREInPredecessors && DREInSuccessors) {
      return false;
    }
  }
  return true;
}
#endif
