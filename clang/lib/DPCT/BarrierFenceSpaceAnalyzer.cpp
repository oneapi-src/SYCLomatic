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

// #define __DEBUG_BARRIER_FENCE_SPACE_ANALYZER

bool clang::dpct::BarrierFenceSpaceAnalyzer::Visit(const clang::IfStmt *IS) {
  // No special process, treat as one block
  return true;
}
void clang::dpct::BarrierFenceSpaceAnalyzer::PostVisit(
    const clang::IfStmt *IS) {
  // No special process, treat as one block
}
bool clang::dpct::BarrierFenceSpaceAnalyzer::Visit(
    const clang::SwitchStmt *SS) {
  // No special process, treat as one block
  return true;
}
void clang::dpct::BarrierFenceSpaceAnalyzer::PostVisit(
    const clang::SwitchStmt *SS) {
  // No special process, treat as one block
}

bool clang::dpct::BarrierFenceSpaceAnalyzer::Visit(const clang::ForStmt *FS) {
  LoopRange.push_back(FS->getSourceRange());
  return true;
}
void clang::dpct::BarrierFenceSpaceAnalyzer::PostVisit(
    const clang::ForStmt *FS) {
  LoopRange.pop_back();
}
bool clang::dpct::BarrierFenceSpaceAnalyzer::Visit(const clang::DoStmt *DS) {
  LoopRange.push_back(DS->getSourceRange());
  return true;
}
void clang::dpct::BarrierFenceSpaceAnalyzer::PostVisit(
    const clang::DoStmt *DS) {
  LoopRange.pop_back();
}
bool clang::dpct::BarrierFenceSpaceAnalyzer::Visit(const clang::WhileStmt *WS) {
  LoopRange.push_back(WS->getSourceRange());
  return true;
}
void clang::dpct::BarrierFenceSpaceAnalyzer::PostVisit(
    const clang::WhileStmt *WS) {
  LoopRange.pop_back();
}
bool clang::dpct::BarrierFenceSpaceAnalyzer::Visit(const clang::CallExpr *CE) {
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
#ifdef __DEBUG_BARRIER_FENCE_SPACE_ANALYZER
        std::cout << "Return False case A: "
                  << CE->getBeginLoc().printToString(
                         clang::dpct::DpctGlobalInfo::getSourceManager())
                  << std::endl;
#endif
        return false;
      }
    }
  }
  return true;
}
void clang::dpct::BarrierFenceSpaceAnalyzer::PostVisit(
    const clang::CallExpr *) {}

bool clang::dpct::BarrierFenceSpaceAnalyzer::Visit(
    const clang::DeclRefExpr *DRE) {
  const ValueDecl *VD = DRE->getDecl();
  if (!dyn_cast<FunctionDecl>(VD) &&
      DRE->getDecl()->hasAttr<clang::CUDADeviceAttr>() &&
      !(VD->getName().str() == "threadIdx" ||
        VD->getName().str() == "blockIdx" ||
        VD->getName().str() == "blockDim" ||
        VD->getName().str() == "gridDim")) {
    setFalseForThisFunctionDecl();
#ifdef __DEBUG_BARRIER_FENCE_SPACE_ANALYZER
    std::cout << "Return False case B: "
              << DRE->getBeginLoc().printToString(
                     clang::dpct::DpctGlobalInfo::getSourceManager())
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
void clang::dpct::BarrierFenceSpaceAnalyzer::PostVisit(
    const clang::DeclRefExpr *) {}

bool clang::dpct::BarrierFenceSpaceAnalyzer::Visit(const clang::GotoStmt *GS) {
  // We will further refine it if meet real request.
  // By default, goto/label stmt is not supported.
#ifdef __DEBUG_BARRIER_FENCE_SPACE_ANALYZER
  std::cout << "Return False case F: "
            << GS->getBeginLoc().printToString(
                   clang::dpct::DpctGlobalInfo::getSourceManager())
            << std::endl;
#endif
  return false;
}
void clang::dpct::BarrierFenceSpaceAnalyzer::PostVisit(
    const clang::GotoStmt *) {}

bool clang::dpct::BarrierFenceSpaceAnalyzer::Visit(const clang::LabelStmt *LS) {
  // We will further refine it if meet real request.
  // By default, goto/label stmt is not supported.
#ifdef __DEBUG_BARRIER_FENCE_SPACE_ANALYZER
  std::cout << "Return False case G: "
            << LS->getBeginLoc().printToString(
                   clang::dpct::DpctGlobalInfo::getSourceManager())
            << std::endl;
#endif
  return false;
}
void clang::dpct::BarrierFenceSpaceAnalyzer::PostVisit(
    const clang::LabelStmt *) {}

bool clang::dpct::BarrierFenceSpaceAnalyzer::Visit(
    const clang::MemberExpr *ME) {
  if (ME->getType()->isPointerType() || ME->getType()->isArrayType()) {
#ifdef __DEBUG_BARRIER_FENCE_SPACE_ANALYZER
    std::cout << "Return False case H: "
              << ME->getBeginLoc().printToString(
                     clang::dpct::DpctGlobalInfo::getSourceManager())
              << std::endl;
#endif
    return false;
  }
  return true;
}
void clang::dpct::BarrierFenceSpaceAnalyzer::PostVisit(
    const clang::MemberExpr *) {}
bool clang::dpct::BarrierFenceSpaceAnalyzer::Visit(
    const clang::CXXDependentScopeMemberExpr *CDSME) {
#ifdef __DEBUG_BARRIER_FENCE_SPACE_ANALYZER
  std::cout << "Return False case I: "
            << CDSME->getBeginLoc().printToString(
                   clang::dpct::DpctGlobalInfo::getSourceManager())
            << std::endl;
#endif
  return false;
}
void clang::dpct::BarrierFenceSpaceAnalyzer::PostVisit(
    const clang::CXXDependentScopeMemberExpr *) {}
bool clang::dpct::BarrierFenceSpaceAnalyzer::Visit(
    const clang::CXXConstructExpr *CCE) {
  auto Ctor = CCE->getConstructor();
  std::string CtorName = Ctor->getParent()->getQualifiedNameAsString();
  if (AllowedDeviceFunctions.count(CtorName) && !isUserDefinedDecl(Ctor))
    return true;
#ifdef __DEBUG_BARRIER_FENCE_SPACE_ANALYZER
  std::cout << "Return False case J: "
            << CCE->getBeginLoc().printToString(
                   clang::dpct::DpctGlobalInfo::getSourceManager())
            << std::endl;
#endif
  return false;
}
void clang::dpct::BarrierFenceSpaceAnalyzer::PostVisit(
    const clang::CXXConstructExpr *) {}

bool clang::dpct::BarrierFenceSpaceAnalyzer::traverseFunction(
    const clang::FunctionDecl *FD) {
  if (!this->TraverseDecl(const_cast<clang::FunctionDecl *>(FD))) {
    return false;
  }
  return true;
}

std::set<const clang::DeclRefExpr *>
clang::dpct::BarrierFenceSpaceAnalyzer::matchAllDRE(
    const clang::VarDecl *TargetDecl, const clang::Stmt *Range) {
  std::set<const clang::DeclRefExpr *> Set;
  if (!TargetDecl || !Range) {
    return Set;
  }
  auto DREMatcher = ast_matchers::findAll(
      ast_matchers::declRefExpr(ast_matchers::isDeclSameAs(TargetDecl))
          .bind("DRE"));
  auto MatchedResults =
      ast_matchers::match(DREMatcher, *Range, DpctGlobalInfo::getContext());
  for (auto Node : MatchedResults) {
    if (auto DRE = Node.getNodeAs<clang::DeclRefExpr>("DRE"))
      Set.insert(DRE);
  }
  return Set;
}

std::set<const clang::DeclRefExpr *>
clang::dpct::BarrierFenceSpaceAnalyzer::isAssignedToAnotherDRE(
    const clang::DeclRefExpr *CurrentDRE) {
  std::set<const clang::DeclRefExpr *> ResultSet;
  auto &Context = clang::dpct::DpctGlobalInfo::getContext();
  clang::DynTypedNode Current = clang::DynTypedNode::create(*CurrentDRE);
  clang::DynTypedNodeList Parents = Context.getParents(Current);
  while (!Parents.empty()) {
    if (Parents[0].get<FunctionDecl>() && Parents[0].get<FunctionDecl>() == FD)
      break;
    const auto BO = Parents[0].get<BinaryOperator>();
    if (BO && BO->isAssignmentOp() &&
        (BO->getRHS() == Current.get<clang::Expr>())) {
      auto DREMatcher =
          ast_matchers::findAll(ast_matchers::declRefExpr().bind("DRE"));
      auto MatchedResults = ast_matchers::match(DREMatcher, *(BO->getRHS()),
                                                DpctGlobalInfo::getContext());
      for (auto Node : MatchedResults) {
        if (auto DRE = Node.getNodeAs<clang::DeclRefExpr>("DRE"))
          ResultSet.insert(DRE);
      }
    }
    Current = Parents[0];
    Parents = Context.getParents(Current);
  }
  return ResultSet;
}

clang::dpct::BarrierFenceSpaceAnalyzer::AccessMode
clang::dpct::BarrierFenceSpaceAnalyzer::getAccessKind(
    const clang::DeclRefExpr *CurrentDRE) {
  using namespace ast_matchers;
  auto Matcher = findAll(
      declRefExpr(
          anyOf(hasAncestor(unaryOperator(
                    hasOperatorName("*"),
                    hasAncestor(binaryOperator(isAssignmentOperator(),
                                               hasLHS(hasDescendant(declRefExpr(
                                                   isSameAs(CurrentDRE)))))
                                    .bind("BO1")))),
                hasAncestor(arraySubscriptExpr(
                    hasAncestor(binaryOperator(isAssignmentOperator(),
                                               hasLHS(hasDescendant(declRefExpr(
                                                   isSameAs(CurrentDRE)))))
                                    .bind("BO2"))))),
          declRefExpr(isSameAs(CurrentDRE)))
          .bind("DRE"));
  auto MatchedResults =
      ast_matchers::match(Matcher, *FD, DpctGlobalInfo::getContext());
  if (MatchedResults.empty())
    return clang::dpct::BarrierFenceSpaceAnalyzer::AccessMode::Read;
  auto Node = MatchedResults.begin();
  const clang::BinaryOperator *BO = nullptr;
  BO = Node->getNodeAs<clang::BinaryOperator>("BO1");
  if (!BO) {
    BO = Node->getNodeAs<clang::BinaryOperator>("BO2");
  }
  if (BO->getOpcode() == clang::BinaryOperatorKind::BO_Assign) {
    return clang::dpct::BarrierFenceSpaceAnalyzer::AccessMode::Write;
  }
  return clang::dpct::BarrierFenceSpaceAnalyzer::AccessMode::ReadWrite;
}

bool clang::dpct::BarrierFenceSpaceAnalyzer::canSetLocalFenceSpace(
    const clang::CallExpr *CE) {
#ifdef __DEBUG_BARRIER_FENCE_SPACE_ANALYZER
  std::cout << "BarrierFenceSpaceAnalyzer Analyzing ..." << std::endl;
#endif
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

#ifdef __DEBUG_BARRIER_FENCE_SPACE_ANALYZER
  std::cout << "Before traverse" << std::endl;
#endif

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

  auto getSize =
      [](const std::unordered_map<const clang::ParmVarDecl *,
                                  std::set<const clang::DeclRefExpr *>>
             &DefUseMap) -> std::size_t {
    std::size_t Size = 0;
    for (const auto &Pair : DefUseMap) {
      Size = Size + Pair.second.size();
    }
    return Size;
  };

#ifdef __DEBUG_BARRIER_FENCE_SPACE_ANALYZER
  std::cout << "DefUseMap init value:" << std::endl;
  for (const auto &Pair : DefUseMap) {
    const auto &SM = clang::dpct::DpctGlobalInfo::getSourceManager();
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
    std::set<const clang::DeclRefExpr *> NewDRESet;
    for (auto &Pair : DefUseMap) {
      const clang::ParmVarDecl *CurDecl = Pair.first;
      std::set<const clang::DeclRefExpr *> CurDRESet = Pair.second;
      std::set<const clang::DeclRefExpr *> MatchedResult =
          matchAllDRE(CurDecl, FD->getBody());
      CurDRESet.insert(MatchedResult.begin(), MatchedResult.end());
      NewDRESet.clear();
      for (const auto &DRE : CurDRESet) {
        std::set<const clang::DeclRefExpr *> AssignedDREs =
            isAssignedToAnotherDRE(DRE);
        for (const auto AnotherDRE : AssignedDREs) {
          std::set<const clang::DeclRefExpr *> AnotherDREMatchedResult =
              matchAllDRE(
                  dyn_cast_or_null<clang::VarDecl>(AnotherDRE->getDecl()),
                  FD->getBody());
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
    const auto &SM = clang::dpct::DpctGlobalInfo::getSourceManager();
    std::cout << "Decl:" << Pair.first->getBeginLoc().printToString(SM)
              << std::endl;
    for (const auto &Item : Pair.second) {
      std::cout << "    DRE:" << Item->getBeginLoc().printToString(SM)
                << std::endl;
    }
  }
#endif

  // Convert DRE to <Location, AccessMode> pair for comparing
  std::map<
      const clang::ParmVarDecl *,
      std::set<std::pair<clang::SourceLocation,
                         clang::dpct::BarrierFenceSpaceAnalyzer::AccessMode>>>
      DRELocs;
  for (const auto &Pair : DefUseMap) {
    for (const auto &Item : Pair.second) {
      DRELocs[Pair.first].insert(
          std::make_pair(Item->getBeginLoc(), getAccessKind(Item)));
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
    for (auto &DeclLocPair : DRELocs) {
      if (DeclLocPair.second.size() == 1) {
        continue;
      }
      std::optional<bool> DREInPredecessors;
      for (auto &LocModePair : DeclLocPair.second) {
        if (containsMacro(LocModePair.first, SyncCall.second.Predecessors) ||
            containsMacro(LocModePair.first, SyncCall.second.Successors)) {
          Result = false;
          break;
        }
        if (isInRanges(LocModePair.first, SyncCall.second.Predecessors)) {
          if (DREInPredecessors.has_value()) {
            if (DREInPredecessors.value() != true) {
              Result = false;
              break;
            }
          } else {
            DREInPredecessors = true;
          }
        }
        if (isInRanges(LocModePair.first, SyncCall.second.Successors)) {
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
        "uint4",
        "sqrtf",
        "__expf",
        "fmaf"};
