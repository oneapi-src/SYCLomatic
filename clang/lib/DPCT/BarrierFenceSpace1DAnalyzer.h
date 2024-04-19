//===--------------- BarrierFenceSpace1DAnalyzer.h ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DPCT_BARRIER_FENCE_SPACE_1D_ANALYZER_H
#define DPCT_BARRIER_FENCE_SPACE_1D_ANALYZER_H

#include "AnalysisInfo.h"
#include "Utility.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include <map>
#include <stack>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace clang {
namespace dpct {

// Depends on 1D kernel
struct BarrierFenceSpace1DAnalyzerResult {
  BarrierFenceSpace1DAnalyzerResult() {}
  BarrierFenceSpace1DAnalyzerResult(bool CanUseLocalBarrier,
                                    std::string GlobalFunctionName,
                                    bool CanUseLocalBarrierWithCondition,
                                    std::string Condition = "")
      : CanUseLocalBarrier(CanUseLocalBarrier),
        GlobalFunctionName(GlobalFunctionName),
        CanUseLocalBarrierWithCondition(CanUseLocalBarrierWithCondition),
        Condition(Condition) {}
  bool CanUseLocalBarrier = false;
  std::string GlobalFunctionName;
  bool CanUseLocalBarrierWithCondition = false;
  std::string Condition;
};

class BarrierFenceSpace1DAnalyzer
    : public RecursiveASTVisitor<BarrierFenceSpace1DAnalyzer> {
public:
  bool shouldVisitImplicitCode() const { return true; }
  bool shouldTraversePostOrder() const { return false; }

#define VISIT_NODE(CLASS)                                                      \
  bool Visit(const CLASS *Node);                                               \
  void PostVisit(const CLASS *FS);                                             \
  bool Traverse##CLASS(CLASS *Node) {                                          \
    if (!Visit(Node))                                                          \
      return false;                                                            \
    if (!RecursiveASTVisitor<BarrierFenceSpace1DAnalyzer>::Traverse##CLASS(    \
            Node))                                                             \
      return false;                                                            \
    PostVisit(Node);                                                           \
    return true;                                                               \
  }

  VISIT_NODE(SwitchStmt)
  VISIT_NODE(IfStmt)
  VISIT_NODE(CallExpr)
  VISIT_NODE(DeclRefExpr)
  VISIT_NODE(GotoStmt)
  VISIT_NODE(LabelStmt)
  VISIT_NODE(MemberExpr)
  VISIT_NODE(CXXDependentScopeMemberExpr)
#undef VISIT_NODE

public:
  BarrierFenceSpace1DAnalyzerResult analyzeFor1DKernel(const CallExpr *CE);

private:
  std::pair<std::set<const DeclRefExpr *>, std::set<const VarDecl *>>
  isAssignedToAnotherDREOrVD(const DeclRefExpr *);
  bool isAccessingMemory(const DeclRefExpr *);

  struct DREInfo {
    DREInfo(const DeclRefExpr *DRE, SourceLocation SL) : DRE(DRE), SL(SL) {}
    const DeclRefExpr *DRE;
    SourceLocation SL;
    bool operator<(const DREInfo &Other) const { return DRE < Other.DRE; }
  };

  std::tuple<bool /*CanUseLocalBarrier*/,
             bool /*CanUseLocalBarrierWithCondition*/,
             std::string /*Condition*/>
  isSafeToUseLocalBarrier(
      const std::map<const ParmVarDecl *, std::set<DREInfo>> &DefDREInfoMap);
  bool containsMacro(const SourceLocation &SL);
  bool hasOverlappingAccessAmongWorkItems(int KernelDim,
                                          const DeclRefExpr *DRE);
  std::vector<const CallExpr *> SyncCallsVec;
  int KernelDim = 3;          // 3 or 1
  int KernelCallBlockDim = 3; // 3 or 1
  const FunctionDecl *FD = nullptr;
  std::string GlobalFunctionName;

  std::unordered_map<const ParmVarDecl *, std::set<const DeclRefExpr *>>
      DefUseMap;
  void constructDefUseMap();
  void
  simplifyMap(std::map<const ParmVarDecl *, std::set<DREInfo>> &DefDREInfoMap);

  template <class TargetTy, class NodeTy>
  static inline const TargetTy *findAncestorInFunctionScope(
      const NodeTy *N, const FunctionDecl *Scope,
      const std::function<const void *(const DynTypedNode &,
                                       const DynTypedNode &)> &Operation) {
    auto &Context = DpctGlobalInfo::getContext();
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
  std::string isAnalyzableWriteInLoop(
      const std::set<const DeclRefExpr *> &WriteInLoopDRESet);

  // This map contains pairs meet below pattern:
  // loop {
  //   ...
  //   DRE[idx] = ...;
  //   ...
  //   idx += step;
  //   ...
  // }
  std::map<const DeclRefExpr *, std::string> DREIncStepMap;

  class TypeAnalyzer {
  public:
    enum class ParamterTypeKind : int {
      NeedAnalysis = 0,
      CanSkipAnalysis,
      Unsupported
    };
    ParamterTypeKind getInputParamterTypeKind(clang::QualType QT) {
      bool Res = canBeAnalyzed(QT.getTypePtr());
      if (!Res)
        return ParamterTypeKind::Unsupported;
      if (PointerLevel) {
        if (IsConstPtr)
          return ParamterTypeKind::CanSkipAnalysis;
        return ParamterTypeKind::NeedAnalysis;
      }
      return ParamterTypeKind::CanSkipAnalysis;
    }

  private:
    int PointerLevel = 0;
    bool IsConstPtr = false;
    bool IsClass = false;
    bool canBeAnalyzed(const clang::Type *TypePtr) {
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
      default:
        if (TypePtr->isFundamentalType())
          return true;
        else
          return false;
      }
    }
  };
};
} // namespace dpct
} // namespace clang

#endif // DPCT_BARRIER_FENCE_SPACE_1D_ANALYZER_H
