//===--------------- BarrierFenceSpaceAnalyzer.h --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DPCT_BARRIER_FENCE_SPACE_ANALYZER_H
#define DPCT_BARRIER_FENCE_SPACE_ANALYZER_H

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

struct BarrierFenceSpaceAnalyzerResult {
  BarrierFenceSpaceAnalyzerResult() {}
  BarrierFenceSpaceAnalyzerResult(bool CanUseLocalBarrier,
                                  bool CanUseLocalBarrierWithCondition,
                                  bool MayDependOn1DKernel,
                                  std::string GlobalFunctionName,
                                  std::string Condition = "")
      : CanUseLocalBarrier(CanUseLocalBarrier),
        CanUseLocalBarrierWithCondition(CanUseLocalBarrierWithCondition),
        MayDependOn1DKernel(MayDependOn1DKernel),
        GlobalFunctionName(GlobalFunctionName),
        Condition(Condition) {}
  bool CanUseLocalBarrier = false;
  bool CanUseLocalBarrierWithCondition = false;
  bool MayDependOn1DKernel = false;
  std::string GlobalFunctionName;
  std::string Condition;
};

class BarrierFenceSpaceAnalyzer
    : public RecursiveASTVisitor<BarrierFenceSpaceAnalyzer> {
public:
  bool shouldVisitImplicitCode() const { return true; }
  bool shouldTraversePostOrder() const { return false; }

#define VISIT_NODE(CLASS)                                                      \
  bool Visit(const CLASS *Node);                                               \
  void PostVisit(const CLASS *FS);                                             \
  bool Traverse##CLASS(CLASS *Node) {                                          \
    if (!Visit(Node))                                                          \
      return false;                                                            \
    if (!RecursiveASTVisitor<BarrierFenceSpaceAnalyzer>::Traverse##CLASS(      \
            Node))                                                             \
      return false;                                                            \
    PostVisit(Node);                                                           \
    return true;                                                               \
  }

  VISIT_NODE(ForStmt)
  VISIT_NODE(DoStmt)
  VISIT_NODE(WhileStmt)
  VISIT_NODE(SwitchStmt)
  VISIT_NODE(IfStmt)
  VISIT_NODE(CallExpr)
  VISIT_NODE(DeclRefExpr)
  VISIT_NODE(GotoStmt)
  VISIT_NODE(LabelStmt)
  VISIT_NODE(MemberExpr)
  VISIT_NODE(CXXDependentScopeMemberExpr)
  VISIT_NODE(CXXConstructExpr)
#undef VISIT_NODE

public:
  BarrierFenceSpaceAnalyzerResult analyze(const CallExpr *CE,
                                          bool SkipCacheInAnalyzer = false);

private:
  enum class AccessMode : int { Read = 0, Write, ReadWrite };
  std::set<const DeclRefExpr *> matchAllDRE(const VarDecl *TargetDecl,
                                            const Stmt *Range);
  std::pair<std::set<const DeclRefExpr *>, std::set<const VarDecl *>>
  isAssignedToAnotherDREOrVD(const DeclRefExpr *);
  bool isAccessingMemory(const DeclRefExpr *);
  AccessMode getAccessKind(const DeclRefExpr *);
  using Ranges = std::vector<SourceRange>;
  struct SyncCallInfo {
    SyncCallInfo() {}
    SyncCallInfo(Ranges Predecessors, Ranges Successors)
        : Predecessors(Predecessors), Successors(Successors){};
    Ranges Predecessors;
    Ranges Successors;
  };
  std::tuple<bool /*CanUseLocalBarrier*/,
             bool /*CanUseLocalBarrierWithCondition*/,
             std::string /*Condition*/>
  isSafeToUseLocalBarrier(
      const std::map<const ParmVarDecl *,
                     std::set<std::pair<SourceLocation, AccessMode>>>
          &DefLocInfoMap,
      const SyncCallInfo &SCI);
  bool containsMacro(const SourceLocation &SL, const SyncCallInfo &SCI);
  bool hasOverlappingAccessAmongWorkItems(int KernelDim,
                                          const DeclRefExpr *DRE);
  std::vector<std::pair<const CallExpr *, SyncCallInfo>> SyncCallsVec;
  std::deque<SourceRange> LoopRange;
  int KernelDim = 3; // 3 or 1
  int KernelCallBlockDim = 3; // 3 or 1
  const FunctionDecl *FD = nullptr;
  std::string GlobalFunctionName;

  std::unordered_map<const ParmVarDecl *, std::set<const DeclRefExpr *>>
      DefUseMap;
  std::string CELoc;
  std::string FDLoc;
  void constructDefUseMap();
  void simplifyAndConvertToDefLocInfoMap(
      std::map<const ParmVarDecl *,
               std::set<std::pair<SourceLocation, AccessMode>>> &DefLocInfoMap);

  /// (FD location, (Call location, result))
  static std::unordered_map<
      std::string,
      std::unordered_map<std::string, BarrierFenceSpaceAnalyzerResult>>
      CachedResults;
  bool SkipCacheInAnalyzer = false;
  bool MayDependOn1DKernel = false;

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
  bool isInRanges(SourceLocation SL, std::vector<SourceRange> Ranges);

  std::set<const Expr *> DeviceFunctionCallArgs;

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

#endif // DPCT_BARRIER_FENCE_SPACE_ANALYZER_H
