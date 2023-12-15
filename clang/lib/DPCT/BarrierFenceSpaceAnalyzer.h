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

template <> struct std::hash<clang::SourceRange> {
  std::size_t operator()(const clang::SourceRange &SR) const noexcept {
    return llvm::hash_combine(SR.getBegin().getRawEncoding(),
                              SR.getEnd().getRawEncoding());
  }
};
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
        GlobalFunctionName(GlobalFunctionName), Condition(Condition) {
    IsDefaultValue = false;
  }
  void merge(const BarrierFenceSpaceAnalyzerResult &Another);

  bool CanUseLocalBarrier = false;
  bool CanUseLocalBarrierWithCondition = false;
  bool MayDependOn1DKernel = false;
  std::string GlobalFunctionName;
  std::string Condition;
private:
  bool IsDefaultValue = true;
};

class BarrierFenceSpaceAnalyzerInterface {
  std::set<const FunctionDecl *> TopLevelGlobalFunctions;

public:
  BarrierFenceSpaceAnalyzerResult analyze(const CallExpr *CE,
                                          bool SkipCacheInAnalyzer = false);
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

  VISIT_NODE(SwitchStmt)
  VISIT_NODE(IfStmt)
  VISIT_NODE(CallExpr)
  VISIT_NODE(DeclRefExpr)
  VISIT_NODE(GotoStmt)
  VISIT_NODE(LabelStmt)
  VISIT_NODE(MemberExpr)
  VISIT_NODE(CXXDependentScopeMemberExpr)
  VISIT_NODE(CXXConstructExpr)
  VISIT_NODE(FunctionTemplateDecl)
#undef VISIT_NODE

public:
  BarrierFenceSpaceAnalyzerResult
  analyze_internal(const CallExpr *CE, const FunctionDecl *FD,
                   bool SkipCacheInAnalyzer = false);

private:
  enum class AccessMode : int { Read = 0, Write, ReadWrite };
  std::set<const DeclRefExpr *> matchAllDRE(const VarDecl *TargetDecl,
                                            const Stmt *Range);
  std::pair<std::set<const DeclRefExpr *>, std::set<const VarDecl *>>
  isAssignedToAnotherDREOrVD(const DeclRefExpr *);
  std::pair<const ParmVarDecl *, const FunctionDecl *>
  isPassedToAnotherFunction(const DeclRefExpr *);
  bool isAccessingMemory(const DeclRefExpr *);
  AccessMode getAccessKind(const DeclRefExpr *);
  using Ranges = std::unordered_set<SourceRange>;
  struct SyncCallInfo {
    SyncCallInfo() {}
    SyncCallInfo(Ranges Predecessors, Ranges Successors)
        : Predecessors(Predecessors), Successors(Successors){};
    Ranges Predecessors;
    Ranges Successors;
  };

  struct DREInfo {
    DREInfo(const DeclRefExpr *DRE, SourceLocation SL, AccessMode AM)
        : DRE(DRE), SL(SL), AM(AM) {}
    const DeclRefExpr *DRE;
    SourceLocation SL;
    AccessMode AM;
    bool operator<(const DREInfo &Other) const { return DRE < Other.DRE; }
  };

  std::tuple<bool /*CanUseLocalBarrier*/,
             bool /*CanUseLocalBarrierWithCondition*/,
             std::string /*Condition*/>
  isSafeToUseLocalBarrier(
      const std::map<const ParmVarDecl *, std::set<DREInfo>> &DefDREInfoMap,
      const SyncCallInfo &SCI);
  bool containsMacro(const SourceLocation &SL, const SyncCallInfo &SCI);
  bool hasOverlappingAccessAmongWorkItems(int KernelDim,
                                          const DeclRefExpr *DRE);
  std::map<const CallExpr *, SyncCallInfo> SyncCallsMap;
  int KernelDim = 3; // 3 or 1
  int KernelCallBlockDim = 3; // 3 or 1
  const FunctionDecl *FD = nullptr;
  std::string GlobalFunctionName;

  std::unordered_map<const ParmVarDecl *, std::set<const DeclRefExpr *>>
      DefUseMap;
  std::string CELoc;
  std::string FDLoc;
  void constructDefUseMap();
  void
  simplifyMap(std::map<const ParmVarDecl *, std::set<DREInfo>> &DefDREInfoMap);

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
  bool isInRanges(SourceLocation SL, Ranges Ranges);
  std::string isSafeWriteInLoop(const std::set<const DeclRefExpr *> &WILDRESet);

  std::set<const Expr *> DeviceFunctionCallArgs;

  bool IsDifferenceBetweenThreadIdxXAndIndexConstant = false;

  // This map contains pairs meet below pattern:
  // loop {
  //   ...
  //   DRE[idx] = ...;
  //   ...
  //   idx += step;
  //   ...
  // }
  std::map<const DeclRefExpr*, std::string> DREIncStepMap;
  std::unordered_set<const FunctionDecl *> TraversedSet;
  bool VisitingGlobalFunction = true;

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
