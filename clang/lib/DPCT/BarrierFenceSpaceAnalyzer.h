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
  bool canSetLocalFenceSpace(const CallExpr *CE, bool IsDryRun = false);

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
  bool isValidAccessPattern(
      const std::map<const ParmVarDecl *,
                     std::set<std::pair<SourceLocation, AccessMode>>>
          &DeclUsedLocsMap,
      const SyncCallInfo &SCI);
  bool containsMacro(const SourceLocation &SL, const SyncCallInfo &SCI);
  bool isNoOverlappingAccessAmongWorkItems(int KernelDim,
                                           const DeclRefExpr *DRE);
  std::vector<std::pair<const CallExpr *, SyncCallInfo>> SyncCallsVec;
  std::deque<SourceRange> LoopRange;
  int KernelDim = 3; // 3 or 1
  const FunctionDecl *FD = nullptr;

  std::unordered_map<const ParmVarDecl *, std::set<const DeclRefExpr *>>
      DefUseMap;
  std::string CELoc;
  std::string FDLoc;

  // If meets exit condition when visit AST nodes, all __syncthreads() in this
  // kernel function cannot set local fence space.
  // FDLoc is in the map means this kernel function is analyzed.
  // CELoc is not in the map means cannot set local fence space.
  void setFalseForThisFunctionDecl() {
    if (!IsDryRun)
      CachedResults[FDLoc] = std::unordered_map<std::string, bool>();
  }

  /// (FD location, (Call location, result))
  static std::unordered_map<std::string, std::unordered_map<std::string, bool>>
      CachedResults;
  bool IsDryRun = false;

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

  bool isPotentialGlobalMemoryAccess(std::shared_ptr<DeviceFunctionInfo> DFI,
                                     bool IsInGlobalFunction);
  std::set<const Expr *> DeviceFunctionCallArgs;

  class TypeAnalyzer {
  public:
    /// Decide the input parameter type class
    /// \return One of below 3 int values:
    ///  1: can skip analysis
    ///  0: need analysis
    /// -1: unsupport to analyze
    int getInputParamterTypeKind(clang::QualType QT) {
      bool Res = getTypeInfo(QT.getTypePtr());
      if (!Res)
        return -1;
      if (PointerLevel) {
        if (IsConstPtr)
          return 1;
        return 0;
      }
      return 1;
    }

  private:
    int PointerLevel = 0;
    bool IsConstPtr = false;
    bool IsClass = false;
    bool getTypeInfo(const clang::Type *TypePtr) {
      switch (TypePtr->getTypeClass()) {
      case clang::Type::TypeClass::ConstantArray:
        return getTypeInfo(dyn_cast<clang::ConstantArrayType>(TypePtr)
                               ->getElementType()
                               .getTypePtr());
      case clang::Type::TypeClass::Pointer:
        PointerLevel++;
        if (PointerLevel >= 2 || IsClass)
          return false;
        IsConstPtr = TypePtr->getPointeeType().isConstQualified();
        return getTypeInfo(TypePtr->getPointeeType().getTypePtr());
      case clang::Type::TypeClass::Elaborated:
        return getTypeInfo(
            dyn_cast<clang::ElaboratedType>(TypePtr)->desugar().getTypePtr());
      case clang::Type::TypeClass::Typedef:
        return getTypeInfo(dyn_cast<clang::TypedefType>(TypePtr)
                               ->getDecl()
                               ->getUnderlyingType()
                               .getTypePtr());
      case clang::Type::TypeClass::Record:
        IsClass = true;
        for (const auto &Field :
             dyn_cast<clang::RecordType>(TypePtr)->getDecl()->fields()) {
          if (!getTypeInfo(Field->getType().getTypePtr())) {
            return false;
          }
        }
        return true;
      case clang::Type::TypeClass::SubstTemplateTypeParm:
        return getTypeInfo(dyn_cast<clang::SubstTemplateTypeParmType>(TypePtr)
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

  // TODO: Implement more accuracy Predecessors and Successors. Then below code
  //       can be used for checking.
#if 0
  bool isInRanges(SourceLocation SL, std::vector<SourceRange> Ranges);
  bool isValidLocationSet(
      const std::set<std::pair<SourceLocation, AccessMode>> &LocationSet,
      const SyncCallInfo &SCI);
#endif
};
} // namespace dpct
} // namespace clang

#endif // DPCT_BARRIER_FENCE_SPACE_ANALYZER_H
