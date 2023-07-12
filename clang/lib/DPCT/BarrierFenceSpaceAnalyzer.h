//===--------------- BarrierFenceSpaceAnalyzer.h --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DPCT_BARRIER_FENCE_SPACE_ANALYZER_H
#define DPCT_BARRIER_FENCE_SPACE_ANALYZER_H

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
  bool canSetLocalFenceSpace(const CallExpr *CE);

private:
  enum class AccessMode : int { Read = 0, Write, ReadWrite };
  bool traverseFunction(const FunctionDecl *FD);
  std::set<const DeclRefExpr *> matchAllDRE(const VarDecl *TargetDecl,
                                            const Stmt *Range);
  std::set<const DeclRefExpr *> isAssignedToAnotherDRE(const DeclRefExpr *);
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
  const BinaryOperator *getAssignmentBinaryOP(const DeclRefExpr *DRE);
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
    CachedResults[FDLoc] = std::unordered_map<std::string, bool>();
  }

  /// (FD location, (Call location, result))
  static std::unordered_map<std::string, std::unordered_map<std::string, bool>>
      CachedResults;
  static const std::unordered_set<std::string> AllowedDeviceFunctions;

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
