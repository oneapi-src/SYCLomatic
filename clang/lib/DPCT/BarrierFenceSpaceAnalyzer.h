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
    : public clang::RecursiveASTVisitor<BarrierFenceSpaceAnalyzer> {
public:
  bool shouldVisitImplicitCode() const { return true; }
  bool shouldTraversePostOrder() const { return false; }

#define VISIT_NODE(CLASS)                                                      \
  bool Visit(const CLASS *Node);                                               \
  void PostVisit(const CLASS *FS);                                             \
  bool Traverse##CLASS(CLASS *Node) {                                          \
    if (!Visit(Node))                                                          \
      return false;                                                            \
    if (!clang::RecursiveASTVisitor<                                           \
            BarrierFenceSpaceAnalyzer>::Traverse##CLASS(Node))                 \
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
  bool canSetLocalFenceSpace(const clang::CallExpr *CE);

private:
  enum class AccessMode : int {
    Read,
    Write,
    ReadWrite
  };
  bool traverseFunction(const clang::FunctionDecl *FD);
  std::set<const clang::DeclRefExpr *>
  matchAllDRE(const clang::VarDecl *TargetDecl, const clang::Stmt *Range);
  std::set<const clang::DeclRefExpr *>
  isAssignedToAnotherDRE(const clang::DeclRefExpr *);
  AccessMode getAccessKind(const clang::DeclRefExpr *);
  using Ranges = std::vector<clang::SourceRange>;
  struct SyncCallInfo {
    SyncCallInfo() {}
    SyncCallInfo(Ranges Predecessors, Ranges Successors)
        : Predecessors(Predecessors), Successors(Successors){};
    Ranges Predecessors;
    Ranges Successors;
  };
  bool isValidAccessPattern(
      const std::map<const clang::ParmVarDecl *,
                     std::set<std::pair<clang::SourceLocation, AccessMode>>>
          &DRELocs,
      const SyncCallInfo &SCI);
  bool isInRanges(clang::SourceLocation SL,
                  std::vector<clang::SourceRange> Ranges);
  bool containsMacro(const clang::SourceLocation &SL, const SyncCallInfo &SCI);
  bool isValidLocationSet(
      const std::set<std::pair<clang::SourceLocation, AccessMode>> &LocationSet,
      const SyncCallInfo &SCI);
  std::vector<std::pair<const clang::CallExpr *, SyncCallInfo>> SyncCallsVec;
  std::deque<clang::SourceRange> LoopRange;
  const clang::FunctionDecl *FD = nullptr;

  std::unordered_map<const clang::ParmVarDecl *,
                     std::set<const clang::DeclRefExpr *>>
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
};
} // namespace dpct
} // namespace clang

#endif // DPCT_BARRIER_FENCE_SPACE_ANALYZER_H
