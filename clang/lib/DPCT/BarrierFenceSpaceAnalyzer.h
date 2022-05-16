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
  bool Visit(CLASS *Node);                                                     \
  void PostVisit(CLASS *FS);                                                   \
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
  bool traverseFunction(const clang::FunctionDecl *FD);
  using Ranges = std::vector<clang::SourceRange>;
  struct SyncCallInfo {
    SyncCallInfo(Ranges Predecessors, Ranges Successors)
        : Predecessors(Predecessors), Successors(Successors){};
    Ranges Predecessors;
    Ranges Successors;
  };
  struct Level {
    clang::SourceLocation CurrentLoc;
    clang::SourceLocation LevelBeginLoc;
    clang::SourceLocation FirstSyncBeginLoc;
    std::vector<std::pair<clang::CallExpr *, SyncCallInfo>> SyncCallsVec;
  };

  Level CurrentLevel;
  std::stack<Level> LevelStack;
  std::multimap<unsigned int, Level> LevelMap;
  std::vector<Level> LevelVec;
  std::unordered_map<clang::DeclRefExpr *, clang::ValueDecl *> DREDeclMap;
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
