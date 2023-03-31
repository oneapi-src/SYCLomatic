//===--------------- ReadWriteOrderAnalyzer.h --------------------------===//
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

class BarrierFenceSpaceAnalyzerBase
    : public RecursiveASTVisitor<BarrierFenceSpaceAnalyzerBase> {
protected:
  std::unordered_map<DeclRefExpr *, ValueDecl *> DREDeclMap;
  std::unordered_map<ValueDecl *, std::unordered_set<DeclRefExpr *>>
      DeclDREsMap;
  static const std::unordered_set<std::string> AllowedDeviceFunctions;

public:
  bool shouldVisitImplicitCode() const { return true; }
  bool shouldTraversePostOrder() const { return false; }

#define VISIT_NODE(CLASS)                                                      \
  bool Visit(CLASS *Node);                                                     \
  void PostVisit(CLASS *FS);                                                   \
  bool Traverse##CLASS(CLASS *Node) {                                          \
    if (!Visit(Node))                                                          \
      return false;                                                            \
    if (!RecursiveASTVisitor<BarrierFenceSpaceAnalyzerBase>::Traverse##CLASS(  \
            Node))                                                             \
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
  virtual bool analyze(const CallExpr *CE) = 0;
};

class ReadWriteOrderAnalyzer : public BarrierFenceSpaceAnalyzerBase {
  using Base = BarrierFenceSpaceAnalyzerBase;

  bool traverseFunction(const FunctionDecl *FD);
  using Ranges = std::vector<SourceRange>;
  struct SyncCallInfo {
    SyncCallInfo(Ranges Predecessors, Ranges Successors)
        : Predecessors(Predecessors), Successors(Successors){};
    Ranges Predecessors;
    Ranges Successors;
  };
  struct Level {
    SourceLocation CurrentLoc;
    SourceLocation LevelBeginLoc;
    SourceLocation FirstSyncBeginLoc;
    std::vector<std::pair<CallExpr *, SyncCallInfo>> SyncCallsVec;
  };

  Level CurrentLevel;
  std::stack<Level> LevelStack;
  std::multimap<unsigned int, Level> LevelMap;
  std::vector<Level> LevelVec;
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

public:
  bool Visit(ForStmt *Node);
  void PostVisit(ForStmt *Node);
  bool Visit(CallExpr *Node);
  bool Visit(GotoStmt *Node);
  bool Visit(LabelStmt *Node);
  bool Visit(MemberExpr *Node);
  bool Visit(CXXDependentScopeMemberExpr *Node);
  bool Visit(CXXConstructExpr *Node);

  bool analyze(const CallExpr *CE) override;
};

class NewAnalyzer : public BarrierFenceSpaceAnalyzerBase {
  using Base = BarrierFenceSpaceAnalyzerBase;
  bool HasGlobalDeviceVariable = false;
  bool checkNewPattern(const CallExpr *CE, const FunctionDecl *FD);
  void collectAlias(VarDecl *VD,
                    std::unordered_set<VarDecl *> &NewNonconstPointerDecls);

public:
  bool Visit(DeclRefExpr *Node);
  bool analyze(const CallExpr *CE) override;
};

bool canSetLocalFenceSpace(const CallExpr *CE);

} // namespace dpct
} // namespace clang

#endif // DPCT_BARRIER_FENCE_SPACE_ANALYZER_H
