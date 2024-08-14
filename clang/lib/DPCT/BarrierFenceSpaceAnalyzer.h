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

template <> struct std::hash<clang::SourceRange> {
  std::size_t operator()(const clang::SourceRange &SR) const noexcept {
    return llvm::hash_combine(SR.getBegin().getRawEncoding(),
                              SR.getEnd().getRawEncoding());
  }
};

namespace clang {
namespace dpct {
enum AccessMode : std::uint32_t {
  NotSet = 0,
  Read = 1 << 0,
  Write = 1 << 1,
  ReadWrite = Read | Write,
};

struct AffectedInfo {
  AffectedInfo() {}
  AffectedInfo(bool UsedBefore, bool UsedAfter, AccessMode AM)
      : UsedBefore(UsedBefore), UsedAfter(UsedAfter), AM(AM) {}
  bool UsedBefore = false;
  bool UsedAfter = false;
  AccessMode AM = NotSet;
};

struct IntraproceduralAnalyzerResult {
  using MapT =
      std::unordered_map<
          std::string /*call's combined loc string*/,
          std::tuple<
              bool /*is real sync call*/, bool /*is in loop*/,
              tooling::UnifiedPath, unsigned int,
              std::
                  unordered_map<unsigned int /*parameter idx*/, AffectedInfo /*{bool UsedBefore, bool UsedAfter, AccessMode AM}*/>,
              std::unordered_map<
                  unsigned int /*arg idx*/,
                  std::set<unsigned int> /*caller parameter(s) idx*/>,
              std::unordered_map<std::string /*global var combined loc*/,
                                 AffectedInfo>>>;
  IntraproceduralAnalyzerResult() {}
  IntraproceduralAnalyzerResult(MapT Map, std::string CurrentCtxFuncCombinedLoc,
                                unsigned int POENum)
      : Map(Map), CurrentCtxFuncCombinedLoc(CurrentCtxFuncCombinedLoc),
        POENum(POENum) {}
  MapT Map;
  std::string CurrentCtxFuncCombinedLoc;
  unsigned int POENum = 0;
};

using Ranges = std::unordered_set<SourceRange>;
struct SyncCallInfo {
  SyncCallInfo() {}
  Ranges Predecessors;
  Ranges Successors;
  bool IsRealSyncCall = false;
  bool IsInLoop = false;
};

struct DREInfo {
  DREInfo(const DeclRefExpr *DRE, SourceLocation SL, AccessMode AM)
      : DRE(DRE), SL(SL), AM(AM) {}
  const DeclRefExpr *DRE;
  SourceLocation SL;
  AccessMode AM;
  bool operator<(const DREInfo &Other) const { return DRE < Other.DRE; }
};

bool isInRanges(SourceLocation SL, Ranges Ranges);

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
  bool canBeAnalyzed(const clang::Type *TypePtr);
};

namespace detail {
class AnalyzerBase {
protected:
  std::pair<std::set<const DeclRefExpr *>, std::set<const VarDecl *>>
  isAssignedToAnotherDREOrVD(const DeclRefExpr *);
  void constructDefUseMap();
  const FunctionDecl *FD = nullptr;
  std::unordered_map<const VarDecl *, std::set<const DeclRefExpr *>> DefUseMap;
};
} // namespace detail

class IntraproceduralAnalyzer
    : public RecursiveASTVisitor<IntraproceduralAnalyzer>,
      public detail::AnalyzerBase {
public:
  bool shouldVisitImplicitCode() const { return true; }
  bool shouldTraversePostOrder() const { return false; }

#define VISIT_NODE(CLASS)                                                      \
  bool Visit(const CLASS *Node);                                               \
  void PostVisit(const CLASS *FS);                                             \
  bool Traverse##CLASS(CLASS *Node) {                                          \
    if (!Visit(Node))                                                          \
      return false;                                                            \
    if (!RecursiveASTVisitor<IntraproceduralAnalyzer>::Traverse##CLASS(Node))  \
      return false;                                                            \
    PostVisit(Node);                                                           \
    return true;                                                               \
  }
  VISIT_NODE(GotoStmt)
  VISIT_NODE(LabelStmt)
  VISIT_NODE(MemberExpr)
  VISIT_NODE(CXXDependentScopeMemberExpr)
  VISIT_NODE(ForStmt)
  VISIT_NODE(DoStmt)
  VISIT_NODE(WhileStmt)
  VISIT_NODE(CallExpr)
  VISIT_NODE(DeclRefExpr)
  VISIT_NODE(CXXConstructExpr)
  VISIT_NODE(PseudoObjectExpr)
#undef VISIT_NODE
  IntraproceduralAnalyzerResult analyze(const FunctionDecl *FD,
                                        DeviceFunctionInfo *DFI);

private:
  void simplifyMap(std::map<const VarDecl *, std::set<DREInfo>> &DefDREInfoMap);
  void affectedByWhichParameters(
      const std::map<const VarDecl *, std::set<DREInfo>> &DefDREInfoMap,
      const SyncCallInfo &SCI,
      std::unordered_map<unsigned int /*parameter idx*/, AffectedInfo>
          &AffectingParameters,
      std::unordered_map<std::string /*global var combined loc*/, AffectedInfo>
          &AffectingGlobalVars);
  std::unordered_map<unsigned int /*arg idx*/,
                     std::set<unsigned int> /*caller parameter(s) idx*/>
  getArgCallerParmsMap(const CallExpr *CE);
  bool isAccessingMemory(const DeclRefExpr *);
  AccessMode getAccessKindReadWrite(const DeclRefExpr *);
  void generateDRE2VDMap(const std::map<const VarDecl *, std::set<DREInfo>> &);

  std::string FDLoc;
  /// (FD location, result)
  std::vector<std::pair<const CallExpr *, SyncCallInfo>> SyncCallsVec;
  std::multimap<const DeclRefExpr *, const VarDecl *> DRE2VDMap;
  // This map contains pairs meet below pattern:
  // loop {
  //   ...
  //   DRE[idx] = ...;
  //   ...
  //   idx += step;
  //   ...
  // }
  std::deque<SourceRange> LoopRange;
  std::set<const Expr *> DeviceFunctionCallArgs;
  unsigned int POENum = 0;
};

class InterproceduralAnalyzer {
public:
  bool analyze(const std::shared_ptr<DeviceFunctionInfo> DFI,
               std::string SyncCallCombinedLoc);

private:
  std::vector<std::pair<const CallExpr *, SyncCallInfo>> SyncCallsVec;
  const FunctionDecl *FD = nullptr;
  std::string GlobalFunctionName;
  std::unordered_map<const VarDecl *, std::set<const DeclRefExpr *>> DefUseMap;
  std::string CELoc;
  std::string FDLoc;
  bool SkipCacheInAnalyzer = false;
};

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
    : public RecursiveASTVisitor<BarrierFenceSpace1DAnalyzer>,
      public detail::AnalyzerBase {
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
      const std::map<const VarDecl *, std::set<DREInfo>> &DefDREInfoMap);
  bool containsMacro(const SourceLocation &SL);
  bool hasOverlappingAccessAmongWorkItems(int KernelDim,
                                          const DeclRefExpr *DRE);
  std::vector<const CallExpr *> SyncCallsVec;
  int KernelDim = 3;          // 3 or 1
  int KernelCallBlockDim = 3; // 3 or 1
  std::string GlobalFunctionName;

  void simplifyMap(std::map<const VarDecl *, std::set<DREInfo>> &DefDREInfoMap);

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
};

} // namespace dpct
} // namespace clang

#endif // DPCT_BARRIER_FENCE_SPACE_ANALYZER_H
