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
  Read = 1 << 0,
  Write = 1 << 1,
  ReadWrite = 1 << 2,
};

struct AffectedInfo {
  AffectedInfo() {}
  AffectedInfo(bool UsedBefore, bool UsedAfter, AccessMode AM)
      : UsedBefore(UsedBefore), UsedAfter(UsedAfter), AM(AM) {}
  bool UsedBefore = false;
  bool UsedAfter = false;
  AccessMode AM = Read;
};

struct IntraproceduralAnalyzerResult {
  IntraproceduralAnalyzerResult() : IsDefault(true) {}
  IntraproceduralAnalyzerResult(bool UnsupportedCase)
      : UnsupportedCase(UnsupportedCase) {}
  IntraproceduralAnalyzerResult(
      std::unordered_map<
          std::string /*call's combined loc string*/,
          std::tuple<
              bool /*is real sync call*/, bool /*is in loop*/,
              tooling::UnifiedPath, unsigned int,
              std::unordered_map<unsigned int /*parameter idx*/, AffectedInfo>,
              std::unordered_map<
                  unsigned int /*arg idx*/,
                  std::set<unsigned int> /*caller parameter(s) idx*/>>>
          Map,
      std::string CurrentCtxFuncCombinedLoc)
      : Map(Map), CurrentCtxFuncCombinedLoc(CurrentCtxFuncCombinedLoc),
        UnsupportedCase(false) {}
  bool isDefault() const noexcept { return IsDefault; }
  std::
      unordered_map<
          std::string /*call's combined loc string*/,
          std::tuple<
              bool /*is real sync call*/, bool /*is in loop*/,
              tooling::UnifiedPath, unsigned int,
              std::
                  unordered_map<unsigned int /*parameter idx*/, AffectedInfo /*{bool UsedBefore, bool UsedAfter, AccessMode AM}*/>,
              std::unordered_map<
                  unsigned int /*arg idx*/,
                  std::set<unsigned int> /*caller parameter(s) idx*/>>>
          Map;
  std::string CurrentCtxFuncCombinedLoc;

private:
  bool IsDefault = false;
  bool UnsupportedCase = true;
};

using Ranges = std::unordered_set<SourceRange>;
struct SyncCallInfo {
  SyncCallInfo() {}
  SyncCallInfo(Ranges Predecessors, Ranges Successors, bool IsRealSyncCall,
               bool IsInLoop)
      : Predecessors(Predecessors), Successors(Successors),
        IsRealSyncCall(IsRealSyncCall), IsInLoop(IsInLoop){};
  Ranges Predecessors;
  Ranges Successors;
  bool IsRealSyncCall;
  bool IsInLoop;
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

#define VISIT_NODE(CLASS)                                                      \
  bool Visit(const CLASS *Node);                                               \
  void PostVisit(const CLASS *FS);                                             \
  bool Traverse##CLASS(CLASS *Node) {                                          \
    if (!Visit(Node))                                                          \
      return false;                                                            \
    if (!RecursiveASTVisitor<IntraproceduralAnalyzer>::Traverse##CLASS(        \
            Node))                                                             \
      return false;                                                            \
    PostVisit(Node);                                                           \
    return true;                                                               \
  }

class IntraproceduralAnalyzer
    : public RecursiveASTVisitor<IntraproceduralAnalyzer> {
public:
  bool shouldVisitImplicitCode() const { return true; }
  bool shouldTraversePostOrder() const { return false; }

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
#undef VISIT_NODE
  IntraproceduralAnalyzerResult analyze(const FunctionDecl *FD,
                                        DeviceFunctionInfo *DFI);

private:
  void constructDefUseMap();
  void simplifyMap(
      std::map<const ParmVarDecl *, std::set<DREInfo>> &DefDREInfoMap);
  std::unordered_map<unsigned int /*parameter idx*/, AffectedInfo>
  affectedByWhichParameters(
      const std::map<const ParmVarDecl *, std::set<DREInfo>> &DefDREInfoMap,
      const SyncCallInfo &SCI);
  std::unordered_map<unsigned int /*arg idx*/,
                     std::set<unsigned int> /*caller parameter(s) idx*/>
  getArgCallerParmsMap(const CallExpr* CE);
  std::pair<std::set<const DeclRefExpr *>, std::set<const VarDecl *>>
  isAssignedToAnotherDREOrVD(const DeclRefExpr *);
  bool isAccessingMemory(const DeclRefExpr *);
  AccessMode getAccessKindReadWrite(const DeclRefExpr *);
  void generateDRE2PVDMap(const std::map<const ParmVarDecl *, std::set<DREInfo>>&);

  const FunctionDecl *FD = nullptr;
  std::string FDLoc;
  /// (FD location, result)
  std::vector<std::pair<const CallExpr *, SyncCallInfo>> SyncCallsVec;
  std::unordered_map<const ParmVarDecl *, std::set<const DeclRefExpr *>>
      DefUseMap;
  std::multimap<const DeclRefExpr *, const ParmVarDecl *> DRE2PVDMap;
  // This map contains pairs meet below pattern:
  // loop {
  //   ...
  //   DRE[idx] = ...;
  //   ...
  //   idx += step;
  //   ...
  // }
  std::map<const DeclRefExpr *, std::string> DREIncStepMap;
  std::deque<SourceRange> LoopRange;
  std::set<const Expr *> DeviceFunctionCallArgs;
};

class InterproceduralAnalyzer {
public:
  bool analyze(const std::shared_ptr<DeviceFunctionInfo> DFI,
               std::string SyncCallCombinedLoc);

private:
  std::tuple<bool /*CanUseLocalBarrier*/,
             bool /*CanUseLocalBarrierWithCondition*/,
             std::string /*Condition*/>
  isSafeToUseLocalBarrier(
      const std::map<const ParmVarDecl *, std::set<DREInfo>> &DefDREInfoMap,
      const SyncCallInfo &SCI);

  std::vector<std::pair<const CallExpr *, SyncCallInfo>> SyncCallsVec;
  const FunctionDecl *FD = nullptr;
  std::string GlobalFunctionName;
  std::unordered_map<const ParmVarDecl *, std::set<const DeclRefExpr *>>
      DefUseMap;
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

#endif // DPCT_BARRIER_FENCE_SPACE_ANALYZER_H
