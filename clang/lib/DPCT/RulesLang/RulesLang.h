//===--------------------------- RulesLang.h ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DPCT_RULES_LANG_H
#define DPCT_RULES_LANG_H

#include "AnalysisInfo.h"
#include "ASTTraversal.h"
#include "RuleInfra/MapNames.h"
#include "TextModification.h"
#include "Utility.h"

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Frontend/CompilerInstance.h"

#include <algorithm>
#include <unordered_set>

namespace clang {
namespace dpct {
// utils
bool isAssignOperator(const Stmt *);

const Expr *getRhs(const Stmt *);
TextModification *ReplaceMemberAssignAsSetMethod(
    SourceLocation EndLoc, const MemberExpr *ME, StringRef MethodName,
    StringRef ReplacedArg, StringRef ExtraArg = "", StringRef ExtraFeild = "");

TextModification *ReplaceMemberAssignAsSetMethod(const Expr *E,
                                                 const MemberExpr *ME,
                                                 StringRef MethodName,
                                                 StringRef ReplacedArg = "",
                                                 StringRef ExtraArg = "",
                                                 StringRef ExtraFeild = "");

/// Migration rule for iteration space built-in variables (threadIdx, etc).
class IterationSpaceBuiltinRule
    : public NamedMigrationRule<IterationSpaceBuiltinRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);

private:
  bool renameBuiltinName(const DeclRefExpr *DRE, std::string &NewName);
};

/// Migration rule for atomic functions.
class AtomicFunctionRule : public NamedMigrationRule<AtomicFunctionRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);

private:
  void ReportUnsupportedAtomicFunc(const CallExpr *CE);
  void MigrateAtomicFunc(const CallExpr *CE,
                         const ast_matchers::MatchFinder::MatchResult &Result);
};

class ZeroLengthArrayRule
    : public NamedMigrationRule<ZeroLengthArrayRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

class MiscAPIRule : public NamedMigrationRule<MiscAPIRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

/// Migration rule for types replacements in var. declarations.
class TypeInDeclRule : public NamedMigrationRule<TypeInDeclRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);

private:
  struct TypeLocHash {
    std::size_t operator()(TypeLoc const &TL) const noexcept {
      return std::hash<unsigned>{}(TL.getBeginLoc().getRawEncoding());
    }
  };
  struct TypeLocEqual {
    bool operator()(TypeLoc const &TL1, TypeLoc const &TL2) const {
      return (TL1.getBeginLoc() == TL2.getBeginLoc()) &&
             (TL1.getEndLoc() == TL2.getEndLoc());
    }
  };
  // Holds the set of TypeLocs that have been processed.
  // Used to prevent them from being processed multiple times
  std::unordered_set<TypeLoc, TypeLocHash, TypeLocEqual> ProcessedTypeLocs;

  void processConstFFTHandleType(const DeclaratorDecl *DD,
                                 SourceLocation BeginLoc,
                                 SourceLocation EndLoc,
                                 bool HasGlobalNSPrefix);
  void processCudaStreamType(const DeclaratorDecl *DD);
  bool replaceTemplateSpecialization(SourceManager *SM, LangOptions &LOpts,
                                     SourceLocation BeginLoc,
                                     const TemplateSpecializationTypeLoc TSL);
  bool replaceDependentNameTypeLoc(SourceManager *SM, LangOptions &LOpts,
                                   const TypeLoc *TL);
  bool replaceTransformIterator(SourceManager *SM, LangOptions &LOpts,
                                const TypeLoc *TL);
};

class TemplateSpecializationTypeLocRule
    : public clang::dpct::NamedMigrationRule<
          TemplateSpecializationTypeLocRule> {

public:
  void registerMatcher(clang::ast_matchers::MatchFinder &MF) override;
  void runRule(const clang::ast_matchers::MatchFinder::MatchResult &Result);
};

class UserDefinedAPIRule
    : public clang::dpct::NamedMigrationRule<UserDefinedAPIRule> {
  std::string APIName;
  bool HasExplicitTemplateArgs;

public:
  UserDefinedAPIRule(std::string APIName, bool HasExplicitTemplateArguments)
      : APIName(std::move(APIName)),
        HasExplicitTemplateArgs(HasExplicitTemplateArguments){};
  void registerMatcher(clang::ast_matchers::MatchFinder &MF) override;
  void runRule(const clang::ast_matchers::MatchFinder::MatchResult &Result);
};

class UserDefinedTypeRule
    : public clang::dpct::NamedMigrationRule<UserDefinedTypeRule> {
  std::string TypeName;

public:
  UserDefinedTypeRule(std::string TypeName) : TypeName(TypeName){};
  void registerMatcher(clang::ast_matchers::MatchFinder &MF) override;
  void runRule(const clang::ast_matchers::MatchFinder::MatchResult &Result);
};

class UserDefinedClassFieldRule
    : public clang::dpct::NamedMigrationRule<UserDefinedClassFieldRule> {
  std::string BaseName;
  std::string FieldName;

public:
  UserDefinedClassFieldRule(std::string BaseName, std::string FieldName)
      : BaseName(BaseName), FieldName(FieldName){};
  void registerMatcher(clang::ast_matchers::MatchFinder &MF) override;
  void runRule(const clang::ast_matchers::MatchFinder::MatchResult &Result);
};

class UserDefinedClassMethodRule
    : public clang::dpct::NamedMigrationRule<UserDefinedClassMethodRule> {
  std::string BaseName;
  std::string MethodName;

public:
  UserDefinedClassMethodRule(std::string BaseName, std::string MethodName)
      : BaseName(BaseName), MethodName(MethodName){};
  void registerMatcher(clang::ast_matchers::MatchFinder &MF) override;
  void runRule(const clang::ast_matchers::MatchFinder::MatchResult &Result);
};

class UserDefinedEnumRule
    : public clang::dpct::NamedMigrationRule<UserDefinedEnumRule> {
  std::string EnumName;

public:
  UserDefinedEnumRule(std::string EnumName) : EnumName(EnumName){};
  void registerMatcher(clang::ast_matchers::MatchFinder &MF) override;
  void runRule(const clang::ast_matchers::MatchFinder::MatchResult &Result);
};


/// Migration rule for inserting namespace for vector types
class VectorTypeNamespaceRule
    : public NamedMigrationRule<VectorTypeNamespaceRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

/// Migration rule for vector type member access
class VectorTypeMemberAccessRule
    : public NamedMigrationRule<VectorTypeMemberAccessRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);

public:
  void renameMemberField(const MemberExpr *ME);
  static const std::map<std::string, std::string> MemberNamesMap;
};

/// Migration rule for vector type operator
class VectorTypeOperatorRule
    : public NamedMigrationRule<VectorTypeOperatorRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);

private:
  void MigrateOverloadedOperatorDecl(
      const ast_matchers::MatchFinder::MatchResult &Result,
      const FunctionDecl *FD);
  void MigrateOverloadedOperatorCall(
      const ast_matchers::MatchFinder::MatchResult &Result,
      const CXXOperatorCallExpr *CE, bool InOverloadedOperator);

private:
  static const char NamespaceName[];
};

class CudaExtentRule : public NamedMigrationRule<CudaExtentRule> {
  CharSourceRange getConstructorRange(const CXXConstructExpr *Ctor);
  void replaceConstructor(const CXXConstructExpr *Ctor);
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
  std::optional<TraversalKind> getCheckTraversalKind() const override {
    return TK_IgnoreUnlessSpelledInSource;
  }
};

class CudaUuidRule : public NamedMigrationRule<CudaUuidRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

/// Migration rule for return types replacements.
class ReturnTypeRule : public NamedMigrationRule<ReturnTypeRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

/// Migration rule for removing of error handling if-stmt
class ErrorHandlingIfStmtRule
    : public NamedMigrationRule<ErrorHandlingIfStmtRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

/// Migration rule for adding try-catch for host APIs calls
class ErrorHandlingHostAPIRule
    : public NamedMigrationRule<ErrorHandlingHostAPIRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
  void insertTryCatch(const FunctionDecl *FD);
};

/// Migration rule for CUDA device property and attribute.
/// E.g. cudaDeviceProp, cudaPointerAttributes.
class DeviceInfoVarRule : public NamedMigrationRule<DeviceInfoVarRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);

public:
  static const std::map<std::string, std::string> PropNamesMap;
};

/// Migration rule for enums constants.
class EnumConstantRule : public NamedMigrationRule<EnumConstantRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
  void handleComputeMode(std::string EnumName, const DeclRefExpr *E);
};

/// Migration rule for Error enums constants.
class ErrorConstantsRule : public NamedMigrationRule<ErrorConstantsRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

class LinkageSpecDeclRule : public NamedMigrationRule<LinkageSpecDeclRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};


/// Migration rule for CU_JIT enums.
class CU_JITEnumsRule : public NamedMigrationRule<CU_JITEnumsRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

/// Migration rule for general function calls.
class FunctionCallRule : public NamedMigrationRule<FunctionCallRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
  std::string findValueofAttrVar(const Expr *AttrArg, const CallExpr *CE);
};

class EventAPICallRule;
class EventQueryTraversal {
  EventAPICallRule *Rule;
  ASTContext &Context;

  bool QueryCallUsed = false;

  using ResultTy = std::vector<std::pair<const Stmt *, TextModification *>>;

  const VarDecl *getAssignTarget(const CallExpr *);

  bool checkVarDecl(const VarDecl *, const FunctionDecl *);
  bool isEventQuery(const CallExpr *);
  std::string getReplacedEnumValue(const DeclRefExpr *);

  TextModification *buildCallReplacement(const CallExpr *);

  bool traverseFunction(const FunctionDecl *, const VarDecl *);
  bool traverseStmt(const Stmt *, const VarDecl *, ResultTy &);
  bool traverseAssignRhs(const Expr *, ResultTy &);
  bool traverseEqualStmt(const Stmt *, const VarDecl *, ResultTy &);

  void handleDirectEqualStmt(const DeclRefExpr *, const CallExpr *);

  bool startFromStmt(const Stmt *, const std::function<const VarDecl *()> &);

public:
  EventQueryTraversal(EventAPICallRule *R)
      : Rule(R), Context(DpctGlobalInfo::getContext()) {}
  bool startFromQuery(const CallExpr *);
  bool startFromEnumRef(const DeclRefExpr *);
  bool startFromTypeLoc(TypeLoc TL);
};
/// Migration rule for event API calls
class EventAPICallRule : public NamedMigrationRule<EventAPICallRule> {
public:
  EventAPICallRule() { CurrentRule = this; }
  ~EventAPICallRule() {
    if (CurrentRule == this)
      CurrentRule = nullptr;
  }
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
  void handleEventRecord(const CallExpr *CE,
                         const ast_matchers::MatchFinder::MatchResult &Result,
                         bool IsAssigned);
  void handleEventElapsedTime(bool IsAssigned);
  void handleTimeMeasurement();
  void handleTargetCalls(const Stmt *Parent, const Stmt *Last = nullptr);
  void handleKernelCalls(const Stmt *Parent, const CUDAKernelCallExpr *KCall);
  void handleOrdinaryCalls(const CallExpr *Call);
  bool IsEventArgArraySubscriptExpr(const Expr *E);
  const Expr *findNextRecordedEvent(const Stmt *Parent, unsigned KCallLoc);

  static EventQueryTraversal getEventQueryTraversal();

private:
  void handleEventRecordWithProfilingEnabled(
      const CallExpr *CE, const ast_matchers::MatchFinder::MatchResult &Result,
      bool IsAssigned);
  void handleEventRecordWithProfilingDisabled(
      const CallExpr *CE, const ast_matchers::MatchFinder::MatchResult &Result,
      bool IsAssigned);
  void findEventAPI(const Stmt *Node, const CallExpr *&Call,
                    const std::string EventAPIName);
  void processAsyncJob(const Stmt *Node);
  void updateAsyncRange(const Stmt *FuncBody, const std::string EventAPIName);
  void updateAsyncRangRecursive(const Stmt *Node, const CallExpr *AsyncCE,
                                const std::string EventAPIName);

  void findThreadSyncLocation(const Stmt *Node);
  const clang::Stmt *getRedundantParenExpr(const CallExpr *Call);
  bool isEventElapsedTimeFollowed(const CallExpr *Expr);
  // Since the state of a rule is shared between multiple matches, it is
  // necessary to clear the previous migration status.
  // The call is supposed to be called whenever a migration on time measurement
  // is triggered.
  void reset() {
    RecordBegin = nullptr;
    RecordEnd = nullptr;
    TimeElapsedCE = nullptr;
    RecordBeginLoc = 0;
    RecordEndLoc = 0;
    TimeElapsedLoc = 0;
    ThreadSyncLoc = 0;
    Events2Wait.clear();
    QueueCounter.clear();
    Queues2Wait.clear();
    DefaultQueueAdded = false;
    IsKernelInLoopStmt = false;
    IsKernelSync = false;
  }
  const Stmt *RecordBegin = nullptr, *RecordEnd = nullptr;
  const CallExpr *TimeElapsedCE = nullptr;
  unsigned RecordBeginLoc = 0;
  unsigned RecordEndLoc = 0;
  unsigned TimeElapsedLoc = 0;

  // To store the location of "cudaThreadSynchronize"
  unsigned ThreadSyncLoc = 0;
  std::vector<std::string> Events2Wait;
  std::map<std::string, int> QueueCounter;
  std::vector<std::pair<std::string, const CallExpr *>> Queues2Wait;
  bool DefaultQueueAdded = false;

  // To check whether kernel call is in loop stmt between RecordBeginLoc and
  // RecordEndLoc
  bool IsKernelInLoopStmt = false;

  // To check whether kernel call needs wait between RecordBeginLoc and
  // RecordEndLoc
  bool IsKernelSync = false;

  std::map<const Stmt *, bool> ExprCache;
  std::map<const VarDecl *, bool> VarDeclCache;

  friend class EventQueryTraversal;
  static EventAPICallRule *CurrentRule;
};

/// Migration rule for stream API calls
class StreamAPICallRule : public NamedMigrationRule<StreamAPICallRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

/// Migration rule for kernel API calls
class KernelCallRule : public NamedMigrationRule<KernelCallRule> {
  std::unordered_set<unsigned> Insertions;
  std::set<clang::SourceLocation> CodePinInstrumentation;

public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
  SourceLocation
  removeTrailingSemicolon(const CallExpr *KCall,
                          const ast_matchers::MatchFinder::MatchResult &Result);
  void instrumentKernelLogsForCodePin(const CUDAKernelCallExpr *KCall,
                                      SourceLocation &EpilogLocation);
};

/// Migration rule for device function calls
class DeviceFunctionDeclRule
    : public NamedMigrationRule<DeviceFunctionDeclRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

/// Migration rule for __constant__/__shared__/__device__ memory variables.
class MemVarAnalysisRule : public NamedMigrationRule<MemVarAnalysisRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

class MemVarMigrationRule : public NamedMigrationRule<MemVarMigrationRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);

private:
  void processTypeDeclaredLocal(const VarDecl *MemVar,
                                std::shared_ptr<MemVarInfo> Info);
};

class ConstantMemVarMigrationRule : public NamedMigrationRule<ConstantMemVarMigrationRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);

private:
  void previousHCurrentD(const VarDecl *VD, tooling::Replacement &R);
  void previousDCurrentH(const VarDecl *VD, tooling::Replacement &R);
  void removeHostConstantWarning(tooling::Replacement &R);
  bool currentIsDevice(const VarDecl *MemVar, std::shared_ptr<MemVarInfo> Info);
  bool currentIsHost(const VarDecl *VD, std::string VarName);
};

class MemVarRefMigrationRule : public NamedMigrationRule<MemVarRefMigrationRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

class ProfilingEnableOnDemandRule
    : public NamedMigrationRule<ProfilingEnableOnDemandRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

/// Migration rule for memory management routine.
/// Current implementation is intentionally simplistic. The following things
/// need a more detailed design:
///   - interplay with error handling (possible solution is that we keep
///   function
///     signature as close to original as possible, so return error codes when
///     original functions return them);
///   - SYCL memory buffers are typed. Using a "char" type is definitely a
///   tradeoff.
///     Using better type information requires some kind of global analysis and
///     heuristics, as well as a mechanism for user hint (like "treat all
///     buffers as float-typed")'
///   - interplay with streams need to be designed.
///   - transformation rules are currently unordered, which create potential
///     ambiguity, so need to understand how to handle function call arguments,
///     which are modified by other rules.
///
class MemoryMigrationRule : public NamedMigrationRule<MemoryMigrationRule> {

public:
  MemoryMigrationRule();
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);

  /// Get helper function name with namespace which has 'dpct_' in dpct helper
  /// functions and w/o in syclcompat.
  /// If has "_async" suffix, the name in dpct helper function will have
  /// 'async_' prefix and remove the suffix.
  /// If `ExperimentalInSYCLCompat` is true, will add `experimental` namespace
  /// in syclcompat.
  static std::string
  getMemoryHelperFunctionName(StringRef RawName,
                              bool ExperimentalInSYCLCompat = false);

private:
  void mallocMigration(const ast_matchers::MatchFinder::MatchResult &Result,
                       const CallExpr *C,
                       const UnresolvedLookupExpr *ULExpr = NULL,
                       bool IsAssigned = false);
  void memcpyMigration(const ast_matchers::MatchFinder::MatchResult &Result,
                       const CallExpr *C,
                       const UnresolvedLookupExpr *ULExpr = NULL,
                       bool IsAssigned = false);
  void freeMigration(const ast_matchers::MatchFinder::MatchResult &Result,
                     const CallExpr *C,
                     const UnresolvedLookupExpr *ULExpr = NULL,
                     bool IsAssigned = false);
  void memsetMigration(const ast_matchers::MatchFinder::MatchResult &Result,
                       const CallExpr *C,
                       const UnresolvedLookupExpr *ULExpr = NULL,
                       bool IsAssigned = false);
  void arrayMigration(const ast_matchers::MatchFinder::MatchResult &Result,
                      const CallExpr *C,
                      const UnresolvedLookupExpr *ULExpr = NULL,
                      bool IsAssigned = false);
  void getSymbolAddressMigration(
      const ast_matchers::MatchFinder::MatchResult &Result, const CallExpr *C,
      const UnresolvedLookupExpr *ULExpr = NULL, bool IsAssigned = false);
  void getSymbolSizeMigration(
      const ast_matchers::MatchFinder::MatchResult &Result, const CallExpr *C,
      const UnresolvedLookupExpr *ULExpr = NULL, bool IsAssigned = false);
  void prefetchMigration(const ast_matchers::MatchFinder::MatchResult &Result,
                         const CallExpr *C,
                         const UnresolvedLookupExpr *ULExpr = NULL,
                         bool IsAssigned = false);
  void miscMigration(const ast_matchers::MatchFinder::MatchResult &Result,
                     const CallExpr *C,
                     const UnresolvedLookupExpr *ULExpr = NULL,
                     bool IsAssigned = false);
  void cudaArrayGetInfo(const ast_matchers::MatchFinder::MatchResult &Result,
                        const CallExpr *C,
                        const UnresolvedLookupExpr *ULExpr = NULL,
                        bool IsAssigned = false);
  void cudaHostGetFlags(const ast_matchers::MatchFinder::MatchResult &Result,
                        const CallExpr *C,
                        const UnresolvedLookupExpr *ULExpr = NULL,
                        bool IsAssigned = false);
  void cudaMemAdvise(const ast_matchers::MatchFinder::MatchResult &Result,
                     const CallExpr *C,
                     const UnresolvedLookupExpr *ULExpr = NULL,
                     bool IsAssigned = false);
  void handleAsync(const CallExpr *C, unsigned i,
                   const ast_matchers::MatchFinder::MatchResult &Result);
  void handleDirection(const CallExpr *C, unsigned i);
  void replaceMemAPIArg(const Expr *E,
                        const ast_matchers::MatchFinder::MatchResult &Result,
                        const std::string &StreamStr,
                        std::string OffsetFromBaseStr = "");
  const ArraySubscriptExpr *getArraySubscriptExpr(const Expr *E);
  const Expr *getUnaryOperatorExpr(const Expr *E);
  void memcpySymbolMigration(
      const ast_matchers::MatchFinder::MatchResult &Result, const CallExpr *C,
      const UnresolvedLookupExpr *ULExpr = NULL, bool IsAssigned = false);
  std::unordered_map<
      std::string,
      std::function<void(const ast_matchers::MatchFinder::MatchResult &Result,
                         const CallExpr *C, const UnresolvedLookupExpr *ULExpr,
                         bool IsAssigned)>>
      MigrationDispatcher;
  std::string getTypeStrRemovedAddrOf(const Expr *E, bool isCOCE = false);
  std::string getAssignedStr(const Expr *E, const std::string &Arg0Str);
  void mallocArrayMigration(const CallExpr *C, const std::string &Name,
                            const std::string &Flag, SourceManager &SM);
  void mallocMigrationWithTransformation(SourceManager &SM, const CallExpr *C,
                                         const std::string &CallName,
                                         std::string &&ReplaceName,
                                         const std::string &PaddingArgs = "",
                                         bool NeedTypeCast = true,
                                         size_t AllocatedArgIndex = 0,
                                         size_t SizeArgIndel = 1);
  bool canUseTemplateStyleMigration(const Expr *AllocatedExpr,
                                    const Expr *SizeExpr, std::string &ReplType,
                                    std::string &ReplSize);
  std::string getTransformedMallocPrefixStr(const Expr *MallocOutArg,
                                            bool NeedTypeCast,
                                            bool TemplateStyle = false);
  void aggregatePitchedData(const CallExpr *C, size_t DataArgIndex,
                            size_t SizeArgIndex, SourceManager &SM,
                            bool ExcludeSizeArg = false);
  void aggregate3DVectorClassCtor(const CallExpr *C, StringRef ClassName,
                                  size_t ArgStartIndex, StringRef DefaultValue,
                                  SourceManager &SM, size_t ArgsNum = 2);
  void aggregateArgsToCtor(const CallExpr *C, const std::string &ClassName,
                           size_t StartArgIndex, size_t EndArgIndex,
                           const std::string &PaddingArgs, SourceManager &SM);
  void insertToPitchedData(const CallExpr *C, size_t ArgIndex) {
    if (C->getNumArgs() > ArgIndex) {
      if (C->getArg(ArgIndex)->IgnoreImplicit()->getStmtClass() !=
          Stmt::StmtClass::DeclRefExprClass)
        insertAroundStmt(C->getArg(ArgIndex), "(", ")");
      requestFeature(HelperFeatureEnum::device_ext);
      emplaceTransformation(
          new InsertAfterStmt(C->getArg(ArgIndex), "->to_pitched_data()"));
    }
  }
  void insertZeroOffset(const CallExpr *C, size_t InsertArgIndex) {
    static std::string InsertedText =
        buildString(MapNames::getClNamespace(),
                    DpctGlobalInfo::getCtadClass("id", 3), "(0, 0, 0), ");
    if (C->getNumArgs() > InsertArgIndex)
      emplaceTransformation(new InsertBeforeStmt(C->getArg(InsertArgIndex),
                                                 std::string(InsertedText)));
  }
  void instrumentAddressToSizeRecordForCodePin(const CallExpr *C, int PtrArgLoc,
                                               int AllocMemSizeLoc);
};

class MemoryDataTypeRule : public NamedMigrationRule<MemoryDataTypeRule> {
  static inline std::string getCtadType(StringRef BaseTypeName) {
    return buildString(DpctGlobalInfo::getCtadClass(
        buildString(MapNames::getClNamespace(), BaseTypeName), 3));
  }
  template <class... Args>
  void emplaceParamDecl(const VarDecl *VD, StringRef ParamType,
                        bool HasInitialZeroCtor, std::string InitValue = "0",
                        Args &&...ParamNames) {
    std::string ParamDecl;
    llvm::raw_string_ostream OS(ParamDecl);
    OS << ParamType << " ";
    unsigned Index = 0;
    printParamNameWithInitArgs(OS, VD->getName(), HasInitialZeroCtor, Index,
                               InitValue, std::forward<Args>(ParamNames)...);
    OS << ";";
    emplaceTransformation(
        ReplaceVarDecl::getVarDeclReplacement(VD, std::move(OS.str())));
  }
  static inline llvm::raw_ostream &printParamName(llvm::raw_ostream &OS,
                                                  StringRef BaseName,
                                                  StringRef ParamName) {
    return OS << BaseName << "_" << ParamName << getCTFixedSuffix();
  }
  static inline llvm::raw_ostream &
  printParamNameWithInitArgs(llvm::raw_ostream &OS, StringRef BaseName,
                             bool HasInitialZeroCtor, unsigned &Index,
                             StringRef InitValue) {
    return OS;
  }
  template <class... RestNamesT>
  static inline llvm::raw_ostream &
  printParamNameWithInitArgs(llvm::raw_ostream &OS, StringRef BaseName,
                             bool HasInitialZeroCtor, unsigned &Index,
                             StringRef InitValue, StringRef FirstName,
                             RestNamesT &&...Rest) {
    if (Index++)
      OS << ", ";
    printParamName(OS, BaseName, FirstName);
    if (HasInitialZeroCtor)
      OS << "(" << InitValue << ", " << InitValue << ", " << InitValue << ")";
    return printParamNameWithInitArgs(OS, BaseName, HasInitialZeroCtor, Index,
                                      InitValue,
                                      std::forward<RestNamesT>(Rest)...);
  }

  const static MapNames::MapTy DirectReplMemberNames;
  const static MapNames::MapTy GetSetReplMemberNames;
  const static MapNames::MapTy ExtentMemberNames;
  const static MapNames::MapTy PitchMemberNames;
  const static MapNames::MapTy ArrayDescMemberNames;
  const static std::vector<std::string> RemoveMember;

public:
  static std::string getArrayDescMemberName(StringRef BaseName,
                                            const std::string &Member) {
    auto Itr = ArrayDescMemberNames.find(Member);
    if (Itr != ArrayDescMemberNames.end()) {
      std::string ReplacedName;
      llvm::raw_string_ostream OS(ReplacedName);
      printParamName(OS, BaseName, Itr->second);
      return OS.str();
    }
    return Member;
  }

  static bool isRemove(std::string Name) {
    return std::find(RemoveMember.begin(), RemoveMember.end(), Name) !=
           RemoveMember.end();
  }

  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

class CMemoryAPIRule : public NamedMigrationRule<CMemoryAPIRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

/// Name all unnamed types.
class UnnamedTypesRule : public NamedMigrationRule<UnnamedTypesRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

/// Guess original code indent width.
class GuessIndentWidthRule : public NamedMigrationRule<GuessIndentWidthRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

/// Migration for math functions
class MathFunctionsRule : public NamedMigrationRule<MathFunctionsRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
  void handleExceptionalFunctions(
      const CallExpr *CE, const ast_matchers::MatchFinder::MatchResult &Result);
  void
  handleHalfFunctions(const CallExpr *CE,
                      const ast_matchers::MatchFinder::MatchResult &Result);
  void handleSingleDoubleFunctions(
      const CallExpr *CE, const ast_matchers::MatchFinder::MatchResult &Result);
  void
  handleTypecastFunctions(const CallExpr *CE,
                          const ast_matchers::MatchFinder::MatchResult &Result);
  void
  handleMiscFunctions(const CallExpr *CE,
                      const ast_matchers::MatchFinder::MatchResult &Result);
};

/// Migration for warp functions
class WarpFunctionsRule : public NamedMigrationRule<WarpFunctionsRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

class CooperativeGroupsFunctionRule
    : public NamedMigrationRule<CooperativeGroupsFunctionRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

/// Migration rule for replacing __syncthreads() function call.
///
/// This rule replace __syncthreads() with item.barrier()
class SyncThreadsRule : public NamedMigrationRule<SyncThreadsRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

class SyncThreadsMigrationRule
    : public NamedMigrationRule<SyncThreadsMigrationRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

/// Migrate Function Attributes to Sycl kernel info, defined in
/// runtime headers.
class KernelFunctionInfoRule
    : public NamedMigrationRule<KernelFunctionInfoRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);

  static const std::map<std::string, std::string> AttributesNamesMap;
};

/// RecognizeAPINameRule to give comments for the API not in the record table
class RecognizeAPINameRule : public NamedMigrationRule<RecognizeAPINameRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);

private:
  const std::string getFunctionSignature(const FunctionDecl *Func,
                                         std::string ObjName);
  std::vector<std::vector<std::string>>
  splitAPIName(std::vector<std::string> &AllAPINames);
  void processFuncCall(const CallExpr *CE);
};

/// RecognizeTypeRule to emit warning message for known unsupported type
class RecognizeTypeRule : public NamedMigrationRule<RecognizeTypeRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

class TextureMemberSetRule : public NamedMigrationRule<TextureMemberSetRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
  void removeRange(SourceRange R);
};

/// Texture migration rule
class TextureRule : public NamedMigrationRule<TextureRule> {
  // Get the binary operator if E is lhs of an assign expression.
  const Expr *getAssignedBO(const Expr *E, ASTContext &Context);
  const Expr *getParentAsAssignedBO(const Expr *E, ASTContext &Context);
  bool removeExtraMemberAccess(const MemberExpr *ME);
  void replaceTextureMember(const MemberExpr *ME, ASTContext &Context,
                            SourceManager &SM);
  void replaceResourceDataExpr(const MemberExpr *ME, ASTContext &Context);
  inline const MemberExpr *getParentMemberExpr(const Stmt *S) {
    return DpctGlobalInfo::findParent<MemberExpr>(S);
  }
  std::string getTextureFlagsSetterInfo(const Expr *Flags,
                                        StringRef &SetterName);
  std::string getMemberAssignedValue(const Stmt *AssignStmt,
                                     StringRef MemberName,
                                     StringRef &SetMethodName);
  static MapNames::MapTy ResourceTypeNames;

public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);

  static const MapNames::MapTy TextureMemberNames;

private:
  bool processTexVarDeclInDevice(const VarDecl *VD);

  bool tryMerge(const MemberExpr *ME, const Expr *BO);

  class SettersMerger {
    TextureRule *Rule;
    const std::vector<std::string> &MethodNames;
    std::map<const Stmt *, bool> &ProcessedBO;

    const Stmt *Target = nullptr;
    bool Stop = false;
    ValueDecl *D = nullptr;
    bool IsArrow = false;
    std::vector<std::pair<unsigned, const Stmt *>> Result;

    void traverse(const Stmt *);
    void traverseBinaryOperator(const Stmt *);
    bool applyResult();

  public:
    SettersMerger(const std::vector<std::string> &Names, TextureRule *TexRule)
        : Rule(TexRule), MethodNames(Names), ProcessedBO(TexRule->ProcessedBO) {
    }

    bool tryMerge(const Stmt *S);
  };

  std::map<const Stmt *, bool> ProcessedBO;
};

/// CXXNewExprRule is to migrate types in C++ new expressions, e.g.
/// "new cudaStream_t[10]" => "new queue_p[10]"
/// "new cudaStream_t" => "new queue_p"
class CXXNewExprRule : public NamedMigrationRule<CXXNewExprRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

class NamespaceRule : public NamedMigrationRule<NamespaceRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

class RemoveBaseClassRule : public NamedMigrationRule<RemoveBaseClassRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

class PreDefinedStreamHandleRule
    : public NamedMigrationRule<PreDefinedStreamHandleRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};


class DriverModuleAPIRule : public NamedMigrationRule<DriverModuleAPIRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

class VirtualMemRule : public NamedMigrationRule<VirtualMemRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

class DriverDeviceAPIRule : public NamedMigrationRule<DriverDeviceAPIRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

class DriverContextAPIRule : public NamedMigrationRule<DriverContextAPIRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

class CudaArchMacroRule : public NamedMigrationRule<CudaArchMacroRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};
class ComplexAPIRule : public NamedMigrationRule<ComplexAPIRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

class CudaStreamCastRule : public NamedMigrationRule<CudaStreamCastRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

class TypeRemoveRule : public NamedMigrationRule<TypeRemoveRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

class TypeMmberRule : public NamedMigrationRule<TypeMmberRule> {
  std::optional<SourceLocation>
  findTokenEndBeforeColonColon(SourceLocation TokStart,
                               const SourceManager &SM);

public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

class CompatWithClangRule : public NamedMigrationRule<CompatWithClangRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

class GraphRule : public NamedMigrationRule<GraphRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

class AssertRule : public NamedMigrationRule<AssertRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

class GraphicsInteropRule : public NamedMigrationRule<GraphicsInteropRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

class RulesLangAddrSpaceConvRule
    : public NamedMigrationRule<RulesLangAddrSpaceConvRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

TextModification *replaceText(SourceLocation Begin, SourceLocation End,
                              std::string &&Str, const SourceManager &SM);

TextModification *removeArg(const CallExpr *C, unsigned n,
                            const SourceManager &SM) ;
} // namespace dpct
} // namespace clang
#endif // DPCT_RULES_LANG_H
