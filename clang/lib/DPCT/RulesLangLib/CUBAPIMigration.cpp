//===--------------- CUBAPIMigration.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RulesLangLib/CUBAPIMigration.h"
#include "AnalysisInfo.h"
#include "MigrationRuleManager.h"
#include "RuleInfra/ASTmatcherCommon.h"
#include "RuleInfra/ExprAnalysis.h"
#include "TextModification.h"
#include "Utility.h"
#include "clang/AST/Attrs.inc"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/OperationKinds.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/Type.h"
#include "clang/AST/TypeLoc.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Analysis/AnalysisDeclContext.h"
#include "clang/Basic/LLVM.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
#include <memory>
#include <optional>
#include <vector>

using namespace clang;
using namespace dpct;
using namespace clang::dpct;
using namespace tooling;
using namespace ast_matchers;

namespace {


auto isDeviceFuncCallExpr = []() {
  auto hasDeviceFuncName = []() {
    return hasAnyName(
        "Sum", "Min", "Max", "ArgMin", "ArgMax", "Reduce", "ReduceByKey",
        "ExclusiveSum", "InclusiveSum", "InclusiveScan", "ExclusiveScan",
        "InclusiveScanByKey", "InclusiveSumByKey", "ExclusiveScanByKey",
        "ExclusiveSumByKey", "Flagged", "Unique", "UniqueByKey", "Encode",
        "SortKeys", "SortKeysDescending", "SortPairs", "SortPairsDescending",
        "If", "StableSortKeys", "StableSortKeysDescending", "StableSortPairs",
        "StableSortPairsDescending", "NonTrivialRuns", "HistogramEven",
        "MultiHistogramEven", "HistogramRange", "MultiHistogramRange",
        "SortKeysCopy", "StableSortKeys", "CsrMV");
  };
  auto hasDeviceRecordName = []() {
    return hasAnyName("DeviceSegmentedReduce", "DeviceReduce", "DeviceScan",
                      "DeviceSelect", "DeviceRunLengthEncode",
                      "DeviceRadixSort", "DeviceSegmentedRadixSort",
                      "DeviceSegmentedSort", "DeviceHistogram",
                      "DeviceMergeSort", "DevicePartition", "DeviceSpmv");
  };
  return callExpr(callee(functionDecl(allOf(
      hasDeviceFuncName(),
      hasDeclContext(cxxRecordDecl(allOf(
          hasDeviceRecordName(),
          hasAncestor(namespaceDecl(hasName("cub"))))))))));
};

} // namespace

REGISTER_RULE(CubTypeRule, PassKind::PK_Analysis)
REGISTER_RULE(CubMemberCallRule, PassKind::PK_Analysis)
REGISTER_RULE(CubDeviceLevelRule, PassKind::PK_Analysis)
REGISTER_RULE(CubIntrinsicRule, PassKind::PK_Analysis)

#define REPORT_UNSUPPORT_SYCLCOMPACT(S)                                        \
  if (DpctGlobalInfo::useSYCLCompat()) {                                       \
    std::string PrettyStmt;                                                    \
    llvm::raw_string_ostream OS(PrettyStmt);                                   \
    S->printPretty(OS, /*Helper=*/nullptr,                                     \
                   DpctGlobalInfo::getContext().getPrintingPolicy());          \
    report(S->getBeginLoc(), Diagnostics::UNSUPPORT_SYCLCOMPAT,                \
           /*UseTextBegin=*/false, PrettyStmt);                                \
    return;                                                                    \
  }

void CubTypeRule::registerMatcher(ast_matchers::MatchFinder &MF) {
  auto TargetTypeName = [&]() {
    return hasAnyName(
        "cub::Sum", "cub::Max", "cub::Min", "cub::Equality",
        "cub::KeyValuePair", "cub::CountingInputIterator",
        "cub::TransformInputIterator", "cub::ConstantInputIterator",
        "cub::ArgIndexInputIterator", "cub::DiscardOutputIterator",
        "cub::DoubleBuffer", "cub::NullType", "cub::ArgMax", "cub::ArgMin",
        "cub::BlockRadixSort", "cub::BlockExchange", "cub::BlockLoad",
        "cub::BlockStore");
  };

  MF.addMatcher(
      typeLoc(loc(qualType(hasDeclaration(namedDecl(TargetTypeName())))))
          .bind("loc"),
      this);
}

void CubTypeRule::runRule(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  if (const TypeLoc *TL = getAssistNodeAsType<TypeLoc>(Result, "loc")) {
    ExprAnalysis EA;
    auto DNTL = DpctGlobalInfo::findAncestor<DependentNameTypeLoc>(TL);
    auto NNSL = DpctGlobalInfo::findAncestor<NestedNameSpecifierLoc>(TL);
    if (NNSL) {
      EA.analyze(*TL, *NNSL);
    } else if (DNTL) {
      EA.analyze(*TL, *DNTL);
    } else {
      EA.analyze(*TL);
    }
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
  }
}

bool CubTypeRule::CanMappingToSyclNativeBinaryOp(StringRef OpTypeName) {
  return OpTypeName == "cub::Sum" || OpTypeName == "cub::Max" ||
         OpTypeName == "cub::Min";
}

bool CubTypeRule::CanMappingToSyclType(StringRef OpTypeName) {
  return CanMappingToSyclNativeBinaryOp(OpTypeName) ||
         OpTypeName == "cub::Equality" || OpTypeName == "cub::NullType" ||

         // Ignore template arguments, .e.g cub::KeyValuePair<int, int>
         OpTypeName.starts_with("cub::KeyValuePair");
}

void CubDeviceLevelRule::registerMatcher(ast_matchers::MatchFinder &MF) {
  MF.addMatcher(isDeviceFuncCallExpr().bind("FuncCall"), this);
}

void CubDeviceLevelRule::runRule(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  if (const auto *CE = getNodeAsType<CallExpr>(Result, "FuncCall")) {
    ExprAnalysis EA;
    EA.analyze(CE);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
    return;
  }
}

void CubMemberCallRule::registerMatcher(ast_matchers::MatchFinder &MF) {
  MF.addMatcher(
      cxxMemberCallExpr(
          allOf(on(hasType(hasCanonicalType(qualType(hasDeclaration(namedDecl(
                    hasAnyName("cub::ArgIndexInputIterator",
                               "cub::BlockRadixSort", "cub::BlockExchange",
                               "cub::BlockLoad", "cub::BlockStore"))))))),
                callee(cxxMethodDecl(hasAnyName(
                    "normalize", "Sort", "SortDescending", "BlockedToStriped",
                    "StripedToBlocked", "ScatterToBlocked", "ScatterToStriped",
                    "SortBlockedToStriped", "SortDescendingBlockedToStriped",
                    "Load", "Store")))))
          .bind("memberCall"),
      this);

  MF.addMatcher(
      memberExpr(
          hasObjectExpression(hasType(hasCanonicalType(qualType(
              hasDeclaration(namedDecl(hasName("cub::DoubleBuffer"))))))),
          member(hasAnyName("Current", "Alternate", "d_buffers")))
          .bind("memberExpr"),
      this);
}

static std::pair<const VarDecl *, TypeLoc>
getTempstorageVarAndValueTypeLoc(const CXXMemberCallExpr *MC) {
  Expr *Obj = MC->getImplicitObjectArgument();
  const VarDecl *TempStorage = nullptr;

  auto FindTempStorageVarInCtor = [](const Expr *E) -> const VarDecl * {
    if (auto *Ctor = dyn_cast<CXXConstructExpr>(E)) {
      if (auto *DRE = dyn_cast<DeclRefExpr>(Ctor->getArg(0))) {
        if (auto *VD = dyn_cast<VarDecl>(DRE->getDecl())) {
          if (VD->hasAttr<CUDASharedAttr>() && isCubVar(VD)) {
            return VD;
          }
        }
      }
    }
    return nullptr;
  };

  auto FindDataTypeLoc = [](TypeLoc Loc) -> TypeLoc {
    if (Loc.isNull())
      return Loc;
    while (true) {
      switch (Loc.getTypeLocClass()) {
      case TypeLoc::Elaborated:
        Loc = Loc.getNextTypeLoc();
        break;
      case TypeLoc::Typedef: {
        auto NewLoc = Loc.castAs<TypedefTypeLoc>();
        Loc = NewLoc.getTypedefNameDecl()->getTypeSourceInfo()->getTypeLoc();
        break;
      }
      case TypeLoc::TemplateSpecialization: {
        auto NewLoc = Loc.getAs<TemplateSpecializationTypeLoc>();
        return NewLoc.getArgLocInfo(0).getAsTypeSourceInfo()->getTypeLoc();
        break;
      }
      default:
        return Loc;
      }
    }
  };

  TypeLoc DataTypeLoc;
  if (const auto *MTE = dyn_cast<MaterializeTemporaryExpr>(Obj)) {
    if (auto *TOE = dyn_cast<CXXTemporaryObjectExpr>(MTE->getSubExpr())) {
      DataTypeLoc = FindDataTypeLoc(TOE->getTypeSourceInfo()->getTypeLoc());
    } else if (auto *FC = dyn_cast<CXXFunctionalCastExpr>(MTE->getSubExpr())) {
      DataTypeLoc = FindDataTypeLoc(FC->getTypeInfoAsWritten()->getTypeLoc());
    }
    TempStorage = FindTempStorageVarInCtor(MTE->getSubExpr()->IgnoreCasts());
  } else if (const auto *DRE = dyn_cast<DeclRefExpr>(Obj)) {
    if (const auto *VD = dyn_cast<VarDecl>(DRE->getDecl())) {
      DataTypeLoc = FindDataTypeLoc(VD->getTypeSourceInfo()->getTypeLoc());
      if (isCubCollectiveRecordType(VD->getType()) && VD->hasInit())
        TempStorage = FindTempStorageVarInCtor(VD->getInit());
    }
  }
  return {TempStorage, DataTypeLoc};
}

void CubMemberCallRule::runRule(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  ExprAnalysis EA;
  if (const auto *BlockMC =
          getNodeAsType<CXXMemberCallExpr>(Result, "memberCall")) {
    EA.analyze(BlockMC);
    StringRef Name = BlockMC->getMethodDecl()->getName();
    bool isBlockRadixSort = Name == "Sort" || Name == "SortDescending" ||
                            Name == "SortBlockedToStriped" ||
                            Name == "SortDescendingBlockedToStriped";
    bool isBlockExchange =
        Name == "BlockedToStriped" || Name == "StripedToBlocked" ||
        Name == "StripedToBlocked" || Name == "ScatterToBlocked" ||
        Name == "ScatterToStriped";
    if (isBlockRadixSort || isBlockExchange || Name == "Load" ||
        Name == "Store") {
      std::string HelpFuncName;
      if (isBlockRadixSort)
        HelpFuncName = "group_radix_sort";
      else if (isBlockExchange)
        HelpFuncName = "exchange";
      else if (Name == "Load")
        HelpFuncName = "group_load";
      else if (Name == "Store")
        HelpFuncName = "group_store";
      auto [TempStorage, DataTypeLoc] =
          getTempstorageVarAndValueTypeLoc(BlockMC);
      auto *FD = DpctGlobalInfo::findAncestor<FunctionDecl>(TempStorage);
      if (!FD || !TempStorage || DataTypeLoc.isNull())
        return;
      QualType CanTy = BlockMC->getObjectType().getCanonicalType();
      auto *ClassSpecDecl = dyn_cast<ClassTemplateSpecializationDecl>(
          CanTy->getAs<RecordType>()->getDecl());
      const auto &ValueTyArg = ClassSpecDecl->getTemplateArgs()[0];
      const auto &ItemsPreThreadArg = ClassSpecDecl->getTemplateArgs()[2];
      ValueTyArg.getAsType().getAsString();
      std::string Fn;
      llvm::raw_string_ostream OS(Fn);
      OS << MapNames::getDpctNamespace() << "group::" << HelpFuncName << "<"
         << ValueTyArg.getAsType().getAsString() << ", "
         << ItemsPreThreadArg.getAsIntegral() << ">::get_local_memory_size";
      if (auto FuncInfo = DeviceFunctionDecl::LinkRedecls(FD)) {
        auto LocInfo = DpctGlobalInfo::getLocInfo(TempStorage);
        ExprAnalysis EA;
        EA.analyze(DataTypeLoc);
        FuncInfo->getVarMap().addCUBTempStorage(
            std::make_shared<TempStorageVarInfo>(
                LocInfo.second, TempStorageVarInfo::BlockRadixSort,
                TempStorage->getName(), Fn,
                EA.getTemplateDependentStringInfo()));
      }
    }
  } else if (const auto *E2 = getNodeAsType<MemberExpr>(Result, "memberExpr")) {
    EA.analyze(E2);
  }
  emplaceTransformation(EA.getReplacement());
  EA.applyAllSubExprRepl();
}

void CubIntrinsicRule::registerMatcher(ast_matchers::MatchFinder &MF) {
  MF.addMatcher(
      callExpr(
          callee(functionDecl(allOf(
              hasAnyName("IADD3", "SHR_ADD", "SHL_ADD", "BFE", "BFI", "LaneId",
                         "WarpId", "SyncStream", "CurrentDevice", "DeviceCount",
                         "DeviceCountUncached", "DeviceCountCachedValue",
                         "PtxVersion", "PtxVersionUncached", "SmVersion",
                         "SmVersionUncached", "RowMajorTid",
                         "LoadDirectBlocked", "LoadDirectStriped",
                         "StoreDirectBlocked", "StoreDirectStriped",
                         "ShuffleDown", "ShuffleUp", "Debug"),
              hasAncestor(namespaceDecl(hasName("cub")))))))
          .bind("IntrinsicCall"),
      this);
}

void CubIntrinsicRule::runRule(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  if (const auto *CE = getNodeAsType<CallExpr>(Result, "IntrinsicCall")) {
    auto &SM = DpctGlobalInfo::getSourceManager();
    if (!DpctGlobalInfo::isInAnalysisScope(
            SM.getSpellingLoc(CE->getBeginLoc()))) {
      return;
    }
    ExprAnalysis EA;
    EA.analyze(CE);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
    return;
  }
}

static bool isNullPointerConstant(const clang::Expr *E) {
  assert(E && "Expr can not be nullptr");
  return E->isNullPointerConstant(clang::dpct::DpctGlobalInfo::getContext(),
                                  Expr::NPC_ValueDependentIsNull) !=
         Expr::NPCK_NotNull;
}

static bool isCubDeviceFunctionCallExpr(const CallExpr *C) {
  if (!C)
    return false;
  return !ast_matchers::match(isDeviceFuncCallExpr(), *C,
                              DpctGlobalInfo::getContext())
              .empty();
}

static bool isCudaMemoryAPICallExpr(const CallExpr *C) {
  if (!C)
    return false;

  if (const auto *FD = C->getDirectCallee())
    return FD->getName() == "cudaMalloc" || FD->getName() == "cuMemAlloc_v2" ||
           FD->getName() == "cudaFree";
  return false;
}

template <typename NodeType>
static void findLoop(const NodeType *Node, std::vector<const Stmt *> &LoopList,
                     bool OnlyFindFirstLevel = false) {
  const Stmt *Loop = nullptr;
  auto Conditon = [&](const DynTypedNode &LoopNode) -> bool {
    if (const auto *DoLoop = LoopNode.get<DoStmt>()) {
      if (DoLoop->getBody() == Node ||
          DpctGlobalInfo::isAncestor(DoLoop->getBody(), Node)) {
        return true;
      }
    } else if (const auto *ForLoop = LoopNode.get<ForStmt>()) {
      if (ForLoop->getBody() == Node ||
          DpctGlobalInfo::isAncestor(ForLoop->getBody(), Node)) {
        return true;
      }
    } else if (const auto *WhileLoop = LoopNode.get<WhileStmt>()) {
      if (WhileLoop->getBody() == Node ||
          DpctGlobalInfo::isAncestor(WhileLoop->getBody(), Node)) {
        return true;
      }
    }
    return false;
  };
  if (Loop = DpctGlobalInfo::findAncestor<Stmt>(Node, Conditon)) {
    LoopList.push_back(Loop);
    if (!OnlyFindFirstLevel) {
      findLoop(Loop, LoopList);
    }
  }
}

bool CubDeviceLevelRule::isRedundantCallExpr(const CallExpr *CE) {
  const auto *FuncArgs = CE->getArgs();
  const auto *TempStorage = FuncArgs[0]->IgnoreCasts();
  SourceLocation InitLoc;
  std::vector<const DeclRefExpr *> TempStorageMatchResult;
  std::vector<const DeclRefExpr *> TempStorageSizeMatchResult;

  // Check the 1st arg of cub device api, if it's a null pointer or
  // zero(0/NULL/nullptr), this call expression used to calculate temp_storage
  // size, so it's a redundant expression.
  if (isNullPointerConstant(TempStorage))
    return true;

  if (const auto *DRE = dyn_cast<DeclRefExpr>(TempStorage)) {
    if (const auto *VD = dyn_cast<VarDecl>(DRE->getDecl())) {
      const Expr *Init = nullptr;
      SourceLocation SearchEndLoc =
          DpctGlobalInfo::getSourceManager().getExpansionLoc(
              DRE->getBeginLoc());
      SourceLocation LastModifiedLoc;
      std::vector<const Stmt *> DRELoopList;
      std::vector<const Stmt *> CELoopList;
      findAllVarRef(DRE, TempStorageMatchResult);
      findLoop(CE, CELoopList);
      if (VD->hasInit()) {
        // tempstorage = nullptr/NULL/0/...
        if (VD->getInitStyle() == VarDecl::InitializationStyle::CInit) {
          Init = VD->getInit()->IgnoreImplicitAsWritten();
          if (isNullPointerConstant(Init)) {
            InitLoc = DpctGlobalInfo::getSourceManager().getExpansionLoc(
                VD->getBeginLoc());
          }
          // tempstorage = { nullptr/NULL/0/... }
        } else if (VD->getInitStyle() ==
                   VarDecl::InitializationStyle::ListInit) {
          if (const auto *InitList = dyn_cast<InitListExpr>(VD->getInit())) {
            if (const auto *Init0 = InitList->getInit(0)) {
              Init = Init0->IgnoreImplicitAsWritten();
              if (isNullPointerConstant(Init)) {
                InitLoc = DpctGlobalInfo::getSourceManager().getExpansionLoc(
                    VD->getBeginLoc());
              }
            }
          }
        }
      }
      for (auto &Element : TempStorageMatchResult) {
        if (Element == DRE)
          continue;
        SourceLocation CurLoc =
            DpctGlobalInfo::getSourceManager().getExpansionLoc(
                Element->getBeginLoc());
        bool IsModified = isModifiedRef(Element);
        bool IsAssignedWithNull = false;
        if (IsModified) {
          if (const auto *BO =
                  DpctGlobalInfo::findAncestor<BinaryOperator>(Element)) {
            if (BO->getLHS() == Element &&
                isNullPointerConstant(
                    BO->getRHS()->IgnoreImplicitAsWritten())) {
              IsAssignedWithNull = true;
            }
          }
        }
        if (IsAssignedWithNull && (CurLoc < SearchEndLoc) &&
            (InitLoc.isInvalid() || CurLoc > InitLoc)) {
          InitLoc = CurLoc;
          Init = Element;
        }
        if (IsModified && !IsAssignedWithNull) {
          if (CurLoc < SearchEndLoc) {
            LastModifiedLoc = CurLoc;
          } else {
            findLoop(Element, DRELoopList);
          }
        }
      }
      bool IsSafeToRemoveCallExpr = true;
      if (!CELoopList.empty()) {
        int CELoopListSize = CELoopList.size();
        for (int I = 0; I < CELoopListSize; I++) {
          if (DpctGlobalInfo::isAncestor(CELoopList[I], Init)) {
            break;
          }
          if (!DRELoopList.empty() &&
              std::find(DRELoopList.begin(), DRELoopList.end(),
                        CELoopList[I]) != DRELoopList.end()) {
            IsSafeToRemoveCallExpr = false;
            break;
          }
        }
      }
      if (!InitLoc.isInvalid() &&
          (LastModifiedLoc.isInvalid() || InitLoc > LastModifiedLoc) &&
          IsSafeToRemoveCallExpr) {
        return true;
      }
    }
  }
  return false;
}

/// Remove cub device level api temp_storage/temp_storage_size variable
/// declaration
void removeVarDecl(const VarDecl *VD) {
  static std::unordered_map<std::string, std::vector<bool>> DeclStmtBitMap;
  if (!VD)
    return;
  auto &SM = DpctGlobalInfo::getSourceManager();
  auto &Context = DpctGlobalInfo::getContext();
  if (const auto *DS = Context.getParents(*VD)[0].get<DeclStmt>()) {
    auto LocInfo = DpctGlobalInfo::getLocInfo(DS->getBeginLoc());
    std::string Key =
        LocInfo.first.getCanonicalPath().str() + std::to_string(LocInfo.second);
    auto Decls = DS->decls();
    unsigned int DeclNum = DS->decls().end() - DS->decls().begin();
    // if this declstmt has one sub decl, then we just need to remove whole
    // declstmt simply.
    if (DeclNum == 1) {
      const auto &Mgr = DpctGlobalInfo::getSourceManager();
      const auto Range = DS->getSourceRange();
      const CharSourceRange CRange(Range, true);
      auto Replacement =
          std::make_shared<ExtReplacement>(Mgr, CRange, "", nullptr);
      DpctGlobalInfo::getInstance().addReplacement(Replacement);
      return;
    }
    if (!DeclStmtBitMap.count(Key)) {
      DeclStmtBitMap[Key] =
          std::vector<bool>(DS->decls().end() - DS->decls().begin(), true);
    }
    auto NameLength = VD->getNameAsString().length();
    const auto *DeclBegPtr = SM.getCharacterData(VD->getBeginLoc());
    for (const auto *Iter = Decls.begin(); Iter != Decls.end(); Iter++) {
      if (auto *SubDecl = llvm::dyn_cast<VarDecl>(*Iter)) {
        if (SubDecl == VD) {
          int InitLength = 0;
          if (SubDecl->hasInit()) {
            ExprAnalysis InitEA(SubDecl->getInit());
            InitLength = InitEA.getReplacedString().length();
          }
          /// Example1(for non first decl):              Example2(for first
          /// decl): Step 1: Init Beg and End                   Step 1: Init Beg
          /// and End int *a = nullptr, b = 100;                 int *a =
          /// nullptr, b = 100;
          ///                   ^     ^                       ^         ^
          ///                  Beg   End                     Beg       End
          ///
          /// Step 2: Adjust the Beg to previous comma   Ste p2: Adjust the Beg
          /// to the
          ///                                            begin of
          ///                                            prt-declarator, the End
          ///                                            to the behind comma.
          /// int **a = nullptr, b = 100;                int **a = nullptr, b =
          /// 100;
          ///                  ^       ^                     ^            ^
          ///                 Beg     End                   Beg          End
          ///
          /// Step 3: Remove code from Beg to End
          SourceLocation Beg = SM.getExpansionLoc(SubDecl->getEndLoc());
          SourceLocation End =
              SubDecl->hasInit()
                  ? SM.getExpansionLoc(SubDecl->getInit()->getBeginLoc())
                        .getLocWithOffset(InitLength - 1)
                  : SubDecl->getEndLoc().getLocWithOffset(NameLength - 1);
          if (Iter != Decls.begin()) {
            const auto *BegPtr = SM.getCharacterData(Beg);
            const auto *CommaPtr = BegPtr;
            while (CommaPtr && (CommaPtr > DeclBegPtr) && *(CommaPtr) != ',') {
              CommaPtr--;
            };
            if (CommaPtr == DeclBegPtr) {
              return;
            }
            if (Iter - Decls.begin() == 1 && !DeclStmtBitMap[Key][0]) {
              CommaPtr++;
            }
            Beg = Beg.getLocWithOffset(CommaPtr - BegPtr);

          } else {
            QualType TypeTemp = VD->getType();
            while (TypeTemp->isPointerType()) {
              const auto *BegPtr = SM.getCharacterData(Beg);
              const auto *StarPtr = BegPtr;
              while (StarPtr && (StarPtr > DeclBegPtr) &&
                     (*(StarPtr) != '*' || StarPtr == BegPtr)) {
                StarPtr--;
              };
              if (StarPtr == DeclBegPtr) {
                return;
              }
              Beg = Beg.getLocWithOffset(StarPtr - BegPtr);
              TypeTemp = TypeTemp->getPointeeType();
            };
            auto tok = Lexer::findNextToken(End, SM, Context.getLangOpts());
            if (tok.has_value() && tok.value().is(tok::comma)) {
              End = tok.value().getLocation();
            } else {
              return;
            }
          }
          DeclStmtBitMap[Key][Iter - Decls.begin()] = false;
          // if all sub decls are removed, we need to remove this declstmt
          if (std::find(DeclStmtBitMap[Key].begin(), DeclStmtBitMap[Key].end(),
                        true) == DeclStmtBitMap[Key].end()) {
            auto DeclPRInfo = std::make_shared<PriorityReplInfo>();
            DeclPRInfo->Repls.emplace_back(
                (new ReplaceStmt(DS, ""))->getReplacement(Context));
            DeclPRInfo->Priority = 1;
            DpctGlobalInfo::addPriorityReplInfo(Key, DeclPRInfo);
            // Insert header here since PriorityReplInfo delay the replacement
            // addation to the post process. At that time, the MainFile is
            // invalid.
            DpctGlobalInfo::getInstance().insertHeader(DS->getBeginLoc(),
                                                       HeaderType::HT_SYCL);
          } else {
            auto SubDeclPRInfo = std::make_shared<PriorityReplInfo>();
            SubDeclPRInfo->Repls.emplace_back(
                replaceText(Beg, End.getLocWithOffset(1), "", SM)
                    ->getReplacement(Context));
            DpctGlobalInfo::addPriorityReplInfo(Key, SubDeclPRInfo);
            // Insert header here since PriorityReplInfo delay the replacement
            // addation to the post process. At that time, the MainFile is
            // invalid.
            DpctGlobalInfo::getInstance().insertHeader(Beg,
                                                       HeaderType::HT_SYCL);
          }
          break;
        }
      }
    }
  }
}

void CubDeviceLevelRule::removeRedundantTempVar(const CallExpr *CE) {
  if (!CE || CE->getNumArgs() < 2)
    return;
  const auto *FuncArgs = CE->getArgs();
  const auto *TempStorage = FuncArgs[0]->IgnoreCasts();
  const auto *TempStorageSize = FuncArgs[1]->IgnoreCasts();
  std::vector<const DeclRefExpr *> TempStorageMatchResult;
  std::vector<const DeclRefExpr *> TempStorageSizeMatchResult;
  std::vector<const CallExpr *> TempStorageRelatedMalloc;
  std::vector<const CallExpr *> TempStorageSizeRelatedMalloc;

  auto TempStorageVarAnalysis =
      [](const DeclRefExpr *DRE,
         std::vector<const CallExpr *> &RelatedMalloc) -> bool {
    if (const auto *CE = DpctGlobalInfo::findAncestor<CallExpr>(DRE)) {
      if (isCubDeviceFunctionCallExpr(CE))
        return true;
      if (isCudaMemoryAPICallExpr(CE)) {
        RelatedMalloc.push_back(CE);
        return true;
      }
    }
    return false;
  };

  // Analyze and try to remove temp_storage
  if (const auto *DRE = llvm::dyn_cast<DeclRefExpr>(TempStorage)) {
    if (const auto *VD = llvm::dyn_cast<VarDecl>(DRE->getDecl())) {
      findAllVarRef(DRE, TempStorageMatchResult);
      for (auto &Element : TempStorageMatchResult) {
        if (Element == DRE)
          continue;

        // Analyze whether it is safe to remove temp_storage, if not, do nothing
        if (!TempStorageVarAnalysis(Element, TempStorageRelatedMalloc))
          return;
      }

      removeVarDecl(VD);
      for (auto Itr = TempStorageRelatedMalloc.begin();
           Itr != TempStorageRelatedMalloc.end();) {
        bool IsUsed = false;
        if (!isExprUsed(*Itr, IsUsed)) {
          Itr = TempStorageRelatedMalloc.erase(Itr);
          continue;
        }
        auto LocInfo = DpctGlobalInfo::getLocInfo((*Itr)->getBeginLoc());
        auto Info = std::make_shared<PriorityReplInfo>();
        Info->Priority = 1;
        // Insert header here since PriorityReplInfo delay the replacement
        // addation to the post process. At that time, the MainFile is
        // invalid.
        DpctGlobalInfo::getInstance().insertHeader((*Itr)->getBeginLoc(),
                                                   HeaderType::HT_SYCL);
        if (IsUsed) {
          Info->Repls.emplace_back(ReplaceStmt(*Itr, "0").getReplacement(
              DpctGlobalInfo::getContext()));
        } else {
          Info->Repls.emplace_back(ReplaceStmt(*Itr, "").getReplacement(
              DpctGlobalInfo::getContext()));
        }
        DpctGlobalInfo::addPriorityReplInfo(
            LocInfo.first.getCanonicalPath().str() +
                std::to_string(LocInfo.second),
            Info);
        Itr++;
      }
    }
  }

  // Analyze and try to remove temp_storage_size
  if (const auto *DRE = llvm::dyn_cast<DeclRefExpr>(TempStorageSize)) {
    if (const auto *VD = llvm::dyn_cast<VarDecl>(DRE->getDecl())) {
      findAllVarRef(DRE, TempStorageSizeMatchResult);
      for (auto &Element : TempStorageSizeMatchResult) {
        if (Element == DRE)
          continue;

        // Analyze whether it is safe to remove temp_storage_size, if not, do
        // nothing
        if (!TempStorageVarAnalysis(Element, TempStorageSizeRelatedMalloc))
          return;
      }

      // Check if there is an expression that does not refer to both
      // temp_storage and temp_storage_size, if there is, it is unsafe to delete
      // temp_storage_size
      for (const auto *Element : TempStorageSizeRelatedMalloc) {
        if (std::find(TempStorageRelatedMalloc.begin(),
                      TempStorageRelatedMalloc.end(),
                      Element) == TempStorageRelatedMalloc.end()) {
          return;
        }
      }

      // Ok, it's safe to remove temp_storage_size now
      removeVarDecl(VD);
    }
  }
}

void CubRule::registerMatcher(ast_matchers::MatchFinder &MF) {
  MF.addMatcher(usingDirectiveDecl().bind("CubUsingNamespace"), this);
  MF.addMatcher(
      typeLoc(loc(qualType(hasDeclaration(namedDecl(hasAnyName(
                  "WarpScan", "WarpReduce", "BlockScan", "BlockReduce"))))))
          .bind("TypeLoc"),
      this);

  MF.addMatcher(
      typedefDecl(
          hasType(hasCanonicalType(qualType(hasDeclaration(namedDecl(hasAnyName(
              "WarpScan", "WarpReduce", "BlockScan", "BlockReduce")))))))
          .bind("TypeDefDecl"),
      this);

  auto isTempStorage = hasDeclaration(namedDecl(hasAnyName("TempStorage")));
  MF.addMatcher(declStmt(has(varDecl(anyOf(
                             hasType(hasCanonicalType(qualType(isTempStorage))),
                             hasType(arrayType(hasElementType(
                                 hasCanonicalType(qualType(isTempStorage))))),
                             hasType(hasCanonicalType(qualType(hasDeclaration(
                                 recordDecl(isUnion(), has(fieldDecl())))))),
                             hasType(typeContainsString("TempStorage"))))))
                    .bind("DeclStmt"),
                this);

  MF.addMatcher(cxxMemberCallExpr(has(memberExpr(member(hasAnyName(
                                      "InclusiveSum", "ExclusiveSum",
                                      "InclusiveScan", "ExclusiveScan",
                                      "Reduce", "Sum", "Broadcast", "Scan")))))
                    .bind("MemberCall"),
                this);

  MF.addMatcher(
      cxxDependentScopeMemberExpr(
          anyOf(hasMemberName("InclusiveSum"), hasMemberName("ExclusiveSum"),
                hasMemberName("InclusiveScan"), hasMemberName("ExclusiveScan"),
                hasMemberName("Reduce"), hasMemberName("Sum"),
                hasMemberName("Broadcast"), hasMemberName("Scan")))
          .bind("DependentMemberCall"),
      this);

  MF.addMatcher(
      callExpr(allOf(callee(functionDecl(hasAnyName(
                         "ShuffleIndex", "ThreadLoad", "ThreadStore", "Sum",
                         "Reduce", "ExclusiveSum", "InclusiveSum",
                         "InclusiveScan", "ExclusiveScan"))),
                     parentStmtCub()))
          .bind("FuncCall"),
      this);

  MF.addMatcher(
      callExpr(allOf(callee(functionDecl(hasAnyName(
                         "Sum", "Reduce", "ThreadLoad", "ShuffleIndex",
                         "ExclusiveSum", "InclusiveSum", "InclusiveScan",
                         "ExclusiveScan"))),
                     unless(parentStmtCub())))
          .bind("FuncCallUsed"),
      this);
}

std::string CubRule::getOpRepl(const Expr *Operator) {
  std::string OpRepl;
  if (!Operator) {
    return MapNames::getClNamespace() + "plus<>()";
  }

  auto processOperatorExpr = [&](const Expr *Obj) {
    std::string OpType = DpctGlobalInfo::getUnqualifiedTypeName(
        Obj->getType().getCanonicalType());
    if (OpType == "cub::Sum" || OpType == "cuda::std::plus<void>") {
      OpRepl = MapNames::getClNamespace() + "plus<>()";
    } else if (OpType == "cub::Max") {
      OpRepl = MapNames::getClNamespace() + "maximum<>()";
    } else if (OpType == "cub::Min") {
      OpRepl = MapNames::getClNamespace() + "minimum<>()";
    }
  };

  if (auto Op = dyn_cast<CXXConstructExpr>(Operator)) {
    if (auto CXXTempObj = dyn_cast<CXXTemporaryObjectExpr>(Op)) {
      processOperatorExpr(CXXTempObj);
    } else {
      auto CtorArg = Op->getArg(0)->IgnoreImplicitAsWritten();
      if (auto DRE = dyn_cast<DeclRefExpr>(CtorArg)) {
        auto D = DRE->getDecl();
        if (!D)
          return OpRepl;
        std::string OpType = DpctGlobalInfo::getUnqualifiedTypeName(
            D->getType().getCanonicalType());
        if (OpType == "cub::Sum" || OpType == "cub::Max" ||
            OpType == "cub::Min" || OpType == "cuda::std::plus<void>") {
          ExprAnalysis EA(Operator);
          OpRepl = EA.getReplacedString();
        }
      } else if (auto CXXTempObj = dyn_cast<CXXTemporaryObjectExpr>(CtorArg)) {
        processOperatorExpr(CXXTempObj);
      } else if (const Expr *Inner = CtorArg->IgnoreCasts();
                 isa<CXXFunctionalCastExpr>(CtorArg) &&
                 isa<InitListExpr>(Inner)) {
        processOperatorExpr(Inner);
      }
    }
  } else if (const Expr *Inner = Operator->IgnoreCasts();
             isa<CXXFunctionalCastExpr>(Operator) && isa<InitListExpr>(Inner)) {
    processOperatorExpr(Inner);
  }
  return OpRepl;
}
void CubRule::processCubDeclStmt(const DeclStmt *DS) {
  std::string Repl;
  for (auto Decl : DS->decls()) {
    auto VDecl = dyn_cast<VarDecl>(Decl);
    if (!VDecl)
      return;
    std::string VarType =
        VDecl->getTypeSourceInfo()->getType().getCanonicalType().getAsString();
    std::string VarName = VDecl->getNameAsString();
    bool isUnion = VDecl->getType()->isUnionType();
    auto MatcherScope = DpctGlobalInfo::findAncestor<CompoundStmt>(Decl);
    if (!isCubVar(VDecl)) {
      if (isUnion) {
        const TagDecl *RD =
            VDecl->getType()->getAsUnionType()->getDecl()->getCanonicalDecl();
        for (const auto *D : RD->decls())
          if (const auto *FD = dyn_cast<FieldDecl>(D))
            if (isCubTempStorageType(FD->getType()))
              emplaceTransformation(new ReplaceDecl(FD, ""));
      }
      return;
    }

    if (isUnion) {
      const TagDecl *RD =
          VDecl->getType()->getAsUnionType()->getDecl()->getCanonicalDecl();
      emplaceTransformation(new ReplaceDecl(RD, ""));
    }

    // always remove TempStorage variable declaration
    emplaceTransformation(new ReplaceStmt(DS, ""));

    // process TempStorage used in class constructor
    auto TempVarMatcher = compoundStmt(forEachDescendant(
        declRefExpr(to(varDecl(hasName(VarName)))).bind("TempVar")));
    auto MatchResult = ast_matchers::match(TempVarMatcher, *MatcherScope,
                                           DpctGlobalInfo::getContext());
    for (auto &Element : MatchResult) {
      auto DRE = Element.getNodeAs<DeclRefExpr>("TempVar");
      auto ObjDecl = DpctGlobalInfo::findAncestor<VarDecl>(DRE);
      if (!ObjDecl || !DRE) {
        continue;
      }
      auto ArrayOfDRE = DpctGlobalInfo::findAncestor<ArraySubscriptExpr>(DRE);
      auto ObjCanonicalType = ObjDecl->getType().getCanonicalType();
      std::string ObjTypeStr = ObjCanonicalType.getAsString();
      if (isTypeInAnalysisScope(ObjCanonicalType.getTypePtr())) {
        continue;
      } else if (ObjTypeStr.find("class cub::WarpScan") == 0 ||
                 ObjTypeStr.find("class cub::WarpReduce") == 0) {
        Repl = DpctGlobalInfo::getSubGroup(DRE);
      } else if (ObjTypeStr.find("class cub::BlockScan") == 0 ||
                 ObjTypeStr.find("class cub::BlockReduce") == 0) {
        Repl = DpctGlobalInfo::getGroup(DRE);
      } else {
        continue;
      }
      if (ArrayOfDRE) {
        emplaceTransformation(new ReplaceStmt(ArrayOfDRE, Repl));
      } else {
        emplaceTransformation(new ReplaceStmt(DRE, Repl));
      }
    }
  }
}
void CubRule::processCubTypeDef(const TypedefDecl *TD) {
  auto CanonicalType = TD->getUnderlyingType().getCanonicalType();
  std::string CanonicalTypeStr = CanonicalType.getAsString();
  if (isTypeInAnalysisScope(CanonicalType.getTypePtr()))
    return;
  if (!isCubCollectiveRecordType(TD->getUnderlyingType().getDesugaredType(
          DpctGlobalInfo::getContext())) &&
      CanonicalTypeStr.find("class cub::") != 0)
    return;

  std::string TypeName = TD->getNameAsString();
  auto &Context = dpct::DpctGlobalInfo::getContext();
  auto &SM = dpct::DpctGlobalInfo::getSourceManager();
  auto MyMatcher = compoundStmt(forEachDescendant(
      typeLoc(loc(qualType(hasDeclaration(typedefDecl(hasName(TypeName))))))
          .bind("typeLoc")));
  auto MatcherScope = DpctGlobalInfo::findAncestor<CompoundStmt>(TD);
  if (!MatcherScope)
    return;
  auto TypeLocMatchResult =
      ast_matchers::match(MyMatcher, *MatcherScope, Context);
  bool DeleteFlag = true;
  // Currently, typedef decl can be deleted in following cases
  for (auto &Element : TypeLocMatchResult) {
    if (auto TL = Element.getNodeAs<TypeLoc>("typeLoc")) {
      // 1. Used in TempStorage variable declaration
      if (auto AncestorVD = DpctGlobalInfo::findAncestor<VarDecl>(TL)) {
        auto VarType = AncestorVD->getType().getCanonicalType();
        std::string VarTypeStr =
            AncestorVD->getType().getCanonicalType().getAsString();
        if (isTypeInAnalysisScope(VarType.getTypePtr()) ||
            !(VarTypeStr.find("TempStorage") != std::string::npos &&
              VarTypeStr.find("struct cub::") == 0)) {
          DeleteFlag = false;
          break;
        }
      } // 2. Used in temporary class constructor
      else if (auto AncestorMTE =
                   DpctGlobalInfo::findAncestor<MaterializeTemporaryExpr>(TL)) {
        auto MC = DpctGlobalInfo::findAncestor<CXXMemberCallExpr>(AncestorMTE);
        if (MC) {
          auto ObjType = MC->getObjectType().getCanonicalType();
          std::string ObjTypeStr = ObjType.getAsString();
          if (isTypeInAnalysisScope(ObjType.getTypePtr()) ||
              !(ObjTypeStr.find("class cub::WarpScan") == 0 ||
                ObjTypeStr.find("class cub::WarpReduce") == 0 ||
                ObjTypeStr.find("class cub::BlockScan") == 0 ||
                ObjTypeStr.find("class cub::BlockReduce") == 0)) {
            DeleteFlag = false;
            break;
          }
        }
      } // 3. Used in self typedef decl
      else if (auto AncestorTD =
                   DpctGlobalInfo::findAncestor<TypedefDecl>(TL)) {
        if (AncestorTD != TD) {
          DeleteFlag = false;
          break;
        }
      } else if (auto *FD = DpctGlobalInfo::findAncestor<FieldDecl>(TL)) {
        if (!isCubTempStorageType(FD->getType())) {
          DeleteFlag = false;
          break;
        }
      } else {
        DeleteFlag = false;
        break;
      }
    }
  }
  if (DeleteFlag) {
    emplaceTransformation(new ReplaceDecl(TD, ""));
  } else {
    auto BeginLoc =
        SM.getExpansionLoc(TD->getTypeSourceInfo()->getTypeLoc().getBeginLoc());
    auto EndLoc =
        SM.getExpansionLoc(TD->getTypeSourceInfo()->getTypeLoc().getEndLoc());
    if (CanonicalTypeStr.find("Warp") != std::string::npos) {
      emplaceTransformation(
          replaceText(BeginLoc, EndLoc.getLocWithOffset(1),
                      MapNames::getClNamespace() + "sub_group", SM));
    } else if (CanonicalTypeStr.find("Block") != std::string::npos) {
      auto DeviceFuncDecl = DpctGlobalInfo::findAncestor<FunctionDecl>(TD);
      if (DeviceFuncDecl && (DeviceFuncDecl->hasAttr<CUDADeviceAttr>() ||
                             DeviceFuncDecl->hasAttr<CUDAGlobalAttr>())) {
        if (auto DI = DeviceFunctionDecl::LinkRedecls(DeviceFuncDecl)) {
          auto &Map = DpctGlobalInfo::getInstance().getCubPlaceholderIndexMap();
          Map.insert({PlaceholderIndex, DI});
          emplaceTransformation(
              replaceText(BeginLoc, EndLoc.getLocWithOffset(1),
                          MapNames::getClNamespace() + "group<{{NEEDREPLACEC" +
                              std::to_string(PlaceholderIndex++) + "}}>",
                          SM));
        }
      }
    }
  }
}

void CubRule::processThreadLevelFuncCall(const CallExpr *CE,
                                         bool FuncCallUsed) {
  std::string Repl;
  auto DC = CE->getDirectCallee();
  std::string FuncName = DC->getNameAsString();
  if (FuncName == "ThreadLoad") {
    auto FuncArgs = CE->getArgs();
    const Expr *InData = FuncArgs[0];
    ExprAnalysis InEA(InData);
    Repl = "*(" + InEA.getReplacedString() + ")";
    emplaceTransformation(new ReplaceStmt(CE, Repl));
  } else if (FuncName == "ThreadStore") {
    auto FuncArgs = CE->getArgs();
    const Expr *OutputIterator = FuncArgs[0];
    const Expr *Value = FuncArgs[1];
    ExprAnalysis ItrEA(OutputIterator);
    ExprAnalysis ValueEA(Value);
    Repl =
        "*(" + ItrEA.getReplacedString() + ") = " + ValueEA.getReplacedString();
    emplaceTransformation(new ReplaceStmt(CE, Repl));
  }
}

static std::string GetFunctionName(const CallExpr *CE) {
  std::string s;
  llvm::raw_string_ostream OS(s);
  if (isa<CXXMemberCallExpr>(CE)) {
    CE->getDirectCallee()->getNameForDiagnostic(
      OS, DpctGlobalInfo::getContext().getLangOpts(), /*Qualified=*/true);
  } else {
    OS << "cub::" << CE->getDirectCallee()->getName();
  }

  OS << '(';
  for (unsigned I = 0, E = CE->getNumArgs(); I != E; ++I) {
    auto *Arg = CE->getArg(I);
    Arg->getType().print(OS, DpctGlobalInfo::getContext().getLangOpts());
    if (I < E - 1)
      OS << ", ";
  }
  OS << ')';
  return s;
}

void CubRule::processWarpLevelFuncCall(const CallExpr *CE, bool FuncCallUsed) {
  std::string Repl;
  size_t WarpSize = 32;
  auto DC = CE->getDirectCallee();
  std::string FuncName = DC->getNameAsString();
  if (FuncName == "ShuffleIndex") {
    auto TA = DC->getTemplateSpecializationArgs();
    if (!TA)
      return;
    WarpSize = TA->get(0).getAsIntegral().getExtValue();
    std::string ValueType =
        TA->get(1).getAsType().getUnqualifiedType().getAsString();
    const auto *MemberMask = CE->getArg(2);
    const auto *Mask = dyn_cast<IntegerLiteral>(MemberMask);
    const Expr *Value = CE->getArg(0);
    const Expr *Lane = CE->getArg(1);
    const auto *DeviceFuncDecl = getImmediateOuterFuncDecl(CE);
    ExprAnalysis ValueEA(Value);
    ExprAnalysis LaneEA(Lane);
    llvm::raw_string_ostream OS(Repl);
    if (Mask) {
      if (Mask->getValue().getZExtValue() == 0xffffffff) {
        OS << MapNames::getDpctNamespace() << "select_from_sub_group("
           << DpctGlobalInfo::getSubGroup(CE, DeviceFuncDecl) << ", "
           << ValueEA.getReplacedString() << ", " << LaneEA.getReplacedString();
        if (WarpSize != 32)
          OS << ", " << WarpSize;
        OS << ')';
      } else {
        OS << MapNames::getDpctNamespace() << "experimental::"
           << "select_from_sub_group(" << getStmtSpelling(Mask) << ", "
           << DpctGlobalInfo::getSubGroup(CE, DeviceFuncDecl) << ", "
           << ValueEA.getReplacedString() << ", " << LaneEA.getReplacedString();
        if (WarpSize != 32)
          OS << ", " << WarpSize;
        OS << ')';
      }
      emplaceTransformation(new ReplaceStmt(CE, Repl));
    } else {
      report(CE->getBeginLoc(), Diagnostics::API_NOT_MIGRATED, false,
             GetFunctionName(CE));
    }
  }
}

void CubRule::processCubFuncCall(const CallExpr *CE, bool FuncCallUsed) {
  const auto *DC = CE->getDirectCallee();
  if (!DC)
    return;

  llvm::StringRef FuncName = DC->getName();
  if (FuncName == "ShuffleIndex") {
    processWarpLevelFuncCall(CE, FuncCallUsed);
  } else if (FuncName == "ThreadLoad" || FuncName == "ThreadStore") {
    processThreadLevelFuncCall(CE, FuncCallUsed);
  }
}

void CubRule::processBlockLevelMemberCall(const CXXMemberCallExpr *BlockMC) {
  if (!BlockMC || !BlockMC->getMethodDecl()) {
    return;
  }
  std::string Repl, NewFuncName, ParamList, InitRepl, OpRepl, Indent,
      GroupOrWorkitem, AggregateOrCallback;
  ParamAssembler CubParamAs(ParamList);
  std::string FuncName = BlockMC->getMethodDecl()->getNameAsString();
  std::string ValueType;
  int NumArgs = BlockMC->getNumArgs();
  auto FuncArgs = BlockMC->getArgs();
  auto MD = BlockMC->getMethodDecl()->getParent();
  if (auto CTS = dyn_cast<ClassTemplateSpecializationDecl>(MD)) {
    auto &TA = CTS->getTemplateArgs();
    ValueType = TA[0].getAsType().getUnqualifiedType().getAsString();
  }
  Indent = getIndent(BlockMC->getBeginLoc(), DpctGlobalInfo::getSourceManager())
               .str();
  if (BlockMC->getObjectType()->getTypeClass() ==
      Type::TypeClass::SubstTemplateTypeParm) {
    auto DRE =
        dyn_cast_or_null<DeclRefExpr>(BlockMC->getImplicitObjectArgument());
    if (DRE) {
      GroupOrWorkitem = DRE->getNameInfo().getAsString();
    }
  }
  if (GroupOrWorkitem.empty()) {
    GroupOrWorkitem = DpctGlobalInfo::getGroup(BlockMC);
  }
  if (FuncName == "InclusiveSum" || FuncName == "ExclusiveSum" ||
      FuncName == "InclusiveScan" || FuncName == "ExclusiveScan") {
    const Expr *InData = FuncArgs[0];
    const Expr *OutData = FuncArgs[1];
    ExprAnalysis InEA(InData);
    ExprAnalysis OutEA(OutData);
    bool IsReferenceOutput = false;
    if (FuncName == "ExclusiveScan") {
      if (NumArgs == 4) {
        if (BlockMC->getMethodDecl()
                ->getParamDecl(0)
                ->getType()
                ->isLValueReferenceType()) {
          if (BlockMC->getMethodDecl()->getPrimaryTemplate() &&
              BlockMC->getMethodDecl()
                      ->getPrimaryTemplate()
                      ->getTemplateParameters()
                      ->size() == 2) {
            ExprAnalysis InitEA(FuncArgs[2]);
            InitRepl = InitEA.getReplacedString();
            GroupOrWorkitem = DpctGlobalInfo::getItem(BlockMC);
            OpRepl = getOpRepl(FuncArgs[3]);
            NewFuncName =
                MapNames::getDpctNamespace() + "group::exclusive_scan";
            requestFeature(HelperFeatureEnum::device_ext);
            DpctGlobalInfo::getInstance().insertHeader(BlockMC->getBeginLoc(),
                                                       HT_DPCT_DPL_Utils);
            IsReferenceOutput = true;
          } else {
            report(BlockMC->getBeginLoc(), Diagnostics::API_NOT_MIGRATED, false,
                   GetFunctionName(BlockMC));
            return;
          }
        } else {
          if (BlockMC->getMethodDecl()
                  ->getParamDecl(0)
                  ->getType()
                  .getAsString() == BlockMC->getMethodDecl()
                                        ->getParamDecl(2)
                                        ->getType()
                                        .getAsString()) {
            ExprAnalysis InitEA(FuncArgs[2]);
            InitRepl = InitEA.getReplacedString();
            OpRepl = getOpRepl(FuncArgs[3]);
            NewFuncName =
                MapNames::getClNamespace() + "exclusive_scan_over_group";
          } else {
            ExprAnalysis AggregateOrCallbackEA(FuncArgs[3]);
            GroupOrWorkitem = DpctGlobalInfo::getItem(BlockMC);
            AggregateOrCallback = AggregateOrCallbackEA.getReplacedString();
            OpRepl = getOpRepl(FuncArgs[2]);
            NewFuncName =
                MapNames::getDpctNamespace() + "group::exclusive_scan";
            requestFeature(HelperFeatureEnum::device_ext);
            DpctGlobalInfo::getInstance().insertHeader(BlockMC->getBeginLoc(),
                                                       HT_DPCT_DPL_Utils);
          }
        }
      } else if (NumArgs == 5) {
        if (!BlockMC->getMethodDecl()
                 ->getParamDecl(0)
                 ->getType()
                 ->isLValueReferenceType()) {
          GroupOrWorkitem = DpctGlobalInfo::getItem(BlockMC);
          ExprAnalysis InitEA(FuncArgs[2]);
          ExprAnalysis AggregateOrCallbackEA(FuncArgs[4]);
          InitRepl = InitEA.getReplacedString();
          OpRepl = getOpRepl(FuncArgs[3]);
          AggregateOrCallback = AggregateOrCallbackEA.getReplacedString();
          NewFuncName = MapNames::getDpctNamespace() + "group::exclusive_scan";
          requestFeature(HelperFeatureEnum::device_ext);
          DpctGlobalInfo::getInstance().insertHeader(BlockMC->getBeginLoc(),
                                                     HT_DPCT_DPL_Utils);
        } else {
          report(BlockMC->getBeginLoc(), Diagnostics::API_NOT_MIGRATED, false,
                 GetFunctionName(BlockMC));
          return;
        }
      }
    } else if (FuncName == "InclusiveScan") {
      if (NumArgs == 3) {
        if (BlockMC->getMethodDecl()
                ->getParamDecl(0)
                ->getType()
                ->isLValueReferenceType()) {
          GroupOrWorkitem = DpctGlobalInfo::getItem(BlockMC);
          OpRepl = getOpRepl(FuncArgs[2]);
          NewFuncName = MapNames::getDpctNamespace() + "group::inclusive_scan";
          requestFeature(HelperFeatureEnum::device_ext);
          DpctGlobalInfo::getInstance().insertHeader(BlockMC->getBeginLoc(),
                                                     HT_DPCT_DPL_Utils);
          IsReferenceOutput = true;
        } else {
          OpRepl = getOpRepl(FuncArgs[2]);
          NewFuncName =
              MapNames::getClNamespace() + "inclusive_scan_over_group";
        }
      } else if (NumArgs == 4) {
        if (BlockMC->getMethodDecl()
                ->getParamDecl(0)
                ->getType()
                ->isLValueReferenceType()) {
          report(BlockMC->getBeginLoc(), Diagnostics::API_NOT_MIGRATED, false,
                 GetFunctionName(BlockMC));
          return;
        }
        GroupOrWorkitem = DpctGlobalInfo::getItem(BlockMC);
        OpRepl = getOpRepl(FuncArgs[2]);
        ExprAnalysis AggregateOrCallbackEA(FuncArgs[3]);
        AggregateOrCallback = AggregateOrCallbackEA.getReplacedString();
        NewFuncName = MapNames::getDpctNamespace() + "group::inclusive_scan";
        requestFeature(HelperFeatureEnum::device_ext);
        DpctGlobalInfo::getInstance().insertHeader(BlockMC->getBeginLoc(),
                                                   HT_DPCT_DPL_Utils);
      }
    } else if (FuncName == "ExclusiveSum") {
      if (NumArgs == 2) {
        OpRepl = getOpRepl(nullptr);
        InitRepl = "0";
        if (BlockMC->getMethodDecl()
                ->getParamDecl(0)
                ->getType()
                ->isLValueReferenceType()) {
          NewFuncName = MapNames::getDpctNamespace() + "group::exclusive_scan";
          requestFeature(HelperFeatureEnum::device_ext);
          DpctGlobalInfo::getInstance().insertHeader(BlockMC->getBeginLoc(),
                                                     HT_DPCT_DPL_Utils);
          GroupOrWorkitem = DpctGlobalInfo::getItem(BlockMC);
          IsReferenceOutput = true;
        } else {
          NewFuncName =
              MapNames::getClNamespace() + "exclusive_scan_over_group";
        }
      } else if (NumArgs == 3) {
        if (!BlockMC->getMethodDecl()
                 ->getParamDecl(0)
                 ->getType()
                 ->isLValueReferenceType()) {
          GroupOrWorkitem = DpctGlobalInfo::getItem(BlockMC);
          OpRepl = getOpRepl(nullptr);
          if (BlockMC->getMethodDecl()
                  ->getParamDecl(1)
                  ->getType()
                  .getAsString() == BlockMC->getMethodDecl()
                                        ->getParamDecl(2)
                                        ->getType()
                                        .getAsString()) {
            InitRepl = "0";
          }
          ExprAnalysis AggregateOrCallbackEA(FuncArgs[2]);
          AggregateOrCallback = AggregateOrCallbackEA.getReplacedString();
          NewFuncName = MapNames::getDpctNamespace() + "group::exclusive_scan";
          requestFeature(HelperFeatureEnum::device_ext);
          DpctGlobalInfo::getInstance().insertHeader(BlockMC->getBeginLoc(),
                                                     HT_DPCT_DPL_Utils);
        } else {
          report(BlockMC->getBeginLoc(), Diagnostics::API_NOT_MIGRATED, false,
                 GetFunctionName(BlockMC));
          return;
        }
      }
    } else if (FuncName == "InclusiveSum") {
      if (NumArgs == 2) {
        if (BlockMC->getMethodDecl()
                ->getParamDecl(0)
                ->getType()
                ->isLValueReferenceType()) {
          GroupOrWorkitem = DpctGlobalInfo::getItem(BlockMC);
          OpRepl = getOpRepl(nullptr);
          NewFuncName = MapNames::getDpctNamespace() + "group::inclusive_scan";
          requestFeature(HelperFeatureEnum::device_ext);
          DpctGlobalInfo::getInstance().insertHeader(BlockMC->getBeginLoc(),
                                                     HT_DPCT_DPL_Utils);
          IsReferenceOutput = true;
        } else {
          OpRepl = getOpRepl(nullptr);
          NewFuncName =
              MapNames::getClNamespace() + "inclusive_scan_over_group";
        }
      } else if (NumArgs == 3) {
        if (BlockMC->getMethodDecl()
                ->getParamDecl(0)
                ->getType()
                ->isLValueReferenceType()) {
          report(BlockMC->getBeginLoc(), Diagnostics::API_NOT_MIGRATED, false,
                 GetFunctionName(BlockMC));
          return;
        }
        GroupOrWorkitem = DpctGlobalInfo::getItem(BlockMC);
        OpRepl = getOpRepl(nullptr);
        ExprAnalysis AggregateOrCallbackEA(FuncArgs[2]);
        AggregateOrCallback = AggregateOrCallbackEA.getReplacedString();
        NewFuncName = MapNames::getDpctNamespace() + "group::inclusive_scan";
        requestFeature(HelperFeatureEnum::device_ext);
        DpctGlobalInfo::getInstance().insertHeader(BlockMC->getBeginLoc(),
                                                   HT_DPCT_DPL_Utils);
      }
    } else {
      report(BlockMC->getBeginLoc(), Diagnostics::API_NOT_MIGRATED, false,
             GetFunctionName(BlockMC));
      return;
    }
    if (IsReferenceOutput) {
      CubParamAs << GroupOrWorkitem << InEA.getReplacedString()
                 << OutEA.getReplacedString() << InitRepl << OpRepl
                 << AggregateOrCallback;
      Repl = NewFuncName + "(" + ParamList + ")";
    } else {
      CubParamAs << GroupOrWorkitem << InEA.getReplacedString() << InitRepl
                 << OpRepl << AggregateOrCallback;
      Repl = OutEA.getReplacedString() + " = " + NewFuncName + "(" + ParamList +
             ")";
    }
    emplaceTransformation(new ReplaceStmt(BlockMC, Repl));
  } else if (FuncName == "Sum" || FuncName == "Reduce") {
    if (BlockMC->getMethodDecl()
            ->getParamDecl(0)
            ->getType()
            ->isLValueReferenceType()) {
      GroupOrWorkitem = DpctGlobalInfo::getItem(BlockMC);
      NewFuncName = MapNames::getDpctNamespace() + "group::reduce";
      requestFeature(HelperFeatureEnum::device_ext);
      DpctGlobalInfo::getInstance().insertHeader(BlockMC->getBeginLoc(),
                                                 HT_DPCT_DPL_Utils);
    } else {
      NewFuncName = MapNames::getClNamespace() + "reduce_over_group";
    }
    const Expr *InData = FuncArgs[0];
    ExprAnalysis InEA(InData);
    bool IsPartialReduce = false;
    unsigned ValidItemParamIdx = 0;
    if (FuncName == "Reduce") {
      OpRepl = getOpRepl(FuncArgs[1]);
      IsPartialReduce = NumArgs == 3;
      ValidItemParamIdx = 2;
      const auto *CK = dyn_cast<ImplicitCastExpr>(FuncArgs[1]);
      if (DpctGlobalInfo::useUserDefineReductions() && OpRepl.empty() && CK &&
          CK->getCastKind() == CK_FunctionToPointerDecay) {
        ExprAnalysis EA;
        EA.analyze(CK);
        OpRepl =
            "[](auto&& x, auto&& y) { return " + EA.getReplacedString() +
            "(std::forward<decltype(x)>(x), std::forward<decltype(y)>(y)); }";

        NewFuncName = MapNames::getClNamespace() +
                      "ext::oneapi::experimental::reduce_over_group";
        auto [TempStorage, DataTypeLoc] =
            getTempstorageVarAndValueTypeLoc(BlockMC);
        auto *FD = DpctGlobalInfo::findAncestor<FunctionDecl>(TempStorage);
        if (!FD || !TempStorage || DataTypeLoc.isNull())
          return;
        if (auto FuncInfo = DeviceFunctionDecl::LinkRedecls(FD)) {
          auto LocInfo = DpctGlobalInfo::getLocInfo(TempStorage);
          ExprAnalysis EA;
          EA.analyze(DataTypeLoc);
          FuncInfo->getVarMap().addCUBTempStorage(
              std::make_shared<TempStorageVarInfo>(
                  LocInfo.second, TempStorageVarInfo::BlockReduce,
                  TempStorage->getName(), "",
                  EA.getTemplateDependentStringInfo()));
        }
        std::string Span = MapNames::getClNamespace() + "span<std::byte, 1>" +
                           "(&" + TempStorage->getNameAsString() + "[0], " +
                           TempStorage->getNameAsString() + ".size())";
        GroupOrWorkitem = MapNames::getClNamespace() +
                          "ext::oneapi::experimental::group_with_scratchpad(" +
                          GroupOrWorkitem + ", " + Span + ")";
      }
    } else if (FuncName == "Sum") {
      OpRepl = getOpRepl(nullptr);
      IsPartialReduce = NumArgs == 2;
      ValidItemParamIdx = 1;
    } else {
      report(BlockMC->getBeginLoc(), Diagnostics::API_NOT_MIGRATED, false,
             GetFunctionName(BlockMC));
      return;
    }
    std::string In;
    if (IsPartialReduce) {
      std::string tmp;
      llvm::raw_string_ostream OS(tmp);
      ExprAnalysis ValidItemsEA(BlockMC->getArg(ValidItemParamIdx));
      ValidItemsEA.analyze();
      OS << '(' << GroupOrWorkitem << ".get_local_linear_id() < "
         << ValidItemsEA.getReplacedString() << ") ? "
         << InEA.getReplacedString() << " : " << MapNames::getClNamespace()
         << "known_identity_v<" << StringRef(OpRepl).drop_back(2) << ", "
         << DpctGlobalInfo::getTypeName(InData->getType()) << ">";
      In = std::move(tmp);
    } else
      In = InEA.getReplacedString();
    CubParamAs << GroupOrWorkitem << In << OpRepl;
    Repl = NewFuncName + "(" + ParamList + ")";
    emplaceTransformation(new ReplaceStmt(BlockMC, Repl));
  }
}

void CubRule::processWarpLevelMemberCall(const CXXMemberCallExpr *WarpMC) {
  if (!WarpMC || !WarpMC->getMethodDecl()) {
    return;
  }
  size_t WarpSize = 32;
  std::string Repl, NewFuncName, ParamList, InitRepl, OpRepl, Indent,
      GroupOrWorkitem, AggregateOrCallback;
  ParamAssembler CubParamAs(ParamList);
  std::string FuncName = WarpMC->getMethodDecl()->getNameAsString();
  std::string ValueType;
  int NumArgs = WarpMC->getNumArgs();
  auto MD = WarpMC->getMethodDecl()->getParent();
  if (auto CTS = dyn_cast<ClassTemplateSpecializationDecl>(MD)) {
    auto &TA = CTS->getTemplateArgs();
    ValueType = TA[0].getAsType().getUnqualifiedType().getAsString();
    WarpSize = TA[1].getAsIntegral().getExtValue();
  }
  Indent = getIndent(WarpMC->getBeginLoc(), DpctGlobalInfo::getSourceManager())
               .str();
  auto FD = DpctGlobalInfo::getParentFunction(WarpMC);
  if (WarpMC->getObjectType()->getTypeClass() ==
      Type::TypeClass::SubstTemplateTypeParm) {
    auto DRE =
        dyn_cast_or_null<DeclRefExpr>(WarpMC->getImplicitObjectArgument());
    if (DRE) {
      GroupOrWorkitem = DRE->getNameInfo().getAsString();
    }
  }
  if (GroupOrWorkitem.empty()) {
    GroupOrWorkitem = DpctGlobalInfo::getSubGroup(WarpMC, FD);
  }
  if (FuncName == "InclusiveSum" || FuncName == "ExclusiveSum" ||
      FuncName == "InclusiveScan" || FuncName == "ExclusiveScan") {
    auto FuncArgs = WarpMC->getArgs();
    const Expr *InData = FuncArgs[0];
    const Expr *OutData = FuncArgs[1];
    if (FuncName == "ExclusiveScan") {
      if (NumArgs == 3) {
        OpRepl = getOpRepl(FuncArgs[2]);
      } else if (NumArgs == 4 &&
                 WarpMC->getMethodDecl()
                         ->getParamDecl(0)
                         ->getType()
                         .getCanonicalType()
                         .getAsString() == WarpMC->getMethodDecl()
                                               ->getParamDecl(2)
                                               ->getType()
                                               .getCanonicalType()
                                               .getAsString()) {
        ExprAnalysis InitEA(FuncArgs[2]);
        InitRepl = ", " + InitEA.getReplacedString();
        OpRepl = getOpRepl(FuncArgs[3]);
      } else {
        report(WarpMC->getBeginLoc(), Diagnostics::API_NOT_MIGRATED, false,
               GetFunctionName(WarpMC));
        return;
      }
      NewFuncName = "exclusive_scan_over_group";
    } else if (FuncName == "InclusiveScan" && NumArgs == 3) {
      OpRepl = getOpRepl(FuncArgs[2]);
      NewFuncName = "inclusive_scan_over_group";
    } else if (FuncName == "ExclusiveSum" && NumArgs == 2) {
      OpRepl = getOpRepl(nullptr);
      NewFuncName = "exclusive_scan_over_group";
    } else if (FuncName == "InclusiveSum" && NumArgs == 2) {
      OpRepl = getOpRepl(nullptr);
      NewFuncName = "inclusive_scan_over_group";
    } else {
      report(WarpMC->getBeginLoc(), Diagnostics::API_NOT_MIGRATED, false,
             GetFunctionName(WarpMC));
      return;
    }
    ExprAnalysis InEA(InData);
    ExprAnalysis OutEA(OutData);
    Repl = OutEA.getReplacedString() + " = " + MapNames::getClNamespace() +
           NewFuncName + "(" + GroupOrWorkitem + ", " +
           InEA.getReplacedString() + InitRepl + ", " + OpRepl + ")";
    emplaceTransformation(new ReplaceStmt(WarpMC, Repl));
  } else if (FuncName == "Broadcast") {
    auto FuncArgs = WarpMC->getArgs();
    const Expr *InData = FuncArgs[0];
    const Expr *SrcLane = FuncArgs[1];
    ExprAnalysis InEA(InData);
    ExprAnalysis SrcLaneEA(SrcLane);
    Repl = MapNames::getClNamespace() + "group_broadcast(" +
           DpctGlobalInfo::getSubGroup(WarpMC) + ", " +
           InEA.getReplacedString() + ", " + SrcLaneEA.getReplacedString() +
           ")";
    NewFuncName = "group_broadcast";
    emplaceTransformation(new ReplaceStmt(WarpMC, Repl));
    analyzeUninitializedDeviceVar(WarpMC, InData);
  } else if (FuncName == "Reduce") {
    ExprAnalysis InDateEA(WarpMC->getArg(0));
    switch (NumArgs) {
    case 2: {
      OpRepl = getOpRepl(WarpMC->getArg(1));
      Repl = MapNames::getClNamespace() + "reduce_over_group(" +
             GroupOrWorkitem + ", " + InDateEA.getReplacedString() + ", " +
             OpRepl + ")";
      break;
    }
    case 3: {
      DpctGlobalInfo::getInstance().insertHeader(WarpMC->getBeginLoc(),
                                                 HeaderType::HT_DPCT_DPL_Utils);
      GroupOrWorkitem = DpctGlobalInfo::getItem(WarpMC, FD);
      ExprAnalysis ValidItemEA(WarpMC->getArg(2));
      OpRepl = getOpRepl(WarpMC->getArg(1));
      Repl = MapNames::getDpctNamespace() +
             "group::reduce_over_partial_group(" + GroupOrWorkitem + ", " +
             InDateEA.getReplacedString() + ", " +
             ValidItemEA.getReplacedString() + ", " + OpRepl + ")";
      break;
    }
    default:
      report(WarpMC->getBeginLoc(), Diagnostics::API_NOT_MIGRATED, false,
             GetFunctionName(WarpMC));
      return;
    }
    emplaceTransformation(new ReplaceStmt(WarpMC, Repl));
  } else if (FuncName == "Sum") {
    const auto *FuncArgs = WarpMC->getArgs();
    const auto *InData = FuncArgs[0];
    ExprAnalysis InEA(InData);
    if (NumArgs == 1) {
      OpRepl = getOpRepl(nullptr);
      NewFuncName = "reduce_over_group";
      Repl = MapNames::getClNamespace() + "reduce_over_group(" +
             GroupOrWorkitem + ", " + InEA.getReplacedString() + ", " + OpRepl +
             ")";
    } else if (NumArgs == 2) {
      OpRepl = getOpRepl(nullptr);
      DpctGlobalInfo::getInstance().insertHeader(WarpMC->getBeginLoc(),
                                                 HeaderType::HT_DPCT_DPL_Utils);
      GroupOrWorkitem = DpctGlobalInfo::getItem(WarpMC, FD);
      ExprAnalysis ValidItemEA(WarpMC->getArg(1));
      Repl = MapNames::getDpctNamespace() +
             "group::reduce_over_partial_group(" + GroupOrWorkitem + ", " +
             InEA.getReplacedString() + ", " + ValidItemEA.getReplacedString() +
             ", " + OpRepl + ")";
    } else {
      report(WarpMC->getBeginLoc(), Diagnostics::API_NOT_MIGRATED, false,
             GetFunctionName(WarpMC));
      return;
    }
    emplaceTransformation(new ReplaceStmt(WarpMC, Repl));
  }

  if (auto FuncInfo = DeviceFunctionDecl::LinkRedecls(FD)) {
    FuncInfo->addSubGroupSizeRequest(WarpSize, WarpMC->getBeginLoc(),
                                     NewFuncName);
  }
}

void CubRule::processCubMemberCall(const CXXMemberCallExpr *MC) {
  auto ObjType = MC->getObjectType().getCanonicalType();
  std::string ObjTypeStr = ObjType.getAsString();

  if (isTypeInAnalysisScope(ObjType.getTypePtr())) {
    return;
  } else if (ObjTypeStr.find("class cub::WarpScan") == 0 ||
             ObjTypeStr.find("class cub::WarpReduce") == 0) {
    processWarpLevelMemberCall(MC);
  } else if (ObjTypeStr.find("class cub::BlockScan") == 0 ||
             ObjTypeStr.find("class cub::BlockReduce") == 0) {
    processBlockLevelMemberCall(MC);
  } else {
    report(MC->getBeginLoc(), Diagnostics::API_NOT_MIGRATED, false, ObjTypeStr);
    return;
  }
}

void CubRule::processTypeLoc(const TypeLoc *TL) {
  auto TD = DpctGlobalInfo::findAncestor<TypedefDecl>(TL);
  if (TD ||
      isTypeInAnalysisScope(TL->getType().getCanonicalType().getTypePtr()))
    return;
  auto &SM = DpctGlobalInfo::getSourceManager();
  auto Range = getDefinitionRange(TL->getBeginLoc(), TL->getEndLoc());
  auto BeginLoc = Range.getBegin();
  auto EndLoc = Range.getEnd();
  std::string TypeName = TL->getType().getCanonicalType().getAsString();
  if (TypeName.find("class cub::WarpScan") == 0 ||
      TypeName.find("class cub::WarpReduce") == 0) {
    emplaceTransformation(replaceText(BeginLoc, EndLoc.getLocWithOffset(1),
                                      MapNames::getClNamespace() + "sub_group",
                                      SM));
  } else if (TypeName.find("class cub::BlockScan") == 0 ||
             TypeName.find("class cub::BlockReduce") == 0) {
    auto DeviceFuncDecl = DpctGlobalInfo::findAncestor<FunctionDecl>(TL);
    if (DeviceFuncDecl && (DeviceFuncDecl->hasAttr<CUDADeviceAttr>() ||
                           DeviceFuncDecl->hasAttr<CUDAGlobalAttr>())) {
      if (auto DI = DeviceFunctionDecl::LinkRedecls(DeviceFuncDecl)) {
        auto &Map = DpctGlobalInfo::getInstance().getCubPlaceholderIndexMap();
        Map.insert({PlaceholderIndex, DI});
        emplaceTransformation(
            replaceText(BeginLoc, EndLoc.getLocWithOffset(1),
                        MapNames::getClNamespace() + "group<{{NEEDREPLACEC" +
                            std::to_string(PlaceholderIndex++) + "}}>",
                        SM));
      }
    }
  }
}

void CubRule::processDependentMemberCall(
    const CXXDependentScopeMemberExpr *DMC) {
  if (!isCubCollectiveRecordType(DMC->getBaseType())) {
    return;
  }
  if (auto FTD = DpctGlobalInfo::findAncestor<FunctionTemplateDecl>(DMC)) {
    if (FTD->spec_end() == FTD->spec_begin()) {
      auto MemeberName = DMC->getMember().getAsString();
      report(DMC->getBeginLoc(), Diagnostics::NOT_SUPPORTED_PARAMETER, false,
             MemeberName + " member function call",
             "the caller function may not instantiated");
    }
  }
  return;
}

int CubRule::PlaceholderIndex = 1;

void CubRule::runRule(const ast_matchers::MatchFinder::MatchResult &Result) {
  if (const CXXMemberCallExpr *MC =
          getNodeAsType<CXXMemberCallExpr>(Result, "MemberCall")) {
    processCubMemberCall(MC);
  } else if (const CXXDependentScopeMemberExpr *DMC =
                 getNodeAsType<CXXDependentScopeMemberExpr>(
                     Result, "DependentMemberCall")) {
    processDependentMemberCall(DMC);
  } else if (const DeclStmt *DS =
                 getAssistNodeAsType<DeclStmt>(Result, "DeclStmt")) {
    processCubDeclStmt(DS);
  } else if (const CallExpr *CE = getNodeAsType<CallExpr>(Result, "FuncCall")) {
    processCubFuncCall(CE);
  } else if (const CallExpr *CE =
                 getNodeAsType<CallExpr>(Result, "FuncCallUsed")) {
    processCubFuncCall(CE, true);
  } else if (const TypedefDecl *TD =
                 getNodeAsType<TypedefDecl>(Result, "TypeDefDecl")) {
    processCubTypeDef(TD);
  } else if (auto TL = getNodeAsType<TypeLoc>(Result, "cudaTypeDef")) {
    processTypeLoc(TL);
  } else if (auto *UDD = getNodeAsType<UsingDirectiveDecl>(
                 Result, "CubUsingNamespace")) {
    llvm::StringRef NamespaceName = UDD->getNominatedNamespace()->getName();
    if (NamespaceName == "cub") {
      if (const auto *NSD =
              dyn_cast<NamespaceDecl>(UDD->getNominatedNamespace())) {
        if (DpctGlobalInfo::isInCudaPath(NSD->getLocation())) {
          emplaceTransformation(new ReplaceDecl(UDD, ""));
        }
      }
    }
  }
}

REGISTER_RULE(CubRule, PassKind::PK_Analysis)
