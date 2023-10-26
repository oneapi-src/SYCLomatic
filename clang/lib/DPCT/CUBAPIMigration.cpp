//===--------------- CUBAPIMigration.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CUBAPIMigration.h"
#include "ASTTraversal.h"
#include "AnalysisInfo.h"
#include "CallExprRewriter.h"
#include "MigrationRuleManager.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/TypeLoc.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Analysis/AnalysisDeclContext.h"
#include "clang/Basic/CharInfo.h"
#include "clang/Basic/LLVM.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Path.h"
#include <iterator>

using namespace clang;
using namespace dpct;
using namespace tooling;
using namespace ast_matchers;

namespace {
auto parentStmt = []() {
  return anyOf(hasParent(compoundStmt()), hasParent(forStmt()),
               hasParent(whileStmt()), hasParent(doStmt()),
               hasParent(ifStmt()));
};

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
        "SortKeysCopy", "StableSortKeys");
  };
  auto hasDeviceRecordName = []() {
    return hasAnyName("DeviceSegmentedReduce", "DeviceReduce", "DeviceScan",
                      "DeviceSelect", "DeviceRunLengthEncode",
                      "DeviceRadixSort", "DeviceSegmentedRadixSort",
                      "DeviceSegmentedSort", "DeviceHistogram", "DeviceMergeSort");
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

void CubTypeRule::registerMatcher(ast_matchers::MatchFinder &MF) {
  auto TargetTypeName = [&]() {
    return hasAnyName(
        "cub::Sum", "cub::Max", "cub::Min", "cub::Equality",
        "cub::KeyValuePair", "cub::CountingInputIterator",
        "cub::TransformInputIterator", "cub::ConstantInputIterator",
        "cub::ArgIndexInputIterator", "cub::DiscardOutputIterator",
        "cub::DoubleBuffer", "cub::NullType");
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
    EA.analyze(*TL);
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
         OpTypeName.startswith("cub::KeyValuePair");
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
          allOf(on(hasType(hasCanonicalType(qualType(hasDeclaration(
                    namedDecl(hasName("cub::ArgIndexInputIterator"))))))),
                callee(cxxMethodDecl(hasName("normalize")))))
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

void CubMemberCallRule::runRule(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  ExprAnalysis EA;
  if (const auto *E1 = getNodeAsType<CXXMemberCallExpr>(Result, "memberCall")) {
    EA.analyze(E1);
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
              hasAnyName("IADD3", "SHR_ADD", "SHL_ADD", "LaneId", "WarpId",
                         "SyncStream", "CurrentDevice", "DeviceCount",
                         "DeviceCountUncached", "DeviceCountCachedValue"),
              hasAncestor(namespaceDecl(hasName("cub")))))))
          .bind("IntrinsicCall"),
      this);
}

void CubIntrinsicRule::runRule(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  if (const auto *CE = getNodeAsType<CallExpr>(Result, "IntrinsicCall")) {
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
    std::string Key = LocInfo.first + std::to_string(LocInfo.second);
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
          } else {
            auto SubDeclPRInfo = std::make_shared<PriorityReplInfo>();
            SubDeclPRInfo->Repls.emplace_back(
                replaceText(Beg, End.getLocWithOffset(1), "", SM)
                    ->getReplacement(Context));
            DpctGlobalInfo::addPriorityReplInfo(Key, SubDeclPRInfo);
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
        if (IsUsed) {
          Info->Repls.emplace_back(ReplaceStmt(*Itr, "0").getReplacement(
              DpctGlobalInfo::getContext()));
        } else {
          Info->Repls.emplace_back(ReplaceStmt(*Itr, "").getReplacement(
              DpctGlobalInfo::getContext()));
        }
        DpctGlobalInfo::addPriorityReplInfo(
            LocInfo.first + std::to_string(LocInfo.second), Info);
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

  MF.addMatcher(
      declStmt(
          has(varDecl(anyOf(
              hasType(hasCanonicalType(qualType(
                  hasDeclaration(namedDecl(hasAnyName("TempStorage")))))),
              hasType(arrayType(hasElementType(hasCanonicalType(qualType(
                  hasDeclaration(namedDecl(hasAnyName("TempStorage"))))))))))))
          .bind("DeclStmt"),
      this);

  MF.addMatcher(cxxMemberCallExpr(has(memberExpr(member(hasAnyName(
                                      "InclusiveSum", "ExclusiveSum",
                                      "InclusiveScan", "ExclusiveScan",
                                      "Reduce", "Sum", "Broadcast", "Scan")))))
                    .bind("MemberCall"),
                this);

  MF.addMatcher(
      callExpr(allOf(callee(functionDecl(hasAnyName(
                         "ShuffleIndex", "ThreadLoad", "ThreadStore", "Sum",
                         "Reduce", "ExclusiveSum", "InclusiveSum",
                         "InclusiveScan", "ExclusiveScan"))),
                     parentStmt()))
          .bind("FuncCall"),
      this);

  MF.addMatcher(
      callExpr(allOf(callee(functionDecl(hasAnyName(
                         "Sum", "Reduce", "ThreadLoad", "ShuffleIndex",
                         "ExclusiveSum", "InclusiveSum", "InclusiveScan",
                         "ExclusiveScan"))),
                     unless(parentStmt())))
          .bind("FuncCallUsed"),
      this);
}

std::string CubRule::getOpRepl(const Expr *Operator) {
  std::string OpRepl;
  if (!Operator) {
    return MapNames::getClNamespace() + "plus<>()";
  }

  auto processCXXTemporaryObjectExpr =
      [&](const CXXTemporaryObjectExpr *CXXTempObj) {
        std::string OpType = DpctGlobalInfo::getUnqualifiedTypeName(
            CXXTempObj->getType().getCanonicalType());
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
      processCXXTemporaryObjectExpr(CXXTempObj);
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
        processCXXTemporaryObjectExpr(CXXTempObj);
      }
    }
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

    auto MatcherScope = DpctGlobalInfo::findAncestor<CompoundStmt>(Decl);
    if (!isCubVar(VDecl)) {
      return;
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
  
  if (maybeDependentCubType(TD->getTypeSourceInfo())) {
    emplaceTransformation(new ReplaceDecl(TD, ""));
    return;
  }

  if (CanonicalTypeStr.find("class cub::") != 0) {
    return;
  }

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
    auto MemberMask = CE->getArg(2);
    auto Mask = dyn_cast<IntegerLiteral>(MemberMask);
    if (Mask && Mask->getValue().getZExtValue() == 0xffffffff) {
      const Expr *Value = CE->getArg(0);
      const Expr *Lane = CE->getArg(1);
      ExprAnalysis ValueEA(Value);
      ExprAnalysis LaneEA(Lane);
      auto DeviceFuncDecl = getImmediateOuterFuncDecl(CE);
      Repl = DpctGlobalInfo::getSubGroup(CE, DeviceFuncDecl) + ".shuffle(" +
             ValueEA.getReplacedString() + ", " + LaneEA.getReplacedString() +
             ")";
      emplaceTransformation(new ReplaceStmt(CE, Repl));
      if (DeviceFuncDecl) {
        auto DI = DeviceFunctionDecl::LinkRedecls(DeviceFuncDecl);
        if (DI) {
          DI->addSubGroupSizeRequest(WarpSize, CE->getBeginLoc(), "shuffle");
        }
      }
    } else {
      report(CE->getBeginLoc(), Diagnostics::API_NOT_MIGRATED, false,
             "cub::" + FuncName);
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
                   "cub::" + FuncName);
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
                 "cub::" + FuncName);
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
                 "cub::" + FuncName);
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
                 "cub::" + FuncName);
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
                 "cub::" + FuncName);
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
             "cub::" + FuncName);
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
    if (FuncName == "Reduce" && NumArgs == 2) {
      OpRepl = getOpRepl(FuncArgs[1]);
    } else if (FuncName == "Sum" && NumArgs == 1) {
      OpRepl = getOpRepl(nullptr);
    } else {
      report(BlockMC->getBeginLoc(), Diagnostics::API_NOT_MIGRATED, false,
             "cub::" + FuncName);
      return;
    }
    CubParamAs << GroupOrWorkitem << InEA.getReplacedString() << OpRepl;
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
               "cub::" + FuncName);
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
             "cub::" + FuncName);
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
             "cub::" + FuncName);
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
             "cub::" + FuncName);
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

int CubRule::PlaceholderIndex = 1;

void CubRule::runRule(const ast_matchers::MatchFinder::MatchResult &Result) {
  if (const CXXMemberCallExpr *MC =
          getNodeAsType<CXXMemberCallExpr>(Result, "MemberCall")) {
    processCubMemberCall(MC);
  } else if (const DeclStmt *DS = getNodeAsType<DeclStmt>(Result, "DeclStmt")) {
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
  }
}
REGISTER_RULE(CubRule, PassKind::PK_Analysis)
