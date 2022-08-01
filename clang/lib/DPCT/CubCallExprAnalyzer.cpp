//===------------------- CubCallExprAnalyzer.cpp --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CubCallExprAnalyzer.h"
#include "AnalysisInfo.h"
#include "TextModification.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include <memory>

using namespace clang::dpct;

static bool isNullptrOrZero(const clang::Expr *E) {
  if (!E)
    return false;
  if (llvm::isa<clang::GNUNullExpr, clang::CXXNullPtrLiteralExpr>(E))
    return true;
  if (E->isEvaluatable(DpctGlobalInfo::getContext())) {
    clang::Expr::EvalResult Result;
    E->EvaluateAsRValue(Result, DpctGlobalInfo::getContext());
    if (Result.Val.isInt() && Result.Val.getInt() == 0) {
      return true;
    }
  }
  return false;
}

static const clang::Expr *getNonCasExpr(const clang::Expr *E) {
  // There may be multiple type conversion(ImplicitCast/CStyleCast...)
  const clang::Expr *NonCastExpr = E;
  while (NonCastExpr != nullptr && llvm::isa<clang::CastExpr>(NonCastExpr)) {
    NonCastExpr = llvm::dyn_cast<clang::CastExpr>(NonCastExpr)->getSubExpr();
  }
  return NonCastExpr;
}

template <typename NodeType>
static void findLoop(const NodeType *Node,
                     std::vector<const clang::Stmt *> &LoopList,
                     bool OnlyFindFirstLevel = false) {
  const clang::Stmt *Loop = nullptr;
  auto Conditon = [&](const clang::DynTypedNode &LoopNode) -> bool {
    if (const auto *DoLoop = LoopNode.get<clang::DoStmt>()) {
      if (DoLoop->getBody() == Node ||
          DpctGlobalInfo::isAncestor(DoLoop->getBody(), Node)) {
        return true;
      }
    } else if (const auto *ForLoop = LoopNode.get<clang::ForStmt>()) {
      if (ForLoop->getBody() == Node ||
          DpctGlobalInfo::isAncestor(ForLoop->getBody(), Node)) {
        return true;
      }
    } else if (const auto *WhileLoop = LoopNode.get<clang::WhileStmt>()) {
      if (WhileLoop->getBody() == Node ||
          DpctGlobalInfo::isAncestor(WhileLoop->getBody(), Node)) {
        return true;
      }
    }
    return false;
  };
  if (Loop = DpctGlobalInfo::findAncestor<clang::Stmt>(Node, Conditon)) {
    LoopList.push_back(Loop);
    if (!OnlyFindFirstLevel) {
      findLoop(Loop, LoopList);
    }
  }
}

bool CubRedundantCallAnalyzer::isRedundantCallExpr(const CallExpr *CE) {
  const auto *FuncArgs = CE->getArgs();
  const auto *TempStorage = FuncArgs[0]->IgnoreImplicitAsWritten();
  SourceLocation InitLoc;
  std::vector<const DeclRefExpr *> TempStorageMatchResult;
  std::vector<const DeclRefExpr *> TempStorageSizeMatchResult;
  const Expr *NonCastExpr = getNonCasExpr(TempStorage);

  // Check the 1st arg of cub device api, if it's a null pointer or
  // zero(0/NULL/nullptr), this call expression used to calculate temp_storage
  // size, so it's a redundant expression.
  if (isNullptrOrZero(NonCastExpr))
    return true;

  if (!NonCastExpr || !llvm::isa<DeclRefExpr>(NonCastExpr))
    return false;

  const auto *DRE = dyn_cast<DeclRefExpr>(NonCastExpr);
  const auto *VD = dyn_cast<VarDecl>(DRE->getDecl());
  if (!DRE || !VD)
    return false;
  const Expr *Init = nullptr;
  SourceLocation SearchEndLoc =
      DpctGlobalInfo::getSourceManager().getExpansionLoc(DRE->getBeginLoc());
  SourceLocation LastModifiedLoc;
  std::vector<const Stmt *> DRELoopList;
  std::vector<const Stmt *> CELoopList;
  findAllVarRef(DRE, TempStorageMatchResult);
  findLoop(CE, CELoopList);
  if (VD->hasInit()) {
    // tempstorage = nullptr/NULL/0/...
    if (VD->getInitStyle() == VarDecl::InitializationStyle::CInit) {
      Init = VD->getInit()->IgnoreImplicitAsWritten();
      if (isNullptrOrZero(Init)) {
        InitLoc = DpctGlobalInfo::getSourceManager().getExpansionLoc(
            VD->getBeginLoc());
      }
      // tempstorage = { nullptr/NULL/0/... }
    } else if (VD->getInitStyle() == VarDecl::InitializationStyle::ListInit) {
      if (const auto *InitList = dyn_cast<InitListExpr>(VD->getInit())) {
        if (const auto *Init0 = InitList->getInit(0)) {
          Init = Init0->IgnoreImplicitAsWritten();
          if (isNullptrOrZero(Init)) {
            InitLoc = DpctGlobalInfo::getSourceManager().getExpansionLoc(
                VD->getBeginLoc());
          }
        }
      }
    }
  }
  for (auto &Element : TempStorageMatchResult) {
    if (Element == DRE) {
      continue;
    }
    SourceLocation CurLoc = DpctGlobalInfo::getSourceManager().getExpansionLoc(
        Element->getBeginLoc());
    bool IsModified = isModifiedRef(Element);
    bool IsAssignedWithNull = false;
    if (IsModified) {
      if (const auto *BO =
              DpctGlobalInfo::findAncestor<BinaryOperator>(Element)) {
        if (BO->getLHS() == Element &&
            isNullptrOrZero(BO->getRHS()->IgnoreImplicitAsWritten())) {
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
          std::find(DRELoopList.begin(), DRELoopList.end(), CELoopList[I]) !=
              DRELoopList.end()) {
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
  return false;
}

static TextModification *replaceText(clang::SourceLocation Begin,
                                     clang::SourceLocation End,
                                     std::string &&Str,
                                     const clang::SourceManager &SM) {
  auto Length = SM.getFileOffset(End) - SM.getFileOffset(Begin);
  if (Length > 0) {
    return new ReplaceText(Begin, Length, std::move(Str));
  }
  return nullptr;
}

/// Remove cub device level api temp_storage/temp_storage_size variable
/// declaration
void removeVarDecl(const clang::VarDecl *VD) {
  static std::unordered_map<std::string, std::vector<bool>> DeclStmtBitMap;
  if (!VD)
    return;
  auto &SM = DpctGlobalInfo::getSourceManager();
  auto &Context = DpctGlobalInfo::getContext();
  if (const auto *DS = Context.getParents(*VD)[0].get<clang::DeclStmt>()) {
    auto LocInfo = DpctGlobalInfo::getLocInfo(DS->getBeginLoc());
    std::string Key = LocInfo.first + std::to_string(LocInfo.second);
    auto Decls = DS->decls();
    unsigned int DeclNum = DS->decls().end() - DS->decls().begin();
    // if this declstmt has one sub decl, then we just need to remove whole
    // declstmt simply.
    if (DeclNum == 1) {
      const auto &Mgr = DpctGlobalInfo::getSourceManager();
      const auto Range = DS->getSourceRange();
      const clang::CharSourceRange CRange(Range, true);
      auto Replacement = std::make_shared<ExtReplacement>(
          Mgr, CRange, "", new ReplaceStmt(DS, ""));
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
      if (auto *SubDecl = llvm::dyn_cast<clang::VarDecl>(*Iter)) {
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
          clang::SourceLocation Beg = SM.getExpansionLoc(SubDecl->getEndLoc());
          clang::SourceLocation End =
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
            clang::QualType TypeTemp = VD->getType();
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
            auto tok =
                clang::Lexer::findNextToken(End, SM, Context.getLangOpts());
            if (tok.hasValue() && tok.getValue().is(clang::tok::comma)) {
              End = tok.getValue().getLocation();
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

static void
TempVarAnalysis(const clang::DeclRefExpr *DRE, bool &IsSafeToRemove,
                std::vector<const clang::CallExpr *> &RelatedMalloc) {
  if (const auto *CE = DpctGlobalInfo::findAncestor<clang::CallExpr>(DRE)) {
    if (const auto *FuncDecl = CE->getDirectCallee()) {
      llvm::StringRef FuncName = FuncDecl->getName();
      if (FuncName == "Reduce" || FuncName == "Min" || FuncName == "Max" ||
          FuncName == "Sum" || FuncName == "ExclusiveSum" ||
          FuncName == "InclusiveSum" || FuncName == "Flagged") {
        const clang::DeclContext *FuncDeclContext = FuncDecl->getDeclContext();
        if (const auto *CXXRD =
                llvm::dyn_cast<clang::CXXRecordDecl>(FuncDeclContext)) {
          llvm::StringRef CXXRDName = CXXRD->getName();
          if (CXXRDName == "DeviceSegmentedReduce" ||
              CXXRDName == "DeviceReduce" || CXXRDName == "DeviceScan" ||
              CXXRDName == "DeviceSelect") {
            return;
          }
        }
      } else if (FuncName == "cudaMalloc" || FuncName == "cuMemAlloc_v2" ||
                 FuncName == "cudaFree") {
        RelatedMalloc.push_back(CE);
        return;
      }
    }
  };
  IsSafeToRemove = false;
  return;
}

void CubRedundantTempStorageAnalyzer::removeRedundantTempVar(
    const CallExpr *CE) {
  const auto *FuncArgs = CE->getArgs();
  const auto *TempStorage =
      getNonCasExpr(FuncArgs[0]->IgnoreImplicitAsWritten());
  const auto *TempStorageSize = FuncArgs[1]->IgnoreImplicitAsWritten();
  std::vector<const DeclRefExpr *> TempStorageMatchResult;
  std::vector<const DeclRefExpr *> TempStorageSizeMatchResult;
  std::vector<const CallExpr *> TempStorageRelatedMalloc;
  std::vector<const CallExpr *> TempStorageSizeRelatedMalloc;
  bool IsSafeToRemoveTempStorage = true;
  bool IsSafeToRemoveTempStorageSize = true;

  if (const auto *DRE = llvm::dyn_cast<DeclRefExpr>(TempStorage)) {
    if (const auto *VD = llvm::dyn_cast<VarDecl>(DRE->getDecl())) {
      findAllVarRef(DRE, TempStorageMatchResult);
      for (auto &Element : TempStorageMatchResult) {
        if (Element == DRE)
          continue;
        if (!IsSafeToRemoveTempStorage)
          break;
        TempVarAnalysis(Element, IsSafeToRemoveTempStorage,
                        TempStorageRelatedMalloc);
      }
      if (!IsSafeToRemoveTempStorage)
        return;
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

  if (const auto *DRE = llvm::dyn_cast<DeclRefExpr>(TempStorageSize)) {
    if (const auto *VD = llvm::dyn_cast<VarDecl>(DRE->getDecl())) {
      findAllVarRef(DRE, TempStorageSizeMatchResult);
      for (auto &Element : TempStorageSizeMatchResult) {
        if (Element == DRE) {
          continue;
        }
        if (IsSafeToRemoveTempStorageSize) {
          TempVarAnalysis(Element, IsSafeToRemoveTempStorageSize,
                          TempStorageSizeRelatedMalloc);
        } else {
          break;
        }
      }
      for (const auto *Element : TempStorageSizeRelatedMalloc) {
        if (std::find(TempStorageRelatedMalloc.begin(),
                      TempStorageRelatedMalloc.end(),
                      Element) == TempStorageRelatedMalloc.end()) {
          IsSafeToRemoveTempStorageSize = false;
        }
      }
      if (IsSafeToRemoveTempStorageSize) {
        removeVarDecl(VD);
      }
    }
  }
}
