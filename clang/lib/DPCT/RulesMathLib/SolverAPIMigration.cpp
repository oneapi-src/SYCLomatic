//===------------------ SolverAPIMigration.cpp ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SolverAPIMigration.h"
#include "MapNamesSolver.h"
#include "RuleInfra/ASTmatcherCommon.h"
#include "RuleInfra/CallExprRewriter.h"
#include "RuleInfra/CallExprRewriterCommon.h"
#include "RulesLang/RulesLang.h"

namespace clang {
namespace dpct {

using namespace clang::ast_matchers;

// Rule for SOLVER enums.
// Migrate SOLVER status values to corresponding int values
// Other SOLVER named values are migrated to corresponding named values
void SOLVEREnumsRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(declRefExpr(to(enumConstantDecl(matchesName(
                                "(CUSOLVER_STATU.*)|(CUSOLVER_ALG.*)|("
                                "CUSOLVERDN_GETRF)|(CUSOLVERDN_POTRF)"))))
                    .bind("SOLVERConstants"),
                this);
  MF.addMatcher(
      declRefExpr(to(enumConstantDecl(matchesName(
                      "(CUSOLVER_EIG_TYPE.*)|(CUSOLVER_EIG_MODE.*)"))))
          .bind("SLOVERNamedValueConstants"),
      this);
}

void SOLVEREnumsRule::runRule(const MatchFinder::MatchResult &Result) {
  if (const DeclRefExpr *DE =
          getNodeAsType<DeclRefExpr>(Result, "SOLVERConstants")) {
    auto *EC = cast<EnumConstantDecl>(DE->getDecl());
    emplaceTransformation(new ReplaceStmt(DE, toString(EC->getInitVal(), 10)));
  }

  if (const DeclRefExpr *DE =
          getNodeAsType<DeclRefExpr>(Result, "SLOVERNamedValueConstants")) {
    auto *EC = cast<EnumConstantDecl>(DE->getDecl());
    std::string Name = EC->getNameAsString();
    auto Search = MapNamesSolver::SOLVEREnumsMap.find(Name);
    if (Search == MapNamesSolver::SOLVEREnumsMap.end()) {
      llvm::dbgs() << "[" << getName()
                   << "] Unexpected enum variable: " << Name;
      return;
    }
    std::string Replacement = Search->second;
    emplaceTransformation(new ReplaceStmt(DE, std::move(Replacement)));
  }
}


void SOLVERFunctionCallRule::registerMatcher(MatchFinder &MF) {
  
  auto functionName = [&]() {
    return hasAnyName(
        "cusolverDnSetAdvOptions", "cusolverDnGetStream", "cusolverDnSetStream",
        "cusolverDnCreateParams", "cusolverDnDestroyParams", "cusolverDnCreate",
        "cusolverDnDestroy", "cusolverDnSpotrf_bufferSize",
        "cusolverDnDpotrf_bufferSize", "cusolverDnCpotrf_bufferSize",
        "cusolverDnZpotrf_bufferSize", "cusolverDnSpotri_bufferSize",
        "cusolverDnDpotri_bufferSize", "cusolverDnCpotri_bufferSize",
        "cusolverDnZpotri_bufferSize", "cusolverDnSgetrf_bufferSize",
        "cusolverDnDgetrf_bufferSize", "cusolverDnCgetrf_bufferSize",
        "cusolverDnZgetrf_bufferSize", "cusolverDnSpotrf", "cusolverDnDpotrf",
        "cusolverDnCpotrf", "cusolverDnZpotrf", "cusolverDnSpotrs",
        "cusolverDnDpotrs", "cusolverDnCpotrs", "cusolverDnZpotrs",
        "cusolverDnSpotri", "cusolverDnDpotri", "cusolverDnCpotri",
        "cusolverDnZpotri", "cusolverDnSgetrf", "cusolverDnDgetrf",
        "cusolverDnCgetrf", "cusolverDnZgetrf", "cusolverDnSgetrs",
        "cusolverDnDgetrs", "cusolverDnCgetrs", "cusolverDnZgetrs",
        "cusolverDnSgeqrf_bufferSize", "cusolverDnDgeqrf_bufferSize",
        "cusolverDnCgeqrf_bufferSize", "cusolverDnZgeqrf_bufferSize",
        "cusolverDnSgeqrf", "cusolverDnDgeqrf", "cusolverDnCgeqrf",
        "cusolverDnZgeqrf", "cusolverDnSormqr_bufferSize",
        "cusolverDnDormqr_bufferSize", "cusolverDnSormqr", "cusolverDnDormqr",
        "cusolverDnCunmqr_bufferSize", "cusolverDnZunmqr_bufferSize",
        "cusolverDnCunmqr", "cusolverDnZunmqr", "cusolverDnSorgqr_bufferSize",
        "cusolverDnDorgqr_bufferSize", "cusolverDnCungqr_bufferSize",
        "cusolverDnZungqr_bufferSize", "cusolverDnSorgqr", "cusolverDnDorgqr",
        "cusolverDnCungqr", "cusolverDnZungqr", "cusolverDnSsytrf_bufferSize",
        "cusolverDnDsytrf_bufferSize", "cusolverDnCsytrf_bufferSize",
        "cusolverDnZsytrf_bufferSize", "cusolverDnSsytrf", "cusolverDnDsytrf",
        "cusolverDnCsytrf", "cusolverDnZsytrf", "cusolverDnSgebrd_bufferSize",
        "cusolverDnDgebrd_bufferSize", "cusolverDnCgebrd_bufferSize",
        "cusolverDnZgebrd_bufferSize", "cusolverDnSgebrd", "cusolverDnDgebrd",
        "cusolverDnCgebrd", "cusolverDnZgebrd", "cusolverDnSorgbr_bufferSize",
        "cusolverDnDorgbr_bufferSize", "cusolverDnCungbr_bufferSize",
        "cusolverDnZungbr_bufferSize", "cusolverDnSorgbr", "cusolverDnDorgbr",
        "cusolverDnCungbr", "cusolverDnZungbr", "cusolverDnSsytrd_bufferSize",
        "cusolverDnDsytrd_bufferSize", "cusolverDnChetrd_bufferSize",
        "cusolverDnZhetrd_bufferSize", "cusolverDnSsytrd", "cusolverDnDsytrd",
        "cusolverDnChetrd", "cusolverDnZhetrd", "cusolverDnSormtr_bufferSize",
        "cusolverDnDormtr_bufferSize", "cusolverDnCunmtr_bufferSize",
        "cusolverDnZunmtr_bufferSize", "cusolverDnSormtr", "cusolverDnDormtr",
        "cusolverDnCunmtr", "cusolverDnZunmtr", "cusolverDnSorgtr_bufferSize",
        "cusolverDnDorgtr_bufferSize", "cusolverDnCungtr_bufferSize",
        "cusolverDnZungtr_bufferSize", "cusolverDnSorgtr", "cusolverDnDorgtr",
        "cusolverDnCungtr", "cusolverDnZungtr", "cusolverDnSgesvd_bufferSize",
        "cusolverDnDgesvd_bufferSize", "cusolverDnCgesvd_bufferSize",
        "cusolverDnZgesvd_bufferSize", "cusolverDnSgesvd", "cusolverDnDgesvd",
        "cusolverDnCgesvd", "cusolverDnZgesvd", "cusolverDnSpotrfBatched",
        "cusolverDnDpotrfBatched", "cusolverDnCpotrfBatched",
        "cusolverDnZpotrfBatched", "cusolverDnSpotrsBatched",
        "cusolverDnDpotrsBatched", "cusolverDnCpotrsBatched",
        "cusolverDnZpotrsBatched", "cusolverDnSsygvd", "cusolverDnDsygvd",
        "cusolverDnSsygvd_bufferSize", "cusolverDnDsygvd_bufferSize",
        "cusolverDnChegvd", "cusolverDnZhegvd", "cusolverDnChegvd_bufferSize",
        "cusolverDnZhegvd_bufferSize", "cusolverDnXgetrf",
        "cusolverDnXgetrf_bufferSize", "cusolverDnXgetrs", "cusolverDnXgeqrf",
        "cusolverDnXgeqrf_bufferSize", "cusolverDnGetrf",
        "cusolverDnGetrf_bufferSize", "cusolverDnGetrs", "cusolverDnGeqrf",
        "cusolverDnGeqrf_bufferSize", "cusolverDnCreateGesvdjInfo",
        "cusolverDnDestroyGesvdjInfo", "cusolverDnCreateSyevjInfo",
        "cusolverDnDestroySyevjInfo", "cusolverDnSgesvdj_bufferSize",
        "cusolverDnDgesvdj_bufferSize", "cusolverDnCgesvdj_bufferSize",
        "cusolverDnZgesvdj_bufferSize", "cusolverDnXgesvd_bufferSize",
        "cusolverDnGesvd_bufferSize", "cusolverDnSgesvdj", "cusolverDnDgesvdj",
        "cusolverDnCgesvdj", "cusolverDnZgesvdj", "cusolverDnXgesvd",
        "cusolverDnGesvd", "cusolverDnXpotrf_bufferSize",
        "cusolverDnPotrf_bufferSize", "cusolverDnXpotrf", "cusolverDnPotrf",
        "cusolverDnXpotrs", "cusolverDnPotrs", "cusolverDnSsyevdx",
        "cusolverDnDsyevdx", "cusolverDnSsyevdx_bufferSize",
        "cusolverDnDsyevdx_bufferSize", "cusolverDnCheevdx",
        "cusolverDnZheevdx", "cusolverDnCheevdx_bufferSize",
        "cusolverDnZheevdx_bufferSize", "cusolverDnSsygvdx",
        "cusolverDnDsygvdx", "cusolverDnSsygvdx_bufferSize",
        "cusolverDnDsygvdx_bufferSize", "cusolverDnChegvdx",
        "cusolverDnZhegvdx", "cusolverDnChegvdx_bufferSize",
        "cusolverDnZhegvdx_bufferSize", "cusolverDnSsygvj", "cusolverDnDsygvj",
        "cusolverDnSsygvj_bufferSize", "cusolverDnDsygvj_bufferSize",
        "cusolverDnChegvj", "cusolverDnZhegvj", "cusolverDnChegvj_bufferSize",
        "cusolverDnZhegvj_bufferSize", "cusolverDnXsyevdx",
        "cusolverDnXsyevdx_bufferSize", "cusolverDnSyevdx",
        "cusolverDnSyevdx_bufferSize", "cusolverDnSsyevj", "cusolverDnDsyevj",
        "cusolverDnSsyevj_bufferSize", "cusolverDnDsyevj_bufferSize",
        "cusolverDnCheevj", "cusolverDnZheevj", "cusolverDnCheevj_bufferSize",
        "cusolverDnZheevj_bufferSize", "cusolverDnXsyevd",
        "cusolverDnXsyevd_bufferSize", "cusolverDnSyevd",
        "cusolverDnSyevd_bufferSize", "cusolverDnXtrtri",
        "cusolverDnXtrtri_bufferSize", "cusolverDnSsyevd_bufferSize",
        "cusolverDnDsyevd_bufferSize", "cusolverDnCheevd_bufferSize",
        "cusolverDnZheevd_bufferSize", "cusolverDnSsyevd", "cusolverDnDsyevd",
        "cusolverDnCheevd", "cusolverDnZheevd");
  };

  MF.addMatcher(callExpr(allOf(callee(functionDecl(functionName())),
                               hasAncestor(functionDecl(
                                   anyOf(hasAttr(attr::CUDADevice),
                                         hasAttr(attr::CUDAGlobal))))))
                    .bind("kernelCall"),
                this);

  MF.addMatcher(
      callExpr(
          allOf(callee(functionDecl(functionName())), parentStmt(),
                hasAncestor(functionDecl(unless(allOf(
                    hasAttr(attr::CUDADevice), hasAttr(attr::CUDAGlobal)))))))
          .bind("FunctionCall"),
      this);
  MF.addMatcher(
      callExpr(
          allOf(callee(functionDecl(functionName())), unless(parentStmt()),
                unless(hasParent(varDecl())),
                hasAncestor(functionDecl(unless(allOf(
                    hasAttr(attr::CUDADevice), hasAttr(attr::CUDAGlobal)))))))
          .bind("FunctionCallUsedNotInitializeVarDecl"),
      this);

  MF.addMatcher(
      callExpr(
          allOf(callee(functionDecl(functionName())), hasParent(varDecl()),
                hasAncestor(functionDecl(unless(allOf(
                    hasAttr(attr::CUDADevice), hasAttr(attr::CUDAGlobal)))))))
          .bind("FunctionCallUsedInitializeVarDecl"),
      this);
}

void SOLVERFunctionCallRule::runRule(const MatchFinder::MatchResult &Result) {
  bool IsAssigned = false;
  bool IsInitializeVarDecl = false;
  bool HasDeviceAttr = false;
  const CallExpr *CE = getNodeAsType<CallExpr>(Result, "kernelCall");
  if (CE) {
    HasDeviceAttr = true;
  } else if (!(CE = getNodeAsType<CallExpr>(Result, "FunctionCall"))) {
    if ((CE = getNodeAsType<CallExpr>(
             Result, "FunctionCallUsedNotInitializeVarDecl"))) {
      IsAssigned = true;
    } else if ((CE = getNodeAsType<CallExpr>(
                    Result, "FunctionCallUsedInitializeVarDecl"))) {
      IsAssigned = true;
      IsInitializeVarDecl = true;
    } else {
      return;
    }
  }

  const SourceManager *SM = Result.SourceManager;
  auto SL = SM->getExpansionLoc(CE->getBeginLoc());
  std::string Key = SM->getFilename(SL).str() +
                    std::to_string(SM->getDecomposedLoc(SL).second);
  DpctGlobalInfo::updateInitSuffixIndexInRule(
      DpctGlobalInfo::getSuffixIndexInitValue(Key));

  // Collect sourceLocations of the function call
  SourceLocation FuncNameBegin(CE->getBeginLoc());
  SourceLocation FuncCallEnd(CE->getEndLoc());

  // Correct sourceLocations for macros
  if (FuncNameBegin.isMacroID())
    FuncNameBegin = SM->getExpansionLoc(FuncNameBegin);
  if (FuncCallEnd.isMacroID())
    FuncCallEnd = SM->getExpansionLoc(FuncCallEnd);
  // Offset 1 is the length of the last token ")"
  FuncCallEnd = FuncCallEnd.getLocWithOffset(1);

  // Collect sourceLocations for creating new scope
  std::string PrefixBeforeScope, PrefixInsertStr, SuffixInsertStr;
  auto SR = getScopeInsertRange(CE, FuncNameBegin, FuncCallEnd);
  SourceLocation StmtBegin = SR.getBegin(), StmtEndAfterSemi = SR.getEnd();
  std::string IndentStr =
      getIndent(StmtBegin, (Result.Context)->getSourceManager()).str();
  Token Tok;
  Lexer::getRawToken(FuncNameBegin, Tok, *SM, LangOptions());
  SourceLocation FuncNameEnd = Tok.getEndLoc();
  auto FuncNameLength =
      SM->getCharacterData(FuncNameEnd) - SM->getCharacterData(FuncNameBegin);

  // Prepare the prefix and the postfix for assignment
  if (!CE->getDirectCallee())
    return;
  std::string FuncName =
      CE->getDirectCallee()->getNameInfo().getName().getAsString();
  std::string AssignPrefix = "";
  std::string AssignPostfix = "";

  if (IsAssigned) {
    AssignPrefix = MapNames::getCheckErrorMacroName() + "(";
    AssignPostfix = ")";
  }

  if (HasDeviceAttr) {
    report(CE->getBeginLoc(), Diagnostics::FUNCTION_CALL_IN_DEVICE, false,
           MapNames::ITFName.at(FuncName),
           MemoryMigrationRule::getMemoryHelperFunctionName("memcpy"));
    return;
  }

  const VarDecl *VD = 0;
  if (IsInitializeVarDecl) {
    // Create the prefix for VarDecl before scope and remove the VarDecl inside
    // the scope
    VD = getAncestralVarDecl(CE);
    std::string VarType, VarName;
    if (VD) {
      VarType = VD->getType().getAsString();
      VarName = VD->getNameAsString();

      requestHelperFeatureForTypeNames(VarType);
      insertHeaderForTypeRule(VarType, VD->getBeginLoc());
      auto Itr = MapNames::TypeNamesMap.find(VarType);
      if (Itr == MapNames::TypeNamesMap.end())
        return;
      VarType = Itr->second->NewName;
      PrefixBeforeScope = VarType + " " + VarName + ";" + getNL() + IndentStr +
                          PrefixBeforeScope;
      SourceLocation typeBegin =
          VD->getTypeSourceInfo()->getTypeLoc().getBeginLoc();
      SourceLocation nameBegin = VD->getLocation();
      SourceLocation nameEnd = Lexer::getLocForEndOfToken(
          nameBegin, 0, *SM, Result.Context->getLangOpts());
      auto replLen =
          SM->getCharacterData(nameEnd) - SM->getCharacterData(typeBegin);
      for (auto ItHeader = Itr->second->Includes.begin();
           ItHeader != Itr->second->Includes.end(); ItHeader++) {
        DpctGlobalInfo::getInstance().insertHeader(typeBegin, *ItHeader);
      }
      emplaceTransformation(
          new ReplaceText(typeBegin, replLen, std::move(VarName)));
    } else {
      assert(0 && "Fail to get VarDecl information");
      return;
    }
  }

  if (MapNamesSolver::SOLVERAPIWithRewriter.find(FuncName) !=
      MapNamesSolver::SOLVERAPIWithRewriter.end()) {
    ExprAnalysis EA(CE);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
    return;
  } else if (MapNamesSolver::SOLVERFuncReplInfoMap.find(FuncName) !=
             MapNamesSolver::SOLVERFuncReplInfoMap.end()) {
    // Find replacement string
    auto ReplInfoPair = MapNamesSolver::SOLVERFuncReplInfoMap.find(FuncName);
    MapNamesSolver::SOLVERFuncReplInfo ReplInfo = ReplInfoPair->second;
    std::string Replacement = ReplInfo.ReplName;

    // Migrate arguments one by one
    int ArgNum = CE->getNumArgs();
    for (int i = 0; i < ArgNum; ++i) {
      int IndexTemp = -1;
      // MyFunction(float* a);
      // In usm: Keep it and type cast if needed.
      // In non-usm: Create a buffer, pass the buffer
      // If type is int: MKL takes int64_t, so need to create a temp
      //                 buffer/variable and copy the value back
      //                 after the function call.
      // Some API migration requires MoveFrom and MoveTo.
      // e.g., move arg#1 to arg#0
      // MyFunction(float* a, float* b);
      // ---> MyFunction(float* b, float*a);
      if (isReplIndex(i, ReplInfo.BufferIndexInfo, IndexTemp)) {
        if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_None) {
          requestFeature(HelperFeatureEnum::device_ext);
          std::string BufferDecl;
          std::string BufferName = getBufferNameAndDeclStr(
              CE->getArg(i), *(Result.Context),
              ReplInfo.BufferTypeInfo[IndexTemp], StmtBegin, BufferDecl, i);
          PrefixInsertStr = PrefixInsertStr + BufferDecl;
          if (ReplInfo.BufferTypeInfo[IndexTemp] == "int") {
            PrefixInsertStr =
                PrefixInsertStr + IndentStr + MapNames::getClNamespace() +
                "buffer<int64_t> "
                "result_temp_buffer" +
                std::to_string(i) + "(" + MapNames::getClNamespace() +
                "range<1>(1));" + getNL();
            SuffixInsertStr = SuffixInsertStr + BufferName + ".get_access<" +
                              MapNames::getClNamespace() +
                              "access_mode::write>()[0] = "
                              "(int)result_temp_buffer" +
                              std::to_string(i) + ".get_access<" +
                              MapNames::getClNamespace() +
                              "access_mode::read>()[0];" + getNL() + IndentStr;
            BufferName = "result_temp_buffer" + std::to_string(i);
          }
          bool Moved = false;
          for (size_t j = 0; j < ReplInfo.MoveFrom.size(); j++) {
            if (ReplInfo.MoveFrom[j] == i) {
              Moved = true;
              if (CE->getArg(ReplInfo.MoveTo[j])) {
                emplaceTransformation(new InsertAfterStmt(
                    CE->getArg(ReplInfo.MoveTo[j] - 1),
                    ", result_temp_buffer" + std::to_string(i)));
              }
              ReplInfo.RedundantIndexInfo.push_back(i);
              break;
            }
          }
          if (!Moved) {
            emplaceTransformation(new ReplaceStmt(CE->getArg(i), BufferName));
          }
        } else {
          std::string ArgName = ExprAnalysis::ref(CE->getArg(i));
          if (ReplInfo.BufferTypeInfo[IndexTemp] == "int") {
            PrefixInsertStr = IndentStr + "int64_t result_temp_pointer" +
                              std::to_string(i) + ";" + getNL();
            SuffixInsertStr = SuffixInsertStr + " *" +
                              ExprAnalysis::ref(CE->getArg(i)) +
                              " = result_temp_pointer" + std::to_string(i) +
                              ";" + getNL() + IndentStr;
            ArgName = "&result_temp_pointer" + std::to_string(i);
          }
          bool Moved = false;
          for (size_t j = 0; j < ReplInfo.MoveFrom.size(); j++) {
            if (ReplInfo.MoveFrom[j] == i) {
              Moved = true;
              if (CE->getArg(ReplInfo.MoveTo[j])) {
                emplaceTransformation(new InsertAfterStmt(
                    CE->getArg(ReplInfo.MoveTo[j] - 1), ", " + ArgName));
              }
              ReplInfo.RedundantIndexInfo.push_back(i);
              break;
            }
          }
          if (!Moved) {
            std::string TypeStr =
                ReplInfo.BufferTypeInfo[IndexTemp].compare("int")
                    ? "(" + ReplInfo.BufferTypeInfo[IndexTemp] + "*)"
                    : "";
            emplaceTransformation(
                new ReplaceStmt(CE->getArg(i), TypeStr + ArgName));
          }
        }
      }
      // Remove the redundant args including the leading ","
      if (isReplIndex(i, ReplInfo.RedundantIndexInfo, IndexTemp)) {
        SourceLocation RemoveBegin, RemoveEnd;
        if (i == 0) {
          RemoveBegin = CE->getArg(i)->getBeginLoc();
        } else {
          RemoveBegin = CE->getArg(i - 1)->getEndLoc().getLocWithOffset(
              Lexer::MeasureTokenLength(
                  CE->getArg(i - 1)->getEndLoc(), *SM,
                  dpct::DpctGlobalInfo::getContext().getLangOpts()));
        }
        RemoveEnd = CE->getArg(i)->getEndLoc().getLocWithOffset(
            Lexer::MeasureTokenLength(
                CE->getArg(i)->getEndLoc(), *SM,
                dpct::DpctGlobalInfo::getContext().getLangOpts()));
        auto ParameterLength =
            SM->getCharacterData(RemoveEnd) - SM->getCharacterData(RemoveBegin);
        emplaceTransformation(
            new ReplaceText(RemoveBegin, ParameterLength, ""));
      }
      // OldFoo(float* out); --> *(out) = NewFoo();
      // In current case, return value is always the last arg
      if (ReplInfo.ReturnValue && i == ArgNum - 1) {
        Replacement = "*(" +
                      ExprAnalysis::ref(CE->getArg(CE->getNumArgs() - 1)) +
                      ") = " + Replacement;
      }
      // The arg#0 is always the handler and will always be migrated to queue.
      if (i == 0) {
        // process handle argument
        emplaceTransformation(new ReplaceStmt(
            CE->getArg(i), "*" + ExprAnalysis::ref(CE->getArg(i))));
      }
    }
    // Declare new args if it is used in MKL
    if (!ReplInfo.MissedArgumentFinalLocation.empty()) {
      std::string ReplStr;
      for (size_t i = 0; i < ReplInfo.MissedArgumentFinalLocation.size(); ++i) {
        if (ReplInfo.MissedArgumentIsBuffer[i]) {
          PrefixInsertStr =
              PrefixInsertStr + IndentStr + MapNames::getClNamespace() +
              "buffer<" + ReplInfo.MissedArgumentType[i] + "> " +
              ReplInfo.MissedArgumentName[i] + "(" +
              MapNames::getClNamespace() + "range<1>(1));" + getNL();
        } else {
          PrefixInsertStr = PrefixInsertStr + IndentStr +
                            ReplInfo.MissedArgumentType[i] + " " +
                            ReplInfo.MissedArgumentName[i] + ";" + getNL();
        }
        ReplStr = ReplStr + ", " + ReplInfo.MissedArgumentName[i];
        if (i == ReplInfo.MissedArgumentFinalLocation.size() - 1 ||
            ReplInfo.MissedArgumentInsertBefore[i + 1] !=
                ReplInfo.MissedArgumentInsertBefore[i]) {
          if (ReplInfo.MissedArgumentInsertBefore[i] > 0) {
            emplaceTransformation(new InsertAfterStmt(
                CE->getArg(ReplInfo.MissedArgumentInsertBefore[i] - 1),
                std::move(ReplStr)));
          }
          ReplStr = "";
        }
      }
    }

    // Copy an arg. e.g. copy arg#0 to arg#2
    // OldFoo(int m, int n); --> NewFoo(int m, int n, int m)
    if (!ReplInfo.CopyFrom.empty()) {
      std::string InsStr = "";
      for (size_t i = 0; i < ReplInfo.CopyFrom.size(); ++i) {
        InsStr =
            InsStr + ", " + ExprAnalysis::ref(CE->getArg(ReplInfo.CopyFrom[i]));
        if (i == ReplInfo.CopyTo.size() - 1 ||
            ReplInfo.CopyTo[i + 1] != ReplInfo.CopyTo[i]) {
          emplaceTransformation(new InsertAfterStmt(
              CE->getArg(ReplInfo.CopyTo[i - 1]), std::move(InsStr)));
          InsStr = "";
        }
      }
    }
    // Type cast, for enum type migration
    if (!ReplInfo.CastIndexInfo.empty()) {
      for (size_t i = 0; i < ReplInfo.CastIndexInfo.size(); ++i) {
        std::string CastStr = "(" + ReplInfo.CastTypeInfo[i] + ")";
        emplaceTransformation(new InsertBeforeStmt(
            CE->getArg(ReplInfo.CastIndexInfo[i]), std::move(CastStr)));
      }
    }
    // Create scratchpad and scratchpad_size if required in MKL
    if (!ReplInfo.WorkspaceIndexInfo.empty()) {
      std::string BufferSizeArgStr = "";
      for (size_t i = 0; i < ReplInfo.WorkspaceSizeInfo.size(); ++i) {
        BufferSizeArgStr += i ? " ," : "";
        BufferSizeArgStr +=
            ExprAnalysis::ref(CE->getArg(ReplInfo.WorkspaceSizeInfo[i]));
      }
      std::string ScratchpadSizeNameStr =
          "scratchpad_size_ct" +
          std::to_string(dpct::DpctGlobalInfo::getSuffixIndexInRuleThenInc());
      std::string ScratchpadNameStr =
          "scratchpad_ct" +
          std::to_string(dpct::DpctGlobalInfo::getSuffixIndexInRuleThenInc());
      std::string EventNameStr =
          "event_ct" +
          std::to_string(dpct::DpctGlobalInfo::getSuffixIndexInRuleThenInc());
      std::string WSVectorNameStr =
          "ws_vec_ct" +
          std::to_string(dpct::DpctGlobalInfo::getSuffixIndexInRuleThenInc());
      PrefixInsertStr += IndentStr + "std::int64_t " + ScratchpadSizeNameStr +
                         " = " + ReplInfo.WorkspaceSizeFuncName + "(*" +
                         BufferSizeArgStr + ");" + getNL();
      std::string BufferTypeStr = "float";
      if (ReplInfo.BufferTypeInfo.size() > 0) {
        BufferTypeStr = ReplInfo.BufferTypeInfo[0];
      }
      if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_Restricted) {
        DpctGlobalInfo::getInstance().insertHeader(CE->getBeginLoc(),
                                                   HT_Thread);

        PrefixInsertStr +=
            IndentStr + BufferTypeStr + " *" + ScratchpadNameStr + " = " +
            MapNames::getClNamespace() + "malloc_device<" + BufferTypeStr +
            ">(" + ScratchpadSizeNameStr + ", *" +
            ExprAnalysis::ref(CE->getArg(0)) + ");" + getNL();
        PrefixInsertStr += IndentStr + MapNames::getClNamespace() + "event " +
                           EventNameStr + ";" + getNL();

        Replacement = EventNameStr + " = " + Replacement;

        SuffixInsertStr += "std::vector<void *> " + WSVectorNameStr + "{" +
                           ScratchpadNameStr + "};" + getNL() + IndentStr;
        SuffixInsertStr +=
            MapNames::getDpctNamespace() +
            (DpctGlobalInfo::useSYCLCompat() ? "enqueue_free("
                                             : "async_dpct_free(") +
            WSVectorNameStr + ", {" + EventNameStr + "}, *" +
            ExprAnalysis::ref(CE->getArg(0)) + ");" + getNL() + IndentStr;
        requestFeature(HelperFeatureEnum::device_ext);
      } else {
        PrefixInsertStr += IndentStr + MapNames::getClNamespace() + "buffer<" +
                           BufferTypeStr + ", 1> " + ScratchpadNameStr + "{" +
                           MapNames::getClNamespace() + "range<1>(" +
                           ScratchpadSizeNameStr + ")};" + getNL();
      }
      if (ReplInfo.WorkspaceIndexInfo[0] > 0) {
        emplaceTransformation(new InsertAfterStmt(
            CE->getArg(ReplInfo.WorkspaceIndexInfo[0]),
            ", " + ScratchpadNameStr + ", " + ScratchpadSizeNameStr));
      }
    }

    // Create scratchpad_size if only scratchpad_size is required in MKL
    if (!ReplInfo.WSSizeInsertAfter.empty()) {
      std::string BufferSizeArgStr = "";
      for (size_t i = 0; i < ReplInfo.WSSizeInfo.size(); ++i) {
        BufferSizeArgStr += i ? " ," : "";
        BufferSizeArgStr +=
            ExprAnalysis::ref(CE->getArg(ReplInfo.WSSizeInfo[i]));
      }
      std::string ScratchpadSizeNameStr =
          "scratchpad_size_ct" +
          std::to_string(dpct::DpctGlobalInfo::getSuffixIndexInRuleThenInc());
      PrefixInsertStr += IndentStr + "std::int64_t " + ScratchpadSizeNameStr +
                         " = " + ReplInfo.WSSFuncName + "(*" +
                         BufferSizeArgStr + ");" + getNL();
      if (ReplInfo.WSSizeInsertAfter[0] > 0) {
        emplaceTransformation(
            new InsertAfterStmt(CE->getArg(ReplInfo.WSSizeInsertAfter[0]),
                                ", " + ScratchpadSizeNameStr));
      }
    }

    // Check PrefixInsertStr and SuffixInsertStr to decide whether to add
    // bracket
    std::string PrefixWithBracket = "";
    std::string SuffixWithBracket = "";
    if (!PrefixInsertStr.empty() || !SuffixInsertStr.empty()) {
      PrefixWithBracket =
          "{" + std::string(getNL()) + PrefixInsertStr + IndentStr;
      SuffixWithBracket = getNL() + IndentStr + SuffixInsertStr + "}";
    }

    std::string ReplaceFuncName = Replacement;
    emplaceTransformation(
        new ReplaceText(FuncNameBegin, FuncNameLength, std::move(Replacement)));
    insertAroundRange(StmtBegin, StmtEndAfterSemi,
                      PrefixBeforeScope + PrefixWithBracket,
                      std::move(SuffixWithBracket));

    StringRef FuncNameRef(FuncName);
    if (FuncNameRef.ends_with("getrf")) {
      report(StmtBegin, Diagnostics::DIFFERENT_LU_FACTORIZATION, true,
             getStmtSpelling(CE->getArg(6)), ReplaceFuncName,
             MapNames::ITFName.at(FuncName));
    }
    if (IsAssigned) {
      insertAroundRange(FuncNameBegin, FuncCallEnd, std::move(AssignPrefix),
                        std::move(AssignPostfix));
      requestFeature(HelperFeatureEnum::device_ext);
    }
  } else if (FuncName == "cusolverDnCreate" ||
             FuncName == "cusolverDnDestroy") {
    std::string Repl;
    if (FuncName == "cusolverDnCreate") {
      std::string LHS = getDrefName(CE->getArg(0));
      if (isPlaceholderIdxDuplicated(CE))
        return;
      int Index = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();
      buildTempVariableMap(Index, CE, HelperFuncType::HFT_DefaultQueuePtr);
      Repl = LHS + " = {{NEEDREPLACEZ" + std::to_string(Index) + "}}";
    } else if (FuncName == "cusolverDnDestroy") {
      dpct::ExprAnalysis EA(CE->getArg(0));
      Repl = EA.getReplacedString() + " = nullptr";
    } else {
      return;
    }

    if (IsAssigned) {
      requestFeature(HelperFeatureEnum::device_ext);
      emplaceTransformation(
          new ReplaceStmt(CE, true, MapNames::getCheckErrorMacroName() + "(" + Repl + ")"));
    } else {
      emplaceTransformation(new ReplaceStmt(CE, true, Repl));
    }
  }
}

void SOLVERFunctionCallRule::getParameterEnd(
    const SourceLocation &ParameterEnd, SourceLocation &ParameterEndAfterComma,
    const ast_matchers::MatchFinder::MatchResult &Result) {
  std::optional<Token> TokSharedPtr;
  TokSharedPtr = Lexer::findNextToken(ParameterEnd, *(Result.SourceManager),
                                      LangOptions());
  Token TokComma = TokSharedPtr.value();
  if (TokComma.getKind() == tok::comma) {
    ParameterEndAfterComma = TokComma.getEndLoc();
  } else {
    ParameterEndAfterComma = TokComma.getLocation();
  }
}

bool SOLVERFunctionCallRule::isReplIndex(int Input, std::vector<int> &IndexInfo,
                                         int &IndexTemp) {
  for (int i = 0; i < static_cast<int>(IndexInfo.size()); ++i) {
    if (IndexInfo[i] == Input) {
      IndexTemp = i;
      return true;
    }
  }
  return false;
}

std::string SOLVERFunctionCallRule::getBufferNameAndDeclStr(
    const Expr *Arg, const ASTContext &AC, const std::string &TypeAsStr,
    SourceLocation SL, std::string &BufferDecl, int DistinctionID) {

  std::string PointerName = ExprAnalysis::ref(Arg);
  std::string BufferTempName =
      getTempNameForExpr(Arg, true, true) + "buf_ct" +
      std::to_string(dpct::DpctGlobalInfo::getSuffixIndexInRuleThenInc());

  requestFeature(HelperFeatureEnum::device_ext);
  BufferDecl = getIndent(SL, AC.getSourceManager()).str() + "auto " +
               BufferTempName + " = " + MapNames::getDpctNamespace() +
               "get_buffer<" + TypeAsStr + ">(" + PointerName + ");" + getNL();
  return BufferTempName;
}

const clang::VarDecl *
SOLVERFunctionCallRule::getAncestralVarDecl(const clang::CallExpr *CE) {
  auto &Context = dpct::DpctGlobalInfo::getContext();
  auto Parents = Context.getParents(*CE);
  while (Parents.size() == 1) {
    auto *Parent = Parents[0].get<VarDecl>();
    if (Parent) {
      return Parent;
    } else {
      Parents = Context.getParents(Parents[0]);
    }
  }
  return nullptr;
}


} // namespace dpct
} // namespace clang