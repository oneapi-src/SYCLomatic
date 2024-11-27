//===--------------- RulesLangCooperativeGroups.cpp-----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AnalysisInfo.h"
#include "RuleInfra/CallExprRewriter.h"
#include "RuleInfra/ExprAnalysis.h"
#include "RuleInfra/MigrationStatistics.h"
#include "RulesLang.h"
#include "RulesLang/MapNamesLang.h"
#include "TextModification.h"
#include "Utility.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/Type.h"
#include "clang/AST/TypeLoc.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Analysis/AnalysisDeclContext.h"
#include "clang/Analysis/CallGraph.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Lex/MacroArgs.h"

#include <string>
#include <utility>

using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::dpct;
using namespace clang::tooling;

extern clang::tooling::UnifiedPath
    DpctInstallPath; // Installation directory for this tool
extern DpctOption<opt, bool> ProcessAll;
extern DpctOption<opt, bool> AsyncHandler;

namespace clang {
namespace dpct {

void CooperativeGroupsFunctionRule::registerMatcher(MatchFinder &MF) {
  std::vector<std::string> CGAPI;
  CGAPI.insert(CGAPI.end(), MapNamesLang::CooperativeGroupsAPISet.begin(),
               MapNamesLang::CooperativeGroupsAPISet.end());
  MF.addMatcher(
      callExpr(
          allOf(callee(functionDecl(
                    internal::Matcher<NamedDecl>(
                        new internal::HasNameMatcher(CGAPI)),
                    hasAncestor(namespaceDecl(hasName("cooperative_groups"))))),
                hasAncestor(functionDecl(anyOf(hasAttr(attr::CUDADevice),
                                               hasAttr(attr::CUDAGlobal))))))
          .bind("FuncCall"),
      this);
  MF.addMatcher(
      declRefExpr(
          hasAncestor(implicitCastExpr(hasImplicitDestinationType(
              qualType(hasCanonicalType(recordType(hasDeclaration(cxxRecordDecl(
                  hasName("cooperative_groups::__v1::thread_group"))))))))))
          .bind("declRef"),
      this);
}

void CooperativeGroupsFunctionRule::runRule(
    const MatchFinder::MatchResult &Result) {
  const CallExpr *CE = getNodeAsType<CallExpr>(Result, "FuncCall");
  const DeclRefExpr *DR = getNodeAsType<DeclRefExpr>(Result, "declRef");
  const SourceManager &SM = DpctGlobalInfo::getSourceManager();
  if (DR && DpctGlobalInfo::useLogicalGroup()) {
    std::string ReplacedStr = MapNames::getDpctNamespace() +
                              "experimental::group" + "(" +
                              DR->getNameInfo().getAsString() + ", " +
                              DpctGlobalInfo::getItem(DR) + ")";
    SourceRange DefRange =
        getDefinitionRange(DR->getBeginLoc(), DR->getEndLoc());
    SourceLocation Begin = DefRange.getBegin();
    SourceLocation End = DefRange.getEnd();
    End = End.getLocWithOffset(Lexer::MeasureTokenLength(
        End, SM, DpctGlobalInfo::getContext().getLangOpts()));
    emplaceTransformation(replaceText(Begin, End, std::move(ReplacedStr),
                                      DpctGlobalInfo::getSourceManager()));
    return;
  }
  if (!CE)
    return;
  std::string FuncName =
      CE->getDirectCallee()->getNameInfo().getName().getAsString();

  struct ReportUnsupportedWarning {
    ReportUnsupportedWarning(SourceLocation SL, std::string FunctionName,
                             CooperativeGroupsFunctionRule *ThisPtrOfRule)
        : SL(SL), FunctionName(FunctionName), ThisPtrOfRule(ThisPtrOfRule) {}
    ~ReportUnsupportedWarning() {
      if (NeedReport) {
        ThisPtrOfRule->report(SL, Diagnostics::API_NOT_MIGRATED, true,
                              FunctionName);
      }
    }
    ReportUnsupportedWarning(const ReportUnsupportedWarning &) = delete;
    ReportUnsupportedWarning &
    operator=(const ReportUnsupportedWarning &) = delete;
    bool NeedReport = true;

  private:
    SourceLocation SL;
    std::string FunctionName;
    CooperativeGroupsFunctionRule *ThisPtrOfRule = nullptr;
  };

  ReportUnsupportedWarning RUW(CE->getBeginLoc(), FuncName, this);
  if (FuncName == "sync" || FuncName == "thread_rank" || FuncName == "size" ||
      FuncName == "shfl_down" || FuncName == "shfl_up" || FuncName == "shfl" ||
      FuncName == "shfl_xor" || FuncName == "meta_group_rank" ||
      FuncName == "meta_group_size" || FuncName == "reduce" ||
      FuncName == "thread_index" || FuncName == "group_index" ||
      FuncName == "num_threads" || FuncName == "inclusive_scan" ||
      FuncName == "exclusive_scan" || FuncName == "coalesced_threads" ||
      FuncName == "this_grid" || FuncName == "num_blocks" ||
      FuncName == "block_rank") {
    // There are 3 usages of cooperative groups APIs.
    // 1. cg::thread_block tb; tb.sync(); // member function
    // 2. cg::thread_block tb; cg::sync(tb); // free function
    // 3. cg::thread_block::sync(); // static function
    // Value meaning: is_migration_support/is_original_code_support
    // FunctionName  Case1 Case2 Case3
    // sync          1/1   1/1   0/1
    // thread_rank   1/1   1/1   0/1
    // size          1/1   0/0   1/1
    // num_threads   1/1   0/0   1/1
    // shfl_down     1/1   0/0   0/0
    // shfl_up       1/1   0/0   0/0
    // shfl_xor      1/1   0/0   0/0
    // meta_group_rank 1/1   0/0   0/0
    // meta_group_size 1/1   0/0   0/0

    ExprAnalysis EA(CE);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
    RUW.NeedReport = false;
  } else if (FuncName == "this_thread_block") {
    RUW.NeedReport = false;
    emplaceTransformation(new ReplaceStmt(CE, DpctGlobalInfo::getGroup(CE)));
  } else if (FuncName == "tiled_partition") {
    RUW.NeedReport = false;
    ExprAnalysis EA(CE);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();

    CheckParamType Checker1(
        0, "const class cooperative_groups::__v1::thread_block &");
    CheckIntergerTemplateArgValueNE Checker2(0, 32);
    CheckIntergerTemplateArgValueLE Checker3(0, 32);
    if (Checker1(CE) && Checker3(CE)) {
      auto FuncInfo = DeviceFunctionDecl::LinkRedecls(
          DpctGlobalInfo::getParentFunction(CE));
      if (FuncInfo) {
        FuncInfo->getVarMap().Dim = 3;
        if (Checker2(CE) && DpctGlobalInfo::useLogicalGroup()) {
          FuncInfo->addSubGroupSizeRequest(32, CE->getBeginLoc(),
                                           MapNames::getDpctNamespace() +
                                               "experimental::logical_group");
        } else {
          FuncInfo->addSubGroupSizeRequest(32, CE->getBeginLoc(),
                                           DpctGlobalInfo::getSubGroup(CE));
        }
      }
    }
  }
}

} // namespace dpct
} // namespace clang
