//===--------------- RulesLangAtomic.cpp ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RuleInfra/CallExprRewriter.h"
#include "RuleInfra/ExprAnalysis.h"
#include "RuleInfra/MigrationStatistics.h"
#include "RulesLang.h"
#include "RulesLang/MapNamesLang.h"
#include "Utility.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Type.h"
#include "clang/ASTMatchers/ASTMatchers.h"

#include <algorithm>
#include <ostream>
#include <sstream>
#include <string>
#include <unordered_map>
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

void AtomicFunctionRule::registerMatcher(MatchFinder &MF) {
  std::vector<std::string> AtomicFuncNames(
      MapNamesLang::AtomicFuncNamesMap.size());
  std::transform(
      MapNamesLang::AtomicFuncNamesMap.begin(),
      MapNamesLang::AtomicFuncNamesMap.end(), AtomicFuncNames.begin(),
      [](const std::pair<std::string, std::string> &p) { return p.first; });

  auto hasAnyAtomicFuncName = [&]() {
    return internal::Matcher<NamedDecl>(
        new internal::HasNameMatcher(AtomicFuncNames));
  };

  // Support all integer type, float, double, half and half2.
  auto supportedTypes = [&]() {
    return anyOf(hasType(pointsTo(isInteger())),
                 hasType(pointsTo(asString("float"))),
                 hasType(pointsTo(asString("double"))),
                 hasType(pointsTo(asString("__half"))),
                 hasType(pointsTo(asString("__half2"))));
  };

  auto supportedAtomicFunctions = [&]() {
    return allOf(hasAnyAtomicFuncName(), hasParameter(0, supportedTypes()));
  };

  auto unsupportedAtomicFunctions = [&]() {
    return allOf(hasAnyAtomicFuncName(),
                 unless(hasParameter(0, supportedTypes())));
  };

  MF.addMatcher(callExpr(callee(functionDecl(supportedAtomicFunctions())))
                    .bind("supportedAtomicFuncCall"),
                this);

  MF.addMatcher(callExpr(callee(functionDecl(unsupportedAtomicFunctions())))
                    .bind("unsupportedAtomicFuncCall"),
                this);
}

void AtomicFunctionRule::ReportUnsupportedAtomicFunc(const CallExpr *CE) {
  if (!CE)
    return;

  std::ostringstream OSS;
  // Atomic functions with __half are not supported.
  if (!CE->getDirectCallee())
    return;
  OSS << "half version of "
      << MapNames::ITFName.at(CE->getDirectCallee()->getName().str());
  report(CE->getBeginLoc(), Diagnostics::API_NOT_MIGRATED, false, OSS.str());
}

void AtomicFunctionRule::MigrateAtomicFunc(
    const CallExpr *CE, const ast_matchers::MatchFinder::MatchResult &Result) {
  if (!CE)
    return;

  // Don't migrate user defined function
  if (auto *CalleeDecl = CE->getDirectCallee()) {
    if (isUserDefinedDecl(CalleeDecl))
      return;
  } else {
    return;
  };

  const std::string FuncName = CE->getDirectCallee()->getName().str();
  if (!CallExprRewriterFactoryBase::RewriterMap)
    return;
  auto Iter = CallExprRewriterFactoryBase::RewriterMap->find(FuncName);
  if (Iter != CallExprRewriterFactoryBase::RewriterMap->end()) {
    ExprAnalysis EA(CE);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
    return;
  }
}

void AtomicFunctionRule::runRule(const MatchFinder::MatchResult &Result) {
  ReportUnsupportedAtomicFunc(
      getNodeAsType<CallExpr>(Result, "unsupportedAtomicFuncCall"));

  MigrateAtomicFunc(getNodeAsType<CallExpr>(Result, "supportedAtomicFuncCall"),
                    Result);
}

} // namespace dpct
} // namespace clang
