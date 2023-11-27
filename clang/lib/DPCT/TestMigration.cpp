//===----------------------- TestMigration.cpp -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===--------------------------------------------------------------------===//

// This file only used on test.
// If feature is ready
// this file should be removed
#include "TestMigration.h"
#include "Schema.h"

using namespace clang::dpct;
using namespace clang::ast_matchers;

void clang::dpct::TESTRule::registerMatcher(ast_matchers::MatchFinder &MF) {
  MF.addMatcher(
      callExpr(callee(functionDecl(hasAnyName("faketest")))).bind("call"),
      this);
}

void clang::dpct::TESTRule::runRule(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  if (const CallExpr *CE = getNodeAsType<CallExpr>(Result, "call")) {

    if (CE->getNumArgs() != 0) {

      for (const auto *arg : CE->arguments()) {
        if (const auto *Ctor = dyn_cast<CXXConstructExpr>(arg)) {
          for (const auto *it : Ctor->arguments()) {
            // std::cout << it->IgnoreImpCasts()->getStmtClassName() << '\n';
            if (const auto *DRE = dyn_cast<DeclRefExpr>(it->IgnoreImpCasts())) {
              constructVarSchema(DRE);
              // serializeJsonArrayToFile(
              //     serializeSchemaToJsonArray(
              //         getRelatedTypeSchema(DRE->getType())),
              //     std::string("output_") +
              //         DpctGlobalInfo::getTypeName(DRE->getType()) + ".json");
            }
          }
        }
        if (const auto *CBTE = dyn_cast<CXXBindTemporaryExpr>(arg)) {
          if (const auto *Ctor =
                  dyn_cast<CXXConstructExpr>(CBTE->getSubExpr())) {
            for (const auto *it : Ctor->arguments()) {
              if (const auto *DRE =
                      dyn_cast<DeclRefExpr>(it->IgnoreImpCasts())) {
                constructVarSchema(DRE);
                // serializeJsonArrayToFile(
                //     serializeSchemaToJsonArray(
                //         getRelatedTypeSchema(DRE->getType())),
                //     std::string("output_") +
                //         DpctGlobalInfo::getTypeName(DRE->getType()) + ".json");
              }
            }
          }
        }
        if (const auto *DRE = dyn_cast<DeclRefExpr>(arg->IgnoreImpCasts())) {
          constructVarSchema(DRE);
          // serializeJsonArrayToFile(
          //     serializeSchemaToJsonArray(getRelatedTypeSchema(DRE->getType())),
          //     std::string("output_") +
          //         DpctGlobalInfo::getTypeName(DRE->getType()) + ".json");
        }
      }
    }
    serializeJsonArrayToFile(serializeSchemaToJsonArray(CTypeSchemaMap),
                             "output_all_cuda.json");
    serializeJsonArrayToFile(serializeSchemaToJsonArray(STypeSchemaMap),
                             "output_all_sycl.json");
  }
  return;
}
