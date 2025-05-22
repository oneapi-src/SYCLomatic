//===---------------------- NVSHMEMAPIMigration.cpp -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------------===//

#include "NVSHMEMAPIMigration.h"
#include "RuleInfra/ExprAnalysis.h"

using namespace clang::dpct;
using namespace clang::ast_matchers;

void clang::dpct::NVSHMEMRule::registerMatcher(ast_matchers::MatchFinder &MF) {
  auto NvshmemAPI = [&]() {
    return hasAnyName(
        // Library Setup, Exit & Query
        "nvshmem_init", "nvshmem_my_pe", "nvshmem_n_pes", "nvshmem_finalize",
        "nvshmem_ptr", "nvshmem_info_get_version", "nvshmem_info_get_name",
        "nvshmemx_init_attr",
        // Memory Management
        "nvshmem_malloc", "nvshmem_free", "nvshmem_align", "nvshmem_calloc",
        // Team Management
        "nvshmem_team_my_pe", "nvshmem_team_n_pes", "nvshmem_team_get_config",
        "nvshmem_team_translate_pe", "nvshmem_team_split_strided",
        "nvshmem_team_split_2d", "nvshmem_team_destroy",
        // Nonblocking RMA
        "nvshmem_putmem_nbi",
        // Signalling Operations
        "nvshmemx_signal_op", "nvshmem_signal_wait_until",
        "nvshmem_putmem_signal_nbi");
  };

  MF.addMatcher(typeLoc(loc(qualType(hasDeclaration(namedDecl(hasAnyName(
                            "nvshmem_team_t", "nvshmem_team_config_t",
                            "nvshmemx_uniqueid_t", "nvshmemx_uniqueid_args_t",
                            "nvshmemx_init_args_t", "nvshmemx_init_attr_t"))))))
                    .bind("type"),
                this);

  MF.addMatcher(
      memberExpr(hasObjectExpression(declRefExpr(hasType(typedefDecl(hasAnyName(
                     "nvshmemx_uniqueid_t", "nvshmemx_uniqueid_args_t",
                     "nvshmemx_init_args_t", "nvshmemx_init_attr_t"))))))
          .bind("memberAccess"),
      this);

  MF.addMatcher(callExpr(callee(functionDecl(NvshmemAPI()))).bind("call"),
                this);

  MF.addMatcher(
      declRefExpr(to(enumConstantDecl(hasAnyName(
                      "NVSHMEM_TEAM_WORLD", "NVSHMEM_TEAM_INVALID",
                      "NVSHMEM_TEAM_SHARED", "NVSHMEMX_INIT_WITH_MPI_COMM",
                      "NVSHMEMX_INIT_WITH_SHMEM", "NVSHMEM_SIGNAL_SET",
                      "NVSHMEM_SIGNAL_ADD", "NVSHMEM_CMP_EQ", "NVSHMEM_CMP_NE",
                      "NVSHMEM_CMP_GT", "NVSHMEM_CMP_GE", "NVSHMEM_CMP_LT",
                      "NVSHMEM_CMP_LE"))))
          .bind("enumConstant"),
      this);
}

void clang::dpct::NVSHMEMRule::runRule(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  ExprAnalysis EA;

  if (auto ME = getNodeAsType<MemberExpr>(Result, "memberAccess")) {
    auto BaseTy = DpctGlobalInfo::getUnqualifiedTypeName(
        ME->getBase()->getType().getDesugaredType(*Result.Context),
        *Result.Context);
    auto MemberName = ME->getMemberNameInfo().getAsString();

    if (BaseTy == "nvshmemx_init_attr_v1") {
      if (MemberName == "args") {
        report(ME->getBeginLoc(), Diagnostics::API_NOT_MIGRATED, false,
               DpctGlobalInfo::getOriginalTypeName(ME->getBase()->getType()) +
                   "::" + ME->getMemberDecl()->getName().str());
        return;
      }
    } else if (BaseTy == "nvshmemx_init_args_v1" ||
               BaseTy == "nvshmemx_uniqueid_v1" ||
               BaseTy == "nvshmemx_uniqueid_args_v1") {
      report(ME->getBeginLoc(), Diagnostics::API_NOT_MIGRATED, false,
             DpctGlobalInfo::getOriginalTypeName(ME->getBase()->getType()) +
                 "::" + ME->getMemberDecl()->getName().str());
      return;
    }
  }

  if (const auto *TL = getNodeAsType<TypeLoc>(Result, "type")) {
    auto BaseTy = DpctGlobalInfo::getUnqualifiedTypeName(
        TL->getType().getDesugaredType(*Result.Context), *Result.Context);

    if (BaseTy == "nvshmemx_init_args_v1" || BaseTy == "nvshmemx_uniqueid_v1" ||
        BaseTy == "nvshmemx_uniqueid_args_v1") {
      report(TL->getBeginLoc(), Diagnostics::API_NOT_MIGRATED, false,
             DpctGlobalInfo::getOriginalTypeName(TL->getType()));
      return;
    }

    EA.analyze(*TL);
  } else if (const CallExpr *CE = getNodeAsType<CallExpr>(Result, "call")) {
    EA.analyze(CE);
  } else if (const DeclRefExpr *DRE =
                 getNodeAsType<DeclRefExpr>(Result, "enumConstant")) {
    EA.analyze(DRE);
  } else {
    return;
  }

  emplaceTransformation(EA.getReplacement());
  EA.applyAllSubExprRepl();
}
