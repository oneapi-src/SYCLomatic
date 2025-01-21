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
  auto nvSHMEM_API = [&]() {
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
        "nvshmem_team_split_2d", "nvshmem_team_destroy");
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

  MF.addMatcher(callExpr(callee(functionDecl(nvSHMEM_API()))).bind("call"),
                this);

  MF.addMatcher(
      declRefExpr(to(enumConstantDecl(hasAnyName(
                      "NVSHMEM_TEAM_WORLD", "NVSHMEM_TEAM_INVALID",
                      "NVSHMEM_TEAM_SHARED", "NVSHMEMX_INIT_WITH_MPI_COMM",
                      "NVSHMEMX_INIT_WITH_SHMEM"))))
          .bind("enum"),
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
    std::string FuncName = "";
    const FunctionDecl *FD = CE->getDirectCallee();
    if (FD) {
      FuncName = FD->getNameInfo().getName().getAsString();
    }

    if (!FuncName.empty()) {
      if (FuncName == "nvshmemx_init_attr") {
        // Get function arguments
        std::string nvshmem_rt = "";
        std::string nvshmem_init_rt = "";
        std::string attr_arg = "";

        // Get the first argument's data
        const Expr *Arg0 = CE->getArg(0);

        // Binary op on first argument is not supported
        if (dyn_cast<BinaryOperator>(Arg0->IgnoreImpCasts())) {
          report(CE->getBeginLoc(), Diagnostics::API_NOT_MIGRATED, false,
                 FuncName);
          return;
        }

        // Get first argument's init value
        if (auto DRE = dyn_cast<DeclRefExpr>(Arg0->IgnoreImpCasts())) {
          if (const VarDecl *VD = dyn_cast<VarDecl>(DRE->getDecl())) {
            if (VD->hasInit()) {
              // get the init value from definition
              if (auto Init =
                      dyn_cast<DeclRefExpr>(VD->getInit()->IgnoreImplicit())) {
                nvshmem_init_rt = Init->getNameInfo().getName().getAsString();
              }
            }
          }
        }

        // Get the first argument as string
        nvshmem_rt = Lexer::getSourceText(
            CharSourceRange::getTokenRange(Arg0->getSourceRange()),
            DpctGlobalInfo::getSourceManager(), LangOptions());

        if (nvshmem_rt == "0" || nvshmem_init_rt == "0") {
          emplaceTransformation(new ReplaceStmt(CE, "ishmem_init()"));
          return;
        }

        std::string ishmem_rt = "";
        if (nvshmem_rt == "NVSHMEMX_INIT_WITH_MPI_COMM") {
          ishmem_rt = "ISHMEMX_RUNTIME_MPI";
        } else if (nvshmem_rt == "NVSHMEMX_INIT_WITH_SHMEM") {
          ishmem_rt = "ISHMEMX_RUNTIME_OPENSHMEM";
        }

        if (nvshmem_init_rt == "NVSHMEMX_INIT_WITH_MPI_COMM" ||
            nvshmem_init_rt == "NVSHMEMX_INIT_WITH_SHMEM") {
          ishmem_rt = "static_cast<ishmemx_runtime_type_t>(" + nvshmem_rt + ")";
        }

        // Get the second argument as string
        attr_arg = Lexer::getSourceText(
            CharSourceRange::getTokenRange(CE->getArg(1)->getSourceRange()),
            DpctGlobalInfo::getSourceManager(), LangOptions());

        if (ishmem_rt.empty()) {
          report(CE->getBeginLoc(), Diagnostics::API_NOT_MIGRATED, false,
                 FuncName);
          return;
        } else {
          auto &SM = DpctGlobalInfo::getSourceManager();

          auto IndentLoc = CE->getBeginLoc();
          if (IndentLoc.isMacroID())
            IndentLoc = SM.getExpansionLoc(IndentLoc);

          std::string set_ishmem_runtime =
              "(" + attr_arg + ")->runtime = " + ishmem_rt + ";";
          set_ishmem_runtime += getNL();
          set_ishmem_runtime += getIndent(IndentLoc, SM).str();

          emplaceTransformation(
              new InsertBeforeStmt(CE, std::move(set_ishmem_runtime)));
        }
      }
    }

    EA.analyze(CE);
  } else if (const DeclRefExpr *DRE =
                 getNodeAsType<DeclRefExpr>(Result, "enum")) {
    EA.analyze(DRE);
  } else {
    return;
  }

  emplaceTransformation(EA.getReplacement());
  EA.applyAllSubExprRepl();
}
