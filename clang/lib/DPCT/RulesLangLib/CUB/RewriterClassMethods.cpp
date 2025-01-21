//===--------------- RewriterClassMethods.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AnalysisInfo.h"
#include "RuleInfra/CallExprRewriter.h"
#include "CallExprRewriterCUB.h"
#include "RuleInfra/CallExprRewriterCommon.h"
#include "RulesInclude/InclusionHeaders.h"
#include "clang/AST/ExprCXX.h"
#include "llvm/Support/raw_ostream.h"
#include <functional>

using namespace clang::dpct;

namespace {
static inline std::function<std::string(const CallExpr *)>
printCallExprPretty() {
  return [](const CallExpr *C) {
    std::string Buffer;
    llvm::raw_string_ostream OS(Buffer);
    C->printPretty(OS, nullptr,
                   DpctGlobalInfo::getContext().getPrintingPolicy());
    return Buffer;
  };
}

class CheckCUBEnumTemplateArg {
  unsigned Idx;

public:
  CheckCUBEnumTemplateArg(unsigned I) : Idx(I) {}
  bool operator()(const CallExpr *C) const {
    ArrayRef<TemplateArgumentLoc> TemplateArgsList;
    const auto *Callee = C->getCallee()->IgnoreImplicitAsWritten();
    if (const auto *DRE = dyn_cast<DeclRefExpr>(Callee))
      TemplateArgsList = DRE->template_arguments();
    else if (const auto *ULE = dyn_cast<UnresolvedLookupExpr>(Callee))
      TemplateArgsList = ULE->template_arguments();
    if (Idx >= TemplateArgsList.size())
      return true;
    auto Arg = TemplateArgsList[Idx];
    if (Arg.getArgument().isDependent())
      return true;
    const auto *EnumArg = Arg.getArgument().getAsExpr();
    Expr::EvalResult ER;
    int64_t Value = -1;
    if (!EnumArg->isValueDependent() &&
        EnumArg->EvaluateAsInt(ER, DpctGlobalInfo::getContext())) {
      Value = ER.Val.getInt().getExtValue();
      return Value == 1 || Value == 0;
    }
    return false;
  }
};
} // namespace

RewriterMap dpct::createClassMethodsRewriterMap() {
  return RewriterMap{
      // cub::ArgIndexInputIterator.normalize
      FEATURE_REQUEST_FACTORY(
          HelperFeatureEnum::device_ext,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_DPCT_DPL_Utils,
              ASSIGN_FACTORY_ENTRY("cub::ArgIndexInputIterator.normalize",
                                   MemberExprBase(),
                                   MEMBER_CALL(MemberExprBase(), false,
                                               LITERAL("create_normalize")))))
      // cub::BlockRadixSort.Sort
      HEADER_INSERT_FACTORY(
          HeaderType::HT_DPCT_GROUP_Utils,
          CASE_FACTORY_ENTRY(
              CASE(
                  makeCheckAnd(
                      makeCheckAnd(CheckArgCount(3, std::equal_to<>(),
                                                 /*IncludeDefaultArg=*/false),
                                   CheckParamType(1, "int", /*isStrict=*/true)),
                      CheckParamType(2, "int", /*isStrict=*/true)),
                  MEMBER_CALL_FACTORY_ENTRY("cub::BlockRadixSort.Sort",
                                            MemberExprBase(), false, "sort",
                                            NDITEM, ARG(0), ARG(1), ARG(2))),
              CASE(makeCheckAnd(CheckArgCount(2, std::equal_to<>(),
                                              /*IncludeDefaultArg=*/false),
                                CheckParamType(1, "int", /*isStrict=*/true)),
                   MEMBER_CALL_FACTORY_ENTRY("cub::BlockRadixSort.Sort",
                                             MemberExprBase(), false, "sort",
                                             NDITEM, ARG(0), ARG(1))),
              CASE(CheckArgCount(1, std::equal_to<>(),
                                 /*IncludeDefaultArg=*/false),
                   MEMBER_CALL_FACTORY_ENTRY("cub::BlockRadixSort.Sort",
                                             MemberExprBase(), false, "sort",
                                             NDITEM, ARG(0))),
              OTHERWISE(UNSUPPORT_FACTORY_ENTRY("cub::BlockRadixSort.Sort",
                                                Diagnostics::API_NOT_MIGRATED,
                                                printCallExprPretty()))))
      // cub::BlockRadixSort.SortDescending
      HEADER_INSERT_FACTORY(
          HeaderType::HT_DPCT_GROUP_Utils,
          CASE_FACTORY_ENTRY(
              CASE(
                  makeCheckAnd(
                      makeCheckAnd(CheckArgCount(3, std::equal_to<>(),
                                                 /*IncludeDefaultArg=*/false),
                                   CheckParamType(1, "int", /*isStrict=*/true)),
                      CheckParamType(2, "int", /*isStrict=*/true)),
                  MEMBER_CALL_FACTORY_ENTRY(
                      "cub::BlockRadixSort.SortDescending", MemberExprBase(),
                      false, "sort_descending", NDITEM, ARG(0), ARG(1),
                      ARG(2))),
              CASE(makeCheckAnd(CheckArgCount(2, std::equal_to<>(),
                                              /*IncludeDefaultArg=*/false),
                                CheckParamType(1, "int", /*isStrict=*/true)),
                   MEMBER_CALL_FACTORY_ENTRY(
                       "cub::BlockRadixSort.SortDescending", MemberExprBase(),
                       false, "sort_descending", NDITEM, ARG(0), ARG(1))),
              CASE(CheckArgCount(1, std::equal_to<>(),
                                 /*IncludeDefaultArg=*/false),
                   MEMBER_CALL_FACTORY_ENTRY(
                       "cub::BlockRadixSort.SortDescending", MemberExprBase(),
                       false, "sort_descending", NDITEM, ARG(0))),
              OTHERWISE(UNSUPPORT_FACTORY_ENTRY(
                  "cub::BlockRadixSort.SortDescending",
                  Diagnostics::API_NOT_MIGRATED, printCallExprPretty()))))
      // cub::BlockRadixSort.SortBlockedToStriped
      HEADER_INSERT_FACTORY(
          HeaderType::HT_DPCT_GROUP_Utils,
          CASE_FACTORY_ENTRY(
              CASE(
                  makeCheckAnd(
                      makeCheckAnd(CheckArgCount(3, std::equal_to<>(),
                                                 /*IncludeDefaultArg=*/false),
                                   CheckParamType(1, "int", /*isStrict=*/true)),
                      CheckParamType(2, "int", /*isStrict=*/true)),
                  MEMBER_CALL_FACTORY_ENTRY(
                      "cub::BlockRadixSort.SortBlockedToStriped",
                      MemberExprBase(), false, "sort_blocked_to_striped",
                      NDITEM, ARG(0), ARG(1), ARG(2))),
              CASE(makeCheckAnd(CheckArgCount(2, std::equal_to<>(),
                                              /*IncludeDefaultArg=*/false),
                                CheckParamType(1, "int", /*isStrict=*/true)),
                   MEMBER_CALL_FACTORY_ENTRY(
                       "cub::BlockRadixSort.SortBlockedToStriped",
                       MemberExprBase(), false, "sort_blocked_to_striped",
                       NDITEM, ARG(0), ARG(1))),
              CASE(CheckArgCount(1, std::equal_to<>(),
                                 /*IncludeDefaultArg=*/false),
                   MEMBER_CALL_FACTORY_ENTRY(
                       "cub::BlockRadixSort.SortBlockedToStriped",
                       MemberExprBase(), false, "sort_blocked_to_striped",
                       NDITEM, ARG(0))),
              OTHERWISE(UNSUPPORT_FACTORY_ENTRY(
                  "cub::BlockRadixSort.SortBlockedToStriped",
                  Diagnostics::API_NOT_MIGRATED, printCallExprPretty()))))
      // cub::BlockRadixSort.SortDescendingBlockedToStriped
      HEADER_INSERT_FACTORY(
          HeaderType::HT_DPCT_GROUP_Utils,
          CASE_FACTORY_ENTRY(
              CASE(
                  makeCheckAnd(
                      makeCheckAnd(CheckArgCount(3, std::equal_to<>(),
                                                 /*IncludeDefaultArg=*/false),
                                   CheckParamType(1, "int", /*isStrict=*/true)),
                      CheckParamType(2, "int", /*isStrict=*/true)),
                  MEMBER_CALL_FACTORY_ENTRY(
                      "cub::BlockRadixSort.SortDescendingBlockedToStriped",
                      MemberExprBase(), false,
                      "sort_descending_blocked_to_striped", NDITEM, ARG(0),
                      ARG(1), ARG(2))),
              CASE(makeCheckAnd(CheckArgCount(2, std::equal_to<>(),
                                              /*IncludeDefaultArg=*/false),
                                CheckParamType(1, "int", /*isStrict=*/true)),
                   MEMBER_CALL_FACTORY_ENTRY(
                       "cub::BlockRadixSort.SortDescendingBlockedToStriped",
                       MemberExprBase(), false,
                       "sort_descending_blocked_to_striped", NDITEM, ARG(0),
                       ARG(1))),
              CASE(CheckArgCount(1, std::equal_to<>(),
                                 /*IncludeDefaultArg=*/false),
                   MEMBER_CALL_FACTORY_ENTRY(
                       "cub::BlockRadixSort.SortDescendingBlockedToStriped",
                       MemberExprBase(), false,
                       "sort_descending_blocked_to_striped", NDITEM, ARG(0))),
              OTHERWISE(UNSUPPORT_FACTORY_ENTRY(
                  "cub::BlockRadixSort.SortDescendingBlockedToStriped",
                  Diagnostics::API_NOT_MIGRATED, printCallExprPretty()))))
      // cub::BlockExchange.BlockedToStriped
      HEADER_INSERT_FACTORY(HeaderType::HT_DPCT_GROUP_Utils,
                            MEMBER_CALL_FACTORY_ENTRY(
                                "cub::BlockExchange.BlockedToStriped",
                                MemberExprBase(), false, "blocked_to_striped",
                                NDITEM, ARG(0), ARG(1)))
      // cub::BlockExchange.StripedToBlocked
      HEADER_INSERT_FACTORY(HeaderType::HT_DPCT_GROUP_Utils,
                            MEMBER_CALL_FACTORY_ENTRY(
                                "cub::BlockExchange.StripedToBlocked",
                                MemberExprBase(), false, "striped_to_blocked",
                                NDITEM, ARG(0), ARG(1)))
      // cub::BlockExchange.ScatterToBlocked
      HEADER_INSERT_FACTORY(HeaderType::HT_DPCT_GROUP_Utils,
                            MEMBER_CALL_FACTORY_ENTRY(
                                "cub::BlockExchange.ScatterToBlocked",
                                MemberExprBase(), false, "scatter_to_blocked",
                                NDITEM, ARG(0), ARG(1)))
      // cub::BlockExchange.ScatterToStriped
      HEADER_INSERT_FACTORY(HeaderType::HT_DPCT_GROUP_Utils,
                            MEMBER_CALL_FACTORY_ENTRY(
                                "cub::BlockExchange.ScatterToStriped",
                                MemberExprBase(), false, "scatter_to_striped",
                                NDITEM, ARG(0), ARG(1)))
      // cub::BlockShuffle.Offset
      HEADER_INSERT_FACTORY(
          HeaderType::HT_DPCT_GROUP_Utils,
          CASE_FACTORY_ENTRY(
              CASE(CheckArgCount(2, std::equal_to<>(), false),
                   MEMBER_CALL_FACTORY_ENTRY("cub::BlockShuffle.Offset",
                                             MemberExprBase(), false, "select",
                                             NDITEM, ARG(0), ARG(1))),
              CASE(CheckArgCount(3, std::equal_to<>(), false),
                   MEMBER_CALL_FACTORY_ENTRY("cub::BlockShuffle.Offset",
                                             MemberExprBase(), false, "select",
                                             NDITEM, ARG(0), ARG(1), ARG(2))),
              OTHERWISE(UNSUPPORT_FACTORY_ENTRY("cub::BlockShuffle.Offset",
                                                Diagnostics::API_NOT_MIGRATED,
                                                printCallExprPretty()))))
      // cub::BlockShuffle.Rotate
      HEADER_INSERT_FACTORY(
          HeaderType::HT_DPCT_GROUP_Utils,
          CASE_FACTORY_ENTRY(
              CASE(CheckArgCount(2, std::equal_to<>(), false),
                   MEMBER_CALL_FACTORY_ENTRY("cub::BlockShuffle.Rotate",
                                             MemberExprBase(), false, "select2",
                                             NDITEM, ARG(0), ARG(1))),
              CASE(CheckArgCount(3, std::equal_to<>(), false),
                   MEMBER_CALL_FACTORY_ENTRY("cub::BlockShuffle.Rotate",
                                             MemberExprBase(), false, "select2",
                                             NDITEM, ARG(0), ARG(1), ARG(2))),
              OTHERWISE(UNSUPPORT_FACTORY_ENTRY("cub::BlockShuffle.Rotate",
                                                Diagnostics::API_NOT_MIGRATED,
                                                printCallExprPretty()))))
      // cub::BlockShuffle.Up
      HEADER_INSERT_FACTORY(
          HeaderType::HT_DPCT_GROUP_Utils,
          CASE_FACTORY_ENTRY(
              CASE(CheckArgCount(2),
                   MEMBER_CALL_FACTORY_ENTRY(
                       "cub::BlockShuffle.Up", MemberExprBase(), false,
                       "shuffle_right", NDITEM, ARG(0), ARG(1))),
              CASE(CheckArgCount(3),
                   MEMBER_CALL_FACTORY_ENTRY(
                       "cub::BlockShuffle.Up", MemberExprBase(), false,
                       "shuffle_right", NDITEM, ARG(0), ARG(1), ARG(2))),
              OTHERWISE(UNSUPPORT_FACTORY_ENTRY("cub::BlockShuffle.Up",
                                                Diagnostics::API_NOT_MIGRATED,
                                                printCallExprPretty()))))
      // cub::BlockShuffle.Down
      HEADER_INSERT_FACTORY(
          HeaderType::HT_DPCT_GROUP_Utils,
          CASE_FACTORY_ENTRY(
              CASE(CheckArgCount(2),
                   MEMBER_CALL_FACTORY_ENTRY(
                       "cub::BlockShuffle.Down", MemberExprBase(), false,
                       "shuffle_left", NDITEM, ARG(0), ARG(1))),
              CASE(CheckArgCount(3),
                   MEMBER_CALL_FACTORY_ENTRY(
                       "cub::BlockShuffle.Down", MemberExprBase(), false,
                       "shuffle_left", NDITEM, ARG(0), ARG(1), ARG(2))),
              OTHERWISE(UNSUPPORT_FACTORY_ENTRY("cub::BlockShuffle.Down",
                                                Diagnostics::API_NOT_MIGRATED,
                                                printCallExprPretty()))))
      // cub::BlockLoad.Load
      HEADER_INSERT_FACTORY(
          HeaderType::HT_DPCT_GROUP_Utils,
          CASE_FACTORY_ENTRY(
              CASE(makeCheckAnd(CheckArgCount(2), CheckCUBEnumTemplateArg(3)),
                   MEMBER_CALL_FACTORY_ENTRY("cub::BlockLoad.Load",
                                             MemberExprBase(), false, "load",
                                             NDITEM, ARG(0), ARG(1))),
              CASE(makeCheckAnd(CheckArgCount(3), CheckCUBEnumTemplateArg(3)),
                   MEMBER_CALL_FACTORY_ENTRY("cub::BlockLoad.Load",
                                             MemberExprBase(), false, "load",
                                             NDITEM, ARG(0), ARG(1), ARG(2))),
              OTHERWISE(UNSUPPORT_FACTORY_ENTRY("cub::BlockLoad.Load",
                                                Diagnostics::API_NOT_MIGRATED,
                                                printCallExprPretty()))))
      // cub::BlockStore.Store
      HEADER_INSERT_FACTORY(
          HeaderType::HT_DPCT_GROUP_Utils,
          CASE_FACTORY_ENTRY(
              CASE(makeCheckAnd(CheckArgCount(2), CheckCUBEnumTemplateArg(3)),
                   MEMBER_CALL_FACTORY_ENTRY("cub::BlockStore.Store",
                                             MemberExprBase(), false, "store",
                                             NDITEM, ARG(0), ARG(1))),
              CASE(makeCheckAnd(CheckArgCount(3), CheckCUBEnumTemplateArg(3)),
                   MEMBER_CALL_FACTORY_ENTRY("cub::BlockStore.Store",
                                             MemberExprBase(), false, "store",
                                             NDITEM, ARG(0), ARG(1), ARG(2))),
              OTHERWISE(UNSUPPORT_FACTORY_ENTRY("cub::BlockStore.Store",
                                                Diagnostics::API_NOT_MIGRATED,
                                                printCallExprPretty()))))

  };
}
