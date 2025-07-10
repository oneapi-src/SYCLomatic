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

#define STRINGIZE(x) #x
#define EXPAND_AND_STRINGIZE(x) STRINGIZE(x)

#define FOR_ALL_STANDARD_RMA_TYPES(PREFIX, POSTFIX)                            \
  EXPAND_AND_STRINGIZE(nvshmem##PREFIX##_##float##_##POSTFIX),                 \
      EXPAND_AND_STRINGIZE(nvshmem##PREFIX##_##double##_##POSTFIX),            \
      EXPAND_AND_STRINGIZE(nvshmem##PREFIX##_##char##_##POSTFIX),              \
      EXPAND_AND_STRINGIZE(nvshmem##PREFIX##_##schar##_##POSTFIX),             \
      EXPAND_AND_STRINGIZE(nvshmem##PREFIX##_##short##_##POSTFIX),             \
      EXPAND_AND_STRINGIZE(nvshmem##PREFIX##_##int##_##POSTFIX),               \
      EXPAND_AND_STRINGIZE(nvshmem##PREFIX##_##long##_##POSTFIX),              \
      EXPAND_AND_STRINGIZE(nvshmem##PREFIX##_##longlong##_##POSTFIX),          \
      EXPAND_AND_STRINGIZE(nvshmem##PREFIX##_##uchar##_##POSTFIX),             \
      EXPAND_AND_STRINGIZE(nvshmem##PREFIX##_##ushort##_##POSTFIX),            \
      EXPAND_AND_STRINGIZE(nvshmem##PREFIX##_##uint##_##POSTFIX),              \
      EXPAND_AND_STRINGIZE(nvshmem##PREFIX##_##ulong##_##POSTFIX),             \
      EXPAND_AND_STRINGIZE(nvshmem##PREFIX##_##ulonglong##_##POSTFIX),         \
      EXPAND_AND_STRINGIZE(nvshmem##PREFIX##_##int8##_##POSTFIX),              \
      EXPAND_AND_STRINGIZE(nvshmem##PREFIX##_##int16##_##POSTFIX),             \
      EXPAND_AND_STRINGIZE(nvshmem##PREFIX##_##int32##_##POSTFIX),             \
      EXPAND_AND_STRINGIZE(nvshmem##PREFIX##_##int64##_##POSTFIX),             \
      EXPAND_AND_STRINGIZE(nvshmem##PREFIX##_##uint8##_##POSTFIX),             \
      EXPAND_AND_STRINGIZE(nvshmem##PREFIX##_##uint16##_##POSTFIX),            \
      EXPAND_AND_STRINGIZE(nvshmem##PREFIX##_##uint32##_##POSTFIX),            \
      EXPAND_AND_STRINGIZE(nvshmem##PREFIX##_##uint64##_##POSTFIX),            \
      EXPAND_AND_STRINGIZE(nvshmem##PREFIX##_##size##_##POSTFIX),              \
      EXPAND_AND_STRINGIZE(nvshmem##PREFIX##_##ptrdiff##_##POSTFIX)

#define FOR_ALL_SIZES(PREFIX, OP, POSTFIX)                                     \
  EXPAND_AND_STRINGIZE(nvshmem##PREFIX##_##OP##8##POSTFIX),                    \
      EXPAND_AND_STRINGIZE(nvshmem##PREFIX##_##OP##16##POSTFIX),               \
      EXPAND_AND_STRINGIZE(nvshmem##PREFIX##_##OP##32##POSTFIX),               \
      EXPAND_AND_STRINGIZE(nvshmem##PREFIX##_##OP##64##POSTFIX),               \
      EXPAND_AND_STRINGIZE(nvshmem##PREFIX##_##OP##128##POSTFIX)

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
        // RMA
        FOR_ALL_STANDARD_RMA_TYPES(, put) /*nvshmem_TYPENAME_put*/,
        FOR_ALL_STANDARD_RMA_TYPES(
            x, put_on_stream) /*nvshmemx_TYPENAME_put_on_stream*/,
        FOR_ALL_STANDARD_RMA_TYPES(x,
                                   put_block) /*nvshmemx_TYPENAME_put_block*/,
        FOR_ALL_STANDARD_RMA_TYPES(x, put_warp) /*nvshmemx_TYPENAME_put_warp*/,
        FOR_ALL_SIZES(, put, ) /*nvshmem_putSIZE*/,
        FOR_ALL_SIZES(x, put, _on_stream) /*nvshmemx_putSIZE_on_stream*/,
        FOR_ALL_SIZES(x, put, _block) /*nvshmem_putSIZE_block*/,
        FOR_ALL_SIZES(x, put, _warp) /*nvshmem_putSIZE_warp*/,
        FOR_ALL_STANDARD_RMA_TYPES(, iput) /*nvshmem_TYPENAME_iput*/,
        FOR_ALL_STANDARD_RMA_TYPES(
            x, iput_on_stream) /*nvshmemx_TYPENAME_iput_on_stream*/,
        FOR_ALL_STANDARD_RMA_TYPES(x,
                                   iput_block) /*nvshmemx_TYPENAME_iput_block*/,
        FOR_ALL_STANDARD_RMA_TYPES(x,
                                   iput_warp) /*nvshmemx_TYPENAME_iput_warp*/,
        FOR_ALL_SIZES(, iput, ) /*nvshmem_iputSIZE*/,
        FOR_ALL_SIZES(x, iput, _on_stream) /*nvshmem_iputSIZE_on_stream*/,
        FOR_ALL_SIZES(x, iput, _block) /*nvshmem_iputSIZE_block*/,
        FOR_ALL_SIZES(x, iput, _warp) /*nvshmem_iputSIZE_warp*/,
        "nvshmem_putmem", "nvshmemx_putmem_on_stream", "nvshmemx_putmem_block",
        "nvshmemx_putmem_warp",
        FOR_ALL_STANDARD_RMA_TYPES(, p) /*nvshmem_TYPENAME_p*/,
        FOR_ALL_STANDARD_RMA_TYPES(, get) /*nvshmem_TYPENAME_get*/,
        FOR_ALL_STANDARD_RMA_TYPES(
            x, get_on_stream) /*nvshmemx_TYPENAME_get_on_stream*/,
        FOR_ALL_STANDARD_RMA_TYPES(x,
                                   get_block) /*nvshmemx_TYPENAME_get_block*/,
        FOR_ALL_STANDARD_RMA_TYPES(x, get_warp) /*nvshmemx_TYPENAME_get_warp*/,
        FOR_ALL_SIZES(, get, ) /*nvshmem_getSIZE*/,
        FOR_ALL_SIZES(x, get, _on_stream) /*nvshmem_getSIZE_on_stream*/,
        FOR_ALL_SIZES(x, get, _block) /*nvshmem_getSIZE_block*/,
        FOR_ALL_SIZES(x, get, _warp) /*nvshmem_getSIZE_warp*/,
        FOR_ALL_STANDARD_RMA_TYPES(, iget) /*nvshmem_TYPENAME_iget*/,
        FOR_ALL_STANDARD_RMA_TYPES(
            x, iget_on_stream) /*nvshmemx_TYPENAME_iget_on_stream*/,
        FOR_ALL_STANDARD_RMA_TYPES(x,
                                   iget_block) /*nvshmemx_TYPENAME_iget_block*/,
        FOR_ALL_STANDARD_RMA_TYPES(x,
                                   iget_warp) /*nvshmemx_TYPENAME_iget_warp*/,
        FOR_ALL_SIZES(, iget, ) /*nvshmem_igetSIZE*/,
        FOR_ALL_SIZES(x, iget, _on_stream) /*nvshmem_igetSIZE_on_stream*/,
        FOR_ALL_SIZES(x, iget, _block) /*nvshmem_igetSIZE_block*/,
        FOR_ALL_SIZES(x, iget, _warp) /*nvshmem_igetSIZE_warp*/,
        "nvshmem_getmem", "nvshmemx_getmem_on_stream", "nvshmemx_getmem_block",
        "nvshmemx_getmem_warp",
        FOR_ALL_STANDARD_RMA_TYPES(, g) /*nvshmem_TYPENAME_g*/,
        // Nonblocking RMA
        FOR_ALL_STANDARD_RMA_TYPES(, put_nbi) /*nvshmem_TYPENAME_put_nbi*/,
        FOR_ALL_STANDARD_RMA_TYPES(
            x, put_nbi_on_stream) /*nvshmemx_TYPENAME_put_nbi_on_stream*/,
        FOR_ALL_STANDARD_RMA_TYPES(
            x, put_nbi_block) /*nvshmemx_TYPENAME_put_nbi_block*/,
        FOR_ALL_STANDARD_RMA_TYPES(
            x, put_nbi_warp) /*nvshmemx_TYPENAME_put_nbi_warp*/,
        FOR_ALL_SIZES(, put, _nbi) /*nvshmem_putSIZE_nbi*/,
        FOR_ALL_SIZES(x, put,
                      _nbi_on_stream) /*nvshmemx_putSIZE_nbi_on_stream*/,
        FOR_ALL_SIZES(x, put, _nbi_block) /*nvshmem_putSIZE_nbi_block*/,
        FOR_ALL_SIZES(x, put, _nbi_warp) /*nvshmem_putSIZE_nbi_warp*/,
        "nvshmem_putmem_nbi", "nvshmemx_putmem_nbi_on_stream",
        "nvshmemx_putmem_nbi_block", "nvshmemx_putmem_nbi_warp",
        FOR_ALL_STANDARD_RMA_TYPES(, get_nbi) /*nvshmem_TYPENAME_get_nbi*/,
        FOR_ALL_STANDARD_RMA_TYPES(
            x, get_nbi_on_stream) /*nvshmemx_TYPENAME_get_nbi_on_stream*/,
        FOR_ALL_STANDARD_RMA_TYPES(
            x, get_nbi_block) /*nvshmemx_TYPENAME_get_nbi_block*/,
        FOR_ALL_STANDARD_RMA_TYPES(
            x, get_nbi_warp) /*nvshmemx_TYPENAME_get_nbi_warp*/,
        FOR_ALL_SIZES(, get, _nbi) /*nvshmem_getSIZE_nbi*/,
        FOR_ALL_SIZES(x, get,
                      _nbi_on_stream) /*nvshmemx_getSIZE_nbi_on_stream*/,
        FOR_ALL_SIZES(x, get, _nbi_block) /*nvshmem_getSIZE_nbi_block*/,
        FOR_ALL_SIZES(x, get, _nbi_warp) /*nvshmem_getSIZE_nbi_warp*/,
        "nvshmem_getmem_nbi", "nvshmemx_getmem_nbi_on_stream",
        "nvshmemx_getmem_nbi_block", "nvshmemx_getmem_nbi_warp",
        // Memory Ordering
        "nvshmem_fence", "nvshmem_quiet", "nvshmemx_quiet_on_stream",
        // Collective Operations
        "nvshmemx_barrier_all_on_stream",
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

#undef STRINGIZE
#undef EXPAND_AND_STRINGIZE
#undef FOR_ALL_STANDARD_RMA_TYPES
#undef FOR_ALL_SIZES
