//===--------------- CallExprRewriterWarp.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CallExprRewriter.h"
#include "CallExprRewriterCommon.h"

namespace clang {
namespace dpct {

void CallExprRewriterFactoryBase::initRewriterMapWarp() {
  RewriterMap->merge(
  std::unordered_map<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>(
  {// __all_sync
   SUBGROUPSIZE_FACTORY(
   UINT_MAX, MapNames::getClNamespace() + "all_of_group",
   CALL_FACTORY_ENTRY(
   "__all_sync",
   CALL(
   MapNames::getClNamespace() + "all_of_group", SUBGROUP,
   makeCombinedArg(
   ARG("(~"), makeCombinedArg(
              makeCallArgCreatorWithCall(0),
              makeCombinedArg(
              makeCombinedArg(
              ARG(" & (0x1 << "),
              makeCombinedArg(SUBGROUP, ARG(".get_local_linear_id())) || "))),
              makeCallArgCreatorWithCall(1)))))))

   // __all
   SUBGROUPSIZE_FACTORY(
   UINT_MAX, MapNames::getClNamespace() + "all_of_group",
   CALL_FACTORY_ENTRY("__all", CALL(MapNames::getClNamespace() + "all_of_group",
                                    SUBGROUP, makeCallArgCreatorWithCall(0))))

   // __any_sync
   SUBGROUPSIZE_FACTORY(
   UINT_MAX, MapNames::getClNamespace() + "any_of_group",
   CALL_FACTORY_ENTRY(
   "__any_sync", CALL(MapNames::getClNamespace() + "any_of_group", SUBGROUP,
                      makeCombinedArg(
                      ARG("("), makeCombinedArg(
                                makeCallArgCreatorWithCall(0),
                                makeCombinedArg(
                                makeCombinedArg(
                                ARG(" & (0x1 << "),
                                makeCombinedArg(
                                SUBGROUP, ARG(".get_local_linear_id())) && "))),
                                makeCallArgCreatorWithCall(1)))))))

   // __any
   SUBGROUPSIZE_FACTORY(
   UINT_MAX, MapNames::getClNamespace() + "any_of_group",
   CALL_FACTORY_ENTRY("__any", CALL(MapNames::getClNamespace() + "any_of_group",
                                    SUBGROUP, makeCallArgCreatorWithCall(0))))

   // __shfl_up_sync
   CONDITIONAL_FACTORY_ENTRY(
   CheckArgIsConstantIntWithValue(3, 32),
   SUBGROUPSIZE_FACTORY(
   UINT_MAX, MapNames::getDpctNamespace() + "shift_sub_group_right",
   WARNING_FACTORY_ENTRY(
   "__shfl_up_sync",
   FEATURE_REQUEST_FACTORY(
   HelperFeatureEnum::Util_shift_sub_group_right,
   CALL_FACTORY_ENTRY(
   "__shfl_up_sync",
   CALL(MapNames::getDpctNamespace() + "shift_sub_group_right", SUBGROUP,
        makeCallArgCreatorWithCall(1), makeCallArgCreatorWithCall(2)))),
   Diagnostics::MASK_UNSUPPORTED,
   ARG(MapNames::getDpctNamespace() + "shift_sub_group_right"))),
   SUBGROUPSIZE_FACTORY(
   UINT_MAX, MapNames::getDpctNamespace() + "shift_sub_group_right",
   WARNING_FACTORY_ENTRY(
   "__shfl_up_sync",
   FEATURE_REQUEST_FACTORY(
   HelperFeatureEnum::Util_shift_sub_group_right,
   CALL_FACTORY_ENTRY(
   "__shfl_up_sync",
   CALL(MapNames::getDpctNamespace() + "shift_sub_group_right", SUBGROUP,
        makeCallArgCreatorWithCall(1), makeCallArgCreatorWithCall(2),
        makeCallArgCreatorWithCall(3)))),
   Diagnostics::MASK_UNSUPPORTED,
   ARG(MapNames::getDpctNamespace() + "shift_sub_group_right"))))

   // __shfl_up
   CONDITIONAL_FACTORY_ENTRY(
   CheckArgIsConstantIntWithValue(2, 32),
   SUBGROUPSIZE_FACTORY(
   UINT_MAX, MapNames::getDpctNamespace() + "shift_sub_group_right",
   FEATURE_REQUEST_FACTORY(
   HelperFeatureEnum::Util_shift_sub_group_right,
   CALL_FACTORY_ENTRY(
   "__shfl_up",
   CALL(MapNames::getDpctNamespace() + "shift_sub_group_right", SUBGROUP,
        makeCallArgCreatorWithCall(0), makeCallArgCreatorWithCall(1))))),
   SUBGROUPSIZE_FACTORY(
   UINT_MAX, MapNames::getDpctNamespace() + "shift_sub_group_right",
   FEATURE_REQUEST_FACTORY(
   HelperFeatureEnum::Util_shift_sub_group_right,
   CALL_FACTORY_ENTRY(
   "__shfl_up",
   CALL(MapNames::getDpctNamespace() + "shift_sub_group_right", SUBGROUP,
        makeCallArgCreatorWithCall(0), makeCallArgCreatorWithCall(1),
        makeCallArgCreatorWithCall(2))))))

   // __shfl_down_sync
   CONDITIONAL_FACTORY_ENTRY(
   CheckArgIsConstantIntWithValue(3, 32),
   SUBGROUPSIZE_FACTORY(
   UINT_MAX, MapNames::getDpctNamespace() + "shift_sub_group_left",
   WARNING_FACTORY_ENTRY(
   "__shfl_down_sync",
   FEATURE_REQUEST_FACTORY(
   HelperFeatureEnum::Util_shift_sub_group_left,
   CALL_FACTORY_ENTRY(
   "__shfl_down_sync",
   CALL(MapNames::getDpctNamespace() + "shift_sub_group_left", SUBGROUP,
        makeCallArgCreatorWithCall(1), makeCallArgCreatorWithCall(2)))),
   Diagnostics::MASK_UNSUPPORTED,
   ARG(MapNames::getDpctNamespace() + "shift_sub_group_left"))),
   SUBGROUPSIZE_FACTORY(
   UINT_MAX, MapNames::getDpctNamespace() + "shift_sub_group_left",
   WARNING_FACTORY_ENTRY(
   "__shfl_down_sync",
   FEATURE_REQUEST_FACTORY(
   HelperFeatureEnum::Util_shift_sub_group_left,
   CALL_FACTORY_ENTRY(
   "__shfl_down_sync",
   CALL(MapNames::getDpctNamespace() + "shift_sub_group_left", SUBGROUP,
        makeCallArgCreatorWithCall(1), makeCallArgCreatorWithCall(2),
        makeCallArgCreatorWithCall(3)))),
   Diagnostics::MASK_UNSUPPORTED,
   ARG(MapNames::getDpctNamespace() + "shift_sub_group_left"))))

   // __shfl_down
   CONDITIONAL_FACTORY_ENTRY(
   CheckArgIsConstantIntWithValue(2, 32),
   SUBGROUPSIZE_FACTORY(
   UINT_MAX, MapNames::getDpctNamespace() + "shift_sub_group_left",
   FEATURE_REQUEST_FACTORY(
   HelperFeatureEnum::Util_shift_sub_group_left,
   CALL_FACTORY_ENTRY(
   "__shfl_down",
   CALL(MapNames::getDpctNamespace() + "shift_sub_group_left", SUBGROUP,
        makeCallArgCreatorWithCall(0), makeCallArgCreatorWithCall(1))))),
   SUBGROUPSIZE_FACTORY(
   UINT_MAX, MapNames::getDpctNamespace() + "shift_sub_group_left",
   FEATURE_REQUEST_FACTORY(
   HelperFeatureEnum::Util_shift_sub_group_left,
   CALL_FACTORY_ENTRY(
   "__shfl_down",
   CALL(MapNames::getDpctNamespace() + "shift_sub_group_left", SUBGROUP,
        makeCallArgCreatorWithCall(0), makeCallArgCreatorWithCall(1),
        makeCallArgCreatorWithCall(2))))))

   //__shfl_sync
   CONDITIONAL_FACTORY_ENTRY(
   CheckArgIsConstantIntWithValue(3, 32),
   SUBGROUPSIZE_FACTORY(
   UINT_MAX, MapNames::getDpctNamespace() + "select_from_sub_group",
   WARNING_FACTORY_ENTRY(
   "__shfl_sync",
   FEATURE_REQUEST_FACTORY(
   HelperFeatureEnum::Util_select_from_sub_group,
   CALL_FACTORY_ENTRY(
   "__shfl_sync",
   CALL(MapNames::getDpctNamespace() + "select_from_sub_group", SUBGROUP,
        makeCallArgCreatorWithCall(1), makeCallArgCreatorWithCall(2)))),
   Diagnostics::MASK_UNSUPPORTED,
   ARG(MapNames::getDpctNamespace() + "select_from_sub_group"))),
   SUBGROUPSIZE_FACTORY(
   UINT_MAX, MapNames::getDpctNamespace() + "select_from_sub_group",
   WARNING_FACTORY_ENTRY(
   "__shfl_sync",
   FEATURE_REQUEST_FACTORY(
   HelperFeatureEnum::Util_select_from_sub_group,
   CALL_FACTORY_ENTRY(
   "__shfl_sync",
   CALL(MapNames::getDpctNamespace() + "select_from_sub_group", SUBGROUP,
        makeCallArgCreatorWithCall(1), makeCallArgCreatorWithCall(2),
        makeCallArgCreatorWithCall(3)))),
   Diagnostics::MASK_UNSUPPORTED,
   ARG(MapNames::getDpctNamespace() + "select_from_sub_group"))))

   // __shfl
   CONDITIONAL_FACTORY_ENTRY(
   CheckArgIsConstantIntWithValue(2, 32),
   SUBGROUPSIZE_FACTORY(
   UINT_MAX, MapNames::getDpctNamespace() + "select_from_sub_group",
   FEATURE_REQUEST_FACTORY(
   HelperFeatureEnum::Util_select_from_sub_group,
   CALL_FACTORY_ENTRY(
   "__shfl",
   CALL(MapNames::getDpctNamespace() + "select_from_sub_group", SUBGROUP,
        makeCallArgCreatorWithCall(0), makeCallArgCreatorWithCall(1))))),
   SUBGROUPSIZE_FACTORY(
   UINT_MAX, MapNames::getDpctNamespace() + "select_from_sub_group",
   FEATURE_REQUEST_FACTORY(
   HelperFeatureEnum::Util_select_from_sub_group,
   CALL_FACTORY_ENTRY(
   "__shfl",
   CALL(MapNames::getDpctNamespace() + "select_from_sub_group", SUBGROUP,
        makeCallArgCreatorWithCall(0), makeCallArgCreatorWithCall(1),
        makeCallArgCreatorWithCall(2))))))

   // __shfl_xor_sync
   CONDITIONAL_FACTORY_ENTRY(
   CheckArgIsConstantIntWithValue(3, 32),
   SUBGROUPSIZE_FACTORY(
   UINT_MAX, MapNames::getDpctNamespace() + "permute_sub_group_by_xor",
   WARNING_FACTORY_ENTRY(
   "__shfl_xor_sync",
   FEATURE_REQUEST_FACTORY(
   HelperFeatureEnum::Util_permute_sub_group_by_xor,
   CALL_FACTORY_ENTRY(
   "__shfl_xor_sync",
   CALL(MapNames::getDpctNamespace() + "permute_sub_group_by_xor", SUBGROUP,
        makeCallArgCreatorWithCall(1), makeCallArgCreatorWithCall(2)))),
   Diagnostics::MASK_UNSUPPORTED,
   ARG(MapNames::getDpctNamespace() + "permute_sub_group_by_xor"))),
   SUBGROUPSIZE_FACTORY(
   UINT_MAX, MapNames::getDpctNamespace() + "permute_sub_group_by_xor",
   WARNING_FACTORY_ENTRY(
   "__shfl_xor_sync",
   FEATURE_REQUEST_FACTORY(
   HelperFeatureEnum::Util_permute_sub_group_by_xor,
   CALL_FACTORY_ENTRY(
   "__shfl_xor_sync",
   CALL(MapNames::getDpctNamespace() + "permute_sub_group_by_xor", SUBGROUP,
        makeCallArgCreatorWithCall(1), makeCallArgCreatorWithCall(2),
        makeCallArgCreatorWithCall(3)))),
   Diagnostics::MASK_UNSUPPORTED,
   ARG(MapNames::getDpctNamespace() + "permute_sub_group_by_xor"))))

   // __shfl_xor
   CONDITIONAL_FACTORY_ENTRY(
   CheckArgIsConstantIntWithValue(2, 32),
   SUBGROUPSIZE_FACTORY(
   UINT_MAX, MapNames::getDpctNamespace() + "permute_sub_group_by_xor",
   FEATURE_REQUEST_FACTORY(
   HelperFeatureEnum::Util_permute_sub_group_by_xor,
   CALL_FACTORY_ENTRY(
   "__shfl_xor",
   CALL(MapNames::getDpctNamespace() + "permute_sub_group_by_xor", SUBGROUP,
        makeCallArgCreatorWithCall(0), makeCallArgCreatorWithCall(1))))),
   SUBGROUPSIZE_FACTORY(
   UINT_MAX, MapNames::getDpctNamespace() + "permute_sub_group_by_xor",
   FEATURE_REQUEST_FACTORY(
   HelperFeatureEnum::Util_permute_sub_group_by_xor,
   CALL_FACTORY_ENTRY(
   "__shfl_xor",
   CALL(MapNames::getDpctNamespace() + "permute_sub_group_by_xor", SUBGROUP,
        makeCallArgCreatorWithCall(0), makeCallArgCreatorWithCall(1),
        makeCallArgCreatorWithCall(2))))))

   WARNING_FACTORY_ENTRY(
   "__activemask",
   TOSTRING_FACTORY_ENTRY("__activemask", LITERAL("0xffffffff")),
   Diagnostics::ACTIVE_MASK)

   // __ballot
   SUBGROUPSIZE_FACTORY(
   UINT_MAX, MapNames::getClNamespace() + "reduce_over_group",
   CALL_FACTORY_ENTRY(
   "__ballot",
   CALL(MapNames::getClNamespace() + "reduce_over_group", SUBGROUP,
        makeCombinedArg(
        ARG(0), makeCombinedArg(makeCombinedArg(ARG(" ? (0x1 << "), SUBGROUP),
                                ARG(".get_local_linear_id()) : 0, " +
                                    MapNames::getClNamespace() +
                                    "ext::oneapi::plus<>()"))))))

   // __ballot_sync
   SUBGROUPSIZE_FACTORY(
   UINT_MAX, MapNames::getClNamespace() + "reduce_over_group",
   CALL_FACTORY_ENTRY(
   "__ballot_sync",
   CALL(MapNames::getClNamespace() + "reduce_over_group", SUBGROUP,
        makeCombinedArg(
        ARG("("),
        makeCombinedArg(
        ARG(0),
        makeCombinedArg(
        makeCombinedArg(makeCombinedArg(ARG(" & (0x1 << "), SUBGROUP),
                        ARG(".get_local_linear_id())) && ")),
        makeCombinedArg(
        ARG(1), makeCombinedArg(makeCombinedArg(ARG(" ? (0x1 << "), SUBGROUP),
                                ARG(".get_local_linear_id()) : 0, " +
                                    MapNames::getClNamespace() +
                                    "ext::oneapi::plus<>()")))))))))

  }));
}

} // namespace dpct
} // namespace clang
