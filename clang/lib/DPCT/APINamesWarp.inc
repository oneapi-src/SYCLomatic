//===--------------- APINamesWarp.inc -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// __all_sync
SUBGROUPSIZE_FACTORY(
    UINT_MAX, MapNames::getClNamespace() + "all_of_group",
    CALL_FACTORY_ENTRY(
        "__all_sync",
        CALL(
            MapNames::getClNamespace() + "all_of_group", SUBGROUP,
            makeCombinedArg(
                ARG("(~"),
                makeCombinedArg(
                    makeCallArgCreatorWithCall(0),
                    makeCombinedArg(
                        makeCombinedArg(
                            ARG(" & (0x1 << "),
                            makeCombinedArg(
                                SUBGROUP, ARG(".get_local_linear_id())) || "))),
                        makeCallArgCreatorWithCall(1)))))))

// __all
SUBGROUPSIZE_FACTORY(
    UINT_MAX, MapNames::getClNamespace() + "all_of_group",
    CALL_FACTORY_ENTRY("__all",
                       CALL(MapNames::getClNamespace() + "all_of_group",
                            SUBGROUP, makeCallArgCreatorWithCall(0))))

// __any_sync
SUBGROUPSIZE_FACTORY(
    UINT_MAX, MapNames::getClNamespace() + "any_of_group",
    CALL_FACTORY_ENTRY(
        "__any_sync",
        CALL(
            MapNames::getClNamespace() + "any_of_group", SUBGROUP,
            makeCombinedArg(
                ARG("("),
                makeCombinedArg(
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
    CALL_FACTORY_ENTRY("__any",
                       CALL(MapNames::getClNamespace() + "any_of_group",
                            SUBGROUP, makeCallArgCreatorWithCall(0))))

// __match_any_sync
SUBGROUPSIZE_FACTORY(UINT_MAX,
                     MapNames::getDpctNamespace() + "match_any_over_sub_group",
                     CALL_FACTORY_ENTRY("__match_any_sync",
                                        CALL(MapNames::getDpctNamespace() +
                                                 "match_any_over_sub_group",
                                             SUBGROUP, ARG_WC(0), ARG_WC(1))))

// __match_all_sync
SUBGROUPSIZE_FACTORY(UINT_MAX,
                     MapNames::getDpctNamespace() + "match_all_over_sub_group",
                     CALL_FACTORY_ENTRY("__match_all_sync",
                                        CALL(MapNames::getDpctNamespace() +
                                                 "match_all_over_sub_group",
                                             SUBGROUP, ARG_WC(0), ARG_WC(1),
                                             ARG_WC(2))))

// __reduce_add_sync
SUBGROUPSIZE_FACTORY(
    UINT_MAX, MapNames::getClNamespace() + "reduce_over_group",
    WARNING_FACTORY_ENTRY(
        "__reduce_add_sync",
        CALL_FACTORY_ENTRY(
            "__reduce_add_sync",
            CALL(MapNames::getClNamespace() + "reduce_over_group", SUBGROUP,
                 ARG(1), LITERAL(MapNames::getClNamespace() + "plus<>()"))),
        Diagnostics::MASK_UNSUPPORTED,
        ARG(MapNames::getClNamespace() + "reduce_over_group")))

// __reduce_min_sync
SUBGROUPSIZE_FACTORY(
    UINT_MAX, MapNames::getClNamespace() + "reduce_over_group",
    WARNING_FACTORY_ENTRY(
        "__reduce_min_sync",
        CALL_FACTORY_ENTRY(
            "__reduce_min_sync",
            CALL(MapNames::getClNamespace() + "reduce_over_group", SUBGROUP,
                 ARG(1), LITERAL(MapNames::getClNamespace() + "minimum()"))),
        Diagnostics::MASK_UNSUPPORTED,
        ARG(MapNames::getClNamespace() + "reduce_over_group")))

// __reduce_max_sync
SUBGROUPSIZE_FACTORY(
    UINT_MAX, MapNames::getClNamespace() + "reduce_over_group",
    WARNING_FACTORY_ENTRY(
        "__reduce_max_sync",
        CALL_FACTORY_ENTRY(
            "__reduce_max_sync",
            CALL(MapNames::getClNamespace() + "reduce_over_group", SUBGROUP,
                 ARG(1), LITERAL(MapNames::getClNamespace() + "maximum()"))),
        Diagnostics::MASK_UNSUPPORTED,
        ARG(MapNames::getClNamespace() + "reduce_over_group")))

// __reduce_and_sync
SUBGROUPSIZE_FACTORY(
    UINT_MAX, MapNames::getClNamespace() + "reduce_over_group",
    WARNING_FACTORY_ENTRY(
        "__reduce_and_sync",
        CALL_FACTORY_ENTRY(
            "__reduce_and_sync",
            CALL(MapNames::getClNamespace() + "reduce_over_group", SUBGROUP,
                 ARG(1), LITERAL(MapNames::getClNamespace() + "bit_and<>()"))),
        Diagnostics::MASK_UNSUPPORTED,
        ARG(MapNames::getClNamespace() + "reduce_over_group")))

// __reduce_or_sync
SUBGROUPSIZE_FACTORY(
    UINT_MAX, MapNames::getClNamespace() + "reduce_over_group",
    WARNING_FACTORY_ENTRY(
        "__reduce_or_sync",
        CALL_FACTORY_ENTRY(
            "__reduce_or_sync",
            CALL(MapNames::getClNamespace() + "reduce_over_group", SUBGROUP,
                 ARG(1), LITERAL(MapNames::getClNamespace() + "bit_or<>()"))),
        Diagnostics::MASK_UNSUPPORTED,
        ARG(MapNames::getClNamespace() + "reduce_over_group")))

// __reduce_xor_sync
SUBGROUPSIZE_FACTORY(
    UINT_MAX, MapNames::getClNamespace() + "reduce_over_group",
    WARNING_FACTORY_ENTRY(
        "__reduce_xor_sync",
        CALL_FACTORY_ENTRY(
            "__reduce_xor_sync",
            CALL(MapNames::getClNamespace() + "reduce_over_group", SUBGROUP,
                 ARG(1), LITERAL(MapNames::getClNamespace() + "bit_xor<>()"))),
        Diagnostics::MASK_UNSUPPORTED,
        ARG(MapNames::getClNamespace() + "reduce_over_group")))

// __shfl_up_sync
CONDITIONAL_FACTORY_ENTRY(
    CheckIsMaskedSubGroupFunctionEnabled(),
    CONDITIONAL_FACTORY_ENTRY(
        CheckArgIsConstantIntWithValue(3, 32),
        SUBGROUPSIZE_FACTORY(
            UINT_MAX, MapNames::getDpctNamespace() + "experimental::shift_sub_group_right",
            WARNING_FACTORY_ENTRY(
                "__shfl_up_sync",
                FEATURE_REQUEST_FACTORY(
                    HelperFeatureEnum::device_ext,
                    CALL_FACTORY_ENTRY("__shfl_up_sync",
                                       CALL(MapNames::getDpctNamespace() +
                                                "experimental::shift_sub_group_right",
                                            makeCallArgCreatorWithCall(0),
                                            SUBGROUP,
                                            makeCallArgCreatorWithCall(1),
                                            makeCallArgCreatorWithCall(2)))),
                Diagnostics::EXPERIMENTAL_FEATURE,
                ARG("__shfl_up_sync"),
                ARG("masked sub_group function"))),
        SUBGROUPSIZE_FACTORY(
            UINT_MAX, MapNames::getDpctNamespace() + "experimental::shift_sub_group_right",
            WARNING_FACTORY_ENTRY(
                "__shfl_up_sync",
                FEATURE_REQUEST_FACTORY(
                    HelperFeatureEnum::device_ext,
                    CALL_FACTORY_ENTRY("__shfl_up_sync",
                                       CALL(MapNames::getDpctNamespace() +
                                                "experimental::shift_sub_group_right",
                                            makeCallArgCreatorWithCall(0),
                                            SUBGROUP,
                                            makeCallArgCreatorWithCall(1),
                                            makeCallArgCreatorWithCall(2),
                                            makeCallArgCreatorWithCall(3)))),
                Diagnostics::EXPERIMENTAL_FEATURE,
                ARG("__shfl_up_sync"),
                ARG("masked sub_group function")))),
    CONDITIONAL_FACTORY_ENTRY(
        CheckArgIsConstantIntWithUnsignedValue(0, 0xFFFFFFFF),
        CONDITIONAL_FACTORY_ENTRY(
            CheckArgIsConstantIntWithValue(3, 32),
            SUBGROUPSIZE_FACTORY(
                UINT_MAX,
                MapNames::getDpctNamespace() + "shift_sub_group_right",
                FEATURE_REQUEST_FACTORY(
                    HelperFeatureEnum::device_ext,
                    CALL_FACTORY_ENTRY("__shfl_up_sync",
                                       CALL(MapNames::getDpctNamespace() +
                                                "shift_sub_group_right",
                                            SUBGROUP,
                                            makeCallArgCreatorWithCall(1),
                                            makeCallArgCreatorWithCall(2))))),
            SUBGROUPSIZE_FACTORY(
                UINT_MAX,
                MapNames::getDpctNamespace() + "shift_sub_group_right",
                FEATURE_REQUEST_FACTORY(
                    HelperFeatureEnum::device_ext,
                    CALL_FACTORY_ENTRY("__shfl_up_sync",
                                       CALL(MapNames::getDpctNamespace() +
                                                "shift_sub_group_right",
                                            SUBGROUP,
                                            makeCallArgCreatorWithCall(1),
                                            makeCallArgCreatorWithCall(2),
                                            makeCallArgCreatorWithCall(3)))))),
        CONDITIONAL_FACTORY_ENTRY(
            CheckArgIsConstantIntWithValue(3, 32),
            SUBGROUPSIZE_FACTORY(
                UINT_MAX,
                MapNames::getDpctNamespace() + "shift_sub_group_right",
                WARNING_FACTORY_ENTRY(
                    "__shfl_up_sync",
                    FEATURE_REQUEST_FACTORY(
                        HelperFeatureEnum::device_ext,
                        CALL_FACTORY_ENTRY(
                            "__shfl_up_sync",
                            CALL(MapNames::getDpctNamespace() +
                                     "shift_sub_group_right",
                                 SUBGROUP, makeCallArgCreatorWithCall(1),
                                 makeCallArgCreatorWithCall(2)))),
                    Diagnostics::MASK_UNSUPPORTED,
                    ARG(MapNames::getDpctNamespace() +
                        "shift_sub_group_right." +
                        " You can specify "
                        "\"--use-experimental-features=masked-sub-group-"
                        "operation\""
                        " to use the experimental helper function to migrate "
                        "__shfl_up_sync"))),
            SUBGROUPSIZE_FACTORY(
                UINT_MAX,
                MapNames::getDpctNamespace() + "shift_sub_group_right",
                WARNING_FACTORY_ENTRY(
                    "__shfl_up_sync",
                    FEATURE_REQUEST_FACTORY(
                        HelperFeatureEnum::device_ext,
                        CALL_FACTORY_ENTRY(
                            "__shfl_up_sync",
                            CALL(MapNames::getDpctNamespace() +
                                     "shift_sub_group_right",
                                 SUBGROUP, makeCallArgCreatorWithCall(1),
                                 makeCallArgCreatorWithCall(2),
                                 makeCallArgCreatorWithCall(3)))),
                    Diagnostics::MASK_UNSUPPORTED,
                    ARG(MapNames::getDpctNamespace() +
                        "shift_sub_group_right." +
                        " You can specify "
                        "\"--use-experimental-features=masked-sub-group-"
                        "operation\""
                        " to use the experimental helper function to migrate "
                        "__shfl_up_sync"))))))

// __shfl_up
CONDITIONAL_FACTORY_ENTRY(
    CheckArgIsConstantIntWithValue(2, 32),
    SUBGROUPSIZE_FACTORY(
        UINT_MAX, MapNames::getDpctNamespace() + "shift_sub_group_right",
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::device_ext,
            CALL_FACTORY_ENTRY("__shfl_up",
                               CALL(MapNames::getDpctNamespace() +
                                        "shift_sub_group_right",
                                    SUBGROUP, makeCallArgCreatorWithCall(0),
                                    makeCallArgCreatorWithCall(1))))),
    SUBGROUPSIZE_FACTORY(
        UINT_MAX, MapNames::getDpctNamespace() + "shift_sub_group_right",
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::device_ext,
            CALL_FACTORY_ENTRY("__shfl_up",
                               CALL(MapNames::getDpctNamespace() +
                                        "shift_sub_group_right",
                                    SUBGROUP, makeCallArgCreatorWithCall(0),
                                    makeCallArgCreatorWithCall(1),
                                    makeCallArgCreatorWithCall(2))))))

// __shfl_down_sync
CONDITIONAL_FACTORY_ENTRY(
    CheckIsMaskedSubGroupFunctionEnabled(),
    CONDITIONAL_FACTORY_ENTRY(
        CheckArgIsConstantIntWithValue(3, 32),
        SUBGROUPSIZE_FACTORY(
            UINT_MAX, MapNames::getDpctNamespace() + "experimental::shift_sub_group_left",
            WARNING_FACTORY_ENTRY(
                "__shfl_down_sync",
                FEATURE_REQUEST_FACTORY(
                    HelperFeatureEnum::device_ext,
                    CALL_FACTORY_ENTRY("__shfl_down_sync",
                                       CALL(MapNames::getDpctNamespace() +
                                                "experimental::shift_sub_group_left",
                                            makeCallArgCreatorWithCall(0),
                                            SUBGROUP,
                                            makeCallArgCreatorWithCall(1),
                                            makeCallArgCreatorWithCall(2)))),
                Diagnostics::EXPERIMENTAL_FEATURE,
                ARG("__shfl_down_sync"),
                ARG("masked sub_group function"))),
        SUBGROUPSIZE_FACTORY(
            UINT_MAX, MapNames::getDpctNamespace() + "experimental::shift_sub_group_left",
            WARNING_FACTORY_ENTRY(
                "__shfl_down_sync",
                FEATURE_REQUEST_FACTORY(
                    HelperFeatureEnum::device_ext,
                    CALL_FACTORY_ENTRY("__shfl_down_sync",
                                       CALL(MapNames::getDpctNamespace() +
                                                "experimental::shift_sub_group_left",
                                            makeCallArgCreatorWithCall(0),
                                            SUBGROUP,
                                            makeCallArgCreatorWithCall(1),
                                            makeCallArgCreatorWithCall(2),
                                            makeCallArgCreatorWithCall(3)))),
                Diagnostics::EXPERIMENTAL_FEATURE,
                ARG("__shfl_down_sync"),
                ARG("masked sub_group function")))),
    CONDITIONAL_FACTORY_ENTRY(
        CheckArgIsConstantIntWithUnsignedValue(0, 0xFFFFFFFF),
        CONDITIONAL_FACTORY_ENTRY(
            CheckArgIsConstantIntWithValue(3, 32),
            SUBGROUPSIZE_FACTORY(
                UINT_MAX, MapNames::getDpctNamespace() + "shift_sub_group_left",
                FEATURE_REQUEST_FACTORY(
                    HelperFeatureEnum::device_ext,
                    CALL_FACTORY_ENTRY("__shfl_down_sync",
                                       CALL(MapNames::getDpctNamespace() +
                                                "shift_sub_group_left",
                                            SUBGROUP,
                                            makeCallArgCreatorWithCall(1),
                                            makeCallArgCreatorWithCall(2))))),
            SUBGROUPSIZE_FACTORY(
                UINT_MAX, MapNames::getDpctNamespace() + "shift_sub_group_left",
                FEATURE_REQUEST_FACTORY(
                    HelperFeatureEnum::device_ext,
                    CALL_FACTORY_ENTRY("__shfl_down_sync",
                                       CALL(MapNames::getDpctNamespace() +
                                                "shift_sub_group_left",
                                            SUBGROUP,
                                            makeCallArgCreatorWithCall(1),
                                            makeCallArgCreatorWithCall(2),
                                            makeCallArgCreatorWithCall(3)))))),
        CONDITIONAL_FACTORY_ENTRY(
            CheckArgIsConstantIntWithValue(3, 32),
            SUBGROUPSIZE_FACTORY(
                UINT_MAX, MapNames::getDpctNamespace() + "shift_sub_group_left",
                WARNING_FACTORY_ENTRY(
                    "__shfl_down_sync",
                    FEATURE_REQUEST_FACTORY(
                        HelperFeatureEnum::device_ext,
                        CALL_FACTORY_ENTRY(
                            "__shfl_down_sync",
                            CALL(MapNames::getDpctNamespace() +
                                     "shift_sub_group_left",
                                 SUBGROUP, makeCallArgCreatorWithCall(1),
                                 makeCallArgCreatorWithCall(2)))),
                    Diagnostics::MASK_UNSUPPORTED,
                    ARG(MapNames::getDpctNamespace() + "shift_sub_group_left." +
                        " You can specify "
                        "\"--use-experimental-features=masked-sub-group-"
                        "operation\""
                        " to use the experimental helper function to migrate "
                        "__shfl_down_sync"))),
            SUBGROUPSIZE_FACTORY(
                UINT_MAX, MapNames::getDpctNamespace() + "shift_sub_group_left",
                WARNING_FACTORY_ENTRY(
                    "__shfl_down_sync",
                    FEATURE_REQUEST_FACTORY(
                        HelperFeatureEnum::device_ext,
                        CALL_FACTORY_ENTRY(
                            "__shfl_down_sync",
                            CALL(MapNames::getDpctNamespace() +
                                     "shift_sub_group_left",
                                 SUBGROUP, makeCallArgCreatorWithCall(1),
                                 makeCallArgCreatorWithCall(2),
                                 makeCallArgCreatorWithCall(3)))),
                    Diagnostics::MASK_UNSUPPORTED,
                    ARG(MapNames::getDpctNamespace() + "shift_sub_group_left." +
                        " You can specify "
                        "\"--use-experimental-features=masked-sub-group-"
                        "operation\""
                        " to use the experimental helper function to migrate "
                        "__shfl_down_sync"))))))

// __shfl_down
CONDITIONAL_FACTORY_ENTRY(
    CheckArgIsConstantIntWithValue(2, 32),
    SUBGROUPSIZE_FACTORY(
        UINT_MAX, MapNames::getDpctNamespace() + "shift_sub_group_left",
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::device_ext,
            CALL_FACTORY_ENTRY("__shfl_down",
                               CALL(MapNames::getDpctNamespace() +
                                        "shift_sub_group_left",
                                    SUBGROUP, makeCallArgCreatorWithCall(0),
                                    makeCallArgCreatorWithCall(1))))),
    SUBGROUPSIZE_FACTORY(
        UINT_MAX, MapNames::getDpctNamespace() + "shift_sub_group_left",
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::device_ext,
            CALL_FACTORY_ENTRY("__shfl_down",
                               CALL(MapNames::getDpctNamespace() +
                                        "shift_sub_group_left",
                                    SUBGROUP, makeCallArgCreatorWithCall(0),
                                    makeCallArgCreatorWithCall(1),
                                    makeCallArgCreatorWithCall(2))))))

//__shfl_sync
CONDITIONAL_FACTORY_ENTRY(
    CheckIsMaskedSubGroupFunctionEnabled(),
    CONDITIONAL_FACTORY_ENTRY(
        CheckArgIsConstantIntWithValue(3, 32),
        SUBGROUPSIZE_FACTORY(
            UINT_MAX, MapNames::getDpctNamespace() + "experimental::select_from_sub_group",
            WARNING_FACTORY_ENTRY(
                "__shfl_sync",
                FEATURE_REQUEST_FACTORY(
                    HelperFeatureEnum::device_ext,
                    CALL_FACTORY_ENTRY("__shfl_sync",
                                       CALL(MapNames::getDpctNamespace() +
                                                "experimental::select_from_sub_group",
                                            makeCallArgCreatorWithCall(0),
                                            SUBGROUP,
                                            makeCallArgCreatorWithCall(1),
                                            makeCallArgCreatorWithCall(2)))),
                Diagnostics::EXPERIMENTAL_FEATURE,
                ARG("__shfl_sync"),
                ARG("masked sub_group function"))),
        SUBGROUPSIZE_FACTORY(
            UINT_MAX, MapNames::getDpctNamespace() + "experimental::select_from_sub_group",
            WARNING_FACTORY_ENTRY(
                "__shfl_sync",
                FEATURE_REQUEST_FACTORY(
                    HelperFeatureEnum::device_ext,
                    CALL_FACTORY_ENTRY("__shfl_sync",
                                       CALL(MapNames::getDpctNamespace() +
                                                "experimental::select_from_sub_group",
                                            makeCallArgCreatorWithCall(0),
                                            SUBGROUP,
                                            makeCallArgCreatorWithCall(1),
                                            makeCallArgCreatorWithCall(2),
                                            makeCallArgCreatorWithCall(3)))),
                Diagnostics::EXPERIMENTAL_FEATURE,
                ARG("__shfl_sync"),
                ARG("masked sub_group function")))),
    CONDITIONAL_FACTORY_ENTRY(
        CheckArgIsConstantIntWithUnsignedValue(0, 0xFFFFFFFF),
        CONDITIONAL_FACTORY_ENTRY(
            CheckArgIsConstantIntWithValue(3, 32),
            SUBGROUPSIZE_FACTORY(
                UINT_MAX,
                MapNames::getDpctNamespace() + "select_from_sub_group",
                FEATURE_REQUEST_FACTORY(
                    HelperFeatureEnum::device_ext,
                    CALL_FACTORY_ENTRY("__shfl_sync",
                                       CALL(MapNames::getDpctNamespace() +
                                                "select_from_sub_group",
                                            SUBGROUP,
                                            makeCallArgCreatorWithCall(1),
                                            makeCallArgCreatorWithCall(2))))),
            SUBGROUPSIZE_FACTORY(
                UINT_MAX,
                MapNames::getDpctNamespace() + "select_from_sub_group",
                FEATURE_REQUEST_FACTORY(
                    HelperFeatureEnum::device_ext,
                    CALL_FACTORY_ENTRY("__shfl_sync",
                                       CALL(MapNames::getDpctNamespace() +
                                                "select_from_sub_group",
                                            SUBGROUP,
                                            makeCallArgCreatorWithCall(1),
                                            makeCallArgCreatorWithCall(2),
                                            makeCallArgCreatorWithCall(3)))))),
        CONDITIONAL_FACTORY_ENTRY(
            CheckArgIsConstantIntWithValue(3, 32),
            SUBGROUPSIZE_FACTORY(
                UINT_MAX,
                MapNames::getDpctNamespace() + "select_from_sub_group",
                WARNING_FACTORY_ENTRY(
                    "__shfl_sync",
                    FEATURE_REQUEST_FACTORY(
                        HelperFeatureEnum::device_ext,
                        CALL_FACTORY_ENTRY(
                            "__shfl_sync",
                            CALL(MapNames::getDpctNamespace() +
                                     "select_from_sub_group",
                                 SUBGROUP, makeCallArgCreatorWithCall(1),
                                 makeCallArgCreatorWithCall(2)))),
                    Diagnostics::MASK_UNSUPPORTED,
                    ARG(MapNames::getDpctNamespace() +
                        "select_from_sub_group." +
                        " You can specify "
                        "\"--use-experimental-features=masked-sub-group-"
                        "operation\""
                        " to use the experimental helper function to migrate "
                        "__shfl_sync"))),
            SUBGROUPSIZE_FACTORY(
                UINT_MAX,
                MapNames::getDpctNamespace() + "select_from_sub_group",
                WARNING_FACTORY_ENTRY(
                    "__shfl_sync",
                    FEATURE_REQUEST_FACTORY(
                        HelperFeatureEnum::device_ext,
                        CALL_FACTORY_ENTRY(
                            "__shfl_sync",
                            CALL(MapNames::getDpctNamespace() +
                                     "select_from_sub_group",
                                 SUBGROUP, makeCallArgCreatorWithCall(1),
                                 makeCallArgCreatorWithCall(2),
                                 makeCallArgCreatorWithCall(3)))),
                    Diagnostics::MASK_UNSUPPORTED,
                    ARG(MapNames::getDpctNamespace() +
                        "select_from_sub_group." +
                        " You can specify "
                        "\"--use-experimental-features=masked-sub-group-"
                        "operation\""
                        " to use the experimental helper function to migrate "
                        "__shfl_sync"))))))

// __shfl
CONDITIONAL_FACTORY_ENTRY(
    CheckArgIsConstantIntWithValue(2, 32),
    SUBGROUPSIZE_FACTORY(
        UINT_MAX, MapNames::getDpctNamespace() + "select_from_sub_group",
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::device_ext,
            CALL_FACTORY_ENTRY("__shfl",
                               CALL(MapNames::getDpctNamespace() +
                                        "select_from_sub_group",
                                    SUBGROUP, makeCallArgCreatorWithCall(0),
                                    makeCallArgCreatorWithCall(1))))),
    SUBGROUPSIZE_FACTORY(
        UINT_MAX, MapNames::getDpctNamespace() + "select_from_sub_group",
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::device_ext,
            CALL_FACTORY_ENTRY("__shfl",
                               CALL(MapNames::getDpctNamespace() +
                                        "select_from_sub_group",
                                    SUBGROUP, makeCallArgCreatorWithCall(0),
                                    makeCallArgCreatorWithCall(1),
                                    makeCallArgCreatorWithCall(2))))))

// __shfl_xor_sync
CONDITIONAL_FACTORY_ENTRY(
    CheckIsMaskedSubGroupFunctionEnabled(),
    CONDITIONAL_FACTORY_ENTRY(
        CheckArgIsConstantIntWithValue(3, 32),
        SUBGROUPSIZE_FACTORY(
            UINT_MAX, MapNames::getDpctNamespace() + "experimental::permute_sub_group_by_xor",
            WARNING_FACTORY_ENTRY(
                "__shfl_xor_sync",
                FEATURE_REQUEST_FACTORY(
                    HelperFeatureEnum::device_ext,
                    CALL_FACTORY_ENTRY("__shfl_xor_sync",
                                       CALL(MapNames::getDpctNamespace() +
                                                "experimental::permute_sub_group_by_xor",
                                            makeCallArgCreatorWithCall(0),
                                            SUBGROUP,
                                            makeCallArgCreatorWithCall(1),
                                            makeCallArgCreatorWithCall(2)))),
                Diagnostics::EXPERIMENTAL_FEATURE,
                ARG("__shfl_xor_sync"),
                ARG("masked sub_group function"))),
        SUBGROUPSIZE_FACTORY(
            UINT_MAX, MapNames::getDpctNamespace() + "experimental::permute_sub_group_by_xor",
            WARNING_FACTORY_ENTRY(
                "__shfl_xor_sync",
                FEATURE_REQUEST_FACTORY(
                    HelperFeatureEnum::device_ext,
                    CALL_FACTORY_ENTRY("__shfl_xor_sync",
                                       CALL(MapNames::getDpctNamespace() +
                                                "experimental::permute_sub_group_by_xor",
                                            makeCallArgCreatorWithCall(0),
                                            SUBGROUP,
                                            makeCallArgCreatorWithCall(1),
                                            makeCallArgCreatorWithCall(2),
                                            makeCallArgCreatorWithCall(3)))),
                Diagnostics::EXPERIMENTAL_FEATURE,
                ARG("__shfl_xor_sync"),
                ARG("masked sub_group function")))),
    CONDITIONAL_FACTORY_ENTRY(
        CheckArgIsConstantIntWithUnsignedValue(0, 0xFFFFFFFF),
        CONDITIONAL_FACTORY_ENTRY(
            CheckArgIsConstantIntWithValue(3, 32),
            SUBGROUPSIZE_FACTORY(
                UINT_MAX,
                MapNames::getDpctNamespace() + "permute_sub_group_by_xor",
                FEATURE_REQUEST_FACTORY(
                    HelperFeatureEnum::device_ext,
                    CALL_FACTORY_ENTRY("__shfl_xor_sync",
                                       CALL(MapNames::getDpctNamespace() +
                                                "permute_sub_group_by_xor",
                                            SUBGROUP,
                                            makeCallArgCreatorWithCall(1),
                                            makeCallArgCreatorWithCall(2))))),
            SUBGROUPSIZE_FACTORY(
                UINT_MAX,
                MapNames::getDpctNamespace() + "permute_sub_group_by_xor",
                FEATURE_REQUEST_FACTORY(
                    HelperFeatureEnum::device_ext,
                    CALL_FACTORY_ENTRY("__shfl_xor_sync",
                                       CALL(MapNames::getDpctNamespace() +
                                                "permute_sub_group_by_xor",
                                            SUBGROUP,
                                            makeCallArgCreatorWithCall(1),
                                            makeCallArgCreatorWithCall(2),
                                            makeCallArgCreatorWithCall(3)))))),
        CONDITIONAL_FACTORY_ENTRY(
            CheckArgIsConstantIntWithValue(3, 32),
            SUBGROUPSIZE_FACTORY(
                UINT_MAX,
                MapNames::getDpctNamespace() + "permute_sub_group_by_xor",
                WARNING_FACTORY_ENTRY(
                    "__shfl_xor_sync",
                    FEATURE_REQUEST_FACTORY(
                        HelperFeatureEnum::device_ext,
                        CALL_FACTORY_ENTRY(
                            "__shfl_xor_sync",
                            CALL(MapNames::getDpctNamespace() +
                                     "permute_sub_group_by_xor",
                                 SUBGROUP, makeCallArgCreatorWithCall(1),
                                 makeCallArgCreatorWithCall(2)))),
                    Diagnostics::MASK_UNSUPPORTED,
                    ARG(MapNames::getDpctNamespace() +
                        "permute_sub_group_by_xor." +
                        " You can specify "
                        "\"--use-experimental-features=masked-sub-group-"
                        "operation\""
                        " to use the experimental helper function to migrate "
                        "__shfl_xor_sync"))),
            SUBGROUPSIZE_FACTORY(
                UINT_MAX,
                MapNames::getDpctNamespace() + "permute_sub_group_by_xor",
                WARNING_FACTORY_ENTRY(
                    "__shfl_xor_sync",
                    FEATURE_REQUEST_FACTORY(
                        HelperFeatureEnum::device_ext,
                        CALL_FACTORY_ENTRY(
                            "__shfl_xor_sync",
                            CALL(MapNames::getDpctNamespace() +
                                     "permute_sub_group_by_xor",
                                 SUBGROUP, makeCallArgCreatorWithCall(1),
                                 makeCallArgCreatorWithCall(2),
                                 makeCallArgCreatorWithCall(3)))),
                    Diagnostics::MASK_UNSUPPORTED,
                    ARG(MapNames::getDpctNamespace() +
                        "permute_sub_group_by_xor." +
                        " You can specify "
                        "\"--use-experimental-features=masked-sub-group-"
                        "operation\""
                        " to use the experimental helper function to migrate "
                        "__shfl_xor_sync"))))))

// __shfl_xor
CONDITIONAL_FACTORY_ENTRY(
    CheckArgIsConstantIntWithValue(2, 32),
    SUBGROUPSIZE_FACTORY(
        UINT_MAX, MapNames::getDpctNamespace() + "permute_sub_group_by_xor",
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::device_ext,
            CALL_FACTORY_ENTRY("__shfl_xor",
                               CALL(MapNames::getDpctNamespace() +
                                        "permute_sub_group_by_xor",
                                    SUBGROUP, makeCallArgCreatorWithCall(0),
                                    makeCallArgCreatorWithCall(1))))),
    SUBGROUPSIZE_FACTORY(
        UINT_MAX, MapNames::getDpctNamespace() + "permute_sub_group_by_xor",
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::device_ext,
            CALL_FACTORY_ENTRY("__shfl_xor",
                               CALL(MapNames::getDpctNamespace() +
                                        "permute_sub_group_by_xor",
                                    SUBGROUP, makeCallArgCreatorWithCall(0),
                                    makeCallArgCreatorWithCall(1),
                                    makeCallArgCreatorWithCall(2))))))

WARNING_FACTORY_ENTRY("__activemask",
                      TOSTRING_FACTORY_ENTRY("__activemask",
                                             LITERAL("0xffffffff")),
                      Diagnostics::ACTIVE_MASK)

// __ballot
SUBGROUPSIZE_FACTORY(
    UINT_MAX, MapNames::getClNamespace() + "reduce_over_group",
    CALL_FACTORY_ENTRY(
        "__ballot",
        CALL(MapNames::getClNamespace() + "reduce_over_group", SUBGROUP,
             makeCombinedArg(
                 ARG(0),
                 makeCombinedArg(makeCombinedArg(ARG(" ? (0x1 << "), SUBGROUP),
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
                         makeCombinedArg(makeCombinedArg(ARG(" & (0x1 << "),
                                                         SUBGROUP),
                                         ARG(".get_local_linear_id())) && ")),
                         makeCombinedArg(
                             ARG(1),
                             makeCombinedArg(
                                 makeCombinedArg(ARG(" ? (0x1 << "), SUBGROUP),
                                 ARG(".get_local_linear_id()) : 0, " +
                                     MapNames::getClNamespace() +
                                     "ext::oneapi::plus<>()")))))))))
