//===--------------- APINamesCooperativeGroups.inc ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifdef FUNCTION_CALL

CONDITIONAL_FACTORY_ENTRY(
    CheckArgType(0, "const class cooperative_groups::__v1::thread_block &"),
    WARNING_FACTORY_ENTRY(
        "sync",
        MEMBER_CALL_FACTORY_ENTRY("sync", NDITEM,
                                  false, "barrier"),
        Diagnostics::BARRIER_PERFORMANCE_TUNNING, ARG("nd_item")),
    CONDITIONAL_FACTORY_ENTRY(
        CheckArgType(0, "const class cooperative_groups::__v1::coalesced_group &"),
        WARNING_FACTORY_ENTRY(
            "sync",
            MEMBER_CALL_FACTORY_ENTRY("sync", NDITEM,
                                      false, "barrier"),
            Diagnostics::BARRIER_PERFORMANCE_TUNNING, ARG("nd_item")),
        CONDITIONAL_FACTORY_ENTRY(
            CheckArgType(
                0, "const class cooperative_groups::__v1::thread_block_tile<32> &"),
            WARNING_FACTORY_ENTRY(
                "sync",
                MEMBER_CALL_FACTORY_ENTRY("sync",
                                          SUBGROUP, false, "barrier"),
                Diagnostics::BARRIER_PERFORMANCE_TUNNING, ARG("sub_group")),
            CONDITIONAL_FACTORY_ENTRY(
                CheckArgType(0, "const class cooperative_groups::__v1::grid_group &"),
                CONDITIONAL_FACTORY_ENTRY(
                    UseNDRangeBarrier,
                    FEATURE_REQUEST_FACTORY(
                        HelperFeatureEnum::Util_nd_range_barrier,
                        CALL_FACTORY_ENTRY(
                            "sync",
                            CALL(MapNames::getDpctNamespace() +
                                     "experimental::nd_range_barrier",
                                 NDITEM, ARG(DpctGlobalInfo::getSyncName())))),
                    UNSUPPORT_FACTORY_ENTRY("sync",
                                            Diagnostics::ND_RANGE_BARRIER,
                                            makeCombinedArg(MemberExprBase(), ARG(".sync()")))),
                UNSUPPORT_FACTORY_ENTRY("sync",
                                        Diagnostics::API_NOT_MIGRATED,
                                        ARG("sync"))))))

CONDITIONAL_FACTORY_ENTRY(
        CheckArgType(0, "const class cooperative_groups::__v1::thread_block &"),
        MEMBER_CALL_FACTORY_ENTRY("thread_rank", NDITEM,
                                  false, "get_local_linear_id"),
        CONDITIONAL_FACTORY_ENTRY(
            CheckArgType(0, "const class cooperative_groups::__v1::thread_block_tile<32> &"),
            MEMBER_CALL_FACTORY_ENTRY("thread_rank", SUBGROUP,
                                      false, "get_local_linear_id"),
            CONDITIONAL_FACTORY_ENTRY(
                UseLogicalGroup,
                CONDITIONAL_FACTORY_ENTRY(
                    makeCheckOr(
                        CheckArgType(0, "const class cooperative_groups::__v1::thread_block_tile<16> &"),
                        makeCheckOr(
                            makeCheckOr(CheckArgType(0, "const class cooperative_groups::__v1::thread_block_tile<8> &"),
                                        CheckArgType(0, "const class cooperative_groups::__v1::thread_block_tile<4> &")),
                            makeCheckOr(CheckArgType(0, "const class cooperative_groups::__v1::thread_block_tile<2> &"),
                                        CheckArgType(0, "const class cooperative_groups::__v1::thread_block_tile<1> &")))),
                    FEATURE_REQUEST_FACTORY(
                        HelperFeatureEnum::Util_logical_group_get_local_linear_id,
                        MEMBER_CALL_FACTORY_ENTRY("thread_rank",
                                                  MemberExprBase(), false, "get_local_linear_id")),
                    UNSUPPORT_FACTORY_ENTRY("thread_rank",
                                            Diagnostics::API_NOT_MIGRATED, ARG("thread_rank"))),
                UNSUPPORT_FACTORY_ENTRY("thread_rank",
                                        Diagnostics::API_NOT_MIGRATED, ARG("thread_rank")))))

CONDITIONAL_FACTORY_ENTRY(
    makeCheckAnd(CheckArgType(0, "const class cooperative_groups::__v1::thread_block &"), CheckIntergerTemplateArgValueLE(0, 32)),
        CONDITIONAL_FACTORY_ENTRY(
            makeCheckAnd(CheckIntergerTemplateArgValueNE(0, 32), UseLogicalGroup),
            FEATURE_REQUEST_FACTORY(
                HelperFeatureEnum::Util_logical_group,
                CALL_FACTORY_ENTRY("tiled_partition", CALL(MapNames::getDpctNamespace() + "experimental::logical_group", NDITEM, GROUP, makeCallArgCreatorFromTemplateArg(0)))),
            TOSTRING_FACTORY_ENTRY("tiled_partition", SUBGROUP)),
        UNSUPPORT_FACTORY_ENTRY("tiled_partition", Diagnostics::API_NOT_MIGRATED, ARG("tiled_partition")))
#endif

#ifdef CLASS_METHOD_CALL
// T.sync()
// === cuda header file <= 10.2 === begin ===
WARNING_FACTORY_ENTRY(
    "cooperative_groups::__v1::__thread_block_tile_base<32>.sync",
    MEMBER_CALL_FACTORY_ENTRY("cooperative_groups::__v1::__thread_block_tile_base<32>.sync",
                              SUBGROUP, false, "barrier"),
    Diagnostics::BARRIER_PERFORMANCE_TUNNING, ARG("sub_group"))
// === cuda header file <= 10.2 === end ===
// === cuda header file >= 11.0 === begin ===
WARNING_FACTORY_ENTRY(
    "cooperative_groups::__v1::thread_block_tile<32>.sync",
    MEMBER_CALL_FACTORY_ENTRY("cooperative_groups::__v1::thread_block_tile<32>.sync",
                              SUBGROUP, false, "barrier"),
    Diagnostics::BARRIER_PERFORMANCE_TUNNING, ARG("sub_group"))
// === cuda header file >= 11.0 === end ===

WARNING_FACTORY_ENTRY(
    "cooperative_groups::__v1::thread_block.sync",
    MEMBER_CALL_FACTORY_ENTRY("cooperative_groups::__v1::thread_block.sync", NDITEM,
                              false, "barrier"),
    Diagnostics::BARRIER_PERFORMANCE_TUNNING, ARG("nd_item"))
WARNING_FACTORY_ENTRY(
    "cooperative_groups::__v1::coalesced_group.sync",
    MEMBER_CALL_FACTORY_ENTRY("cooperative_groups::__v1::coalesced_group.sync", NDITEM,
                              false, "barrier"),
    Diagnostics::BARRIER_PERFORMANCE_TUNNING, ARG("nd_item"))
CONDITIONAL_FACTORY_ENTRY(
    UseNDRangeBarrier,
    FEATURE_REQUEST_FACTORY(
        HelperFeatureEnum::Util_nd_range_barrier,
        CALL_FACTORY_ENTRY(
            "cooperative_groups::__v1::grid_group.sync",
            CALL(MapNames::getDpctNamespace() + "experimental::nd_range_barrier", NDITEM, ARG(DpctGlobalInfo::getSyncName())))),
    UNSUPPORT_FACTORY_ENTRY("cooperative_groups::__v1::grid_group.sync", Diagnostics::ND_RANGE_BARRIER, makeCombinedArg(MemberExprBase(), ARG(".sync()"))))


// T.size()
// === cuda header file <= 10.2 === begin ===
MEMBER_CALL_FACTORY_ENTRY("cooperative_groups::__v1::__thread_block_tile_base<32>.size", SUBGROUP, false, "get_local_linear_range")
CONDITIONAL_FACTORY_ENTRY(
    UseLogicalGroup,
    FEATURE_REQUEST_FACTORY(
        HelperFeatureEnum::Util_logical_group_get_local_linear_range,
        MEMBER_CALL_FACTORY_ENTRY("cooperative_groups::__v1::__thread_block_tile_base<16>.size", MemberExprBase(), false, "get_local_linear_range")),
    UNSUPPORT_FACTORY_ENTRY("cooperative_groups::__v1::__thread_block_tile_base<16>.size", Diagnostics::API_NOT_MIGRATED, ARG("size")))
CONDITIONAL_FACTORY_ENTRY(
    UseLogicalGroup,
    FEATURE_REQUEST_FACTORY(
        HelperFeatureEnum::Util_logical_group_get_local_linear_range,
        MEMBER_CALL_FACTORY_ENTRY("cooperative_groups::__v1::__thread_block_tile_base<8>.size", MemberExprBase(), false, "get_local_linear_range")),
    UNSUPPORT_FACTORY_ENTRY("cooperative_groups::__v1::__thread_block_tile_base<8>.size", Diagnostics::API_NOT_MIGRATED, ARG("size")))
CONDITIONAL_FACTORY_ENTRY(
    UseLogicalGroup,
    FEATURE_REQUEST_FACTORY(
        HelperFeatureEnum::Util_logical_group_get_local_linear_range,
        MEMBER_CALL_FACTORY_ENTRY("cooperative_groups::__v1::__thread_block_tile_base<4>.size", MemberExprBase(), false, "get_local_linear_range")),
    UNSUPPORT_FACTORY_ENTRY("cooperative_groups::__v1::__thread_block_tile_base<4>.size", Diagnostics::API_NOT_MIGRATED, ARG("size")))
CONDITIONAL_FACTORY_ENTRY(
    UseLogicalGroup,
    FEATURE_REQUEST_FACTORY(
        HelperFeatureEnum::Util_logical_group_get_local_linear_range,
        MEMBER_CALL_FACTORY_ENTRY("cooperative_groups::__v1::__thread_block_tile_base<2>.size", MemberExprBase(), false, "get_local_linear_range")),
    UNSUPPORT_FACTORY_ENTRY("cooperative_groups::__v1::__thread_block_tile_base<2>.size", Diagnostics::API_NOT_MIGRATED, ARG("size")))
CONDITIONAL_FACTORY_ENTRY(
    UseLogicalGroup,
    FEATURE_REQUEST_FACTORY(
        HelperFeatureEnum::Util_logical_group_get_local_linear_range,
        MEMBER_CALL_FACTORY_ENTRY("cooperative_groups::__v1::__thread_block_tile_base<1>.size", MemberExprBase(), false, "get_local_linear_range")),
    UNSUPPORT_FACTORY_ENTRY("cooperative_groups::__v1::__thread_block_tile_base<1>.size", Diagnostics::API_NOT_MIGRATED, ARG("size")))
// === cuda header file <= 10.2 === end ===
// === cuda header file >= 11.0 === begin ===
MEMBER_CALL_FACTORY_ENTRY("cooperative_groups::__v1::thread_block_tile<32>.size", SUBGROUP, false, "get_local_linear_range")
CONDITIONAL_FACTORY_ENTRY(
    UseLogicalGroup,
    FEATURE_REQUEST_FACTORY(
        HelperFeatureEnum::Util_logical_group_get_local_linear_range,
        MEMBER_CALL_FACTORY_ENTRY("cooperative_groups::__v1::thread_block_tile<16>.size", MemberExprBase(), false, "get_local_linear_range")),
    UNSUPPORT_FACTORY_ENTRY("cooperative_groups::__v1::thread_block_tile<16>.size", Diagnostics::API_NOT_MIGRATED, ARG("size")))
CONDITIONAL_FACTORY_ENTRY(
    UseLogicalGroup,
    FEATURE_REQUEST_FACTORY(
        HelperFeatureEnum::Util_logical_group_get_local_linear_range,
        MEMBER_CALL_FACTORY_ENTRY("cooperative_groups::__v1::thread_block_tile<8>.size", MemberExprBase(), false, "get_local_linear_range")),
    UNSUPPORT_FACTORY_ENTRY("cooperative_groups::__v1::thread_block_tile<8>.size", Diagnostics::API_NOT_MIGRATED, ARG("size")))
CONDITIONAL_FACTORY_ENTRY(
    UseLogicalGroup,
    FEATURE_REQUEST_FACTORY(
        HelperFeatureEnum::Util_logical_group_get_local_linear_range,
        MEMBER_CALL_FACTORY_ENTRY("cooperative_groups::__v1::thread_block_tile<4>.size", MemberExprBase(), false, "get_local_linear_range")),
    UNSUPPORT_FACTORY_ENTRY("cooperative_groups::__v1::thread_block_tile<4>.size", Diagnostics::API_NOT_MIGRATED, ARG("size")))
CONDITIONAL_FACTORY_ENTRY(
    UseLogicalGroup,
    FEATURE_REQUEST_FACTORY(
        HelperFeatureEnum::Util_logical_group_get_local_linear_range,
        MEMBER_CALL_FACTORY_ENTRY("cooperative_groups::__v1::thread_block_tile<2>.size", MemberExprBase(), false, "get_local_linear_range")),
    UNSUPPORT_FACTORY_ENTRY("cooperative_groups::__v1::thread_block_tile<2>.size", Diagnostics::API_NOT_MIGRATED, ARG("size")))
CONDITIONAL_FACTORY_ENTRY(
    UseLogicalGroup,
    FEATURE_REQUEST_FACTORY(
        HelperFeatureEnum::Util_logical_group_get_local_linear_range,
        MEMBER_CALL_FACTORY_ENTRY("cooperative_groups::__v1::thread_block_tile<1>.size", MemberExprBase(), false, "get_local_linear_range")),
    UNSUPPORT_FACTORY_ENTRY("cooperative_groups::__v1::thread_block_tile<1>.size", Diagnostics::API_NOT_MIGRATED, ARG("size")))
// === cuda header file >= 11.0 === end ===

// T.thread_rank()
MEMBER_CALL_FACTORY_ENTRY("cooperative_groups::__v1::thread_block.thread_rank", NDITEM, false, "get_local_linear_id")
// === cuda header file <= 10.2 === begin ===
MEMBER_CALL_FACTORY_ENTRY("cooperative_groups::__v1::__thread_block_tile_base<32>.thread_rank", SUBGROUP, false, "get_local_linear_id")
CONDITIONAL_FACTORY_ENTRY(
    UseLogicalGroup,
    FEATURE_REQUEST_FACTORY(
        HelperFeatureEnum::Util_logical_group_get_local_linear_id,
        MEMBER_CALL_FACTORY_ENTRY("cooperative_groups::__v1::__thread_block_tile_base<16>.thread_rank", MemberExprBase(), false, "get_local_linear_id")),
    UNSUPPORT_FACTORY_ENTRY("cooperative_groups::__v1::__thread_block_tile_base<16>.thread_rank", Diagnostics::API_NOT_MIGRATED, ARG("thread_rank")))
CONDITIONAL_FACTORY_ENTRY(
    UseLogicalGroup,
    FEATURE_REQUEST_FACTORY(
        HelperFeatureEnum::Util_logical_group_get_local_linear_id,
        MEMBER_CALL_FACTORY_ENTRY("cooperative_groups::__v1::__thread_block_tile_base<8>.thread_rank", MemberExprBase(), false, "get_local_linear_id")),
    UNSUPPORT_FACTORY_ENTRY("cooperative_groups::__v1::__thread_block_tile_base<8>.thread_rank", Diagnostics::API_NOT_MIGRATED, ARG("thread_rank")))
CONDITIONAL_FACTORY_ENTRY(
    UseLogicalGroup,
    FEATURE_REQUEST_FACTORY(
        HelperFeatureEnum::Util_logical_group_get_local_linear_id,
        MEMBER_CALL_FACTORY_ENTRY("cooperative_groups::__v1::__thread_block_tile_base<4>.thread_rank", MemberExprBase(), false, "get_local_linear_id")),
    UNSUPPORT_FACTORY_ENTRY("cooperative_groups::__v1::__thread_block_tile_base<4>.thread_rank", Diagnostics::API_NOT_MIGRATED, ARG("thread_rank")))
CONDITIONAL_FACTORY_ENTRY(
    UseLogicalGroup,
    FEATURE_REQUEST_FACTORY(
        HelperFeatureEnum::Util_logical_group_get_local_linear_id,
        MEMBER_CALL_FACTORY_ENTRY("cooperative_groups::__v1::__thread_block_tile_base<2>.thread_rank", MemberExprBase(), false, "get_local_linear_id")),
    UNSUPPORT_FACTORY_ENTRY("cooperative_groups::__v1::__thread_block_tile_base<2>.thread_rank", Diagnostics::API_NOT_MIGRATED, ARG("thread_rank")))
CONDITIONAL_FACTORY_ENTRY(
    UseLogicalGroup,
    FEATURE_REQUEST_FACTORY(
        HelperFeatureEnum::Util_logical_group_get_local_linear_id,
        MEMBER_CALL_FACTORY_ENTRY("cooperative_groups::__v1::__thread_block_tile_base<1>.thread_rank", MemberExprBase(), false, "get_local_linear_id")),
    UNSUPPORT_FACTORY_ENTRY("cooperative_groups::__v1::__thread_block_tile_base<1>.thread_rank", Diagnostics::API_NOT_MIGRATED, ARG("thread_rank")))
// === cuda header file <= 10.2 === end ===
// === cuda header file >= 11.0 === begin ===
MEMBER_CALL_FACTORY_ENTRY("cooperative_groups::__v1::thread_block_tile<32>.thread_rank", SUBGROUP, false, "get_local_linear_id")
CONDITIONAL_FACTORY_ENTRY(
    UseLogicalGroup,
    FEATURE_REQUEST_FACTORY(
        HelperFeatureEnum::Util_logical_group_get_local_linear_id,
        MEMBER_CALL_FACTORY_ENTRY("cooperative_groups::__v1::thread_block_tile<16>.thread_rank", MemberExprBase(), false, "get_local_linear_id")),
    UNSUPPORT_FACTORY_ENTRY("cooperative_groups::__v1::thread_block_tile<16>.thread_rank", Diagnostics::API_NOT_MIGRATED, ARG("thread_rank")))
CONDITIONAL_FACTORY_ENTRY(
    UseLogicalGroup,
    FEATURE_REQUEST_FACTORY(
        HelperFeatureEnum::Util_logical_group_get_local_linear_id,
        MEMBER_CALL_FACTORY_ENTRY("cooperative_groups::__v1::thread_block_tile<8>.thread_rank", MemberExprBase(), false, "get_local_linear_id")),
    UNSUPPORT_FACTORY_ENTRY("cooperative_groups::__v1::thread_block_tile<8>.thread_rank", Diagnostics::API_NOT_MIGRATED, ARG("thread_rank")))
CONDITIONAL_FACTORY_ENTRY(
    UseLogicalGroup,
    FEATURE_REQUEST_FACTORY(
        HelperFeatureEnum::Util_logical_group_get_local_linear_id,
        MEMBER_CALL_FACTORY_ENTRY("cooperative_groups::__v1::thread_block_tile<4>.thread_rank", MemberExprBase(), false, "get_local_linear_id")),
    UNSUPPORT_FACTORY_ENTRY("cooperative_groups::__v1::thread_block_tile<4>.thread_rank", Diagnostics::API_NOT_MIGRATED, ARG("thread_rank")))
CONDITIONAL_FACTORY_ENTRY(
    UseLogicalGroup,
    FEATURE_REQUEST_FACTORY(
        HelperFeatureEnum::Util_logical_group_get_local_linear_id,
        MEMBER_CALL_FACTORY_ENTRY("cooperative_groups::__v1::thread_block_tile<2>.thread_rank", MemberExprBase(), false, "get_local_linear_id")),
    UNSUPPORT_FACTORY_ENTRY("cooperative_groups::__v1::thread_block_tile<2>.thread_rank", Diagnostics::API_NOT_MIGRATED, ARG("thread_rank")))
CONDITIONAL_FACTORY_ENTRY(
    UseLogicalGroup,
    FEATURE_REQUEST_FACTORY(
        HelperFeatureEnum::Util_logical_group_get_local_linear_id,
        MEMBER_CALL_FACTORY_ENTRY("cooperative_groups::__v1::thread_block_tile<1>.thread_rank", MemberExprBase(), false, "get_local_linear_id")),
    UNSUPPORT_FACTORY_ENTRY("cooperative_groups::__v1::thread_block_tile<1>.thread_rank", Diagnostics::API_NOT_MIGRATED, ARG("thread_rank")))
// === cuda header file >= 11.0 === end ===

// T.shfl_down()
// === cuda header file <= 10.2 === begin ===
FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Util_shift_sub_group_left,
            CALL_FACTORY_ENTRY("cooperative_groups::__v1::__thread_block_tile_base<1>.shfl_down",
                           CALL(MapNames::getDpctNamespace() +
                                    "shift_sub_group_left",
                                SUBGROUP, ARG(0), ARG(1), ARG("1"))))
FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Util_shift_sub_group_left,
            CALL_FACTORY_ENTRY("cooperative_groups::__v1::__thread_block_tile_base<2>.shfl_down",
                               CALL(MapNames::getDpctNamespace() +
                                        "shift_sub_group_left",
                                    SUBGROUP, ARG(0), ARG(1), ARG("2"))))
FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Util_shift_sub_group_left,
            CALL_FACTORY_ENTRY("cooperative_groups::__v1::__thread_block_tile_base<4>.shfl_down",
                               CALL(MapNames::getDpctNamespace() +
                                        "shift_sub_group_left",
                                    SUBGROUP, ARG(0), ARG(1), ARG("4"))))
FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Util_shift_sub_group_left,
            CALL_FACTORY_ENTRY("cooperative_groups::__v1::__thread_block_tile_base<8>.shfl_down",
                               CALL(MapNames::getDpctNamespace() +
                                        "shift_sub_group_left",
                                    SUBGROUP, ARG(0), ARG(1), ARG("8"))))
FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Util_shift_sub_group_left,
            CALL_FACTORY_ENTRY("cooperative_groups::__v1::__thread_block_tile_base<16>.shfl_down",
                               CALL(MapNames::getDpctNamespace() +
                                        "shift_sub_group_left",
                                    SUBGROUP, ARG(0), ARG(1), ARG("16"))))
MEMBER_CALL_FACTORY_ENTRY("cooperative_groups::__v1::__thread_block_tile_base<32>.shfl_down",
                                MemberExprBase(), false, "shuffle_down", ARG(0), ARG(1))
// === cuda header file <= 10.2 === end ===
// === cuda header file >= 11.0 === begin ===
FEATURE_REQUEST_FACTORY(
        HelperFeatureEnum::Util_shift_sub_group_left,
        CALL_FACTORY_ENTRY("cooperative_groups::__v1::__static_size_thread_block_tile_base<1>.shfl_down",
                           CALL(MapNames::getDpctNamespace() +
                                    "shift_sub_group_left",
                                SUBGROUP, ARG(0), ARG(1), ARG("1"))))
FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Util_shift_sub_group_left,
            CALL_FACTORY_ENTRY("cooperative_groups::__v1::__static_size_thread_block_tile_base<2>.shfl_down",
                               CALL(MapNames::getDpctNamespace() +
                                        "shift_sub_group_left",
                                    SUBGROUP, ARG(0), ARG(1), ARG("2"))))
FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Util_shift_sub_group_left,
            CALL_FACTORY_ENTRY("cooperative_groups::__v1::__static_size_thread_block_tile_base<4>.shfl_down",
                               CALL(MapNames::getDpctNamespace() +
                                        "shift_sub_group_left",
                                    SUBGROUP, ARG(0), ARG(1), ARG("4"))))
FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Util_shift_sub_group_left,
            CALL_FACTORY_ENTRY("cooperative_groups::__v1::__static_size_thread_block_tile_base<8>.shfl_down",
                               CALL(MapNames::getDpctNamespace() +
                                        "shift_sub_group_left",
                                    SUBGROUP, ARG(0), ARG(1), ARG("8"))))
FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Util_shift_sub_group_left,
            CALL_FACTORY_ENTRY("cooperative_groups::__v1::__static_size_thread_block_tile_base<16>.shfl_down",
                               CALL(MapNames::getDpctNamespace() +
                                        "shift_sub_group_left",
                                    SUBGROUP, ARG(0), ARG(1), ARG("16"))))
MEMBER_CALL_FACTORY_ENTRY("cooperative_groups::__v1::__static_size_thread_block_tile_base<32>.shfl_down",
                                MemberExprBase(), false, "shuffle_down", ARG(0), ARG(1))
// === cuda header file >= 11.0 === end ===
#endif
