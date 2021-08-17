//===---APINamesWarp.inc---------------------------------------------===//
//
// Copyright (C) 2018 - 2021 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

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

// __shfl_up_sync
SUBGROUPSIZE_FACTORY(
    3, "shift_group_right",
    WARNING_FACTORY_ENTRY(
        "__shfl_up_sync",
        CALL_FACTORY_ENTRY(
            "__shfl_up_sync",
            CALL(MapNames::getClNamespace() + "shift_group_right", SUBGROUP,
                 makeCallArgCreatorWithCall(1), makeCallArgCreatorWithCall(2))),
        Diagnostics::MASK_UNSUPPORTED,
        ARG(MapNames::getClNamespace() + "shift_group_right")))

// __shfl_up
SUBGROUPSIZE_FACTORY(
    2, "shift_group_right",
    CALL_FACTORY_ENTRY("__shfl_up",
                       CALL(MapNames::getClNamespace() + "shift_group_right",
                            SUBGROUP, makeCallArgCreatorWithCall(0),
                            makeCallArgCreatorWithCall(1))))

// __shfl_down_sync
SUBGROUPSIZE_FACTORY(
    3, "shift_group_left",
    WARNING_FACTORY_ENTRY(
        "__shfl_down_sync",
        CALL_FACTORY_ENTRY("__shfl_down_sync",
                           CALL(MapNames::getClNamespace() + "shift_group_left",
                                SUBGROUP, makeCallArgCreatorWithCall(1),
                                makeCallArgCreatorWithCall(2))),
        Diagnostics::MASK_UNSUPPORTED,
        ARG(MapNames::getClNamespace() + "shift_group_left")))
// __shfl_down
SUBGROUPSIZE_FACTORY(
    2, "shift_group_left",
    CALL_FACTORY_ENTRY("__shfl_down",
                       CALL(MapNames::getClNamespace() + "shift_group_left",
                            SUBGROUP, makeCallArgCreatorWithCall(0),
                            makeCallArgCreatorWithCall(1))))

//__shfl_sync
SUBGROUPSIZE_FACTORY(
    3, "shuffle",
    WARNING_FACTORY_ENTRY(
        "__shfl_sync",
        MEMBER_CALL_FACTORY_ENTRY("__shfl_sync", SUBGROUP, false, "shuffle",
                                  makeCallArgCreatorWithCall(1),
                                  makeCallArgCreatorWithCall(2)),
        Diagnostics::MASK_UNSUPPORTED, ARG("shuffle")))
// __shfl
SUBGROUPSIZE_FACTORY(2, "shuffle",
                     MEMBER_CALL_FACTORY_ENTRY("__shfl", SUBGROUP, false,
                                               "shuffle",
                                               makeCallArgCreatorWithCall(0),
                                               makeCallArgCreatorWithCall(1)))

// __shfl_xor_sync
SUBGROUPSIZE_FACTORY(
    3, "shuffle_xor",
    WARNING_FACTORY_ENTRY(
        "__shfl_xor_sync",
        MEMBER_CALL_FACTORY_ENTRY("__shfl_xor_sync", SUBGROUP, false,
                                  "shuffle_xor", makeCallArgCreatorWithCall(1),
                                  makeCallArgCreatorWithCall(2)),
        Diagnostics::MASK_UNSUPPORTED, ARG("shuffle_xor")))

// __shfl_xor
SUBGROUPSIZE_FACTORY(2, "shuffle_xor",
                     MEMBER_CALL_FACTORY_ENTRY("__shfl_xor", SUBGROUP, false,
                                               "shuffle_xor",
                                               makeCallArgCreatorWithCall(0),
                                               makeCallArgCreatorWithCall(1)))

WARNING_FACTORY_ENTRY("__activemask",
                      CALL_FACTORY_ENTRY("__activemask", CALL("__activemask")),
                      Diagnostics::ACTIVE_MASK_NOT_SUPPORTED)

// __ballot
SUBGROUPSIZE_FACTORY(
    UINT_MAX, MapNames::getClNamespace() + "reduce_over_group",
    CALL_FACTORY_ENTRY(
        "__ballot",
        CALL(MapNames::getClNamespace() + "reduce_over_group", SUBGROUP,
             makeCombinedArg(ARG(0),
                             makeCombinedArg(makeCombinedArg(ARG(" ? (0x1 << "),
                                                             SUBGROUP),
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