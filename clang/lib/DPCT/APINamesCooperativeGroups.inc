//===--------------- APINamesCooperativeGroups.inc ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifdef FUNCTION_CALL


/*
   --- cg::sync(X) ---
   cg::sync(cg::thread_block)
   => item_ct1.barrier()

   cg::sync(cg::coalesced_group)
   => item_ct1.barrier()
   (cg::coalesced_group's are created from
   cg::coalesced_threads(), which is not migrated,
   so the migration here doesn't really matter since
   the code won't be fully migrated anyways)

   cg::sync(cg::thread_block_tile<32>)
   => item_ct1.get_sub_group().barrier()

   cg::sync(cg::grid_group)
   => if UseNDRangeBarrier then
   dpct::experimental::nd_range_barrier(item_ct1, sync_ct1)
   otherwise unsupported

   Also, issue warning about barrier performance tuning.
*/
CONDITIONAL_FACTORY_ENTRY(
    CheckParamType(0, "const class cooperative_groups::__v1::thread_block &"),
    WARNING_FACTORY_ENTRY(
        "sync",
        MEMBER_CALL_FACTORY_ENTRY("sync", NDITEM,
                                  false, "barrier"),
        Diagnostics::BARRIER_PERFORMANCE_TUNNING, ARG("nd_item")),
    CONDITIONAL_FACTORY_ENTRY(
        CheckParamType(0, "const class cooperative_groups::__v1::coalesced_group &"),
        WARNING_FACTORY_ENTRY(
            "sync",
            MEMBER_CALL_FACTORY_ENTRY("sync", NDITEM,
                                      false, "barrier"),
            Diagnostics::BARRIER_PERFORMANCE_TUNNING, ARG("nd_item")),
        CONDITIONAL_FACTORY_ENTRY(
            argHasThreadBlockTileType(0, 32),
            WARNING_FACTORY_ENTRY(
                "sync",
                MEMBER_CALL_FACTORY_ENTRY("sync",
                                          SUBGROUP, false, "barrier"),
                Diagnostics::BARRIER_PERFORMANCE_TUNNING, ARG("sub_group")),
            CONDITIONAL_FACTORY_ENTRY(
                CheckParamType(0, "const class cooperative_groups::__v1::grid_group &"),
                CONDITIONAL_FACTORY_ENTRY(
                    UseNDRangeBarrier,
                    FEATURE_REQUEST_FACTORY(
                        HelperFeatureEnum::device_ext,
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

/*
   --- cg::reduce(group, val, op) ---

   cg::reduce(cg::thread_block_tile<32>, val, op)
   => sycl::reduce_over_group(item.get_sub_group(), val, op)
*/
CONDITIONAL_FACTORY_ENTRY(
    argHasThreadBlockTileType(0, 32),
        CALL_FACTORY_ENTRY("reduce", CALL(MapNames::getClNamespace() + "reduce_over_group",
                                    SUBGROUP, ARG(1), ARG(2))),
        UNSUPPORT_FACTORY_ENTRY("reduce",
                                    Diagnostics::API_NOT_MIGRATED,
                                    ARG("reduce")))

/*
  --- cg::thread_rank(X) ---
  cg::thread_rank(cg::thread_block)
  => item_ct1.get_local_linear_id()

  cg::thread_rank(cg::thread_block_tile<32>)
  => item_ct1.get_sub_group().get_local_linear_id()

  cg::thread_rank(cg::thread_block_tile<N>) if n in {1,2,4,8,16}
  => if UseLogicalGroup then
     ARG(0).get_local_linear_id()
*/
CONDITIONAL_FACTORY_ENTRY(
        CheckParamType(0, "const class cooperative_groups::__v1::thread_block &"),
        MEMBER_CALL_FACTORY_ENTRY("thread_rank", NDITEM,
                                  false, "get_local_linear_id"),
        CONDITIONAL_FACTORY_ENTRY(
            argHasThreadBlockTileType(0, 32),
            MEMBER_CALL_FACTORY_ENTRY("thread_rank", SUBGROUP,
                                      false, "get_local_linear_id"),
            CONDITIONAL_FACTORY_ENTRY(
                UseLogicalGroup,
                CONDITIONAL_FACTORY_ENTRY(
                    makeCheckOr(
                        argHasThreadBlockTileType(0, 16),
                        argHasThreadBlockTileType(0, 8),
                        argHasThreadBlockTileType(0, 4),
                        argHasThreadBlockTileType(0, 2),
                        argHasThreadBlockTileType(0, 1)),
                    FEATURE_REQUEST_FACTORY(
                        HelperFeatureEnum::device_ext,
                        MEMBER_CALL_FACTORY_ENTRY("thread_rank",
                                                  ARG(0), false, "get_local_linear_id")),
                    UNSUPPORT_FACTORY_ENTRY("thread_rank",
                                            Diagnostics::API_NOT_MIGRATED, ARG("thread_rank"))),
                UNSUPPORT_FACTORY_ENTRY("thread_rank",
                                        Diagnostics::API_NOT_MIGRATED, ARG("thread_rank")))))

CONDITIONAL_FACTORY_ENTRY(
    makeCheckAnd(CheckParamType(0, "const class cooperative_groups::__v1::thread_block &"), CheckIntergerTemplateArgValueLE(0, 32)),
        CONDITIONAL_FACTORY_ENTRY(
            makeCheckAnd(CheckIntergerTemplateArgValueNE(0, 32), UseLogicalGroup),
            FEATURE_REQUEST_FACTORY(
                HelperFeatureEnum::device_ext,
                CALL_FACTORY_ENTRY("tiled_partition", CALL(MapNames::getDpctNamespace() + "experimental::logical_group", NDITEM, GROUP, makeCallArgCreatorFromTemplateArg(0)))),
            TOSTRING_FACTORY_ENTRY("tiled_partition", SUBGROUP)),
        UNSUPPORT_FACTORY_ENTRY("tiled_partition", Diagnostics::API_NOT_MIGRATED, ARG("tiled_partition")))
#endif

#ifdef CLASS_METHOD_CALL
/*
  --- X.sync() ---
  cg::thread_block::sync()
  => item_ct1.barrier()

  cg::coalesced_group::sync()
  => item_ct1.barrier()

  cg::thread_block_tile<32>::sync()
  => item_ct1.get_sub_group().sync()

  cg::sync(cg::grid_group)
  => if UseNDRangeBarrier then
     dpct::experimental::nd_range_barrier(item_ct1, sync_ct1)
     otherwise unsupported
*/
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
// for cuda <= 10.2
CONDITIONAL_FACTORY_ENTRY(
    baseHasThreadBlockTileType(32),
    WARNING_FACTORY_ENTRY(
        "cooperative_groups::__v1::__thread_block_tile_base.sync",
        MEMBER_CALL_FACTORY_ENTRY("cooperative_groups::__v1::__thread_block_tile_base.sync",
                                  SUBGROUP, false, "barrier"),
        Diagnostics::BARRIER_PERFORMANCE_TUNNING, ARG("sub_group")),
    UNSUPPORT_FACTORY_ENTRY("cooperative_groups::__v1::__thread_block_tile_base.sync",
                            Diagnostics::API_NOT_MIGRATED,
                            ARG("sync")))
// for cuda >= 11.0
CONDITIONAL_FACTORY_ENTRY(
    baseHasThreadBlockTileType(32),
    WARNING_FACTORY_ENTRY(
        "cooperative_groups::__v1::thread_block_tile.sync",
        MEMBER_CALL_FACTORY_ENTRY("cooperative_groups::__v1::thread_block_tile.sync",
                                  SUBGROUP, false, "barrier"),
        Diagnostics::BARRIER_PERFORMANCE_TUNNING, ARG("sub_group")),
    UNSUPPORT_FACTORY_ENTRY("cooperative_groups::__v1::thread_block_tile.sync",
                            Diagnostics::API_NOT_MIGRATED,
                            ARG("sync")))
CONDITIONAL_FACTORY_ENTRY(
    UseNDRangeBarrier,
    FEATURE_REQUEST_FACTORY(
        HelperFeatureEnum::device_ext,
        CALL_FACTORY_ENTRY(
            "cooperative_groups::__v1::grid_group.sync",
            CALL(MapNames::getDpctNamespace() + "experimental::nd_range_barrier", NDITEM, ARG(DpctGlobalInfo::getSyncName())))),
    UNSUPPORT_FACTORY_ENTRY("cooperative_groups::__v1::grid_group.sync", Diagnostics::ND_RANGE_BARRIER, makeCombinedArg(MemberExprBase(), ARG(".sync()"))))

/*
  --- X.size() ---
  cg::thread_block::size()
  => item_ct1.get_group().get_local_linear_range()

  cg::thread_block_tile<32>::size()
  => item_ct1.get_subgroup().get_local_linear_range()

  cg::thread_block_tile<N>::size() if n in {1,2,4,8,16}
  => if UseLogicalGroup then
     MemberBase().get_local_linear_id()
*/
MEMBER_CALL_FACTORY_ENTRY("cooperative_groups::__v1::thread_block.size", GROUP,
                          false, "get_local_linear_range")
// for cuda <= 10.2
CASE_FACTORY_ENTRY(
    CASE(baseHasThreadBlockTileType(32),
         MEMBER_CALL_FACTORY_ENTRY("cooperative_groups::__v1::__thread_block_tile_base.size",
                                   SUBGROUP, false, "get_local_linear_range")),
    CASE(makeCheckAnd(UseLogicalGroup, makeCheckOr(
                        baseHasThreadBlockTileType(16),
                        baseHasThreadBlockTileType(8),
                        baseHasThreadBlockTileType(4),
                        baseHasThreadBlockTileType(2),
                        baseHasThreadBlockTileType(1))),
         FEATURE_REQUEST_FACTORY(
             HelperFeatureEnum::device_ext,
             MEMBER_CALL_FACTORY_ENTRY("cooperative_groups::__v1::__thread_block_tile_base.size",
                                       MemberExprBase(), false, "get_local_linear_range"))),
    OTHERWISE(
        UNSUPPORT_FACTORY_ENTRY("cooperative_groups::__v1::__thread_block_tile_base.size",
                                Diagnostics::API_NOT_MIGRATED, ARG("size"))))
// for cuda >= 11.0
CASE_FACTORY_ENTRY(
    CASE(baseHasThreadBlockTileType(32),
         MEMBER_CALL_FACTORY_ENTRY("cooperative_groups::__v1::thread_block_tile.size",
                                   SUBGROUP, false, "get_local_linear_range")),
    CASE(makeCheckAnd(UseLogicalGroup, makeCheckOr(
                        baseHasThreadBlockTileType(16),
                        baseHasThreadBlockTileType(8),
                        baseHasThreadBlockTileType(4),
                        baseHasThreadBlockTileType(2),
                        baseHasThreadBlockTileType(1))),
         FEATURE_REQUEST_FACTORY(
             HelperFeatureEnum::device_ext,
             MEMBER_CALL_FACTORY_ENTRY("cooperative_groups::__v1::thread_block_tile.size",
                                       MemberExprBase(), false, "get_local_linear_range"))),
    OTHERWISE(
        UNSUPPORT_FACTORY_ENTRY("cooperative_groups::__v1::thread_block_tile.size",
                                Diagnostics::API_NOT_MIGRATED, ARG("size"))))

/*
  --- X.thread_rank() ---
  cg::thread_block::thread_rank()
  => item_ct1.size()

  cg::thread_block_tile<32>::thread_rank()
  => item_ct1.get_sub_group().get_local_linear_id()

  cg::thread_block_tile<N>::thread_rank() if n in {1,2,4,8,16}
  => if UseLogicalGroup then
     MemberBase().get_local_linear_id()
*/
MEMBER_CALL_FACTORY_ENTRY("cooperative_groups::__v1::thread_block.thread_rank", NDITEM,
                          false, "get_local_linear_id")

// for cuda <= 10.2
CASE_FACTORY_ENTRY(
    CASE(baseHasThreadBlockTileType(32),
         MEMBER_CALL_FACTORY_ENTRY("cooperative_groups::__v1::__thread_block_tile_base.thread_rank",
                                   SUBGROUP, false, "get_local_linear_id")),
    CASE(makeCheckAnd(UseLogicalGroup, makeCheckOr(
                        baseHasThreadBlockTileType(16),
                        baseHasThreadBlockTileType(8),
                        baseHasThreadBlockTileType(4),
                        baseHasThreadBlockTileType(2),
                        baseHasThreadBlockTileType(1))),
         FEATURE_REQUEST_FACTORY(
             HelperFeatureEnum::device_ext,
             MEMBER_CALL_FACTORY_ENTRY("cooperative_groups::__v1::__thread_block_tile_base.thread_rank",
                                       MemberExprBase(), false, "get_local_linear_id"))),
    OTHERWISE(
        UNSUPPORT_FACTORY_ENTRY("cooperative_groups::__v1::__thread_block_tile_base.thread_rank",
                                Diagnostics::API_NOT_MIGRATED, ARG("thread_rank"))))

// for cuda >= 11.0
CASE_FACTORY_ENTRY(
    CASE(baseHasThreadBlockTileType(32),
         MEMBER_CALL_FACTORY_ENTRY("cooperative_groups::__v1::thread_block_tile.thread_rank",
                                   SUBGROUP, false, "get_local_linear_id")),
    CASE(makeCheckAnd(UseLogicalGroup, makeCheckOr(
                        baseHasThreadBlockTileType(16),
                        baseHasThreadBlockTileType(8),
                        baseHasThreadBlockTileType(4),
                        baseHasThreadBlockTileType(2),
                        baseHasThreadBlockTileType(1))),
         FEATURE_REQUEST_FACTORY(
             HelperFeatureEnum::device_ext,
             MEMBER_CALL_FACTORY_ENTRY("cooperative_groups::__v1::thread_block_tile.thread_rank",
                                       MemberExprBase(), false, "get_local_linear_id"))),
    OTHERWISE(
        UNSUPPORT_FACTORY_ENTRY("cooperative_groups::__v1::thread_block_tile.thread_rank",
                                Diagnostics::API_NOT_MIGRATED, ARG("thread_rank"))))

/*
  --- X.meta_group_rank() ---
  cg::thread_block_tile<32>::meta_group_rank()
  => item_ct1.get_sub_group().get_group_linear_id()

  cg::thread_block_tile<N>::meta_group_rank() if n in {1,2,4,8,16}
  => if UseLogicalGroup then
     MemberBase().get_group_linear_id()
*/
CASE_FACTORY_ENTRY(
    CASE(baseHasThreadBlockTileType(32),
         MEMBER_CALL_FACTORY_ENTRY("cooperative_groups::__v1::thread_block_tile.meta_group_rank",
                                   SUBGROUP, false, "get_group_linear_id")),
    CASE(makeCheckAnd(UseLogicalGroup, makeCheckOr(
                            baseHasThreadBlockTileType(16),
                            baseHasThreadBlockTileType(8),
                            baseHasThreadBlockTileType(4),
                            baseHasThreadBlockTileType(2),
                            baseHasThreadBlockTileType(1))),
         FEATURE_REQUEST_FACTORY(
             HelperFeatureEnum::device_ext,
             MEMBER_CALL_FACTORY_ENTRY("cooperative_groups::__v1::thread_block_tile.meta_group_rank",
                                       MemberExprBase(), false, "get_group_linear_id"))),
    OTHERWISE(
        UNSUPPORT_FACTORY_ENTRY("cooperative_groups::__v1::thread_block_tile.meta_group_rank",
                                Diagnostics::API_NOT_MIGRATED, ARG("meta_group_rank"))))

CASE_FACTORY_ENTRY(
    CASE(baseHasThreadBlockTileType(32),
         MEMBER_CALL_FACTORY_ENTRY("cooperative_groups::__v1::__single_warp_thread_block_tile.meta_group_rank",
                                   SUBGROUP, false, "get_group_linear_id")),
    CASE(makeCheckAnd(UseLogicalGroup, makeCheckOr(
                            baseHasThreadBlockTileType(16),
                            baseHasThreadBlockTileType(8),
                            baseHasThreadBlockTileType(4),
                            baseHasThreadBlockTileType(2),
                            baseHasThreadBlockTileType(1))),
         FEATURE_REQUEST_FACTORY(
             HelperFeatureEnum::device_ext,
             MEMBER_CALL_FACTORY_ENTRY("cooperative_groups::__v1::__single_warp_thread_block_tile.meta_group_rank",
                                       MemberExprBase(), false, "get_group_linear_id"))),
    OTHERWISE(
        UNSUPPORT_FACTORY_ENTRY("cooperative_groups::__v1::__single_warp_thread_block_tile.meta_group_rank",
                                Diagnostics::API_NOT_MIGRATED, ARG("meta_group_rank"))))

/*
  --- X.shfl_down() ---
  cg::thread_block_tile<32>::shfl_down(v, delta)
  => sycl::shift_group_left(item_ct1.get_sub_group(), v, delta)

  cg::thread_block_tile<N>::shfl_down(v, delta) if N in {1,2,4,8,16}
  => dpct::shift_sub_group_left(item_ct1.get_sub_group(), v, delta, N)
*/
// for cuda <= 10.2
CASE_FACTORY_ENTRY(
    CASE(baseHasThreadBlockTileType(32),
         CALL_FACTORY_ENTRY(
             "cooperative_groups::__v1::__thread_block_tile_base.shfl_down",
             CALL(MapNames::getClNamespace() + "shift_group_left",
                  SUBGROUP, ARG(0), ARG(1)))),
    CASE(baseHasThreadBlockTileType(16),
         FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::device_ext,
            CALL_FACTORY_ENTRY("cooperative_groups::__v1::__thread_block_tile_base.shfl_down",
                               CALL(MapNames::getDpctNamespace() +
                                        "shift_sub_group_left",
                                    SUBGROUP, ARG(0), ARG(1), ARG("16"))))),
    CASE(baseHasThreadBlockTileType(8),
         FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::device_ext,
            CALL_FACTORY_ENTRY("cooperative_groups::__v1::__thread_block_tile_base.shfl_down",
                               CALL(MapNames::getDpctNamespace() +
                                        "shift_sub_group_left",
                                    SUBGROUP, ARG(0), ARG(1), ARG("8"))))),
    CASE(baseHasThreadBlockTileType(4),
         FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::device_ext,
            CALL_FACTORY_ENTRY("cooperative_groups::__v1::__thread_block_tile_base.shfl_down",
                               CALL(MapNames::getDpctNamespace() +
                                        "shift_sub_group_left",
                                    SUBGROUP, ARG(0), ARG(1), ARG("4"))))),
    CASE(baseHasThreadBlockTileType(2),
         FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::device_ext,
            CALL_FACTORY_ENTRY("cooperative_groups::__v1::__thread_block_tile_base.shfl_down",
                               CALL(MapNames::getDpctNamespace() +
                                        "shift_sub_group_left",
                                    SUBGROUP, ARG(0), ARG(1), ARG("2"))))),
    CASE(baseHasThreadBlockTileType(1),
         FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::device_ext,
            CALL_FACTORY_ENTRY("cooperative_groups::__v1::__thread_block_tile_base.shfl_down",
                               CALL(MapNames::getDpctNamespace() +
                                        "shift_sub_group_left",
                                    SUBGROUP, ARG(0), ARG(1), ARG("1"))))),
    OTHERWISE(
         UNSUPPORT_FACTORY_ENTRY("cooperative_groups::__v1::__thread_block_tile_base.shfl_down",
                                 Diagnostics::API_NOT_MIGRATED, ARG("shfl_down"))))

// for cuda >= 11.0
CASE_FACTORY_ENTRY(
    CASE(baseHasThreadBlockTileType(32),
         CALL_FACTORY_ENTRY(
             "cooperative_groups::__v1::__static_size_thread_block_tile_base.shfl_down",
             CALL(MapNames::getClNamespace() + "shift_group_left",
                  SUBGROUP, ARG(0), ARG(1)))),
    CASE(baseHasThreadBlockTileType(16),
         FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::device_ext,
            CALL_FACTORY_ENTRY("cooperative_groups::__v1::__static_size_thread_block_tile_base.shfl_down",
                               CALL(MapNames::getDpctNamespace() +
                                        "shift_sub_group_left",
                                    SUBGROUP, ARG(0), ARG(1), ARG("16"))))),
    CASE(baseHasThreadBlockTileType(8),
         FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::device_ext,
            CALL_FACTORY_ENTRY("cooperative_groups::__v1::__static_size_thread_block_tile_base.shfl_down",
                               CALL(MapNames::getDpctNamespace() +
                                        "shift_sub_group_left",
                                    SUBGROUP, ARG(0), ARG(1), ARG("8"))))),
    CASE(baseHasThreadBlockTileType(4),
         FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::device_ext,
            CALL_FACTORY_ENTRY("cooperative_groups::__v1::__static_size_thread_block_tile_base.shfl_down",
                               CALL(MapNames::getDpctNamespace() +
                                        "shift_sub_group_left",
                                    SUBGROUP, ARG(0), ARG(1), ARG("4"))))),
    CASE(baseHasThreadBlockTileType(2),
         FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::device_ext,
            CALL_FACTORY_ENTRY("cooperative_groups::__v1::__static_size_thread_block_tile_base.shfl_down",
                               CALL(MapNames::getDpctNamespace() +
                                        "shift_sub_group_left",
                                    SUBGROUP, ARG(0), ARG(1), ARG("2"))))),
    CASE(baseHasThreadBlockTileType(1),
         FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::device_ext,
            CALL_FACTORY_ENTRY("cooperative_groups::__v1::__static_size_thread_block_tile_base.shfl_down",
                               CALL(MapNames::getDpctNamespace() +
                                        "shift_sub_group_left",
                                    SUBGROUP, ARG(0), ARG(1), ARG("1"))))),
    OTHERWISE(
         UNSUPPORT_FACTORY_ENTRY("cooperative_groups::__v1::__static_size_thread_block_tile_base.shfl_down",
                                 Diagnostics::API_NOT_MIGRATED, ARG("shfl_down"))))

/*
  --- X.shfl_up() ---
  cg::thread_block_tile<32>::shfl_up(v, delta)
  => sycl::shift_group_right(item_ct1.get_sub_group(), v, delta)

  cg::thread_block_tile<N>::shfl_up(v, delta) if N in {1,2,4,8,16}
  => dpct::shift_sub_group_right(item_ct1.get_sub_group(), v, delta, N)
*/
// for cuda <= 10.2
CASE_FACTORY_ENTRY(
    CASE(baseHasThreadBlockTileType(32),
         CALL_FACTORY_ENTRY(
             "cooperative_groups::__v1::__thread_block_tile_base.shfl_up",
             CALL(MapNames::getClNamespace() + "shift_group_right", SUBGROUP,
                  ARG(0), ARG(1)))),
    CASE(baseHasThreadBlockTileType(16),
         FEATURE_REQUEST_FACTORY(
             HelperFeatureEnum::device_ext,
             CALL_FACTORY_ENTRY(
                 "cooperative_groups::__v1::__thread_block_tile_base.shfl_up",
                 CALL(MapNames::getDpctNamespace() + "shift_sub_group_right",
                      SUBGROUP, ARG(0), ARG(1), ARG("16"))))),
    CASE(baseHasThreadBlockTileType(8),
         FEATURE_REQUEST_FACTORY(
             HelperFeatureEnum::device_ext,
             CALL_FACTORY_ENTRY(
                 "cooperative_groups::__v1::__thread_block_tile_base.shfl_up",
                 CALL(MapNames::getDpctNamespace() + "shift_sub_group_right",
                      SUBGROUP, ARG(0), ARG(1), ARG("8"))))),
    CASE(baseHasThreadBlockTileType(4),
         FEATURE_REQUEST_FACTORY(
             HelperFeatureEnum::device_ext,
             CALL_FACTORY_ENTRY(
                 "cooperative_groups::__v1::__thread_block_tile_base.shfl_up",
                 CALL(MapNames::getDpctNamespace() + "shift_sub_group_right",
                      SUBGROUP, ARG(0), ARG(1), ARG("4"))))),
    CASE(baseHasThreadBlockTileType(2),
         FEATURE_REQUEST_FACTORY(
             HelperFeatureEnum::device_ext,
             CALL_FACTORY_ENTRY(
                 "cooperative_groups::__v1::__thread_block_tile_base.shfl_up",
                 CALL(MapNames::getDpctNamespace() + "shift_sub_group_right",
                      SUBGROUP, ARG(0), ARG(1), ARG("2"))))),
    CASE(baseHasThreadBlockTileType(1),
         FEATURE_REQUEST_FACTORY(
             HelperFeatureEnum::device_ext,
             CALL_FACTORY_ENTRY(
                 "cooperative_groups::__v1::__thread_block_tile_base.shfl_up",
                 CALL(MapNames::getDpctNamespace() + "shift_sub_group_right",
                      SUBGROUP, ARG(0), ARG(1), ARG("1"))))),
    OTHERWISE(UNSUPPORT_FACTORY_ENTRY(
        "cooperative_groups::__v1::__thread_block_tile_base.shfl_up",
        Diagnostics::API_NOT_MIGRATED, ARG("shfl_up"))))

// for cuda >= 11.0
CASE_FACTORY_ENTRY(
    CASE(baseHasThreadBlockTileType(32),
         CALL_FACTORY_ENTRY("cooperative_groups::__v1::__static_size_thread_block_tile_base.shfl_up",
                            CALL(MapNames::getClNamespace() +
                                     "shift_group_right",
                                 SUBGROUP, ARG(0), ARG(1)))),
    CASE(baseHasThreadBlockTileType(16),
         FEATURE_REQUEST_FACTORY(
             HelperFeatureEnum::device_ext,
             CALL_FACTORY_ENTRY("cooperative_groups::__v1::__static_size_thread_block_tile_base.shfl_up",
                                CALL(MapNames::getDpctNamespace() +
                                         "shift_sub_group_right",
                                     SUBGROUP, ARG(0), ARG(1), ARG("16"))))),
    CASE(baseHasThreadBlockTileType(8),
         FEATURE_REQUEST_FACTORY(
             HelperFeatureEnum::device_ext,
             CALL_FACTORY_ENTRY("cooperative_groups::__v1::__static_size_thread_block_tile_base.shfl_up",
                                CALL(MapNames::getDpctNamespace() +
                                         "shift_sub_group_right",
                                     SUBGROUP, ARG(0), ARG(1), ARG("8"))))),
    CASE(baseHasThreadBlockTileType(4),
         FEATURE_REQUEST_FACTORY(
             HelperFeatureEnum::device_ext,
             CALL_FACTORY_ENTRY("cooperative_groups::__v1::__static_size_thread_block_tile_base.shfl_up",
                                CALL(MapNames::getDpctNamespace() +
                                         "shift_sub_group_right",
                                     SUBGROUP, ARG(0), ARG(1), ARG("4"))))),
    CASE(baseHasThreadBlockTileType(2),
         FEATURE_REQUEST_FACTORY(
             HelperFeatureEnum::device_ext,
             CALL_FACTORY_ENTRY("cooperative_groups::__v1::__static_size_thread_block_tile_base.shfl_up",
                                CALL(MapNames::getDpctNamespace() +
                                         "shift_sub_group_right",
                                     SUBGROUP, ARG(0), ARG(1), ARG("2"))))),
    CASE(baseHasThreadBlockTileType(1),
         FEATURE_REQUEST_FACTORY(
             HelperFeatureEnum::device_ext,
             CALL_FACTORY_ENTRY("cooperative_groups::__v1::__static_size_thread_block_tile_base.shfl_up",
                                CALL(MapNames::getDpctNamespace() +
                                         "shift_sub_group_right",
                                     SUBGROUP, ARG(0), ARG(1), ARG("1"))))),
    OTHERWISE(UNSUPPORT_FACTORY_ENTRY("cooperative_groups::__v1::__static_size_thread_block_tile_base.shfl_up",
                                      Diagnostics::API_NOT_MIGRATED,
                                      ARG("shfl_up"))))

/*
  --- X.shfl() ---
  cg::thread_block_tile<32>::shfl(v, src_rank)
  => sycl::select_from_group(item_ct1.get_sub_group(), v, src_rank)

  cg::thread_block_tile<N>::shfl(v, src_rank) if N in {1,2,4,8,16}
  => dpct::select_from_sub_group(item_ct1.get_sub_group(), v, src_rank, N)
*/
// for cuda <= 10.2
CASE_FACTORY_ENTRY(
    CASE(baseHasThreadBlockTileType(32),
         CALL_FACTORY_ENTRY(
             "cooperative_groups::__v1::__thread_block_tile_base.shfl",
             CALL(MapNames::getClNamespace() + "select_from_group", SUBGROUP,
                  ARG(0), ARG(1)))),
    CASE(baseHasThreadBlockTileType(16),
             CALL_FACTORY_ENTRY(
                 "cooperative_groups::__v1::__thread_block_tile_base.shfl",
                 CALL(MapNames::getDpctNamespace() + "select_from_sub_group",
                      SUBGROUP, ARG(0), ARG(1), ARG("16")))),
    CASE(baseHasThreadBlockTileType(8),
             CALL_FACTORY_ENTRY(
                 "cooperative_groups::__v1::__thread_block_tile_base.shfl",
                 CALL(MapNames::getDpctNamespace() + "select_from_sub_group",
                      SUBGROUP, ARG(0), ARG(1), ARG("8")))),
    CASE(baseHasThreadBlockTileType(4),
             CALL_FACTORY_ENTRY(
                 "cooperative_groups::__v1::__thread_block_tile_base.shfl",
                 CALL(MapNames::getDpctNamespace() + "select_from_sub_group",
                      SUBGROUP, ARG(0), ARG(1), ARG("4")))),
    CASE(baseHasThreadBlockTileType(2),
             CALL_FACTORY_ENTRY(
                 "cooperative_groups::__v1::__thread_block_tile_base.shfl",
                 CALL(MapNames::getDpctNamespace() + "select_from_sub_group",
                      SUBGROUP, ARG(0), ARG(1), ARG("2")))),
    CASE(baseHasThreadBlockTileType(1),
             CALL_FACTORY_ENTRY(
                 "cooperative_groups::__v1::__thread_block_tile_base.shfl",
                 CALL(MapNames::getDpctNamespace() + "select_from_sub_group",
                      SUBGROUP, ARG(0), ARG(1), ARG("1")))),
    OTHERWISE(UNSUPPORT_FACTORY_ENTRY(
        "cooperative_groups::__v1::__thread_block_tile_base.shfl",
        Diagnostics::API_NOT_MIGRATED, ARG("shfl"))))

// for cuda >= 11.0
CASE_FACTORY_ENTRY(
    CASE(baseHasThreadBlockTileType(32),
         CALL_FACTORY_ENTRY("cooperative_groups::__v1::__static_size_thread_block_tile_base.shfl",
                            CALL(MapNames::getClNamespace() +
                                     "select_from_group",
                                 SUBGROUP, ARG(0), ARG(1)))),
    CASE(baseHasThreadBlockTileType(16),
         FEATURE_REQUEST_FACTORY(
             HelperFeatureEnum::device_ext,
             CALL_FACTORY_ENTRY("cooperative_groups::__v1::__static_size_thread_block_tile_base.shfl",
                                CALL(MapNames::getDpctNamespace() +
                                         "select_from_sub_group",
                                     SUBGROUP, ARG(0), ARG(1), ARG("16"))))),
    CASE(baseHasThreadBlockTileType(8),
         FEATURE_REQUEST_FACTORY(
             HelperFeatureEnum::device_ext,
             CALL_FACTORY_ENTRY("cooperative_groups::__v1::__static_size_thread_block_tile_base.shfl",
                                CALL(MapNames::getDpctNamespace() +
                                         "select_from_sub_group",
                                     SUBGROUP, ARG(0), ARG(1), ARG("8"))))),
    CASE(baseHasThreadBlockTileType(4),
         FEATURE_REQUEST_FACTORY(
             HelperFeatureEnum::device_ext,
             CALL_FACTORY_ENTRY("cooperative_groups::__v1::__static_size_thread_block_tile_base.shfl",
                                CALL(MapNames::getDpctNamespace() +
                                         "select_from_sub_group",
                                     SUBGROUP, ARG(0), ARG(1), ARG("4"))))),
    CASE(baseHasThreadBlockTileType(2),
         FEATURE_REQUEST_FACTORY(
             HelperFeatureEnum::device_ext,
             CALL_FACTORY_ENTRY("cooperative_groups::__v1::__static_size_thread_block_tile_base.shfl",
                                CALL(MapNames::getDpctNamespace() +
                                         "select_from_sub_group",
                                     SUBGROUP, ARG(0), ARG(1), ARG("2"))))),
    CASE(baseHasThreadBlockTileType(1),
         FEATURE_REQUEST_FACTORY(
             HelperFeatureEnum::device_ext,
             CALL_FACTORY_ENTRY("cooperative_groups::__v1::__static_size_thread_block_tile_base.shfl",
                                CALL(MapNames::getDpctNamespace() +
                                         "select_from_sub_group",
                                     SUBGROUP, ARG(0), ARG(1), ARG("1"))))),
    OTHERWISE(UNSUPPORT_FACTORY_ENTRY("cooperative_groups::__v1::__static_size_thread_block_tile_base.shfl",
                                      Diagnostics::API_NOT_MIGRATED,
                                      ARG("shfl"))))
/*
  --- X.shfl_xor() ---
  cg::thread_block_tile<32>::shfl_xor(v, mask)
  => sycl::permute_group_by_xor(item_ct1.get_sub_group(), v, mask)

  cg::thread_block_tile<N>::shfl_xor(v, mask) if N in {1,2,4,8,16}
  => dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), v, mask, N)
*/
// for cuda <= 10.2
CASE_FACTORY_ENTRY(
    CASE(baseHasThreadBlockTileType(32),
         CALL_FACTORY_ENTRY(
             "cooperative_groups::__v1::__thread_block_tile_base.shfl_xor",
             CALL(MapNames::getClNamespace() + "permute_group_by_xor", SUBGROUP,
                  ARG(0), ARG(1)))),
    CASE(baseHasThreadBlockTileType(16),
         FEATURE_REQUEST_FACTORY(
             HelperFeatureEnum::device_ext,
             CALL_FACTORY_ENTRY(
                 "cooperative_groups::__v1::__thread_block_tile_base.shfl_xor",
                 CALL(MapNames::getDpctNamespace() + "permute_sub_group_by_xor",
                      SUBGROUP, ARG(0), ARG(1), ARG("16"))))),
    CASE(baseHasThreadBlockTileType(8),
         FEATURE_REQUEST_FACTORY(
             HelperFeatureEnum::device_ext,
             CALL_FACTORY_ENTRY(
                 "cooperative_groups::__v1::__thread_block_tile_base.shfl_xor",
                 CALL(MapNames::getDpctNamespace() + "permute_sub_group_by_xor",
                      SUBGROUP, ARG(0), ARG(1), ARG("8"))))),
    CASE(baseHasThreadBlockTileType(4),
         FEATURE_REQUEST_FACTORY(
             HelperFeatureEnum::device_ext,
             CALL_FACTORY_ENTRY(
                 "cooperative_groups::__v1::__thread_block_tile_base.shfl_xor",
                 CALL(MapNames::getDpctNamespace() + "permute_sub_group_by_xor",
                      SUBGROUP, ARG(0), ARG(1), ARG("4"))))),
    CASE(baseHasThreadBlockTileType(2),
         FEATURE_REQUEST_FACTORY(
             HelperFeatureEnum::device_ext,
             CALL_FACTORY_ENTRY(
                 "cooperative_groups::__v1::__thread_block_tile_base.shfl_xor",
                 CALL(MapNames::getDpctNamespace() + "permute_sub_group_by_xor",
                      SUBGROUP, ARG(0), ARG(1), ARG("2"))))),
    CASE(baseHasThreadBlockTileType(1),
         FEATURE_REQUEST_FACTORY(
             HelperFeatureEnum::device_ext,
             CALL_FACTORY_ENTRY(
                 "cooperative_groups::__v1::__thread_block_tile_base.shfl_xor",
                 CALL(MapNames::getDpctNamespace() + "permute_sub_group_by_xor",
                      SUBGROUP, ARG(0), ARG(1), ARG("1"))))),
    OTHERWISE(UNSUPPORT_FACTORY_ENTRY(
        "cooperative_groups::__v1::__thread_block_tile_base.shfl_xor",
        Diagnostics::API_NOT_MIGRATED, ARG("shfl_xor"))))

// for cuda >= 11.0
CASE_FACTORY_ENTRY(
    CASE(baseHasThreadBlockTileType(32),
         CALL_FACTORY_ENTRY("cooperative_groups::__v1::__static_size_thread_block_tile_base.shfl_xor",
                            CALL(MapNames::getClNamespace() +
                                     "permute_group_by_xor",
                                 SUBGROUP, ARG(0), ARG(1)))),
    CASE(baseHasThreadBlockTileType(16),
         FEATURE_REQUEST_FACTORY(
             HelperFeatureEnum::device_ext,
             CALL_FACTORY_ENTRY("cooperative_groups::__v1::__static_size_thread_block_tile_base.shfl_xor",
                                CALL(MapNames::getDpctNamespace() +
                                         "permute_sub_group_by_xor",
                                     SUBGROUP, ARG(0), ARG(1), ARG("16"))))),
    CASE(baseHasThreadBlockTileType(8),
         FEATURE_REQUEST_FACTORY(
             HelperFeatureEnum::device_ext,
             CALL_FACTORY_ENTRY("cooperative_groups::__v1::__static_size_thread_block_tile_base.shfl_xor",
                                CALL(MapNames::getDpctNamespace() +
                                         "permute_sub_group_by_xor",
                                     SUBGROUP, ARG(0), ARG(1), ARG("8"))))),
    CASE(baseHasThreadBlockTileType(4),
         FEATURE_REQUEST_FACTORY(
             HelperFeatureEnum::device_ext,
             CALL_FACTORY_ENTRY("cooperative_groups::__v1::__static_size_thread_block_tile_base.shfl_xor",
                                CALL(MapNames::getDpctNamespace() +
                                         "permute_sub_group_by_xor",
                                     SUBGROUP, ARG(0), ARG(1), ARG("4"))))),
    CASE(baseHasThreadBlockTileType(2),
         FEATURE_REQUEST_FACTORY(
             HelperFeatureEnum::device_ext,
             CALL_FACTORY_ENTRY("cooperative_groups::__v1::__static_size_thread_block_tile_base.shfl_xor",
                                CALL(MapNames::getDpctNamespace() +
                                         "permute_sub_group_by_xor",
                                     SUBGROUP, ARG(0), ARG(1), ARG("2"))))),
    CASE(baseHasThreadBlockTileType(1),
         FEATURE_REQUEST_FACTORY(
             HelperFeatureEnum::device_ext,
             CALL_FACTORY_ENTRY("cooperative_groups::__v1::__static_size_thread_block_tile_base.shfl_xor",
                                CALL(MapNames::getDpctNamespace() +
                                         "permute_sub_group_by_xor",
                                     SUBGROUP, ARG(0), ARG(1), ARG("1"))))),
    OTHERWISE(UNSUPPORT_FACTORY_ENTRY("cooperative_groups::__v1::__static_size_thread_block_tile_base.shfl_xor",
                                      Diagnostics::API_NOT_MIGRATED,
                                      ARG("shfl_xor"))))
MEMBER_CALL_FACTORY_ENTRY("cooperative_groups::__v1::thread_group.thread_rank",
                          MemberExprBase(), false, "get_local_linear_id")
MEMBER_CALL_FACTORY_ENTRY("cooperative_groups::__v1::thread_group.sync",
                          MemberExprBase(), false, "barrier")
MEMBER_CALL_FACTORY_ENTRY("cooperative_groups::__v1::thread_group.size",
                          MemberExprBase(), false, "get_local_linear_range")
MEMBER_CALL_FACTORY_ENTRY("cooperative_groups::__v1::thread_group.num_threads",
                          MemberExprBase(), false, "get_local_linear_range")
MEMBER_CALL_FACTORY_ENTRY("cooperative_groups::__v1::thread_group.get_type",
                          MemberExprBase(), false, "get_type")
MEMBER_CALL_FACTORY_ENTRY("cooperative_groups::__v1::thread_block.group_index", NDITEM,
                          false, "get_global_id")
MEMBER_CALL_FACTORY_ENTRY("cooperative_groups::__v1::thread_block.thread_index", NDITEM,
                          false, "get_local_id")
#endif
