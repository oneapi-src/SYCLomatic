//===--------------- APINamesNccl.inc -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

SET_NEED_DPCT_DEVICE_EXT_FACTORY(
    ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY("ncclGetVersion", DEREF(0),
                                            CALL(MapNames::getDpctNamespace() +
                                                 "ccl::get_version"))))

SET_NEED_DPCT_DEVICE_EXT_FACTORY(
    ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY("ncclGetUniqueId", DEREF(0),
                                            CALL(MapNames::getDpctNamespace() +
                                                 "ccl::create_kvs_address"))))

SET_NEED_DPCT_DEVICE_EXT_FACTORY(
    ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY("ncclCommInitRank", DEREF(0),
                                            NEW(MapNames::getDpctNamespace() +
                                                    "ccl::communicator_wrapper",
                                                ARG(1), ARG(3), ARG(2)))))

SET_NEED_DPCT_DEVICE_EXT_FACTORY(
                        ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY(
                            "ncclCommCount", DEREF(1),
                            MEMBER_CALL(ARG(0), true, LITERAL("size")))))

SET_NEED_DPCT_DEVICE_EXT_FACTORY(
    ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY(
        "ncclCommCuDevice", DEREF(1),
        CALL(MapNames::getDpctNamespace() + "get_device_id",
             MEMBER_CALL(ARG(0), true, LITERAL("get_device"))))))

SET_NEED_DPCT_DEVICE_EXT_FACTORY(
    CONDITIONAL_FACTORY_ENTRY(
        CheckEnumArgStr(4, "ncclAvg"),
        UNSUPPORT_FACTORY_ENTRY("ncclAllReduce", Diagnostics::UNSUPPORTED_PARAM,
                                ARG(4)),
        MEMBER_CALL_FACTORY_ENTRY("ncclAllReduce", ARG(5), true, "allreduce",
                                  ARG(0), ARG(1), ARG(2), ARG(3), ARG(4),
                                  ARG(6))))

SET_NEED_DPCT_DEVICE_EXT_FACTORY(
                        ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY(
                            "ncclCommUserRank", DEREF(1),
                            MEMBER_CALL(ARG(0), true, LITERAL("rank")))))

SET_NEED_DPCT_DEVICE_EXT_FACTORY(
                        MEMBER_CALL_FACTORY_ENTRY("ncclBroadcast", ARG(5), true,
                                                  "broadcast", ARG(0),ARG(1), ARG(2),
                                                  ARG(3), ARG(4), ARG(6)))

ASSIGNABLE_FACTORY(DELETER_FACTORY_ENTRY("ncclCommDestroy", ARG(0)))

SET_NEED_DPCT_DEVICE_EXT_FACTORY(
    CONDITIONAL_FACTORY_ENTRY(
        CheckEnumArgStr(4, "ncclAvg"),
        UNSUPPORT_FACTORY_ENTRY("ncclReduce", Diagnostics::UNSUPPORTED_PARAM,
                                ARG(4)),
        MEMBER_CALL_FACTORY_ENTRY("ncclReduce", ARG(6), true, "reduce", ARG(0),
                                  ARG(1), ARG(2), ARG(3), ARG(4), ARG(5),
                                  ARG(7))))

SET_NEED_DPCT_DEVICE_EXT_FACTORY(
    CONDITIONAL_FACTORY_ENTRY(
        CheckEnumArgStr(4, "ncclAvg"),
        UNSUPPORT_FACTORY_ENTRY("ncclReduceScatter",
                                Diagnostics::UNSUPPORTED_PARAM, ARG(4)),
        MEMBER_CALL_FACTORY_ENTRY("ncclReduceScatter", ARG(5), true,
                                  "reduce_scatter", ARG(0), ARG(1), ARG(2),
                                  ARG(3), ARG(4), ARG(6))))
