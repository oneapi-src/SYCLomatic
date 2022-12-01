//===--------------- APINamesNccl.inc -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::CclUtils_get_version,
    ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY("ncclGetVersion", DEREF(0),
                                            CALL(MapNames::getDpctNamespace() +
                                                 "ccl::get_version"))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::CclUtils_create_kvs_address,
    ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY("ncclGetUniqueId", DEREF(0),
                                            CALL(MapNames::getDpctNamespace() +
                                                 "ccl::create_kvs_address"))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::CclUtils_create_kvs,
    ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY(
        "ncclCommInitRank", DEREF(0),
        NEW("oneapi::ccl::communicator",
            CALL("oneapi::ccl::create_communicator", ARG(1), ARG(3),
                 CALL(MapNames::getDpctNamespace() + "ccl::create_kvs",
                      ARG(2)))))))

ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY("ncclCommCount", DEREF(1),
                                        MEMBER_CALL(ARG(0), true,
                                                    LITERAL("size"))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::Device_dev_mgr_2,
    FEATURE_REQUEST_FACTORY(
        HelperFeatureEnum::Device_dev_mgr_get_device_id,
        ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY(
            "ncclCommCuDevice", DEREF(1),
            MEMBER_CALL(
                CALL(MapNames::getDpctNamespace() + "dev_mgr::instance"), false,
                "get_device_id",
                MEMBER_CALL(MEMBER_CALL(ARG(0), true, LITERAL("get_device")),
                            false, LITERAL("get_native")))))))

CONDITIONAL_FACTORY_ENTRY(
    CheckArgStr(4, "ncclAvg"),
    UNSUPPORT_FACTORY_ENTRY("ncclAllReduce", Diagnostics::UNSUPPORTED_PARAM,
                            ARG(4)),
    CALL_FACTORY_ENTRY("ncclAllReduce",
                       CALL("oneapi::ccl::allreduce", ARG(0), ARG(1), ARG(2),
                            ARG(3), ARG(4), DEREF(5),
                            CALL("oneapi::ccl::create_stream", STREAM(6)))))
