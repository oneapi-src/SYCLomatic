//===------------------------ APINamesNvshmem.inc -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Library Setup, Exit & Query
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext,
                        CALL_FACTORY_ENTRY("nvshmem_init", CALL("ishmem_init")))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext,
                        CALL_FACTORY_ENTRY("nvshmemx_init_attr",
                                           CALL("ishmemx_init_attr", ARG(1))))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext,
                        CALL_FACTORY_ENTRY("nvshmem_my_pe",
                                           CALL("ishmem_my_pe")))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext,
                        CALL_FACTORY_ENTRY("nvshmem_n_pes",
                                           CALL("ishmem_n_pes")))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext,
                        CALL_FACTORY_ENTRY("nvshmem_finalize",
                                           CALL("ishmem_finalize")))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext,
                        CALL_FACTORY_ENTRY("nvshmem_ptr",
                                           CALL("ishmem_ptr", ARG(0), ARG(1))))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext,
                        CALL_FACTORY_ENTRY("nvshmem_info_get_version",
                                           CALL("ishmem_info_get_version",
                                                ARG(0), ARG(1))))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext,
                        CALL_FACTORY_ENTRY("nvshmem_info_get_name",
                                           CALL("ishmem_info_get_name",
                                                ARG(0))))

// Memory Management
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext,
                        CALL_FACTORY_ENTRY("nvshmem_malloc",
                                           CALL("ishmem_malloc", ARG(0))))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext,
                        CALL_FACTORY_ENTRY("nvshmem_free",
                                           CALL("ishmem_free", ARG(0))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("nvshmem_align", CALL("ishmem_align", ARG(0), ARG(1))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("nvshmem_calloc", CALL("ishmem_calloc", ARG(0), ARG(1))))

// Team Management
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext,
                        CALL_FACTORY_ENTRY("nvshmem_team_my_pe",
                                           CALL("ishmem_team_my_pe", ARG(0))))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext,
                        CALL_FACTORY_ENTRY("nvshmem_team_n_pes",
                                           CALL("ishmem_team_n_pes", ARG(0))))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext,
                        CALL_FACTORY_ENTRY("nvshmem_team_get_config",
                                           CALL("ishmem_team_get_config",
                                                ARG(0), LITERAL("0"), ARG(1))))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext,
                        CALL_FACTORY_ENTRY("nvshmem_team_translate_pe",
                                           CALL("ishmem_team_translate_pe",
                                                ARG(0), ARG(1), ARG(2))))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext,
                        CALL_FACTORY_ENTRY("nvshmem_team_split_strided",
                                           CALL("ishmem_team_split_strided",
                                                ARG(0), ARG(1), ARG(2), ARG(3),
                                                ARG(4), ARG(5), ARG(6))))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext,
                        CALL_FACTORY_ENTRY("nvshmem_team_split_2d",
                                           CALL("ishmem_team_split_2d", ARG(0),
                                                ARG(1), ARG(2), ARG(3), ARG(4),
                                                ARG(5), ARG(6), ARG(7))))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext,
                        CALL_FACTORY_ENTRY("nvshmem_team_destroy",
                                           CALL("ishmem_team_destroy", ARG(0))))
