//===--------------- APINamesMisc.inc -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

CONDITIONAL_FACTORY_ENTRY(
    UseOccupancyCalculation,
    WARNING_FACTORY_ENTRY(
        "cudaOccupancyMaxActiveBlocksPerMultiprocessor",
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::device_ext,
            CALL_FACTORY_ENTRY(
                "cudaOccupancyMaxActiveBlocksPerMultiprocessor",
                CALL(MapNames::getDpctNamespace() +
                         "experimental::calculate_max_active_wg_per_xecore",
                     ARG(0), ARG(2),
                     BO(BinaryOperatorKind::BO_Add, ARG(3),
                        ARG("dpct_placeholder /* total share local memory "
                            "size */"))))),
        Diagnostics::MAX_ACTIVE_WG_ARGS,
        ARG(MapNames::getDpctNamespace() +
            "experimental::calculate_max_active_wg_per_xecore"),
        ARG(1)),
    ENTRY_UNSUPPORTED("cudaOccupancyMaxActiveBlocksPerMultiprocessor",
                      Diagnostics::API_NOT_MIGRATED))

CONDITIONAL_FACTORY_ENTRY(
    UseOccupancyCalculation,
    WARNING_FACTORY_ENTRY(
        "cuOccupancyMaxActiveBlocksPerMultiprocessor",
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::device_ext,
            CALL_FACTORY_ENTRY(
                "cuOccupancyMaxActiveBlocksPerMultiprocessor",
                CALL(MapNames::getDpctNamespace() +
                         "experimental::calculate_max_active_wg_per_xecore",
                     ARG(0), ARG(2),
                     BO(BinaryOperatorKind::BO_Add, ARG(3),
                        ARG("dpct_placeholder /* total share local memory "
                            "size */"))))),
        Diagnostics::MAX_ACTIVE_WG_ARGS,
        ARG(MapNames::getDpctNamespace() +
            "experimental::calculate_max_active_wg_per_xecore"),
        ARG(1)),
    ENTRY_UNSUPPORTED("cuOccupancyMaxActiveBlocksPerMultiprocessor",
                      Diagnostics::API_NOT_MIGRATED))

CONDITIONAL_FACTORY_ENTRY(
    UseOccupancyCalculation,
    WARNING_FACTORY_ENTRY(
        "cudaOccupancyMaxPotentialBlockSize",
        CONDITIONAL_FACTORY_ENTRY(
            CheckArgCount(5, std::equal_to<>(), false),
            CALL_FACTORY_ENTRY(
                "cudaOccupancyMaxPotentialBlockSize",
                CALL(MapNames::getDpctNamespace() +
                         "experimental::calculate_max_potential_wg",
                     ARG(0), ARG(1), ARG(4),
                     BO(BinaryOperatorKind::BO_Add, ARG(3),
                        ARG("dpct_placeholder /* total share local memory "
                            "size */")))),
            CONDITIONAL_FACTORY_ENTRY(
                CheckArgCount(4, std::equal_to<>(), false),
                CALL_FACTORY_ENTRY(
                    "cudaOccupancyMaxPotentialBlockSize",
                    CALL(MapNames::getDpctNamespace() +
                             "experimental::calculate_max_potential_wg",
                         ARG(0), ARG(1), ARG("0"),
                         BO(BinaryOperatorKind::BO_Add, ARG(3),
                            ARG("dpct_placeholder /* total share local memory "
                                "size */")))),
                CALL_FACTORY_ENTRY(
                    "cudaOccupancyMaxPotentialBlockSize",
                    CALL(MapNames::getDpctNamespace() +
                             "experimental::calculate_max_potential_wg",
                         ARG(0), ARG(1), ARG("0"),
                         ARG("dpct_placeholder /* total share local memory "
                             "size */"))))),
        Diagnostics::MAX_ACTIVE_WG_ARGS,
        ARG(MapNames::getDpctNamespace() +
            "experimental::calculate_max_potential_wg"),
        ARG(2)),
    ENTRY_UNSUPPORTED("cudaOccupancyMaxPotentialBlockSize",
                      Diagnostics::API_NOT_MIGRATED))
CALL_FACTORY_ENTRY("__assertfail", CALL("assert", LITERAL("0")))
CALL_FACTORY_ENTRY("__assert_fail", CALL("assert", LITERAL("0")))

CONDITIONAL_FACTORY_ENTRY(
    checkIsUseNoQueueDevice(),
    ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
        HelperFeatureEnum::device_ext,
        CALL_FACTORY_ENTRY(
            "cudaGetDeviceProperties",
            CALL(MapNames::getDpctNamespace() + "get_device_info", DEREF(0),
                 ARG(DpctGlobalInfo::getGlobalDeviceName()))))),
    ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
        HelperFeatureEnum::device_ext,
        CALL_FACTORY_ENTRY(
            "cudaGetDeviceProperties",
            CALL(MapNames::getDpctNamespace() + "get_device_info", DEREF(0),
                 CALL(MapNames::getDpctNamespace() +
                          "dev_mgr::instance().get_device",
                      ARG(1)))))))

CONDITIONAL_FACTORY_ENTRY(
    checkIsUseNoQueueDevice(),
    ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
        HelperFeatureEnum::device_ext,
        CALL_FACTORY_ENTRY(
            "cudaGetDeviceProperties_v2",
            CALL(MapNames::getDpctNamespace() + "get_device_info", DEREF(0),
                 ARG(DpctGlobalInfo::getGlobalDeviceName()))))),
    ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
        HelperFeatureEnum::device_ext,
        CALL_FACTORY_ENTRY(
            "cudaGetDeviceProperties_v2",
            CALL(MapNames::getDpctNamespace() + "get_device_info", DEREF(0),
                 CALL(MapNames::getDpctNamespace() +
                          "dev_mgr::instance().get_device",
                      ARG(1)))))))

REMOVE_API_FACTORY_ENTRY_WITH_MSG("cudaProfilerStart",
                         getRemovedAPIWarningMessage("cudaProfilerStart"))
REMOVE_API_FACTORY_ENTRY_WITH_MSG("cudaProfilerStop",
                         getRemovedAPIWarningMessage("cudaProfilerStop"))
