//===--------------- APINamesDriver.inc -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// clang-format off

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::Kernel_kernel_library,
    ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY(
        "cuModuleLoad", DEREF(0),
        CALL(MapNames::getDpctNamespace() + "load_kernel_library", ARG(1)))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::Kernel_kernel_library,
    ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY("cuModuleLoadData", DEREF(0),
                                            CALL(MapNames::getDpctNamespace() +
                                                     "load_kernel_library_mem",
                                                 ARG(1)))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::Kernel_kernel_library,
    ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY("cuModuleLoadDataEx", DEREF(0),
                                            CALL(MapNames::getDpctNamespace() +
                                                     "load_kernel_library_mem",
                                                 ARG(1)))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::Kernel_kernel_library,
    ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY("cuModuleGetFunction", DEREF(0),
                                            CALL(MapNames::getDpctNamespace() +
                                                     "get_kernel_function",
                                                 ARG(1), ARG(2)))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::Kernel_get_image_wrapper,
    ASSIGNABLE_FACTORY(
        ASSIGN_FACTORY_ENTRY("cuModuleGetTexRef", DEREF(0),
                             CALL(MapNames::getDpctNamespace() +
                                      "get_image_wrapper",
                                  ARG(1), ARG(2)))))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Kernel_kernel_library, ASSIGNABLE_FACTORY(
    CALL_FACTORY_ENTRY("cuModuleUnload",
                       CALL(MapNames::getDpctNamespace() + "unload_kernel_library", ARG(0)))))

#define RANGES_CTOR(RangeType, ...)                                            \
  CALL(DpctGlobalInfo::getCtadClass(                                           \
           buildString(MapNames::getClNamespace(), RangeType), 3),             \
       __VA_ARGS__)
#define NDRANGE_CTOR(Arg0, Arg1)                                               \
  RANGES_CTOR("nd_range", BO(BinaryOperatorKind::BO_Mul, Arg0, Arg1), Arg1)
#define RANGE_CTOR(Arg0, Arg1, Arg2) RANGES_CTOR("range", Arg0, Arg1, Arg2)

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Kernel_kernel_library, ASSIGNABLE_FACTORY(
    CALL_FACTORY_ENTRY(
        "cuLaunchKernel",
        CALL(MapNames::getDpctNamespace() + "invoke_kernel_function", ARG(0), STREAM(8),
             RANGE_CTOR(ARG(3), ARG(2), ARG(1)),
             RANGE_CTOR(ARG(6), ARG(5), ARG(4)), ARG(7), ARG(9), ARG(10)))))

#undef NDRANGE_CTOR
#undef RANGE_CTOR
#undef RANGES_CTOR
FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::Kernel_get_kernel_function_info,
    ASSIGNABLE_FACTORY(
        ASSIGN_FACTORY_ENTRY("cuFuncGetAttribute", DEREF(0),
                            makeMemberExprCreator(
                                CALL(MapNames::getDpctNamespace() +
                                    "get_kernel_function_info",
                                    ARG(2)),
                                false,
                                ARG(1)))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::Device_select_device,
    ASSIGNABLE_FACTORY(
        ASSIGN_FACTORY_ENTRY(
            "cuCtxCreate_v2",
            DEREF(0),
            CALL(MapNames::getDpctNamespace() + "select_device", ARG(2)))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::Device_select_device,
    ASSIGNABLE_FACTORY(
        ASSIGN_FACTORY_ENTRY(
            "cuDevicePrimaryCtxRetain",
            DEREF(0),
            CALL(MapNames::getDpctNamespace() + "select_device", ARG(1)))))

ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY("cudaHostGetFlags", DEREF(0), ARG("0")))

ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY("cuMemHostGetFlags", DEREF(0),
                                        ARG("0")))

CONDITIONAL_FACTORY_ENTRY(
    checkIsCallExprOnly(),
    WARNING_FACTORY_ENTRY("cuMemHostRegister_v2",
        TOSTRING_FACTORY_ENTRY("cuMemHostRegister_v2", LITERAL("")),
         Diagnostics::FUNC_CALL_REMOVED,
        std::string("cuMemHostRegister"), getRemovedAPIWarningMessage("cuMemHostRegister_v2")),
    WARNING_FACTORY_ENTRY("cuMemHostRegister_v2",
        TOSTRING_FACTORY_ENTRY("cuMemHostRegister_v2", LITERAL("0")),
        Diagnostics::FUNC_CALL_REMOVED_0,
        std::string("cuMemHostRegister"), getRemovedAPIWarningMessage("cuMemHostRegister_v2")))


CONDITIONAL_FACTORY_ENTRY(
    checkIsCallExprOnly(),
    WARNING_FACTORY_ENTRY("cuMemHostUnregister",
        TOSTRING_FACTORY_ENTRY("cuMemHostUnregister", LITERAL("")),
         Diagnostics::FUNC_CALL_REMOVED,
        std::string("cuMemHostUnregister"), getRemovedAPIWarningMessage("cuMemHostUnregister")),
    WARNING_FACTORY_ENTRY("cuMemHostUnregister",
        TOSTRING_FACTORY_ENTRY("cuMemHostUnregister", LITERAL("0")),
        Diagnostics::FUNC_CALL_REMOVED_0,
        std::string("cuMemHostUnregister"), getRemovedAPIWarningMessage("cuMemHostUnregister")))

WARNING_FACTORY_ENTRY("cuDriverGetVersion",
    FEATURE_REQUEST_FACTORY(
        HelperFeatureEnum::Device_get_current_device,
        ASSIGNABLE_FACTORY(
            ASSIGN_FACTORY_ENTRY("cuDriverGetVersion",
                DEREF(0), CALL("std::stoi", CALL(MapNames::getDpctNamespace() +
                    "get_current_device().get_info<" + MapNames::getClNamespace() +
                    "info::device::version>"))))),
    Diagnostics::TYPE_MISMATCH)

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::Device_get_current_device_id,
    ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY("cuCtxGetDevice", DEREF(0),
                                            CALL(MapNames::getDpctNamespace() +
                                                 "get_current_device_id"))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::Util_get_sycl_language_version,
    ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY("cuCtxGetApiVersion", DEREF(1),
                                            CALL(MapNames::getDpctNamespace() +
                                                 "get_sycl_language_version"))))

ASSIGNABLE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
    checkIsGetWorkGroupDim(1),
    ASSIGN_FACTORY_ENTRY(
        "cuDeviceGetAttribute", DEREF(0),
        MEMBER_CALL(
            MEMBER_CALL(
                MEMBER_CALL(MEMBER_CALL(CALL(MapNames::getDpctNamespace() +
                                             "dev_mgr::instance"),
                                        false, "get_device", ARG(2)),
                            false, "get_device_info"),
                false, makeFuncNameFromDevAttrCreator(1)),
            false, "get", getWorkGroupDim(1))),
    ASSIGN_FACTORY_ENTRY(
        "cuDeviceGetAttribute", DEREF(0),
        MEMBER_CALL(MEMBER_CALL(CALL(MapNames::getDpctNamespace() +
                                     "dev_mgr::instance"),
                                false, "get_device", ARG(2)),
                    false, makeFuncNameFromDevAttrCreator(1)))))

// clang-format on
