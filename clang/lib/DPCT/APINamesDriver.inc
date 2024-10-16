//===--------------- APINamesDriver.inc -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// clang-format off

ASSIGNABLE_FACTORY(
    ASSIGN_FACTORY_ENTRY("cuGetErrorString", DEREF(1),
                         CALL(MapNames::getDpctNamespace() + "get_error_string_dummy", ARG_WC(0))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY(
        "cuModuleLoad", DEREF(0),
        CALL(MapNames::getDpctNamespace() + "load_kernel_library", ARG(1)))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY("cuModuleLoadData", DEREF(0),
                                            CALL(MapNames::getDpctNamespace() +
                                                     "load_kernel_library_mem",
                                                 ARG(1)))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY("cuModuleLoadDataEx", DEREF(0),
                                            CALL(MapNames::getDpctNamespace() +
                                                     "load_kernel_library_mem",
                                                 ARG(1)))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY("cuModuleGetFunction", DEREF(0),
                                            CALL(MapNames::getDpctNamespace() +
                                                     "get_kernel_function",
                                                 ARG(1), ARG(2)))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    ASSIGNABLE_FACTORY(
        ASSIGN_FACTORY_ENTRY("cuModuleGetTexRef", DEREF(0),
                             CALL(MapNames::getDpctNamespace() +
                                      "get_image_wrapper",
                                  ARG(1), ARG(2)))))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext, ASSIGNABLE_FACTORY(
    CALL_FACTORY_ENTRY("cuModuleUnload",
                       CALL(MapNames::getDpctNamespace() + "unload_kernel_library", ARG(0)))))

#define RANGES_CTOR(RangeType, ...)                                            \
  CALL(DpctGlobalInfo::getCtadClass(                                           \
           buildString(MapNames::getClNamespace(), RangeType), 3),             \
       __VA_ARGS__)
#define NDRANGE_CTOR(Arg0, Arg1)                                               \
  RANGES_CTOR("nd_range", BO(BinaryOperatorKind::BO_Mul, Arg0, Arg1), Arg1)
#define RANGE_CTOR(Arg0, Arg1, Arg2) RANGES_CTOR("range", Arg0, Arg1, Arg2)

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext, ASSIGNABLE_FACTORY(
    CALL_FACTORY_ENTRY(
        "cuLaunchKernel",
        CALL(MapNames::getDpctNamespace() + "invoke_kernel_function", ARG(0), STREAM(8),
             RANGE_CTOR(ARG(3), ARG(2), ARG(1)),
             RANGE_CTOR(ARG(6), ARG(5), ARG(4)), ARG(7), ARG(9), ARG(10)))))

#undef NDRANGE_CTOR
#undef RANGE_CTOR
#undef RANGES_CTOR
FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    ASSIGNABLE_FACTORY(
        ASSIGN_FACTORY_ENTRY("cuFuncGetAttribute", DEREF(0),
                            makeMemberExprCreator(
                                CALL(MapNames::getDpctNamespace() +
                                    "get_kernel_function_info",
                                    ARG(2)),
                                false,
                                ARG(1)))))

// clang-format on
CONTEXT_ENTRY(ASSIGN_FACTORY_ENTRY("cuCtxCreate_v2", DEREF(0), ARG(2)),
              ASSIGN_FACTORY_ENTRY("cuCtxCreate_v2", DEREF(0),
                                   CALL(MapNames::getDpctNamespace() +
                                            "push_device_for_curr_thread",
                                        ARG(2))))

CONTEXT_ENTRY(ASSIGN_FACTORY_ENTRY("cuCtxCreate_v3", DEREF(0), ARG(4)),
              ASSIGN_FACTORY_ENTRY("cuCtxCreate_v3", DEREF(0),
                                   CALL(MapNames::getDpctNamespace() +
                                            "push_device_for_curr_thread",
                                        ARG(4))))

CONTEXT_ENTRY(ASSIGN_FACTORY_ENTRY("cuCtxCreate_v4", DEREF(0), ARG(3)),
              ASSIGN_FACTORY_ENTRY("cuCtxCreate_v4", DEREF(0),
                                   CALL(MapNames::getDpctNamespace() +
                                            "push_device_for_curr_thread",
                                        ARG(3))))

CONDITIONAL_FACTORY_ENTRY(
    checkIsUseNoQueueDevice(),
    WARNING_FACTORY_ENTRY(
        "cuDevicePrimaryCtxRetain",
        ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY("cuDevicePrimaryCtxRetain",
                                                DEREF(0), ARG(1))),
        Diagnostics::FUNC_CALL_REMOVED, std::string("cuDevicePrimaryCtxRetain"),
        std::string("it is redundant if it is migrated with option "
                    "--helper-function-preference=no-queue-device which declares a global SYCL device "
                    "and queue.")),
    FEATURE_REQUEST_FACTORY(
        HelperFeatureEnum::device_ext,
        ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY(
            "cuDevicePrimaryCtxRetain", DEREF(0),
            CALL(MapNames::getDpctNamespace() + "select_device", ARG(1))))))

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
        HelperFeatureEnum::device_ext,
        ASSIGNABLE_FACTORY(
            ASSIGN_FACTORY_ENTRY("cuDriverGetVersion",
                DEREF(0), CALL("std::stoi", CALL(MapNames::getDpctNamespace() +
                    "get_current_device().get_info<" + MapNames::getClNamespace() +
                    "info::device::version>"))))),
    Diagnostics::TYPE_MISMATCH)

CONDITIONAL_FACTORY_ENTRY(
    clang::dpct::checkIsUseNoQueueDevice(),
    WARNING_FACTORY_ENTRY(
        "cuCtxGetDevice",
        ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY("cuCtxGetDevice", DEREF(0),
                                                LITERAL("0"))),
        Diagnostics::FUNC_CALL_REMOVED, std::string("cuCtxGetDevice"),
        std::string("it is redundant if it is migrated with option "
                    "--helper-function-preference=no-queue-device which declares a global SYCL device "
                    "and queue.")),
    FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext,
                            ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY(
                                "cuCtxGetDevice", DEREF(0),
                                CALL(MapNames::getDpctNamespace() +
                                     "get_current_device_id")))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY("cuCtxGetApiVersion", DEREF(1),
                                            CALL(MapNames::getDpctNamespace() +
                                                 "get_sycl_language_version"))))

CONTEXT_ENTRY(ASSIGN_FACTORY_ENTRY("cuCtxPushCurrent_v2", DEREF(0),
                                   LITERAL("0")),
              CALL_FACTORY_ENTRY("cuCtxPushCurrent_v2",
                                 CALL(MapNames::getDpctNamespace() +
                                          "push_device_for_curr_thread",
                                      ARG(0))))

CONTEXT_ENTRY(ASSIGN_FACTORY_ENTRY("cuCtxPopCurrent_v2", DEREF(0),
                                   LITERAL("0")),
              ASSIGN_FACTORY_ENTRY("cuCtxPopCurrent_v2", DEREF(0),
                                   CALL(MapNames::getDpctNamespace() +
                                        "pop_device_for_curr_thread")))

ASSIGNABLE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
    checkIsGetWorkGroupDim(1),
    ASSIGN_FACTORY_ENTRY(
        "cuDeviceGetAttribute", DEREF(0),
        MEMBER_CALL(
            MEMBER_CALL(
                MEMBER_CALL(CALL(MapNames::getDpctNamespace() +
                                     "get_device", ARG(2)),
                            false, "get_device_info"),
                false, makeFuncNameFromDevAttrCreator(1)),
            false, "get", getWorkGroupDim(1))),
    CONDITIONAL_FACTORY_ENTRY(
    checkIsMigratedToHasAspect(1),
    ASSIGN_FACTORY_ENTRY(
        "cuDeviceGetAttribute", DEREF(0),
        MEMBER_EXPR(CALL(MapNames::getDpctNamespace() +
                                     "get_device", ARG(2)),
                    false, makeFuncNameFromDevAttrCreator(1))),
    CONDITIONAL_FACTORY_ENTRY(
        makeCheckAnd(UseSYCLCompat, isTextureAlignment(1)),
        ASSIGN_FACTORY_ENTRY(
            "cuDeviceGetAttribute", DEREF(0),
            BO(BinaryOperatorKind::BO_Div,
                MEMBER_CALL(CALL(MapNames::getDpctNamespace() +
                                             "get_device", ARG(2)),
                            false, "get_mem_base_addr_align"),
                ARG("8"))),
        ASSIGN_FACTORY_ENTRY(
            "cuDeviceGetAttribute", DEREF(0),
            MEMBER_CALL(CALL(MapNames::getDpctNamespace() +
                                         "get_device", ARG(2)),
                        false, makeFuncNameFromDevAttrCreator(1)))
    )
)))

// clang-format on
