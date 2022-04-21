//===---APINamesDriver.inc--------------------------------------------===//
//
// Copyright (C) Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

#ifdef _WIN32

ASSIGN_FACTORY_ENTRY(
    "cuModuleLoad", DEREF(0),
    CALL("LoadLibraryA",
         ARG("dpct_placeholder/*Fix the module file name manually*/")))

ASSIGN_FACTORY_ENTRY(
    "cuModuleLoadData", DEREF(0),
    CALL("LoadLibraryA",
         ARG("dpct_placeholder/*Fix the module file name manually*/")))

ASSIGN_FACTORY_ENTRY("cuModuleGetFunction", DEREF(0),
                     CALL("(dpct::kernel_functor)GetProcAddress", ARG(1),
                          EXTENDSTR(2, "_wrapper")))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Image_image_wrapper_base_p_alias,
                        ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY(
                            "cuModuleGetTexRef", DEREF(0),
                            CALL("(dpct::image_wrapper_base_p)GetProcAddress",
                                 ARG(1), ARG(2)))))

CALL_FACTORY_ENTRY("cuModuleUnload", CALL("FreeLibrary", ARG(0)))

#else

ASSIGN_FACTORY_ENTRY(
    "cuModuleLoad", DEREF(0),
    CALL("dlopen", ARG("dpct_placeholder/*Fix the module file name manually*/"),
         ARG("RTLD_LAZY")))

ASSIGN_FACTORY_ENTRY(
    "cuModuleLoadData", DEREF(0),
    CALL("dlopen", ARG("dpct_placeholder/*Fix the module file name manually*/"),
         ARG("RTLD_LAZY")))

ASSIGN_FACTORY_ENTRY("cuModuleGetFunction", DEREF(0),
                     CALL("(dpct::kernel_functor)dlsym", ARG(1),
                          EXTENDSTR(2, "_wrapper")))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Image_image_wrapper_base_p_alias,
                        ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY(
                            "cuModuleGetTexRef", DEREF(0),
                            CALL("(dpct::image_wrapper_base_p)dlsym", ARG(1),
                                 ARG(2)))))

CALL_FACTORY_ENTRY("cuModuleUnload", CALL("dlclose", ARG(0)))

#endif

#define RANGES_CTOR(RangeType, ...)                                            \
  CALL(DpctGlobalInfo::getCtadClass(                                           \
           buildString(MapNames::getClNamespace(), RangeType), 3),             \
       __VA_ARGS__)
#define NDRANGE_CTOR(Arg0, Arg1)                                               \
  RANGES_CTOR("nd_range", BO(BinaryOperatorKind::BO_Mul, Arg0, Arg1), Arg1)
#define RANGE_CTOR(Arg0, Arg1, Arg2) RANGES_CTOR("range", Arg0, Arg1, Arg2)

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cuLaunchKernel", CALL(ARG(0), STREAM(8),
                           NDRANGE_CTOR(RANGE_CTOR(ARG(3), ARG(2), ARG(1)),
                                        RANGE_CTOR(ARG(6), ARG(5), ARG(4))),
                           ARG(7), ARG(9), ARG(10))))

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
                                    CAST(ARG("const void *"), ARG(2))),
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
