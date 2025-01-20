//===--------------- APINamesMemory.inc -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

ASSIGNABLE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
    checkIsUSM(),
    CONDITIONAL_FACTORY_ENTRY(
        CheckCanUseTemplateMalloc(0, 1),
        ASSIGN_FACTORY_ENTRY("cudaMalloc", DEREF(makeDerefArgCreatorWithCall(0)),
                            CALL(TEMPLATED_CALLEE_WITH_ARGS(
                                MapNames::getClNamespace() + "malloc_device",
                                getDoubleDerefedType(0)),
                                getSizeForMalloc(0, 1), QUEUESTR)),
        ASSIGN_FACTORY_ENTRY("cudaMalloc", DEREF(makeDerefArgCreatorWithCall(0)),
                            CAST(getDerefedType(0),
                                CALL(MapNames::getClNamespace() + "malloc_device",
                                getSizeForMalloc(0, 1), QUEUESTR)
                             ))),
    CONDITIONAL_FACTORY_ENTRY(
        makeCheckOr(CheckDerefedTypeBeforeCast(0, "NULL TYPE"), CheckDerefedTypeBeforeCast(0, "void *")),
        ASSIGN_FACTORY_ENTRY("cudaMalloc", DEREF(makeDerefArgCreatorWithCall(0)),
                             CALL(MapNames::getDpctNamespace() + "dpct_malloc",
                                  makeCallArgCreatorWithCall(1))),
        ASSIGN_FACTORY_ENTRY("cudaMalloc", DEREF(makeDerefArgCreatorWithCall(0)),
                             CAST(getDerefedType(0),
                                  CALL(MapNames::getDpctNamespace() + "dpct_malloc",
                                       makeCallArgCreatorWithCall(1)))))))

ASSIGNABLE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
    checkIsUSM(),
    CONDITIONAL_FACTORY_ENTRY(
        CheckCanUseTemplateMalloc(0, 1),
        ASSIGN_FACTORY_ENTRY("cuMemAlloc_v2", DEREF(makeDerefArgCreatorWithCall(0)),
                            CALL(TEMPLATED_CALLEE_WITH_ARGS(
                                MapNames::getClNamespace() + "malloc_device",
                                getDoubleDerefedType(0)),
                                getSizeForMalloc(0, 1), QUEUESTR)),
        ASSIGN_FACTORY_ENTRY("cuMemAlloc_v2", DEREF(makeDerefArgCreatorWithCall(0)),
                            CAST(getDerefedType(0),
                                CALL(MapNames::getClNamespace() + "malloc_device",
                                getSizeForMalloc(0, 1), QUEUESTR)
                             ))),
    CONDITIONAL_FACTORY_ENTRY(
        makeCheckOr(CheckDerefedTypeBeforeCast(0, "NULL TYPE"),
                    CheckDerefedTypeBeforeCast(0, "void *")),
        ASSIGN_FACTORY_ENTRY("cuMemAlloc_v2", DEREF(makeDerefArgCreatorWithCall(0)),
                             CALL(MapNames::getDpctNamespace() + "dpct_malloc",
                                  makeCallArgCreatorWithCall(1))),
        ASSIGN_FACTORY_ENTRY("cuMemAlloc_v2", DEREF(makeDerefArgCreatorWithCall(0)),
                             CAST(getDerefedType(0),
                                  CALL(MapNames::getDpctNamespace() + "dpct_malloc",
                                       makeCallArgCreatorWithCall(1)))))))

ASSIGNABLE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
    checkIsUSM(),
    CONDITIONAL_FACTORY_ENTRY(
        CheckCanUseTemplateMalloc(0, 1),
        ASSIGN_FACTORY_ENTRY(
            "cudaHostAlloc", DEREF(makeDerefArgCreatorWithCall(0)),
            CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getClNamespace() +
                                                "malloc_host",
                                            getDoubleDerefedType(0)),
                 getSizeForMalloc(0, 1), QUEUESTR)),
        ASSIGN_FACTORY_ENTRY(
            "cudaHostAlloc", DEREF(makeDerefArgCreatorWithCall(0)),
            CAST(getDerefedType(0),
                 CALL(MapNames::getClNamespace() + "malloc_host",
                      getSizeForMalloc(0, 1), QUEUESTR)))),
    ASSIGN_FACTORY_ENTRY("cudaHostAlloc", DEREF(makeDerefArgCreatorWithCall(0)),
                         CAST(getDerefedType(0), CALL("malloc", ARG(1))))))

ASSIGNABLE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
    checkIsUSM(),
    CONDITIONAL_FACTORY_ENTRY(
        CheckCanUseCLibraryMallocOrFree(0, false),
        ASSIGN_FACTORY_ENTRY("cudaMallocHost",
                             DEREF(makeDerefArgCreatorWithCall(0)),
                             CAST(getDerefedType(0), CALL("malloc", ARG(1)))),
        CONDITIONAL_FACTORY_ENTRY(
            CheckCanUseTemplateMalloc(0, 1),
            ASSIGN_FACTORY_ENTRY(
                "cudaMallocHost", DEREF(makeDerefArgCreatorWithCall(0)),
                CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getClNamespace() +
                                                    "malloc_host",
                                                getDoubleDerefedType(0)),
                     getSizeForMalloc(0, 1), QUEUESTR)),
            ASSIGN_FACTORY_ENTRY(
                "cudaMallocHost", DEREF(makeDerefArgCreatorWithCall(0)),
                CAST(getDerefedType(0),
                     CALL(MapNames::getClNamespace() + "malloc_host",
                          getSizeForMalloc(0, 1), QUEUESTR))))),
    ASSIGN_FACTORY_ENTRY("cudaMallocHost",
                         DEREF(makeDerefArgCreatorWithCall(0)),
                         CAST(getDerefedType(0), CALL("malloc", ARG(1))))))

ASSIGNABLE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
    checkIsUSM(),
    CONDITIONAL_FACTORY_ENTRY(
        CheckCanUseTemplateMalloc(0, 1),
        ASSIGN_FACTORY_ENTRY(
            "cuMemHostAlloc", DEREF(makeDerefArgCreatorWithCall(0)),
            CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getClNamespace() +
                                                "malloc_host",
                                            getDoubleDerefedType(0)),
                 getSizeForMalloc(0, 1), QUEUESTR)),
        ASSIGN_FACTORY_ENTRY(
            "cuMemHostAlloc", DEREF(makeDerefArgCreatorWithCall(0)),
            CAST(getDerefedType(0),
                 CALL(MapNames::getClNamespace() + "malloc_host",
                      getSizeForMalloc(0, 1), QUEUESTR)))),
    ASSIGN_FACTORY_ENTRY("cuMemHostAlloc", DEREF(makeDerefArgCreatorWithCall(0)),
                         CAST(getDerefedType(0), CALL("malloc", ARG(1))))))


ASSIGNABLE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
    checkIsUSM(),
    CONDITIONAL_FACTORY_ENTRY(
        CheckCanUseTemplateMalloc(0, 1),
        ASSIGN_FACTORY_ENTRY(
            "cuMemAllocHost_v2", DEREF(makeDerefArgCreatorWithCall(0)),
            CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getClNamespace() +
                                                "malloc_host",
                                            getDoubleDerefedType(0)),
                 getSizeForMalloc(0, 1), QUEUESTR)),
        ASSIGN_FACTORY_ENTRY(
            "cuMemAllocHost_v2", DEREF(makeDerefArgCreatorWithCall(0)),
            CAST(getDerefedType(0),
                 CALL(MapNames::getClNamespace() + "malloc_host",
                      getSizeForMalloc(0, 1), QUEUESTR)))),
    ASSIGN_FACTORY_ENTRY("cuMemAllocHost_v2", DEREF(makeDerefArgCreatorWithCall(0)),
                         CAST(getDerefedType(0), CALL("malloc", ARG(1))))))

ASSIGNABLE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
    CheckCanUseTemplateMalloc(0, 1),
    CONDITIONAL_FACTORY_ENTRY(
        hasManagedAttr(0),
        ASSIGN_FACTORY_ENTRY(
            "cudaMallocManaged",
            makeCombinedArg(
                makeCombinedArg(ARG("*("),
                                DEREF(makeDerefArgCreatorWithCall(0))),
                ARG(".get_ptr())")),
            CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getClNamespace() +
                                                "malloc_shared",
                                            getDoubleDerefedType(0)),
                 getSizeForMalloc(0, 1), QUEUESTR)),
        ASSIGN_FACTORY_ENTRY(
            "cudaMallocManaged", DEREF(makeDerefArgCreatorWithCall(0)),
            CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getClNamespace() +
                                                "malloc_shared",
                                            getDoubleDerefedType(0)),
                 getSizeForMalloc(0, 1), QUEUESTR))),
    CONDITIONAL_FACTORY_ENTRY(
        hasManagedAttr(0),
        ASSIGN_FACTORY_ENTRY(
            "cudaMallocManaged",
            makeCombinedArg(
                makeCombinedArg(ARG("*("),
                                DEREF(makeDerefArgCreatorWithCall(0))),
                ARG(".get_ptr())")),
            CAST(getDerefedType(0),
                 CALL(MapNames::getClNamespace() + "malloc_shared",
                      getSizeForMalloc(0, 1), QUEUESTR))),
        ASSIGN_FACTORY_ENTRY(
            "cudaMallocManaged", DEREF(makeDerefArgCreatorWithCall(0)),
            CAST(getDerefedType(0),
                 CALL(MapNames::getClNamespace() + "malloc_shared",
                      getSizeForMalloc(0, 1), QUEUESTR))))))

CONDITIONAL_FACTORY_ENTRY(
    UseExtBindlessImages,
    ASSIGNABLE_FACTORY(DELETER_FACTORY_ENTRY("cudaFreeMipmappedArray", ARG(0))),
    REMOVE_API_FACTORY_ENTRY_WITH_MSG("cudaFreeMipmappedArray",
                                      MipmapNeedBindlessImage))

CONDITIONAL_FACTORY_ENTRY(
    UseExtBindlessImages,
    ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY(
        "cudaGetMipmappedArrayLevel", DEREF(makeCallArgCreatorWithCall(0)),
        MEMBER_CALL(ARG(1), true, "get_mip_level", ARG(2)))),
    REMOVE_API_FACTORY_ENTRY_WITH_MSG("cudaGetMipmappedArrayLevel",
                                      MipmapNeedBindlessImage))

CONDITIONAL_FACTORY_ENTRY(
    UseExtBindlessImages,
    ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY(
        "cudaMallocMipmappedArray", DEREF(makeCallArgCreatorWithCall(0)),
        NEW(MapNames::getDpctNamespace() + "experimental::image_mem_wrapper",
            DEREF(1), ARG(2),
            LITERAL(MapNames::getClNamespace() +
                    "ext::oneapi::experimental::image_type::mipmap"),
            ARG(3)))),
    REMOVE_API_FACTORY_ENTRY_WITH_MSG("cudaMallocMipmappedArray",
                                      MipmapNeedBindlessImage))

ASSIGNABLE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
    CheckCanUseTemplateMalloc(0, 1),
    ASSIGN_FACTORY_ENTRY("cuMemAllocManaged", DEREF(makeDerefArgCreatorWithCall(0)),
                         CALL(TEMPLATED_CALLEE_WITH_ARGS(
                                  MapNames::getClNamespace() + "malloc_shared",
                                  getDoubleDerefedType(0)),
                              getSizeForMalloc(0, 1), QUEUESTR)),
    ASSIGN_FACTORY_ENTRY("cuMemAllocManaged", DEREF(makeDerefArgCreatorWithCall(0)),
                         CAST(getDerefedType(0),
                              CALL(MapNames::getClNamespace() + "malloc_shared",
                                   getSizeForMalloc(0, 1), QUEUESTR)
                             ))))

// use makeCallArgCreatorWithCall instead of makeDerefArgCreatorWithCall to keep the cast information
ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY("cudaHostGetDevicePointer",
                                        DEREF(makeCallArgCreatorWithCall(0)),
                                        CAST(getDerefedType(0), ARG(1))))

ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY("cuMemHostGetDevicePointer_v2",
                                        DEREF(makeCallArgCreatorWithCall(0)),
                                        CAST(getReplacedTypeNameForDerefExpr(0),
                                             ARG(1))))

ASSIGNABLE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
    checkIsUSM(),
    CONDITIONAL_FACTORY_ENTRY(
        makeCheckNot(CheckArgIsDefaultCudaStream(3)),
        MEMBER_CALL_FACTORY_ENTRY(
            "cuMemcpyDtoDAsync_v2", ARG(3), true, "memcpy", ARG(0), ARG(1), ARG(2)),
        MEMBER_CALL_FACTORY_ENTRY(
            "cuMemcpyDtoDAsync_v2", QUEUESTR, false, "memcpy", ARG(0), ARG(1), ARG(2))),
    CONDITIONAL_FACTORY_ENTRY(
        makeCheckNot(CheckArgIsDefaultCudaStream(3)),
        FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext,
        CALL_FACTORY_ENTRY(
            "cuMemcpyDtoDAsync_v2", CALL(MapNames::getDpctNamespace() + "async_dpct_memcpy",
            ARG(0), ARG(1), ARG(2), ARG(MapNames::getDpctNamespace() + "automatic"), STREAM(3)))),
        FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext,
        CALL_FACTORY_ENTRY(
            "cuMemcpyDtoDAsync_v2", CALL(MapNames::getDpctNamespace() + "async_dpct_memcpy",
            ARG(0), ARG(1), ARG(2), ARG(MapNames::getDpctNamespace() + "automatic")))))))

ASSIGNABLE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
    checkIsUSM(),
    MEMBER_CALL_FACTORY_ENTRY(
        "cuMemcpyDtoD_v2", MEMBER_CALL(QUEUESTR, false, "memcpy", ARG(0), ARG(1), ARG(2)), false, "wait"),
    FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cuMemcpyDtoD_v2", CALL(MapNames::getDpctNamespace() + "dpct_memcpy",
        ARG(0), ARG(1), ARG(2), ARG(MapNames::getDpctNamespace() + "automatic"))))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    ASSIGN_FACTORY_ENTRY(
        "cuMemAllocPitch_v2", DEREF(makeDerefArgCreatorWithCall(0)),
        CAST(getDerefedType(0),
             CALL(getMemoryHelperFunctionName("malloc"),
                  DEREF(makeCallArgCreatorWithCall(1)), ARG(2), ARG(3))))))

ASSIGNABLE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
    makeCheckNot(CheckArgIsDefaultCudaStream(3)),
    MEMBER_CALL_FACTORY_ENTRY("cuMemPrefetchAsync", ARG(3), true, "prefetch",
                              ARG(0), ARG(1)),
    FEATURE_REQUEST_FACTORY(
        HelperFeatureEnum::device_ext,
        MEMBER_CALL_FACTORY_ENTRY(
            "cuMemPrefetchAsync",
            MEMBER_CALL(CALL(MapNames::getDpctNamespace() + "get_device",
                             ARG(2)),
                        false, DpctGlobalInfo::getDefaultQueueMemFuncName()),
            DpctGlobalInfo::useSYCLCompat(), "prefetch", ARG(0), ARG(1)))))

ASSIGNABLE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
    checkArgSpelling(3, "CU_DEVICE_CPU"),
    CONDITIONAL_FACTORY_ENTRY(
        checkIsArgIntegerLiteral(2),
        WARNING_FACTORY_ENTRY(
            "cuMemAdvise",
            FEATURE_REQUEST_FACTORY(
                HelperFeatureEnum::device_ext,
                MEMBER_CALL_FACTORY_ENTRY(
                    "cuMemAdvise",
                    MEMBER_CALL(CALL(MapNames::getDpctNamespace() +
                                     "cpu_device"),
                                false, DpctGlobalInfo::getDefaultQueueMemFuncName()),
                    DpctGlobalInfo::useSYCLCompat(), "mem_advise", ARG(0), ARG(1), ARG("0"))),
            Diagnostics::DEFAULT_MEM_ADVICE, ARG(" and was set to 0")),

        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::device_ext,
            MEMBER_CALL_FACTORY_ENTRY(
                "cuMemAdvise",
                MEMBER_CALL(CALL(MapNames::getDpctNamespace() + "cpu_device"),
                            false, DpctGlobalInfo::getDefaultQueueMemFuncName()),
                DpctGlobalInfo::useSYCLCompat(), "mem_advise", ARG(0), ARG(1), ARG(2)))),

    CONDITIONAL_FACTORY_ENTRY(
        checkIsArgIntegerLiteral(2),
        WARNING_FACTORY_ENTRY(
            "cuMemAdvise",
            FEATURE_REQUEST_FACTORY(
                HelperFeatureEnum::device_ext,
                MEMBER_CALL_FACTORY_ENTRY(
                    "cuMemAdvise",
                    MEMBER_CALL(CALL(MapNames::getDpctNamespace() +
                                         "get_device",
                                     ARG(3)),
                                false, DpctGlobalInfo::getDefaultQueueMemFuncName()),
                    DpctGlobalInfo::useSYCLCompat(), "mem_advise", ARG(0), ARG(1), ARG("0"))),
            Diagnostics::DEFAULT_MEM_ADVICE, ARG(" and was set to 0")),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::device_ext,
            MEMBER_CALL_FACTORY_ENTRY(
                "cuMemAdvise",
                MEMBER_CALL(CALL(MapNames::getDpctNamespace() + "get_device",
                                 ARG(3)),
                            false, DpctGlobalInfo::getDefaultQueueMemFuncName()),
                DpctGlobalInfo::useSYCLCompat(), "mem_advise", ARG(0), ARG(1), ARG(2))))))

ASSIGNABLE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
    checkIsUSM(),
    CALL_FACTORY_ENTRY("cuMemFree_v2", CALL(MapNames::getClNamespace() + "free", ARG(0), QUEUESTR)),
    FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext,
        CALL_FACTORY_ENTRY("cuMemFree_v2", CALL(MapNames::getDpctNamespace() + "dpct_free", ARG(0))))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    ASSIGN_FACTORY_ENTRY(
        "cuDeviceTotalMem_v2", DEREF(makeDerefArgCreatorWithCall(0)),
        MEMBER_CALL(MEMBER_CALL(CALL(MapNames::getDpctNamespace() +
                                         "get_device",
                                     ARG(1)),
                                false, "get_device_info"),
                    false, "get_global_mem_size"))))

ASSIGNABLE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
    checkIsUSM(),
    CONDITIONAL_FACTORY_ENTRY(
        makeCheckNot(CheckArgIsDefaultCudaStream(3)),
        MEMBER_CALL_FACTORY_ENTRY(
            "cuMemcpyHtoDAsync_v2", ARG(3), true, "memcpy", ARG(0), ARG(1), ARG(2)),
        MEMBER_CALL_FACTORY_ENTRY(
            "cuMemcpyHtoDAsync_v2", QUEUESTR, false, "memcpy", ARG(0), ARG(1), ARG(2))),
    CONDITIONAL_FACTORY_ENTRY(
        makeCheckNot(CheckArgIsDefaultCudaStream(3)),
        FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext,
        CALL_FACTORY_ENTRY(
            "cuMemcpyHtoDAsync_v2", CALL(MapNames::getDpctNamespace() + "async_dpct_memcpy",
            ARG(0), ARG(1), ARG(2), ARG(MapNames::getDpctNamespace() + "automatic"), STREAM(3)))),
        FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext,
        CALL_FACTORY_ENTRY(
            "cuMemcpyHtoDAsync_v2", CALL(MapNames::getDpctNamespace() + "async_dpct_memcpy",
            ARG(0), ARG(1), ARG(2), ARG(MapNames::getDpctNamespace() + "automatic")))))))

ASSIGNABLE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
    checkIsUSM(),
    MEMBER_CALL_FACTORY_ENTRY(
        "cuMemcpyHtoD_v2", MEMBER_CALL(QUEUESTR, false, "memcpy", ARG(0), ARG(1), ARG(2)), false, "wait"),
    FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cuMemcpyHtoD_v2", CALL(MapNames::getDpctNamespace() + "dpct_memcpy",
        ARG(0), ARG(1), ARG(2), ARG(MapNames::getDpctNamespace() + "automatic"))))))

ASSIGNABLE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
    checkIsUSM(),
    MEMBER_CALL_FACTORY_ENTRY(
        "cuMemcpy", MEMBER_CALL(QUEUESTR, false, "memcpy", ARG(0), ARG(1), ARG(2)), false, "wait"),
    FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cuMemcpy", CALL(MapNames::getDpctNamespace() + "dpct_memcpy",
	ARG(0), ARG(1), ARG(2), ARG(MapNames::getDpctNamespace() + "automatic"))))))

ASSIGNABLE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
    checkIsUSM(),
    CONDITIONAL_FACTORY_ENTRY(
        makeCheckNot(CheckArgIsDefaultCudaStream(3)),
        MEMBER_CALL_FACTORY_ENTRY(
            "cuMemcpyAsync", ARG(3), true, "memcpy", ARG(0), ARG(1), ARG(2)),
        MEMBER_CALL_FACTORY_ENTRY(
            "cuMemcpyAsync", QUEUESTR, false, "memcpy", ARG(0), ARG(1), ARG(2))),
    CONDITIONAL_FACTORY_ENTRY(
        makeCheckNot(CheckArgIsDefaultCudaStream(3)),
        FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext,
        CALL_FACTORY_ENTRY(
            "cuMemcpyAsync", CALL(MapNames::getDpctNamespace() + "async_dpct_memcpy",
            ARG(0), ARG(1), ARG(2), ARG(MapNames::getDpctNamespace() + "automatic"), STREAM(3)))),
        FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext,
        CALL_FACTORY_ENTRY(
            "cuMemcpyAsync", CALL(MapNames::getDpctNamespace() + "async_dpct_memcpy",
            ARG(0), ARG(1), ARG(2), ARG(MapNames::getDpctNamespace() + "automatic")))))))

CONDITIONAL_FACTORY_ENTRY(
    checkIsCallExprOnly(),
    WARNING_FACTORY_ENTRY("cudaHostUnregister",
        TOSTRING_FACTORY_ENTRY("cudaHostUnregister", LITERAL("")),
         Diagnostics::FUNC_CALL_REMOVED,
        std::string("cudaHostUnregister"), getRemovedAPIWarningMessage("cudaHostUnregister")),
    WARNING_FACTORY_ENTRY("cudaHostUnregister",
        TOSTRING_FACTORY_ENTRY("cudaHostUnregister", LITERAL("0")),
        Diagnostics::FUNC_CALL_REMOVED_0,
        std::string("cudaHostUnregister"), getRemovedAPIWarningMessage("cudaHostUnregister")))

CONDITIONAL_FACTORY_ENTRY(
    checkIsCallExprOnly(),
    WARNING_FACTORY_ENTRY("cudaHostRegister",
        TOSTRING_FACTORY_ENTRY("cudaHostRegister", LITERAL("")),
         Diagnostics::FUNC_CALL_REMOVED,
        std::string("cudaHostRegister"), getRemovedAPIWarningMessage("cudaHostRegister")),
    WARNING_FACTORY_ENTRY("cudaHostRegister",
        TOSTRING_FACTORY_ENTRY("cudaHostRegister", LITERAL("0")),
        Diagnostics::FUNC_CALL_REMOVED_0,
        std::string("cudaHostRegister"), getRemovedAPIWarningMessage("cudaHostRegister")))

ASSIGNABLE_FACTORY(
    FEATURE_REQUEST_FACTORY(
        HelperFeatureEnum::device_ext,
        MEMBER_CALL_FACTORY_ENTRY(
            "cudaPointerGetAttributes", DEREF(0), false, "init", ARG(1))))

ASSIGNABLE_FACTORY(
    FEATURE_REQUEST_FACTORY(
        HelperFeatureEnum::device_ext,
        CALL_FACTORY_ENTRY(
            "cuPointerGetAttributes", CALL(MapNames::getDpctNamespace() + "pointer_attributes::get",
            ARG(0), ARG(1), ARG(2), ARG(3)))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CONDITIONAL_FACTORY_ENTRY(
        CheckDerefedTypeBeforeCast(0, "void *"),
        ASSIGN_FACTORY_ENTRY(
            "cudaMallocPitch", DEREF(makeDerefArgCreatorWithCall(0)),
            CALL(getMemoryHelperFunctionName("malloc"),
                 DEREF(makeCallArgCreatorWithCall(1)),
                 makeCallArgCreatorWithCall(2),
                 makeDerefArgCreatorWithCall(3))),
        ASSIGN_FACTORY_ENTRY(
            "cudaMallocPitch", DEREF(makeDerefArgCreatorWithCall(0)),
            CAST(getDerefedType(0), CALL(getMemoryHelperFunctionName("malloc"),
                                         DEREF(makeCallArgCreatorWithCall(1)),
                                         makeCallArgCreatorWithCall(2),
                                         makeCallArgCreatorWithCall(3)))))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CONDITIONAL_FACTORY_ENTRY(
        UseExtBindlessImages,
        CALL_FACTORY_ENTRY(
            "cuMemcpyAtoH_v2",
            CALL(MapNames::getDpctNamespace() + "experimental::dpct_memcpy",
                 makeCallArgCreatorWithCall(0), makeCallArgCreatorWithCall(1),
                 makeCallArgCreatorWithCall(2), ARG("0"),
                 makeCallArgCreatorWithCall(3))),
        CALL_FACTORY_ENTRY(
            "cuMemcpyAtoH_v2",
            CALL(MapNames::getDpctNamespace() + "dpct_memcpy",
                 makeCallArgCreatorWithCall(0),
                 BO(BinaryOperatorKind::BO_Add,
                    makeCastExprCreator(
                        makeCharPtrCreator(),
                        MEMBER_CALL(MEMBER_CALL(makeCallArgCreatorWithCall(1),
                                                true, "to_pitched_data"),
                                    false, "get_data_ptr"),
                        true),
                    makeCallArgCreatorWithCall(2)),
                 makeCallArgCreatorWithCall(3))))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CONDITIONAL_FACTORY_ENTRY(
        UseExtBindlessImages,
        CALL_FACTORY_ENTRY(
            "cuMemcpyHtoA_v2",
            CALL(MapNames::getDpctNamespace() + "experimental::dpct_memcpy",
                 makeCallArgCreatorWithCall(0), makeCallArgCreatorWithCall(1),
                 ARG("0"), makeCallArgCreatorWithCall(2),
                 makeCallArgCreatorWithCall(3))),
        CALL_FACTORY_ENTRY(
            "cuMemcpyHtoA_v2",
            CALL(MapNames::getDpctNamespace() + "dpct_memcpy",
                 BO(BinaryOperatorKind::BO_Add,
                    makeCastExprCreator(
                        makeCharPtrCreator(),
                        MEMBER_CALL(MEMBER_CALL(makeCallArgCreatorWithCall(0),
                                                true, "to_pitched_data"),
                                    false, "get_data_ptr"),
                        true),
                    makeCallArgCreatorWithCall(1)),
                 makeCallArgCreatorWithCall(2),
                 makeCallArgCreatorWithCall(3))))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CONDITIONAL_FACTORY_ENTRY(
        UseExtBindlessImages,
        CALL_FACTORY_ENTRY("cuMemcpyAtoHAsync_v2",
                           CALL(MapNames::getDpctNamespace() +
                                    "experimental::async_dpct_memcpy",
                                makeCallArgCreatorWithCall(0),
                                makeCallArgCreatorWithCall(1),
                                makeCallArgCreatorWithCall(2), ARG("0"),
                                makeCallArgCreatorWithCall(3), STREAM(4))),
        CALL_FACTORY_ENTRY(
            "cuMemcpyAtoHAsync_v2",
            CALL(MapNames::getDpctNamespace() + "async_dpct_memcpy",
                 makeCallArgCreatorWithCall(0),
                 BO(BinaryOperatorKind::BO_Add,
                    makeCastExprCreator(
                        makeCharPtrCreator(),
                        MEMBER_CALL(MEMBER_CALL(makeCallArgCreatorWithCall(1),
                                                true, "to_pitched_data"),
                                    false, "get_data_ptr"),
                        true),
                    makeCallArgCreatorWithCall(2)),
                 makeCallArgCreatorWithCall(3),
                 ARG(MapNames::getDpctNamespace() + "automatic"),
                 DEREF(makeCallArgCreatorWithCall(4)))))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CONDITIONAL_FACTORY_ENTRY(
        UseExtBindlessImages,
        CALL_FACTORY_ENTRY("cuMemcpyHtoAAsync_v2",
                           CALL(MapNames::getDpctNamespace() +
                                    "experimental::async_dpct_memcpy",
                                makeCallArgCreatorWithCall(0),
                                makeCallArgCreatorWithCall(1), ARG("0"),
                                makeCallArgCreatorWithCall(2),
                                makeCallArgCreatorWithCall(3), STREAM(4))),
        CALL_FACTORY_ENTRY(
            "cuMemcpyHtoAAsync_v2",
            CALL(MapNames::getDpctNamespace() + "async_dpct_memcpy",
                 BO(BinaryOperatorKind::BO_Add,
                    makeCastExprCreator(
                        makeCharPtrCreator(),
                        MEMBER_CALL(MEMBER_CALL(makeCallArgCreatorWithCall(0),
                                                true, "to_pitched_data"),
                                    false, "get_data_ptr"),
                        true),
                    makeCallArgCreatorWithCall(1)),
                 makeCallArgCreatorWithCall(2), makeCallArgCreatorWithCall(3),
                 ARG(MapNames::getDpctNamespace() + "automatic"),
                 DEREF(makeCallArgCreatorWithCall(4)))))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CONDITIONAL_FACTORY_ENTRY(
        UseExtBindlessImages,
        CALL_FACTORY_ENTRY(
            "cuMemcpyAtoD_v2",
            CALL(MapNames::getDpctNamespace() + "experimental::dpct_memcpy",
                 makeCallArgCreatorWithCall(0), makeCallArgCreatorWithCall(1),
                 makeCallArgCreatorWithCall(2), ARG("0"),
                 makeCallArgCreatorWithCall(3))),
        CALL_FACTORY_ENTRY(
            "cuMemcpyAtoD_v2",
            CALL(MapNames::getDpctNamespace() + "dpct_memcpy",
                 makeCallArgCreatorWithCall(0),
                 BO(BinaryOperatorKind::BO_Add,
                    makeCastExprCreator(
                        makeCharPtrCreator(),
                        MEMBER_CALL(MEMBER_CALL(makeCallArgCreatorWithCall(1),
                                                true, "to_pitched_data"),
                                    false, "get_data_ptr"),
                        true),
                    makeCallArgCreatorWithCall(2)),
                 makeCallArgCreatorWithCall(3))))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CONDITIONAL_FACTORY_ENTRY(
        UseExtBindlessImages,
        CALL_FACTORY_ENTRY(
            "cuMemcpyDtoA_v2",
            CALL(MapNames::getDpctNamespace() + "experimental::dpct_memcpy",
                 makeCallArgCreatorWithCall(0), makeCallArgCreatorWithCall(1),
                 ARG("0"), makeCallArgCreatorWithCall(2),
                 makeCallArgCreatorWithCall(3))),
        CALL_FACTORY_ENTRY(
            "cuMemcpyDtoA_v2",
            CALL(MapNames::getDpctNamespace() + "dpct_memcpy",
                 BO(BinaryOperatorKind::BO_Add,
                    makeCastExprCreator(
                        makeCharPtrCreator(),
                        MEMBER_CALL(MEMBER_CALL(makeCallArgCreatorWithCall(0),
                                                true, "to_pitched_data"),
                                    false, "get_data_ptr"),
                        true),
                    makeCallArgCreatorWithCall(1)),
                 makeCallArgCreatorWithCall(2),
                 makeCallArgCreatorWithCall(3))))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CONDITIONAL_FACTORY_ENTRY(
        UseExtBindlessImages,
        CALL_FACTORY_ENTRY(
            "cuMemcpyAtoA_v2",
            CALL(MapNames::getDpctNamespace() + "experimental::dpct_memcpy",
                 makeCallArgCreatorWithCall(0), makeCallArgCreatorWithCall(1),
                 ARG("0"), makeCallArgCreatorWithCall(2),
                 makeCallArgCreatorWithCall(3), ARG("0"),
                 makeCallArgCreatorWithCall(4))),
        CALL_FACTORY_ENTRY(
            "cuMemcpyAtoA_v2",
            CALL(MapNames::getDpctNamespace() + "dpct_memcpy",
                 BO(BinaryOperatorKind::BO_Add,
                    makeCastExprCreator(
                        makeCharPtrCreator(),
                        MEMBER_CALL(MEMBER_CALL(makeCallArgCreatorWithCall(0),
                                                true, "to_pitched_data"),
                                    false, "get_data_ptr"),
                        true),
                    makeCallArgCreatorWithCall(1)),
                 BO(BinaryOperatorKind::BO_Add,
                    makeCastExprCreator(
                        makeCharPtrCreator(),
                        MEMBER_CALL(MEMBER_CALL(makeCallArgCreatorWithCall(2),
                                                true, "to_pitched_data"),
                                    false, "get_data_ptr"),
                        true),
                    makeCallArgCreatorWithCall(3)),
                 makeCallArgCreatorWithCall(4))))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("cuMemsetD8_v2",
                       CALL(getMemoryHelperFunctionName("memset"),
                            ARG(0), ARG(1), ARG(2)))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("cuMemsetD16_v2",
                       CALL(getMemoryHelperFunctionName("memset_d16"),
                            ARG(0), ARG(1), ARG(2)))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("cuMemsetD32_v2",
                       CALL(getMemoryHelperFunctionName("memset_d32"),
                            ARG(0), ARG(1), ARG(2)))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("cuMemsetD8Async",
                       CALL(getMemoryHelperFunctionName("memset_async"),
                            ARG(0), ARG(1), ARG(2),
                            DEREF(makeCallArgCreatorWithCall(3))))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("cuMemsetD16Async",
                       CALL(getMemoryHelperFunctionName("memset_d16_async"),
                            ARG(0), ARG(1), ARG(2),
                            DEREF(makeCallArgCreatorWithCall(3))))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("cuMemsetD32Async",
                       CALL(getMemoryHelperFunctionName("memset_d32_async"),
                            ARG(0), ARG(1), ARG(2),
                            DEREF(makeCallArgCreatorWithCall(3))))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("cuMemsetD2D8_v2",
                       CALL(getMemoryHelperFunctionName("memset"),
                            ARG(0), ARG(1), ARG(2), ARG(3), ARG(4)))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("cuMemsetD2D16_v2",
                       CALL(getMemoryHelperFunctionName("memset_d16"),
                            ARG(0), ARG(1), ARG(2), ARG(3), ARG(4)))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("cuMemsetD2D32_v2",
                       CALL(getMemoryHelperFunctionName("memset_d32"),
                            ARG(0), ARG(1), ARG(2), ARG(3), ARG(4)))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("cuMemsetD2D8Async",
                       CALL(getMemoryHelperFunctionName("memset_async"),
                            ARG(0), ARG(1), ARG(2), ARG(3), ARG(4), 
                            DEREF(makeCallArgCreatorWithCall(5))))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("cuMemsetD2D16Async",
                       CALL(getMemoryHelperFunctionName("memset_d16_async"),
                            ARG(0), ARG(1), ARG(2), ARG(3), ARG(4),
                            DEREF(makeCallArgCreatorWithCall(5))))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("cuMemsetD2D32Async",
                       CALL(getMemoryHelperFunctionName("memset_d32_async"),
                            ARG(0), ARG(1), ARG(2), ARG(3), ARG(4),
                            DEREF(makeCallArgCreatorWithCall(5))))))

CONDITIONAL_FACTORY_ENTRY(
    UseExpVirtualMemory,
    ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
        HelperFeatureEnum::device_ext,
        ASSIGN_FACTORY_ENTRY(
            "cuMemCreate", DEREF(0),
            NEW(MapNames::getClNamespace() +
                    "ext::oneapi::experimental::physical_mem",
                CALL(MapNames::getDpctNamespace() + "get_device",
                     MEMBER_EXPR(DEREF(2), false, LITERAL("location.id"))),
                MEMBER_CALL(CALL(MapNames::getDpctNamespace() + "get_device",
                                 MEMBER_EXPR(DEREF(2), false,
                                             LITERAL("location.id"))),
                            false, "get_context"),
                ARG(1))))),
    UNSUPPORT_FACTORY_ENTRY("cuMemCreate",
                            Diagnostics::TRY_EXPERIMENTAL_FEATURE,
                            ARG("cuMemCreate"),
                            ARG("--use-experimental-features=virtual_mem")))

CONDITIONAL_FACTORY_ENTRY(
    UseExpVirtualMemory,
    ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
        HelperFeatureEnum::device_ext,
        ASSIGN_FACTORY_ENTRY(
            "cuMemGetAllocationPropertiesFromHandle",
            MEMBER_EXPR(DEREF(0), false, LITERAL("location.id")),
            CALL(MapNames::getDpctNamespace() + "get_device_id",
                 MEMBER_CALL(ARG(1), true, "get_device"))))),
    UNSUPPORT_FACTORY_ENTRY("cuMemGetAllocationPropertiesFromHandle",
                            Diagnostics::TRY_EXPERIMENTAL_FEATURE,
                            ARG("cuMemGetAllocationPropertiesFromHandle"),
                            ARG("--use-experimental-features=virtual_mem")))

CONDITIONAL_FACTORY_ENTRY(
    UseExpVirtualMemory,
    ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
        HelperFeatureEnum::device_ext,
        ASSIGN_FACTORY_ENTRY(
            "cuMemAddressReserve", DEREF(0),
            CALL("(" + MapNames::getDpctNamespace() + "device_ptr)" +
                     MapNames::getClNamespace() +
                     "ext::oneapi::experimental::reserve_virtual_mem",
                 CAST(LITERAL("uintptr_t"), ARG(3)), ARG(1),
                 LITERAL(MapNames::getDpctNamespace() +
                         "get_current_device().get_context()"))))),
    UNSUPPORT_FACTORY_ENTRY("cuMemAddressReserve",
                            Diagnostics::TRY_EXPERIMENTAL_FEATURE,
                            ARG("cuMemAddressReserve"),
                            ARG("--use-experimental-features=virtual_mem")))

CONDITIONAL_FACTORY_ENTRY(
    UseExpVirtualMemory,
    ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
        HelperFeatureEnum::device_ext,
        CALL_FACTORY_ENTRY(
            "cuMemAddressFree",
            CALL(MapNames::getClNamespace() +
                     "ext::oneapi::experimental::free_virtual_mem",
                 CAST(LITERAL("uintptr_t"), ARG(0)), ARG(1),
                 LITERAL(MapNames::getDpctNamespace() +
                         "get_current_device().get_context()"))))),
    UNSUPPORT_FACTORY_ENTRY("cuMemAddressFree",
                            Diagnostics::TRY_EXPERIMENTAL_FEATURE,
                            ARG("cuMemAddressFree"),
                            ARG("--use-experimental-features=virtual_mem")))

CONDITIONAL_FACTORY_ENTRY(
    UseExpVirtualMemory,
    ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
        HelperFeatureEnum::device_ext,
        ASSIGN_FACTORY_ENTRY(
            "cuMemGetAllocationGranularity", DEREF(0),
            CALL(MapNames::getClNamespace() +
                     "ext::oneapi::experimental::get_mem_granularity",
                 CALL(MapNames::getDpctNamespace() + "get_device",
                      MEMBER_EXPR(DEREF(1), false, LITERAL("location.id"))),
                 MEMBER_CALL(CALL(MapNames::getDpctNamespace() + "get_device",
                                  MEMBER_EXPR(DEREF(1), false,
                                              LITERAL("location.id"))),
                             false, "get_context"),
                 ARG(2))))),
    UNSUPPORT_FACTORY_ENTRY("cuMemGetAllocationGranularity",
                            Diagnostics::TRY_EXPERIMENTAL_FEATURE,
                            ARG("cuMemGetAllocationGranularity"),
                            ARG("--use-experimental-features=virtual_mem")))

CONDITIONAL_FACTORY_ENTRY(
    UseExpVirtualMemory,
    ASSIGNABLE_FACTORY(
        FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext,
                                DELETE_FACTORY_ENTRY("cuMemRelease", ARG(0)))),
    UNSUPPORT_FACTORY_ENTRY("cuMemRelease",
                            Diagnostics::TRY_EXPERIMENTAL_FEATURE,
                            ARG("cuMemRelease"),
                            ARG("--use-experimental-features=virtual_mem")))

CONDITIONAL_FACTORY_ENTRY(
    UseExpVirtualMemory,
    ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
        HelperFeatureEnum::device_ext,
        MEMBER_CALL_FACTORY_ENTRY(
            "cuMemMap", ARG(3), true, "map", CAST(LITERAL("uintptr_t"), ARG(0)),
            ARG(1),
            LITERAL(
                MapNames::getClNamespace() +
                "ext::oneapi::experimental::address_access_mode::read_write"),
            ARG(2)))),
    UNSUPPORT_FACTORY_ENTRY("cuMemMap", Diagnostics::TRY_EXPERIMENTAL_FEATURE,
                            ARG("cuMemMap"),
                            ARG("--use-experimental-features=virtual_mem")))

CONDITIONAL_FACTORY_ENTRY(
    UseExpVirtualMemory,
    ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
        HelperFeatureEnum::device_ext,
        CALL_FACTORY_ENTRY(
            "cuMemUnmap",
            CALL(MapNames::getClNamespace() +
                     "ext::oneapi::experimental::unmap",
                 ARG(0), ARG(1),
                 LITERAL(MapNames::getDpctNamespace() +
                         "get_current_device().get_context()"))))),
    UNSUPPORT_FACTORY_ENTRY("cuMemUnmap", Diagnostics::TRY_EXPERIMENTAL_FEATURE,
                            ARG("cuMemUnmap"),
                            ARG("--use-experimental-features=virtual_mem")))

CONDITIONAL_FACTORY_ENTRY(
    UseExpVirtualMemory,
    ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
        HelperFeatureEnum::device_ext,
        CALL_FACTORY_ENTRY(
            "cuMemSetAccess",
            CALL(MapNames::getClNamespace() +
                     "ext::oneapi::experimental::set_access_mode",
                 ARG(0), ARG(1), MEMBER_EXPR(DEREF(2), false, LITERAL("flags")),
                 MEMBER_CALL(CALL(MapNames::getDpctNamespace() + "get_device",
                                  MEMBER_EXPR(DEREF(2), false,
                                              LITERAL("location.id"))),
                             false, "get_context"))))),
    UNSUPPORT_FACTORY_ENTRY("cuMemSetAccess",
                            Diagnostics::TRY_EXPERIMENTAL_FEATURE,
                            ARG("cuMemSetAccess"),
                            ARG("--use-experimental-features=virtual_mem")))

CONDITIONAL_FACTORY_ENTRY(
    UsePeerAccess(),
    ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY(
        "cuDeviceCanAccessPeer", DEREF(ARG_WC(0)),
        MEMBER_CALL(
            CALL(MapNames::getDpctNamespace() + "get_device", ARG_WC(1)), false,
            "ext_oneapi_can_access_peer",
            CALL(MapNames::getDpctNamespace() + "get_device", ARG_WC(2))))),
    ASSIGNABLE_FACTORY(
        WARNING_FACTORY_ENTRY("cuDeviceCanAccessPeer",
                              ASSIGN_FACTORY_ENTRY("cuDeviceCanAccessPeer",
                                                   DEREF(ARG_WC(0)), ARG("0")),
                              Diagnostics::EXPLICIT_PEER_ACCESS)))

CONDITIONAL_FACTORY_ENTRY(
    UsePeerAccess(),
    ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY(
        "cudaDeviceCanAccessPeer", DEREF(ARG_WC(0)),
        MEMBER_CALL(
            CALL(MapNames::getDpctNamespace() + "get_device", ARG_WC(1)), false,
            "ext_oneapi_can_access_peer",
            CALL(MapNames::getDpctNamespace() + "get_device", ARG_WC(2))))),
    ASSIGNABLE_FACTORY(
        WARNING_FACTORY_ENTRY("cudaDeviceCanAccessPeer",
                              ASSIGN_FACTORY_ENTRY("cudaDeviceCanAccessPeer",
                                                   DEREF(ARG_WC(0)), ARG("0")),
                              Diagnostics::EXPLICIT_PEER_ACCESS)))

CONDITIONAL_FACTORY_ENTRY(
    UsePeerAccess(),
    ASSIGNABLE_FACTORY(MEMBER_CALL_FACTORY_ENTRY(
        "cudaDeviceEnablePeerAccess",
        CALL(MapNames::getDpctNamespace() + "get_current_device"), false,
        "ext_oneapi_enable_peer_access",
        CALL(MapNames::getDpctNamespace() + "get_device", ARG_WC(0)))),
    REMOVE_API_FACTORY_ENTRY_WITH_MSG(
        "cudaDeviceEnablePeerAccess",
        "SYCL currently does not support memory access across peer devices. "
        "You can migrate the code with peer access extension by not specifying "
        "-no-dpcpp-extensions=peer_access."))

CONDITIONAL_FACTORY_ENTRY(
    UsePeerAccess(),
    ASSIGNABLE_FACTORY(MEMBER_CALL_FACTORY_ENTRY(
        "cudaDeviceDisablePeerAccess",
        CALL(MapNames::getDpctNamespace() + "get_current_device"), false,
        "ext_oneapi_disable_peer_access",
        CALL(MapNames::getDpctNamespace() + "get_device", ARG_WC(0)))),
    REMOVE_API_FACTORY_ENTRY_WITH_MSG(
        "cudaDeviceDisablePeerAccess",
        "SYCL currently does not support memory access across peer devices. "
        "You can migrate the code with peer access extension by not specifying "
        "-no-dpcpp-extensions=peer_access."))

CONDITIONAL_FACTORY_ENTRY(
    UsePeerAccess(),
    ASSIGNABLE_FACTORY(MEMBER_CALL_FACTORY_ENTRY(
        "cuCtxEnablePeerAccess",
        CALL(MapNames::getDpctNamespace() + "get_current_device"), false,
        "ext_oneapi_enable_peer_access",
        CALL(MapNames::getDpctNamespace() + "get_device", ARG_WC(0)))),
    REMOVE_API_FACTORY_ENTRY_WITH_MSG(
        "cuCtxEnablePeerAccess",
        "SYCL currently does not support memory access across peer devices. "
        "You can migrate the code with peer access extension by not specifying "
        "-no-dpcpp-extensions=peer_access."))

CALL_FACTORY_ENTRY(
    "make_cudaExtent",
    CALL(DpctGlobalInfo::getCtadClass(MapNames::getClNamespace() + "range", 3),
         ARG(0), ARG(1), ARG(2)))

CALL_FACTORY_ENTRY("make_cudaPitchedPtr",
                   CALL(MapNames::getDpctNamespace() + "pitched_data", ARG(0),
                        ARG(1), ARG(2), ARG(3)))

CALL_FACTORY_ENTRY(
    "make_cudaPos",
    CALL(DpctGlobalInfo::getCtadClass(MapNames::getClNamespace() + "id", 3),
         ARG(0), ARG(1), ARG(2)))

CONDITIONAL_FACTORY_ENTRY(
    checkIsUSM(),
    CONDITIONAL_FACTORY_ENTRY(
        [](const CallExpr *CE) { return canOmitMemcpyWait(CE); },
        ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::device_ext,
            MEMBER_CALL_FACTORY_ENTRY("cudaMemcpy", QUEUESTR, false, "memcpy",
                                      MEM_ARG(0), MEM_ARG(1), ARG(2)))),
        ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::device_ext,
            MEMBER_CALL_FACTORY_ENTRY("cudaMemcpy",
                                      MEMBER_CALL(QUEUESTR, false, "memcpy",
                                                  MEM_ARG(0), MEM_ARG(1),
                                                  ARG(2)),
                                      false, "wait")))),
    ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
        HelperFeatureEnum::device_ext,
        CALL_FACTORY_ENTRY(
            "cudaMemcpy", CALL(MemoryMigrationRule::getMemoryHelperFunctionName(
                                   "memcpy", false),
                               MEM_ARG(0), MEM_ARG(1), ARG(2), ARG(3))))))

#define CUDA_FREE(NAME)                                                        \
  CONDITIONAL_FACTORY_ENTRY(                                                   \
      hasManagedAttr(0),                                                       \
      CONDITIONAL_FACTORY_ENTRY(                                               \
          checkIsUSM(),                                                        \
          CONDITIONAL_FACTORY_ENTRY(                                           \
              [](const CallExpr *) {                                           \
                return DpctGlobalInfo::isOptimizeMigration();                  \
              },                                                               \
              ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(                      \
                  HelperFeatureEnum::device_ext,                               \
                  CALL_FACTORY_ENTRY(                                          \
                      NAME, CALL(MapNames::getClNamespace() + "free",          \
                                 makeCombinedArg(                              \
                                     makeCombinedArg(ARG("*("), ARG_WC(0)),    \
                                     ARG(".get_ptr())")),                      \
                                 QUEUESTR)))),                                 \
              CONDITIONAL_FACTORY_ENTRY(                                       \
                  [](const CallExpr *) {                                       \
                    return DpctGlobalInfo::useNoQueueDevice();                 \
                  },                                                           \
                  MULTI_STMTS_FACTORY_ENTRY(                                   \
                      NAME, true, true, true, true,                            \
                      MEMBER_CALL(QUEUESTR, false, "wait_and_throw"),          \
                      CALL(MapNames::getClNamespace() + "free",                \
                           makeCombinedArg(                                    \
                               makeCombinedArg(ARG("*("), ARG_WC(0)),          \
                               ARG(".get_ptr())")),                            \
                           QUEUESTR)),                                         \
                  ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(                  \
                      HelperFeatureEnum::device_ext,                           \
                      CALL_FACTORY_ENTRY(                                      \
                          NAME,                                                \
                          CALL(MapNames::getDpctNamespace() +                  \
                                   (DpctGlobalInfo::useSYCLCompat()            \
                                        ? "wait_and_free"                      \
                                        : "dpct_free"),                        \
                               makeCombinedArg(                                \
                                   makeCombinedArg(ARG("*("), ARG_WC(0)),      \
                                   ARG(".get_ptr())")),                        \
                               QUEUESTR)))))),                                 \
          ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(                          \
              HelperFeatureEnum::device_ext,                                   \
              CALL_FACTORY_ENTRY(                                              \
                  NAME,                                                        \
                  CALL(MapNames::getDpctNamespace() +                          \
                           (DpctGlobalInfo::useSYCLCompat() ? "wait_and_free"  \
                                                            : "dpct_free"),    \
                       makeCombinedArg(makeCombinedArg(ARG("*("), ARG_WC(0)),  \
                                       ARG(".get_ptr())"))))))),               \
      CONDITIONAL_FACTORY_ENTRY(                                               \
          checkIsUSM(),                                                        \
          CONDITIONAL_FACTORY_ENTRY(                                           \
              [](const CallExpr *) {                                           \
                return DpctGlobalInfo::isOptimizeMigration();                  \
              },                                                               \
              ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(                      \
                  HelperFeatureEnum::device_ext,                               \
                  CALL_FACTORY_ENTRY(NAME,                                     \
                                     CALL(MapNames::getClNamespace() + "free", \
                                          ARG_WC(0), QUEUESTR)))),             \
              CONDITIONAL_FACTORY_ENTRY(                                       \
                  [](const CallExpr *) {                                       \
                    return DpctGlobalInfo::useNoQueueDevice();                 \
                  },                                                           \
                  MULTI_STMTS_FACTORY_ENTRY(                                   \
                      NAME, true, true, true, true,                            \
                      MEMBER_CALL(QUEUESTR, false, "wait_and_throw"),          \
                      CALL(MapNames::getClNamespace() + "free", ARG_WC(0),     \
                           QUEUESTR)),                                         \
                  ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(                  \
                      HelperFeatureEnum::device_ext,                           \
                      CALL_FACTORY_ENTRY(                                      \
                          NAME, CALL(MapNames::getDpctNamespace() +            \
                                         (DpctGlobalInfo::useSYCLCompat()      \
                                              ? "wait_and_free"                \
                                              : "dpct_free"),                  \
                                     ARG_WC(0), QUEUESTR)))))),                \
          ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(                          \
              HelperFeatureEnum::device_ext,                                   \
              CALL_FACTORY_ENTRY(                                              \
                  NAME,                                                        \
                  CALL(MapNames::getDpctNamespace() +                          \
                           (DpctGlobalInfo::useSYCLCompat() ? "wait_and_free"  \
                                                            : "dpct_free"),    \
                       ARG_WC(0)))))))
CUDA_FREE("cudaFree")
CUDA_FREE("cublasFree")
#undef CUDA_FREE
